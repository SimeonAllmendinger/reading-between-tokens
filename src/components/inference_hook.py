import torch
import gc
import os
import deepspeed

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from torch import Tensor

from typing import Tuple, Any, Dict, Optional
from settings import Settings

SETTINGS = Settings()
LOGGER = SETTINGS.logger()

HF_API_TOKEN = SETTINGS.api_keys["huggingface"]["api_token"]
# set huggingface token in environment variable
os.environ["HF_TOKEN"] = HF_API_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def rms_normalize(x, epsilon=1e-5):
    return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + epsilon)

class Inference_Hook:
    """
    Inference Hook using DeepSpeed for efficient model inference and activation extraction.
    Supports tokenization, prompt formatting, and extraction of model activations.
    """
    def __init__(
        self, 
        model_id: str, 
        inference_config: dict):

        self.inference_config = inference_config
        self.max_tokens_generated = inference_config['inference_hook']['max_tokens_generated']
        self.model_id = model_id
        hook_cfg = inference_config.get("inference_hook", {})
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize DeepSpeed inference (ZeRO-3 for large models)
        # Prefer lower precision to save GPU memory. Use bfloat16 if supported, else float16.
        self.deepspeed_dtype = torch.bfloat16
        self.input_device = hook_cfg.get("input_device")
        if self.input_device is None:
            self.input_device = "cuda" if torch.cuda.is_available() else "cpu"

        mp_size = hook_cfg.get("deepspeed_tp_size", max(1, torch.cuda.device_count()))
        
        # from_pretrained_no_processing(): This method skips all of the post-processing steps 
        # mentioned above. It simply loads the model’s weights without performing any additional 
        # transformations or optimizations. This is useful if you want to work with the raw model 
        # state exactly as it was trained, without any further manipulation.
        self.use_transformer_lens = True
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.model_id,
            device="cpu",  # Load model to CPU first before moving to DeepSpeed
            dtype=self.deepspeed_dtype,
            default_padding_side="right",
        )
        
        self.model = deepspeed.init_inference(
            self.model,
            dtype=self.deepspeed_dtype,
            tensor_parallel={"tp_size": mp_size},
            replace_with_kernel_inject=True
        ).module
        
        self.assistant_start_token_ids = inference_config['activations'][self.model_id]['assistant_start_token_ids'] if inference_config['activations'][self.model_id].get('is_instruct_model', False) else self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" \n\n"))
        self.assistant_stop_token_ids = inference_config['activations'][self.model_id]['assistant_stop_token_ids'] if inference_config['activations'][self.model_id].get('is_instruct_model', False) else [self.tokenizer.eos_token_id]
        LOGGER.debug(f"Assistant start token IDs: {self.assistant_start_token_ids}")
        LOGGER.debug(f"Assistant stop token IDs: {self.assistant_stop_token_ids}")
    
    def tokenize_prompts(
        self,
        messages: list[list[dict[str, str]]],
        padding: bool, 
        truncation: bool, 
        return_tensors: str,
        max_seq_length: int,
        padding_side: str = 'right',
        add_generation_prompt: bool = False,
        ) -> Tensor:

        # Try to use chat template if available and set, else fallback to simple concatenation
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None) is not None
        ):
            prompts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        else:
            prompts = []
            for message in messages:
                prompt = ""
                for msg in message:
                    if msg['role'] == 'system':
                        prompt += f"{msg['content']} "
                    elif msg['role'] == 'user':
                        prompt += f"{msg['content']} \n\n"
                    elif msg['role'] == 'assistant' or msg['role'] == 'model':
                        prompt += f"\n## Response:\n\n{msg['content'][1:]}"
                if add_generation_prompt:
                    prompt += "\n## Response:\n\n"
                    prompts.append(prompt)
                else:
                    prompts.append(prompt.strip())

        enc = self.tokenizer(
            prompts, 
            padding=padding, 
            truncation=truncation, 
            return_tensors=return_tensors,
            max_length=max_seq_length,
            padding_side=padding_side,
            return_attention_mask=True,
        )
        
        token_ids = enc['input_ids']  # Tensor of shape [batch_size, seq_length]
        attention_masks = enc['attention_mask']  # Tensor of shape [batch_size, seq_length]
            
        return token_ids, attention_masks # [batch_size, seq_length], [batch_size, seq_length]
    
    @torch.no_grad()
    def get_responses_and_activations(
        self,
        prompt_batch: Tuple[str],
        answer_batch: Tuple[str] = None,
        decompose_activations: bool = False,
        return_only_logits: bool = False,
        system_prompt: str = SETTINGS.inference['DEFAULT_SYSTEM_PROMPT'],
    ):
            
        # tokenize messages
        messages = []
        for i, prompt in enumerate(prompt_batch):
            if 'llama' in self.model_id.lower() or 'mistral' in self.model_id.lower():
                messages.append(
                    [
                        # if model has system prompt support, add it, else skip
                        {
                            "role": "system", 
                            "content": f"{system_prompt}"
                        },
                        {
                            "role": "user", 
                            "content": f"{prompt}"
                        },
                    ]
                )
            else:
                messages.append(
                    [
                        {
                            "role": "user", 
                            "content": f"{system_prompt} {prompt}"
                        },
                    ]
                )
            
        q_tokens, q_att_masks = self.tokenize_prompts(
            messages=messages,
            **self.inference_config['tokenizer'],
            add_generation_prompt=True
        )
        
        device = torch.device(self.input_device)
        
        n_layer = self.inference_config['activations'][self.model_id]['n_resid_pre_layers']
        embedding_dim = self.inference_config['activations'][self.model_id]['embedding_dim']
        
        if answer_batch is not None:
            
            for i, answer in enumerate(answer_batch):
                # Llama-like models need specific formatting for answers
                if self.model_id.endswith("Instruct") and self.inference_config['activations'][self.model_id].get('is_instruct_model', False):
                    answer = f"\n\n{answer}"
                # Google Gemma instruct models need specific formatting for answers
                elif self.model_id.endswith("it") and self.inference_config['activations'][self.model_id].get('is_instruct_model', False):
                    answer = f"\n{answer}"
                elif self.model_id.startswith("Qwen") and self.inference_config['activations'][self.model_id].get('is_instruct_model', False):
                    answer = f"\n{answer}"
                else:
                    answer = f" {answer}"
                
                messages[i].append(
                    {
                        "role": "assistant" if "llama" in self.model_id.lower() or 'mistral' in self.model_id.lower() or 'qwen' in self.model_id.lower() else "model", 
                        "content": answer
                    },
                )
            
            qa_tokens, qa_att_masks = self.tokenize_prompts(
                messages=messages,
                **self.inference_config['tokenizer'],
                add_generation_prompt=False
            )
            
            qa_tokens = qa_tokens.to(device)
        
        else:
            qa_tokens = q_tokens.to(device)

        # Find the position of the assistant start tokens in qa_tokens and subtract 1 for index
        assistant_start_positions = q_att_masks.sum(dim=-1).cpu().tolist()
        assistant_stop_positions = []
        for k, token_ids in enumerate(qa_tokens):
            token_ids_list = token_ids.tolist()
            # Remove all stop_token_ids that appear after the assistant start position
            response_ids_list = [_id for _id in token_ids_list[assistant_start_positions[k]:] if _id not in self.assistant_stop_token_ids]
            assistant_stop_positions.append(assistant_start_positions[k] + len(response_ids_list))

        torch.cuda.empty_cache()
        gc.collect()

        if return_only_logits:
            logits = self.model(qa_tokens).cpu().detach()
            logits_batch = []
            for i, (_start_pos, _stop_pos) in enumerate(zip(assistant_start_positions, assistant_stop_positions)):
                sample_logits = logits[i, _start_pos-1:_stop_pos-1].cpu().detach()
                logits_batch.append(sample_logits)
            return logits_batch
        
        logits, cache = self.model.run_with_cache(qa_tokens, return_cache_object=True)

        if decompose_activations:
            # get decompose resid for all qa_tokens -> n_layer, batch_size, sequence_length, d_model
            decompose_resid, labels = cache.decompose_resid(
                layer=n_layer-1,
                mlp_input = True,
                mode = "attn",
                apply_ln = True,
                pos_slice = None,
                incl_embeds= False,
                return_labels= True,
            )
            
            # Check if logits and resids have the same sequence length
            if decompose_resid.shape[2] != logits.shape[1]:
                LOGGER.error(f"Shape mismatch: decompose_resid shape {decompose_resid.shape}, logits shape {logits.shape}")
            
            # Extract activations for each sample starting from its assistant position
            all_layer_decomposed_first_pos = []
            logits_batch = []
            
            # First pass: collect activations and find max length
            for i, (_start_pos, _stop_pos) in enumerate(zip(assistant_start_positions, assistant_stop_positions)):
                # n_layer, batch_size, sequence_length, d_model -> n_layer*3 (pre,mid,post), sequence_length, d_model
                sample_activation = decompose_resid[:, i, _start_pos-1, :].cpu().detach()
                # n_layer*3 (pre,mid,post), sequence_length, d_model -> batch in list: [n_layer*3 (pre,mid,post), sequence_length, d_model]
                all_layer_decomposed_first_pos.append(sample_activation)
                
                sample_logits = logits[i, _start_pos-1:_stop_pos].cpu().detach()
                logits_batch.append(sample_logits)

            # batch_size, n_layer, sequence_length, d_model
            resids['decompose_resid'] = torch.stack(all_layer_decomposed_first_pos, dim=0).numpy()
            
            response_batch = ['' for item in range(len(logits_batch))]
            
        else:
            if any(pos[0] == pos[1] for pos in zip(assistant_start_positions, assistant_stop_positions)):
                LOGGER.warning(f"Assistant start positions: {assistant_start_positions} of {[len(qa) for qa in qa_tokens]}")
                LOGGER.warning(f"Assistant stop positions: {assistant_stop_positions} of {[len(qa) for qa in qa_tokens]}")
                LOGGER.warning("Some assistant start positions are equal to stop positions. This may lead to empty activations.")
                LOGGER.warning(f"answer_batch: {answer_batch if answer_batch is not None else self.tokenizer.batch_decode(qa_tokens, skip_special_tokens=False)}")

            resids = {
            'accum_resid': torch.zeros(
                (q_tokens.shape[0], n_layer, embedding_dim),
                 dtype=self.deepspeed_dtype,
                 device='cpu')
            }
            
            # get accum resid for all qa_tokens -> n_layer, batch_size, sequence_length, d_model
            accum_resid, labels = cache.accumulated_resid(layer=n_layer,
                                                          return_labels=True, 
                                                          apply_ln=False,
                                                          )
            
            # Check if logits and resids have the same sequence length
            if accum_resid.shape[2] != logits.shape[1]:
                LOGGER.error(f"Shape mismatch: accum_resid shape {accum_resid.shape}, logits shape {logits.shape}")
            
            # Extract activations for each sample starting from its assistant position
            all_layer_decomposed_mean_pos = []
            response_batch = []
            logits_batch = []
            
            # First pass: collect activations and find max length
            for i, (_start_pos, _stop_pos) in enumerate(zip(assistant_start_positions, assistant_stop_positions)):
                # n_layer, batch_size, sequence_length, d_model -> n_layer, d_model
                sample_activation = accum_resid[:, i, _start_pos-1:_stop_pos-1, :].mean(dim=1).cpu().detach()
                # n_layer, d_model -> batch in list: [n_layer, d_model]
                all_layer_decomposed_mean_pos.append(sample_activation)
                
                sample_response = self.tokenizer.decode(
                    qa_tokens[i, _start_pos:_stop_pos], skip_special_tokens=False
                )
                response_batch.append(sample_response)
                
                sample_logits = logits[i, _start_pos-1:_stop_pos-1].cpu().detach()
                logits_batch.append(sample_logits)
            
            # batch_size, n_layer, d_model
            resids['accum_resid'] = torch.stack(all_layer_decomposed_mean_pos, dim=0)

        del cache, qa_tokens, q_tokens
        torch.cuda.empty_cache()
        gc.collect()
        
        return response_batch, resids, labels, logits_batch
