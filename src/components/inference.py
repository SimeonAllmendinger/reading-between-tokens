import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from schemas import InferenceItem, IssueItem, IssueOwner, Statement
from settings import SETTINGS, LOGGER
from components.inference_hook import Inference_Hook


def get_inference_items(
    issues: list[IssueItem],
    model_id: str,
    path_activation_dir: str,
) -> list[InferenceItem]:
    """
    Get the inference items from the issues.
    
    Args:
    - issues: list[IssueItem]
        The issues to get the inference items from.
    - model_id: str
        The model id.
    - path_activation_dir: str
        The directory to save the activations.
    
    Returns:
    - inference_items: list[InferenceItem]
        The inference items.
    """
    
    # Get the dataloader
    issues_dataloader = DataLoader(issues, **SETTINGS.inference['dataloader'])
    
    inference_items = []
    
    inference_hook = Inference_Hook(
        model_id=model_id, 
        inference_config=SETTINGS.inference
    )
    
    checkpoint_path = os.path.join(SETTINGS.inference['CHECKPOINTS']['path_checkpoint_dir'], model_id, "inference_items.pt")
    if os.path.exists(checkpoint_path) and SETTINGS.inference['CHECKPOINTS']['load_checkpoint']:
        inference_items = torch.load(checkpoint_path)
        LOGGER.info(f"Loaded inference items from checkpoint: {checkpoint_path}")
    else:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
    for j, batch in enumerate(tqdm(issues_dataloader, desc=f"Model {model_id}")):
        
        if j*issues_dataloader.batch_size < len(inference_items):
            continue
        
        io_names = [batch["issue_owner"]["name_mapped"]]
        for io_name in io_names: # iterate over answer formats
            llm_responses, activations, _, logits = inference_hook.get_responses_and_activations(
                prompt_batch=batch["prompt"],
                answer_batch=io_name,
            )
            
            # Create all inference items for the batch in a single step
            batch_inference_items = [
                InferenceItem(
                    election_id=batch["election_id"][k],
                    issue_owner=IssueOwner(**{key: v[k] for key, v in batch["issue_owner"].items()}),
                    statement=Statement(**{key: v[k] for key, v in batch["statement"].items()}),
                    answer=batch["answer"][k],
                    comment=batch["comment"][k],
                    prompt=batch["prompt"][k],
                    prompt_variant_idx=batch["prompt_variant_idx"][k],
                    model_id=model_id,
                    llm_response=llm_response,
                    path_activation_dir=path_activation_dir,
                )
                for k, llm_response in enumerate(llm_responses)
            ]
                
            # Save activations as a batch
            for k, inference_item in enumerate(batch_inference_items):
                activation = activations["accum_resid"][k]
                logits_k = logits[k]
                torch.save(activation, inference_item.path_activation_file)
                torch.save(logits_k, inference_item.path_logits_file)

            inference_items.extend(batch_inference_items)
                
            # save Dataset as checkpoint every 1000 steps
            if j % 1000 == 0:
                torch.save(inference_items, checkpoint_path)
                
            del activations, llm_responses
    
    return inference_items