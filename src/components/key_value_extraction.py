import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from einops import einsum
from tqdm import tqdm
from collections import defaultdict

from components.data_handler import PersonaGenerator
from components.inference_hook import Inference_Hook
from schemas import ValueVector, DiametricValueVector, KeyValue, Probing, Persona
from settings import SETTINGS, LOGGER


class KeyValueExtractor():
    
    def __init__(
        self,
        country: str,
        model_id: str,
        probing: Probing,
        path_key_values_dir: str,
        path_personas_dir: str,
        persona_config: dict,
        batch_size: int = SETTINGS.key_value_extraction["batch_size"],
    ):
        self.probing = probing
        self.country = country
        self.path_key_values_dir = path_key_values_dir
        self.path_personas_dir = path_personas_dir
        self.model_id = model_id
        self.persona_config = persona_config
        self.batch_size = batch_size
        
        self.prompt_variants_template = SETTINGS.personas[self.country]["PROMPT_VARIANTS_TEMPLATE"]
        self.prompt_introduction = SETTINGS.personas[self.country]["PROMPT_INTRODUCTION"]
        self.issue_owner_names_mapped = list(SETTINGS.dataset[self.country]['name_mapping'].keys())
        self.checkpoint = SETTINGS.key_value_extraction["CHECKPOINT"]
        
        # Get value vectors for each issue owner name mapped
        LOGGER.info("Fetching value vectors for issue owner names...")
        self.value_vectors = {}
        for _io_name_mapped in self.issue_owner_names_mapped:
            # Get value vectors for each issue owner name mapped
            self.value_vectors[_io_name_mapped] = self.probing.get_all_value_vectors(
                model_id=self.model_id, issue_owner_name_mapped=_io_name_mapped
            )
            
        self.color_mapping = SETTINGS.dataset[self.country]['color_mapping']
        # file paths for incremental appends (created lazily)
        self._key_values_csv_path = None
 

    def _get_persona_combinations(self) -> list[dict]:
        """
        Generate all combinations of personas based on the persona configuration.
        
        Returns:
            tuple: A tuple containing two lists:
                - List of persona combinations (dicts) for the main personas.
        """
        persona_generator = PersonaGenerator(self.persona_config)
        
        persona_combinations = list(persona_generator.generate_combinations())
        
        return persona_combinations


    def _process_persona_batch(
        self,
        personas,
        batch_prompts,
        inference_hook: Inference_Hook,
        batch_answers: list[str] = None,
    ) -> tuple[list[KeyValue], list[str], list[np.ndarray]]:
        """
        Process a batch of personas and their corresponding prompts to extract key-value contributions.
        
        Args:
            personas (list[Persona]): List of Persona instances for the current batch.
            batch_prompts (list[str]): List of prompts corresponding to the personas.
            inference_hook (Inference_Hook): Inference hook for model interaction.
            value_vectors (dict[str, list[ValueVector]]): Value vectors for each issue owner name mapped.
        
        Returns:
            tuple: A tuple containing:
                - List of KeyValue contributions.
                - List of model responses.
                - List of logits.
        """
        
        key_value_list = []
        
        n_prompt_variants = int(len(batch_prompts) / len(personas))
        
        # Batched inference for prompts
        # activations shape: [batch_size, n_layers, seq_length, d_model]
        responses, activations, labels, logits = inference_hook.get_responses_and_activations(
            prompt_batch=batch_prompts, decompose_activations=True, answer_batch=batch_answers, system_prompt=self.prompt_introduction
        )
        
        # Process activations and calculate key-value contributions
        for _persona_k, persona in enumerate(personas):
            
            for _io_name_mapped in self.issue_owner_names_mapped:
                
                key_values = get_key_values_contribution_with_cache(
                    model=inference_hook.model,
                    decomposed_resid=activations["decompose_resid"][_persona_k*n_prompt_variants:(_persona_k+1)*n_prompt_variants],
                    activation_labels=labels,
                    responses=responses[_persona_k*n_prompt_variants:(_persona_k+1)*n_prompt_variants],
                    logits=logits[_persona_k*n_prompt_variants:(_persona_k+1)*n_prompt_variants],
                    persona=persona,
                    value_vectors=self.value_vectors[_io_name_mapped],
                    path_key_value_dir=self.path_key_values_dir,
                )
                
                key_value_list.extend(key_values)
        
        return key_value_list, responses, logits


    def _get_key_values(
        self,
        persona_batch,
        inference_hook: Inference_Hook,
        is_blocked_value_vectors: bool = False,
    ) -> list[KeyValue]:
        
        batch_prompts = []
        personas = []
        
        for _persona_combination in persona_batch:
            
            persona = Persona(
                **_persona_combination,
                path_persona_dir=self.path_personas_dir,
            )

            prompt_variants = [
                persona.get_prompt_variants(pv) 
                for pv in self.prompt_variants_template
            ]

            batch_prompts.extend(prompt_variants)
            personas.append(persona)
        
        key_values, responses, logits = self._process_persona_batch(
            personas=personas,
            batch_prompts=batch_prompts,
            inference_hook=inference_hook,
            batch_answers=None,
        )
        
        return key_values

    
    def _save_key_values_to_csv(
        self,
        key_values: list[KeyValue],
    ):
        # Save key_values to file
        if not os.path.exists(self.path_key_values_dir):
            os.makedirs(self.path_key_values_dir)
            LOGGER.info(f"Created directory: {self.path_key_values_dir}")
        
        path_results_dir = os.path.join(
            self.path_key_values_dir,
            self.country.lower(),
            self.model_id
        )
        
        if not os.path.exists(path_results_dir):
            os.makedirs(path_results_dir)
            LOGGER.info(f"Created directory: {path_results_dir}")
        
        LOGGER.info(f"Saving key_values to CSV file...")
        KeyValue.save_to_csv(
            key_values, 
            os.path.join(
                path_results_dir,
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_key_values.csv")
        )
    
    
    def _append_key_values_to_csv(self, key_values: list[KeyValue]):
        """
        Append a list of KeyValue objects to a CSV using the same flattened schema as KeyValue.to_dataframe().
        - If file doesn't exist: create it with header.
        - If file exists and schema matches: append without header.
        - If file exists and schema differs: read, concat and rewrite to preserve a consistent header.
        """
        if not key_values:
            return

        # ensure base dir exists
        if not os.path.exists(self.path_key_values_dir):
            os.makedirs(self.path_key_values_dir, exist_ok=True)

        path_results_dir = os.path.join(
            self.path_key_values_dir,
            self.country.lower(),
            self.model_id
        )
        os.makedirs(path_results_dir, exist_ok=True)

        # choose file path (keep same file for the whole run)
        csv_path_attr = "_key_values_csv_path"
        if getattr(self, csv_path_attr) is None:
            if self.checkpoint["FROM_EXISTING_KEY_VALUE_EXTRACTION"]:
                # load from existing checkpoint
                setattr(self, csv_path_attr, os.path.join(
                    path_results_dir,
                    f"{self.checkpoint['YEAR']}-{self.checkpoint['MONTH']:02d}-{self.checkpoint['DAY']:02d}_"
                    f"{self.checkpoint['HOUR']:02d}-{self.checkpoint['MINUTE']:02d}-{self.checkpoint['SECOND']:02d}"
                    f"_key_values.csv"
                ))
                LOGGER.debug(f"Loading key-values from existing checkpoint CSV: {getattr(self, csv_path_attr)}")
            else:
                setattr(self, csv_path_attr, os.path.join(
                    path_results_dir,
                    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_key_values.csv"
                ))
        csv_path = getattr(self, csv_path_attr)

        # Use KeyValue.to_dataframe to produce flat DataFrame with canonical schema
        try:
            df = KeyValue.to_dataframe(key_values)
        except Exception as e:
            LOGGER.error(f"Failed to convert KeyValue list to DataFrame: {e}")
            # fallback: best-effort via list comprehension of to_flat_dict if available
            try:
                rows = [kv.to_flat_dict() for kv in key_values]
                df = pd.DataFrame(rows)
            except Exception as e2:
                LOGGER.error(f"Fallback conversion also failed: {e2}")
                return

        try:
            if not os.path.exists(csv_path):
                # new file: write header
                df.to_csv(csv_path, mode="w", header=True, index=False)
                LOGGER.info(f"Created key-values CSV and wrote {len(df)} rows to {csv_path}")
            else:
                # existing file: check schema
                existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                if set(existing_cols) == set(df.columns):
                    # same columns -> reorder to match file and append
                    df = df.reindex(columns=existing_cols)
                    df.to_csv(csv_path, mode="a", header=False, index=False)
                    LOGGER.info(f"Appended {len(df)} key-values to {csv_path}")
                else:
                    # schema changed -> read entire file, concat, and rewrite (keeps consistent header)
                    df_existing = pd.read_csv(csv_path)
                    combined = pd.concat([df_existing, df], ignore_index=True, sort=False)
                    combined.to_csv(csv_path, index=False)
                    LOGGER.info(f"File schema changed; rewrote {csv_path} with {len(combined)} total rows")
        except Exception as e:
            LOGGER.error(f"Error while writing key-values CSV '{csv_path}': {e}")
    
    
    def _load_key_values_as_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Load key values from existing CSV file.
        """
        path_results_dir = os.path.join(
            self.path_key_values_dir,
            self.country.lower(),
            self.model_id
        )
        
        csv_path = os.path.join(
            path_results_dir,
            f"{self.checkpoint['YEAR']}-{self.checkpoint['MONTH']:02d}-{self.checkpoint['DAY']:02d}_"
            f"{self.checkpoint['HOUR']:02d}-{self.checkpoint['MINUTE']:02d}-{self.checkpoint['SECOND']:02d}"
            f"_key_values.csv"
        )
        
        if not os.path.exists(csv_path):
            LOGGER.error(f"Key values CSV file does not exist: {csv_path}")
            return []
        
        LOGGER.info(f"Loading key values from CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        return df


    def extract(self):
        """
        Main method to extract key-value contributions for all persona combinations.
        """
        
        inference_hook = Inference_Hook(model_id=self.model_id, inference_config=SETTINGS.inference)
        
        # Generate persona combinations (every combination of persona attributes)
        persona_combinations = self._get_persona_combinations()
        
        # RESUME FROM CHECKPOINT IF APPLICABLE. Remove already processed personas.
        if self.checkpoint["FROM_EXISTING_KEY_VALUE_EXTRACTION"]:
            df = self._load_key_values_as_dataframe()
            # Resume after the last persona found in the CSV.
            original_n = len(persona_combinations)
            if df.empty:
                LOGGER.info("Resume check: key values dataframe is empty; no personas to skip.")
            else:
                persona_cols = [
                    "persona.age",
                    "persona.gender",
                    "persona.education",
                    "persona.employment",
                    "persona.hhincome",
                    "persona.left_leaning",
                    "persona.immigration",
                    "persona.inequality",
                    "persona.year_of_election",
                ]
                missing_persona_cols = [c for c in persona_cols if c not in df.columns]
                if missing_persona_cols:
                    LOGGER.warning(
                        "Resume check: required persona columns missing from CSV; will not skip. Missing=%s",
                        ", ".join(missing_persona_cols),
                    )
                else:
                    persona_keys = [c.replace("persona.", "", 1) for c in persona_cols]
                    persona_df = df[persona_cols].astype(str)
                    # Count unique personas in order of appearance and skip that many combinations.
                    unique_personas = persona_df.drop_duplicates(keep="first")
                    n_processed = min(len(unique_personas), len(persona_combinations))
                    if n_processed == 0:
                        LOGGER.warning(
                            "Resume check: no unique personas found in CSV; will not skip."
                        )
                    else:
                        persona_combinations = persona_combinations[n_processed:]
                        LOGGER.warning(
                            "Resuming from checkpoint: %s personas already processed, %s remaining.",
                            n_processed,
                            len(persona_combinations),
                        )
            if len(persona_combinations) == original_n:
                LOGGER.warning(
                    "Resuming from checkpoint: 0 personas already processed, %s remaining.",
                    len(persona_combinations),
                )
        
        #############################
        # PROCESS PERSONAS IN BATCHES
        #############################
        
        LOGGER.info("Processing personas...")
        persona_batches = [
            persona_combinations[
                i:i+self.batch_size] for i in range(
                    0, len(persona_combinations), self.batch_size
                )
        ]

        for _persona_batch in tqdm(persona_batches, desc="Processing persona batches"):

            key_values = self._get_key_values(
                    persona_batch=_persona_batch,
                    inference_hook=inference_hook,
            )
            # append to CSV immediately and free memory
            self._append_key_values_to_csv(key_values)


@torch.no_grad()
def get_key_values_contribution_with_cache(
    model,
    decomposed_resid: torch.tensor, # torch.tensor(batch_size, n_layers, d_model)
    activation_labels: list[str],
    responses: list[str],
    logits: list[np.ndarray],
    value_vectors: list[ValueVector],
    persona: Persona,
    path_key_value_dir: str,
) -> list[KeyValue]:
    """
    Analyze how key vectors contribute to the emphasis of specific value vectors in the generation process.

    Args:
        model: The model containing the transformer layers.
        decomposed_resid: Residual activations decomposed into layers and MLPs.
        activation_labels: List of labels for the activations.
        value_vectors: Dict[List[ValueProbe]] containing layer and MLP indices for value vectors.
        persona: Persona instance for the current persona configuration.
        path_key_value_dir: Path to save key-value information.

    Returns:
        List[KeyValue]: Contributions of each value vector.
    """
    key_values = []

    # Extract unique layers from all value_vectors
    layers = list(set([v.layer for v in value_vectors]))
    layers.sort()  # Sort layers for consistent processing

    # Prepare data structures for batched processing
    all_layer_decomposed_resid = {layer: torch.tensor(decomposed_resid[:,activation_labels.index(f"{layer}_attn_out")]) for layer in layers}
    W_in = {layer: model.blocks[layer].mlp.W_in for layer in layers}
    b_in = {layer: model.blocks[layer].mlp.b_in for layer in layers}
    act_fn = {layer: model.blocks[layer].mlp.act_fn for layer in layers}
    ln_fn = {layer: model.blocks[layer].mlp.ln if hasattr(model.blocks[layer].mlp, 'ln') else None for layer in layers}
    
    sum_value_vector_cos_similarity = sum(v.cos_similarity for v in value_vectors if isinstance(v, ValueVector))
    sum_diametric_value_vector_cos_similarity = sum(v.cos_similarity for v in value_vectors if isinstance(v, DiametricValueVector))

    LOGGER.debug(f"Sum of value vector cos similarities: {sum_value_vector_cos_similarity}")
    LOGGER.debug(f"Sum of diametric value vector cos similarities: {sum_diametric_value_vector_cos_similarity}")
    
    # Process each value_vector in batches by layer
    for layer in layers:
        layer_decomposed_resid = all_layer_decomposed_resid[layer].cuda().to(W_in[layer].dtype) # [batch_size, d_model]
        W_in_layer = W_in[layer].cuda() # [d_model, d_mlp]
        b_in_layer = b_in[layer].cuda() # [d_mlp]
        
        # Filter value vectors for the current layer
        layer_value_vectors = [v for v in value_vectors if v.layer == layer]
        mlp_indices = torch.tensor([v.mlp for v in layer_value_vectors], device="cuda")  # Batch MLP indices

        # Calculate pre-activation
        pre_act = einsum(
            layer_decomposed_resid,
            W_in_layer,
            "batch d_model, d_model d_mlp -> batch d_mlp"
        ) + b_in_layer

        # Apply activation function
        if act_fn[layer] is not None and not act_fn[layer].__name__.endswith("_ln"):
            post_act = act_fn[layer](pre_act)
        else:
            mid_act = act_fn[layer](pre_act)
            post_act = ln_fn[layer](mid_act) if ln_fn[layer] else mid_act
        
        # Select only the MLP indices corresponding to the current layer's value vectors
        post_act_mlps  = post_act[:, mlp_indices] # Shape: [batch, len(mlp_indices)] -> [batch, len(mlp_indices)]
        
        # Calculate metrics for all value_vectors and persona prompt variants in the current layer    
        for i, value_vector in enumerate(layer_value_vectors):
            for prompt_idx in range(decomposed_resid.shape[0]):
            
                key_value = post_act_mlps[prompt_idx, i].detach().cpu().item()

                probs = torch.softmax(logits[prompt_idx].float(), dim=-1)
                target_token_ids = model.tokenizer.encode(value_vector.value_probe.issue_owner.name_mapped)[1:]  # Exclude BOS token
                mean_prob = np.mean([probs[0, token_id].item() for token_id in target_token_ids])
                first_prob = probs[0, target_token_ids[0]].item()
                first_3_probs = np.mean([probs[0, token_id].item() for token_id in target_token_ids[:3]])

                # Determine scaling factor based on vector type
                if isinstance(value_vector, DiametricValueVector):
                    scale = -1 * value_vector.cos_similarity / sum_diametric_value_vector_cos_similarity
                    LOGGER.debug(f"Diametric: kv: {key_value}, scale: {scale}")
                if isinstance(value_vector, ValueVector):
                    scale = value_vector.cos_similarity / sum_value_vector_cos_similarity
                    LOGGER.debug(f"Value Vector: kv: {key_value}, scale: {scale}")
                
                key_values.extend([
                    KeyValue(
                        value_vector=value_vector,
                        persona=persona,
                        value=key_value,
                        scale=scale,
                        mean_probability=mean_prob,
                        first_probability=first_prob,
                        first_3_probability=first_3_probs,
                        persona_prompt_variant_idx=prompt_idx,
                        response=responses[prompt_idx],
                        path_key_value_dir=path_key_value_dir,
                    )
                ]) 

    return key_values
