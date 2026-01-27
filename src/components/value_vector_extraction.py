import os
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from einops import einsum
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import List
from transformer_lens import HookedTransformer

from components.inference_hook import Inference_Hook
from schemas import Probing, Dataset, ValueVector, DiametricValueVector, ValueProbe
from settings import SETTINGS, LOGGER

class ValueVectorExtractor:
    
    def __init__(
        self,
        probing: Probing,
        dataset: Dataset,
        path_value_vector_dir: str,
        top_k_tokens: int = SETTINGS.probing['Value_Probe']['top_k_tokens'],
        iqr_ratio: float = SETTINGS.value_vector_extraction['iqr_ratio'],
        layer_min_ratio: float = SETTINGS.probing['Value_Probe']["activation_layer"]["min_ratio"],
        layer_max_ratio: float = SETTINGS.probing['Value_Probe']["activation_layer"]["max_ratio"],
        n_data_items_in_vector_effect_analysis: int = SETTINGS.value_vector_extraction['n_data_items_in_vector_effect_analysis'],
    ):
        self.probing = probing
        self.dataset = dataset
        self.path_value_vector_dir = path_value_vector_dir
        
        self.top_k_tokens = top_k_tokens
        self.iqr_ratio = iqr_ratio
        self.layer_min_ratio = layer_min_ratio
        self.layer_max_ratio = layer_max_ratio

        self.sample_idxs = np.random.choice(
            len(self.dataset.data), 
            size=min(n_data_items_in_vector_effect_analysis, len(self.dataset.data)), 
            replace=False,
        )
        self.sample_idxs = [int(i) for i in self.sample_idxs]
    
    @staticmethod
    def rms_normalize(x, epsilon=1e-5):
        return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + epsilon)

    
    def _get_upper_lower_bounds(self, cos_sims_np: np.ndarray):
        """
        Calculate the upper and lower bounds for outlier detection using IQR.
        
        Args:
            cos_sims_np (np.ndarray): Array of cosine similarities.
            
        Returns:
            upper_bound (float): Upper bound for outlier detection.
            lower_bound (float): Lower bound for outlier detection.
        """
        
        # Calculate IQR for outlier detection
        q1 = np.percentile(cos_sims_np, 25)
        q3 = np.percentile(cos_sims_np, 75)
        iqr = q3 - q1
        
        # Calculate upper and lower bounds
        upper_bound = q3 + self.iqr_ratio * iqr
        lower_bound = q1 - self.iqr_ratio * iqr

        return upper_bound, lower_bound


    def _get_cos_sim_scores_per_layer(
        self,
        model: HookedTransformer,
        probe_model_weights: torch.Tensor,
        layer: int,
        scores: list,
        diametric_scores: list
    ) -> tuple[list, list, np.ndarray]:
        """
        Get value vectors and diametric value vectors for a specific layer.
        
        Args:
            model (HookedTransformer): The transformer model.
            probe_model_weights (torch.Tensor): Weights of the value probe model.
            layer (int): The layer index to analyze.
            scores (list): List to append value vector scores.
            diametric_scores (list): List to append diametric value vector scores.
        
        Returns:
            scores (list): Updated list of value vector scores.
            diametric_scores (list): Updated list of diametric value vector scores.
            cos_sims_np (np.ndarray): Array of cosine similarities for the layer.
        """
        
        mlp_outs = model.blocks[layer].mlp.W_out.detach().cpu().float()
        probe_weights_cpu = probe_model_weights.detach().cpu().float()

        # Calculate cosine similarities between probe weights and MLP output weights
        cos_sims = F.cosine_similarity(mlp_outs, probe_weights_cpu, dim=1)
        
        # Calculate IQR for outlier detection
        cos_sims_np = cos_sims.cpu().numpy()

        # Calculate IQR for outlier detection
        upper_bound, lower_bound = self._get_upper_lower_bounds(cos_sims_np=cos_sims_np)

        # Value vectors: positive outliers
        value_outlier_idxs = np.where(cos_sims_np > upper_bound)[0]
        value_outlier_scores = cos_sims_np[value_outlier_idxs]
        value_vectors_outliers = list(zip(value_outlier_scores, value_outlier_idxs, [layer] * len(value_outlier_idxs)))
        scores.extend(value_vectors_outliers)

        # Diametric value vectors: negative outliers
        diametric_outlier_idxs = np.where(cos_sims_np < lower_bound)[0]
        diametric_outlier_scores = cos_sims_np[diametric_outlier_idxs]
        diametric_vectors_outliers = list(zip(diametric_outlier_scores, diametric_outlier_idxs, [layer] * len(diametric_outlier_idxs)))
        diametric_scores.extend(diametric_vectors_outliers)

        return scores, diametric_scores, cos_sims_np

    
    def _get_cos_sim_scores(
        self,
        model: HookedTransformer,
        probe_model_weights: torch.Tensor,
    ):
        """
        Get value vectors and diametric value vectors across all layers.
        
        Args:
            model (HookedTransformer): The transformer model.
            probe_model_weights (torch.Tensor): Weights of the value probe model.
        
        Returns:
            scores (list): List of value vector scores.
            diametric_scores (list): List of diametric value vector scores.
            cos_sims_np_dict (dict): Dictionary of cosine similarities per layer.
        """
        
        scores = []
        diametric_scores = []
        cos_sims_np_dict = {}
        
        min_layer = int(self.layer_min_ratio * model.cfg.n_layers - 1)
        max_layer = int(self.layer_max_ratio * model.cfg.n_layers - 1)

        for layer in range(min_layer, max_layer + 1):

            scores, diametric_scores, cos_sims_np = self._get_cos_sim_scores_per_layer(
                model=model,
                probe_model_weights=probe_model_weights,
                layer=layer,
                scores=scores,
                diametric_scores=diametric_scores
            )
            
            cos_sims_np_dict[layer] = cos_sims_np

        LOGGER.info(f"Number of value vectors: {len(scores)}")
        LOGGER.info(f"Number of diametric value vectors: {len(diametric_scores)}")
        
        return scores, diametric_scores, cos_sims_np_dict


    def _get_vectors(
            self,
            sorted_scores: list,
            value_probe: ValueProbe,
            model: HookedTransformer,
            is_diametric: bool = False
        ) -> List[ValueVector] | List[DiametricValueVector]:
        """
        Get value vectors or diametric value vectors based on the sorted scores.
        
        Args:
            sorted_scores (list): List of sorted scores (cos_similarity, mlp_idx, layer).
            value_probe (ValueProbe): The value probe associated with the vectors.
            model (HookedTransformer): The transformer model.
            is_diametric (bool): Whether to extract diametric value vectors.
        
        Returns:
            List[ValueVector] | List[DiametricValueVector]: List of extracted value vectors.
        """
        
        value_vectors = []
        for _vector_id, score in enumerate(sorted_scores):
            
            if is_diametric:
                
                LOGGER.info(f"Diametric Layer: {score[2]}, MLP: {score[1]}, Score: {score[0]}")
                
                value_vector = DiametricValueVector(
                    value_probe=value_probe,
                    cos_similarity=score[0],
                    layer=score[2],
                    mlp=score[1],
                    path_value_vector_dir=self.path_value_vector_dir,
                )
            if not is_diametric:
                
                LOGGER.info(f"Layer: {score[2]}, MLP: {score[1]}, Score: {score[0]}")
                
                value_vector = ValueVector(
                    value_probe=value_probe,
                    cos_similarity=score[0],
                    layer=score[2],
                    mlp=score[1],
                    path_value_vector_dir=self.path_value_vector_dir,
                )
            
            torch.save(
                model.blocks[score[2]].mlp.W_out[score[1]],
                value_vector.path_value_vector)
            
            value_vectors.append(value_vector)
            
        return value_vectors
    

    @torch.no_grad()
    def _get_vector_effect(
        self,
        inference_hook: Inference_Hook,
        layer_idx: int,
        mlp_idx: int,
        io_token_ids: dict[str, torch.Tensor],
        effect_factor: float = SETTINGS.value_vector_extraction['vector_effect_factor'],
        batch_size: int = SETTINGS.value_vector_extraction['batch_size'],
    ):
        """
        Get the effect of a value vector on model predictions, if the vector is distorted by a certain factor.
        
        Args:
            inference_hook (Inference_Hook): The inference hook for the model.
            layer_idx (int): The layer index of the value vector.
            mlp_idx (int): The MLP index of the value vector.
            io_token_ids (dict): Dictionary mapping issue owner names to token IDs.
            effect_factor (float): The factor by which to distort the value vector.
            batch_size (int): The batch size for processing.
        
        Returns:
            vector_effects (dict): Dictionary mapping issue owner names to their vector effects.
        """
        
        inference_hook.model.blocks[layer_idx].mlp.W_out[mlp_idx] = inference_hook.model.blocks[layer_idx].mlp.W_out[mlp_idx] * effect_factor
        vector_effects = {io_name: [] for io_name in io_token_ids.keys()}
        
        # Iterate over precomputed sample indices and index into dataset.data using ints.
        num_samples = len(self.sample_idxs)
        for batch_start in range(0, num_samples, batch_size):
            
            batch_idxs = self.sample_idxs[batch_start:batch_start + batch_size]
            batch_data_items = [self.dataset.data[_idx] for _idx in batch_idxs if self.dataset.data[_idx].issue_owner.name_mapped in io_token_ids]
            batch_prompts = [_data_item.prompt for _data_item in batch_data_items]
            batch_answers = [_data_item.issue_owner.name_mapped for _data_item in batch_data_items]

            # LOGITS WITH VECTOR DISTORTION
            logits_distorted_batch = inference_hook.get_responses_and_activations(
                prompt_batch=batch_prompts,
                answer_batch=batch_answers,
                return_only_logits=True
            )

            for j, (_data_item, logits_distorted) in enumerate(zip(batch_data_items, logits_distorted_batch)):

                # Compare the logits
                token_ids = io_token_ids[_data_item.issue_owner.name_mapped]
                
                # LOGITS WITHOUT VECTOR DISTORTION
                logits_initial = torch.load(_data_item.path_logits_file, weights_only=False, map_location="cpu")
                
                # Check if logits lengths and shapes match
                if len(logits_distorted.shape) != len(logits_initial.shape):
                    raise ValueError(f"Logits length mismatch: distorted {len(logits_distorted)}, initial {len(logits_initial)}")
                
                if logits_distorted[0].shape != logits_initial[0].shape:
                    raise ValueError(f"Logits shape mismatch at index 0: distorted {logits_distorted[0].shape}, initial {logits_initial[0].shape}")
                
                # Convert logits to probabilities using softmax.
                ld = logits_distorted[0]
                if isinstance(ld, np.ndarray):
                    ld = torch.from_numpy(ld)
                logits_distorted_probs = torch.softmax(ld.detach().cpu().float(), dim=-1)

                li = logits_initial[0]
                if isinstance(li, np.ndarray):
                    li = torch.from_numpy(li)
                logits_initial_probs = torch.softmax(li.detach().cpu().float(), dim=-1)
                
                for _pos, _token_id in enumerate(token_ids[0][1:]):
                    
                    if _pos >= logits_initial.shape[0]:
                        LOGGER.warning(f"Position {_pos} exceeds logits_initial sequence length {len(logits_initial)}. Skipping this position.")
                        LOGGER.warning(f"Token: {inference_hook.tokenizer.convert_ids_to_tokens(_token_id.item())}")
                        LOGGER.warning(f"Best token at this position: {inference_hook.tokenizer.convert_ids_to_tokens(torch.argmax(logits_initial[_pos]).item())}")
                        break

                    if _pos >= logits_distorted.shape[0]:
                        LOGGER.warning(f"Position {_pos} exceeds logits_distorted sequence length {len(logits_distorted)}. Skipping this position.")
                        LOGGER.warning(f"Token: {inference_hook.tokenizer.convert_ids_to_tokens(_token_id.item())}")
                        LOGGER.warning(f"Best token at this position: {inference_hook.tokenizer.convert_ids_to_tokens(torch.argmax(logits_distorted[_pos]).item())}")
                        break
                    
                    prob_diff = logits_distorted_probs[_token_id].item()-logits_initial_probs[_token_id].item()
                    prob_realtive_ratio = prob_diff / logits_initial_probs[_token_id].item()
                    vector_effects[_data_item.issue_owner.name_mapped].append(prob_realtive_ratio)

        inference_hook.model.blocks[layer_idx].mlp.W_out[mlp_idx] = inference_hook.model.blocks[layer_idx].mlp.W_out[mlp_idx] / effect_factor
        
        vector_effects = {k: np.median(v) if len(v) > 0 else 0.0 for k, v in vector_effects.items()}
        return vector_effects
    
    
    def _filter_vectors_by_effect(
        self,
        io_name_mapped: str,
        sorted_scores: list,
        inference_hook: Inference_Hook,
        io_token_ids: dict[str, torch.Tensor],
        effect_factor: float = SETTINGS.value_vector_extraction['vector_effect_factor'],
    ) -> list:
        """
        Filter value vectors based on their effect on model predictions.
        
        Args:
            io_name_mapped (str): The mapped name of the issue owner.
            sorted_scores (list): List of sorted scores (cos_similarity, mlp_idx, layer).
            inference_hook (Inference_Hook): The inference hook for the model.
            io_token_ids (dict): Dictionary mapping issue owner names to token IDs.
            effect_factor (float): The factor by which to distort the value vector.
        
        Returns:
            list: Filtered list of scores based on vector effect.
        """
        
        filtered_scores = []
        
        for score in tqdm(sorted_scores, desc=f"Filtering vectors for {io_name_mapped}"):
            
            vector_effects = self._get_vector_effect(
                inference_hook=inference_hook,
                layer_idx=score[2],
                mlp_idx=score[1],
                io_token_ids=io_token_ids
            )
            
            if effect_factor <= 0:
                median_effect = vector_effects.get(io_name_mapped, 0.0)
            else:
                median_effect = -vector_effects.get(io_name_mapped, 0.0)
                
            if median_effect >= 0:
                LOGGER.debug(f"Vector effect for {io_name_mapped} is not negative ({median_effect}). Skipping.")
                continue
            
            # Check if the median effect is less than all other issue owners
            if all(median_effect < v for k, v in vector_effects.items() if k != io_name_mapped):
                LOGGER.debug(f"Strongest effect for {io_name_mapped}: {median_effect} vs others { {k: v for k, v in vector_effects.items() if k != io_name_mapped} }")
            
            filtered_scores.append(score)

            LOGGER.debug(f"Layer: {score[2]}, MLP: {score[1]}, Score: {score[0]}, Median Vector Effect: {median_effect}")
            
        return filtered_scores
    
    
    def plot_value_vector_cos_similarity(
        self,
        scores,
        diametric_scores: torch.Tensor | None,
        _value_probe,
        cos_sims_np
    ):
        """
        Plot a scatter diagram of layer vs. cosine similarity for value and diametric vectors.

        Args:
            scores: List of tuples (cos_similarity, mlp_idx, layer) for value vectors.
            diametric_scores: List of tuples (cos_similarity, mlp_idx, layer) for diametric vectors.
            _value_probe: The value probe object (should have .issue_owner.name_mapped).
            cos_sims: Cosine similarities of the value vectors.
        """
        
        # Unpack layers and scores
        value_layers = [x[2] for x in scores]
        value_cos = [x[0] for x in scores]
        diametric_layers = [x[2] for x in diametric_scores] if diametric_scores is not None else []
        diametric_cos = [x[0] for x in diametric_scores] if diametric_scores is not None else []

        plt.figure(figsize=(10, 6))
        plt.scatter(value_layers, value_cos, c='b', label='Value Vectors', alpha=0.7)
        if diametric_scores is not None:
            plt.scatter(diametric_layers, diametric_cos, c='r', label='Diametric Vectors', alpha=0.7, marker='x')
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity")

        # Add boxplot for cos_sims distribution per layer
        for layer in cos_sims_np.keys():
            plt.boxplot(
                cos_sims_np[layer],
                positions=[layer],
                widths=0.4,
                patch_artist=True, 
                boxprops=dict(facecolor='lightgray', color='black'), 
                medianprops=dict(color='red'), whiskerprops=dict(color='black')
            )
        
        plt.title(f"Cosine Similarity of Value/Diametric Vectors\n{_value_probe.issue_owner.name_mapped}")
        plt.legend()
        plt.grid(True)

        os.makedirs("results", exist_ok=True)
        # Sanitize filename by replacing invalid characters
        safe_name = _value_probe.issue_owner.name_mapped.replace('/', '-').replace('\\', '-')
        
        path_savefig = os.path.join(
            "results",
            _value_probe.country.lower(),
            _value_probe.model_id.replace('/','-'),
            "value_vectors",
            f"cos_similarity_{safe_name}.png"
        )
        
        # Check if directory exists:
        if not os.path.exists(os.path.dirname(path_savefig)):
            os.makedirs(os.path.dirname(path_savefig))
        
        plt.savefig(path_savefig)
        plt.close()


    def extract(self) -> List[ValueVector]:
        """
        Extract the value vectors.
        
        Returns:
            List[ValueVector]: List of extracted value vectors.
        """
        
        for _model_id in self.probing.model_ids:
            
            inference_hook = Inference_Hook(
                model_id=_model_id, 
                inference_config=SETTINGS.inference
            )
            
            self.ios_unique = self.dataset.get_all_unique_issue_owners()
            io_token_ids = {_io.name_mapped: inference_hook.model.to_tokens([_io.name_mapped]) for _io in self.ios_unique}
            
            unembedding_matrice = inference_hook.model.W_U.cpu().detach()
            
            LOGGER.info(f"Unembedding Matrice shape: {unembedding_matrice.shape}")
            
            # Get all value probes for the model
            value_probes = self.probing.get_all_value_probes(model_id=_model_id)
            value_vectors = []
            
            LOGGER.info(f"Number of value probes for model {_model_id}: {len(value_probes)}")
            LOGGER.info("Extracting value vectors...")

            for _value_probe in value_probes:
                
                LOGGER.info(f"Issue Owner: {_value_probe.issue_owner.name_mapped}")
                
                probe_model = torch.load(_value_probe.path_value_probe, weights_only=False)
                LOGGER.info(f"Value Probe shape: {probe_model.linear.weight.shape}")
                
                probe_model_weights, _ = probe_model.get_linear_weights_and_bias()
                probe_model_weights = probe_model_weights.detach().cpu().to(unembedding_matrice.dtype)
                
                # multiply the value_probe_weights with the unembedding_matrice
                layer_unembedded = einsum(
                    self.rms_normalize(probe_model_weights),
                    unembedding_matrice,
                    "layer d_model, d_model d_vocab -> layer d_vocab"
                )
                
                # Get the top logits and their indices (tokens)
                top_logits, top_token_ids = torch.topk(layer_unembedded, self.top_k_tokens)

                # Load tokenizer to map indices to tokens
                top_tokens = [inference_hook.tokenizer.convert_ids_to_tokens(index.item()) for index in top_token_ids.view(-1)]
                _value_probe.top_tokens = top_tokens
                
                LOGGER.info(f"Top Tokens of value probe: {top_tokens}")
            
                scores, diametric_scores, cos_sims_np_dict = self._get_cos_sim_scores(
                    model=inference_hook.model,
                    probe_model_weights=probe_model_weights
                )
                
                # Sort scores in descending order
                sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
                if SETTINGS.value_vector_extraction['filter_with_vector_effect']:
                    sorted_scores_filtered = self._filter_vectors_by_effect(
                        io_name_mapped=_value_probe.issue_owner.name_mapped,
                        sorted_scores=sorted_scores,
                        inference_hook=inference_hook,
                        io_token_ids=io_token_ids,
                    )
                else:
                    sorted_scores_filtered = sorted_scores

                if sorted_scores_filtered == []:
                    LOGGER.error(f"No value vectors passed the effect filter for {_value_probe.issue_owner.name_mapped}.")
                    raise ValueError(f"No value vectors passed the effect filter for {_value_probe.issue_owner.name_mapped}.")
                
                value_vectors.extend(
                    self._get_vectors(
                        sorted_scores=sorted_scores_filtered,
                        value_probe=_value_probe,
                        model=inference_hook.model
                    )
                )
                
                top_vecs = torch.stack([
                    inference_hook.model.blocks[x[2]].mlp.W_out[x[1]] for x in sorted_scores_filtered
                ])
                
                if SETTINGS.value_vector_extraction['extract_diametrically_opposed_vectors']:
                    # Sort diametric scores in ascending order
                    sorted_diametric_scores = sorted(diametric_scores, key=lambda x: x[0], reverse=True)
                    if SETTINGS.value_vector_extraction['filter_with_vector_effect']:
                        sorted_diametric_scores_filtered = self._filter_vectors_by_effect(
                            io_name_mapped=_value_probe.issue_owner.name_mapped,
                            sorted_scores=sorted_diametric_scores,
                            inference_hook=inference_hook,
                            io_token_ids=io_token_ids,
                        )
                    else:
                        sorted_diametric_scores_filtered = sorted_diametric_scores

                    if sorted_diametric_scores_filtered != []:
                        value_vectors.extend(
                            self._get_vectors(
                                sorted_scores=sorted_diametric_scores_filtered,
                                value_probe=_value_probe,
                                model=inference_hook.model,
                                is_diametric=True
                            )
                        )
                        diametric_top_vecs = torch.stack([
                            inference_hook.model.blocks[x[2]].mlp.W_out[x[1]] for x in sorted_diametric_scores_filtered
                        ])
                    else:
                        LOGGER.warning(f"No diametric value vectors passed the effect filter for {_value_probe.issue_owner.name_mapped}. Skipping.")
                
                LOGGER.info(f"Number of value vectors: {len(value_vectors)}")
                
                # Visualize the cosine similarity of value vectors
                self.plot_value_vector_cos_similarity(
                    sorted_scores_filtered,
                    sorted_diametric_scores_filtered if SETTINGS.value_vector_extraction['extract_diametrically_opposed_vectors'] else None,
                    _value_probe,
                    cos_sims_np_dict
                )
                
                # multiply the top value vector with the unembedding_matrice
                last_layer_unembedded = einsum(
                    top_vecs.cpu(),
                    unembedding_matrice.cpu(),
                    "n_vec d_model, d_model d_vocab -> n_vec d_vocab"
                )
                
                # Get the top logits and their indices (tokens)
                top_logits, top_indices = torch.topk(last_layer_unembedded, self.top_k_tokens)
                
                # Load tokenizer to map indices to tokens
                top_tokens = [inference_hook.tokenizer.convert_ids_to_tokens(index.item()) for index in top_indices.view(-1)]
                LOGGER.info(f"Top tokens in value vectors: {top_tokens}")
                
                _value_probe.top_value_vector_tokens = top_tokens
                
                if SETTINGS.value_vector_extraction['extract_diametrically_opposed_vectors']:
                    last_diametric_layer_unembedded = einsum(
                        diametric_top_vecs.cpu(),
                        unembedding_matrice.cpu(),
                        "n_vec d_model, d_model d_vocab -> n_vec d_vocab"
                    )
                
                    diametric_top_logits, diametric_top_indices = torch.topk(last_diametric_layer_unembedded, self.top_k_tokens)
                    LOGGER.debug(f"Top tokens in diametric value vectors: {[inference_hook.tokenizer.convert_ids_to_tokens(index.item()) for index in diametric_top_indices.view(-1)]}")
                
                # Stack all top vec from all value vectors
                if "all_top_vecs" in locals():
                    if SETTINGS.value_vector_extraction['extract_diametrically_opposed_vectors']:
                        all_top_vecs = torch.cat((all_top_vecs, top_vecs, diametric_top_vecs), dim=0)
                    else:
                        all_top_vecs = torch.cat((all_top_vecs, top_vecs), dim=0)
                else:
                    if SETTINGS.value_vector_extraction['extract_diametrically_opposed_vectors']:
                        all_top_vecs = torch.cat((top_vecs, diametric_top_vecs), dim=0)
                    else:
                        all_top_vecs = top_vecs
        
        return value_vectors





