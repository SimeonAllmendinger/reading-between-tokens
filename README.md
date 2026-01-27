# Reading Between Tokens

Mechanistic forecasting of political preferences with large language models (LLMs).
This repository accompanies an ICML paper on **mechanistic forecasting**: a method that aggregates party-aligned latent activations in LLM MLPs to predict population-level preference distributions, and compares them to survey data.

## Code overview

This maps each method subsection in the paper to the concrete code paths in this repo.

- **Pipeline entry point**
  - `src/main.py` orchestrates the full pipeline: dataset → probing → value vector extraction → persona key/value extraction.
  - Global configs live in `configs/` and are loaded via `src/settings.py`.

- **Technical preliminaries** (residual stream + MLP decomposition)
  - Residual stream extraction and activation decomposition happen in `src/components/inference_hook.py`.
  - `get_responses_and_activations(..., decompose_activations=False)` uses TransformerLens `cache.accumulated_resid` and mean-pools over assistant tokens.
  - `get_responses_and_activations(..., decompose_activations=True)` uses `cache.decompose_resid` to recover MLP sub-updates for persona analyses.

- **Constructing probes for identifying party MLP value vectors**
  - **Dataset + prompts for probing**: `src/components/data_handler.py` builds issue/party statements; prompt variants come from `configs/config_dataset.yaml` (`value_probe_variants`).
  - **Activation collection**: `src/components/inference.py` runs prompts, saves `accum_resid` activations and logits for probe training.
  - **Probe training**: `src/components/probing.py` trains per-party linear probes (BCE with class weighting) on intermediate layers defined in `configs/config_probing.yaml`.
  - **Value vector selection + ablation filter**: `src/components/value_vector_extraction.py` computes cosine alignment between probe weights and MLP `W_out`, selects IQR outliers (aligned/diametric), and optionally filters by effect on party token probabilities via vector distortion.

- **Persona-induced activation of party value vectors**
  - **Persona construction + prompt templates**: `configs/config_personas.yaml` and `Persona` in `src/schemas.py`.
  - **Prompting + activation capture**: `src/components/key_value_extraction.py` runs persona prompt variants with `decompose_activations=True`.
  - **Activation weighting**: `get_key_values_contribution_with_cache` computes MLP post-activations for selected vectors and scales them by cosine similarity (stored as `key_value.scale`).
  - **Outputs**: per-persona/key-vector rows are flattened to CSV under `key_values/<country>/<model>/...` for downstream aggregation.

## Repository contents

- `src/`: core code for persona prompting, activation extraction, and aggregation.
- `configs/`: experiment configurations.
- `scripts/`: batch and experiment launch scripts.
- `data/`: datasets used in experiments (see below).

## Data

This repo does **not** include the Wahl-O-Mat (Germany) dataset by default. Instead, pull it from the open-source **qual-o-mat-data** project and place it under `data/qual-o-mat-data/`:

```bash
git clone https://github.com/gockelhahn/qual-o-mat-data.git data/qual-o-mat-data
```

That project provides machine-readable conversions of Wahl-O-Mat data from the Bundeszentrale für politische Bildung (bpb).
Please consult `data/qual-o-mat-data/README.md` for provenance and licensing notes.

Other datasets used in the paper are described in the paper and configuration files.

## Setup

A minimal conda environment is provided:

```bash
conda env create -f conda_env.yml
conda activate reading-between-tokens
```

Add your Hugging Face token to `configs/utils/config_api_keys.yaml` (used by `src/settings.py` / `src/components/inference_hook.py`).

## Usage

The project is organized around running persona prompts, extracting internal activations, and aggregating them into preference distributions.
See `scripts/` and `configs/` for example runs and experiment definitions.

## Citation

If you use this code or data, please cite the ICML paper (citation to be added once finalized).

## License

See `LICENSE` for repository licensing. Note that datasets may have their own terms; the Wahl-O-Mat data is provided via `data/qual-o-mat-data/` as described in its README.
