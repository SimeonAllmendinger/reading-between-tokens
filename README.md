# Reading Between Tokens

Mechanistic forecasting of political preferences with large language models (LLMs).
This repository accompanies an ICML paper on **mechanistic forecasting**: a method that aggregates party-aligned latent activations in LLM MLPs to predict population-level preference distributions, and compares them to survey data.

## Paper overview

Current persona-based LLM forecasting relies on surface-level outputs (next-token probabilities). The paper argues that this misses latent, structured political knowledge inside the model. We therefore:

- identify party-aligned value vectors in MLPs,
- measure how persona prompts shift those internal activations, and
- aggregate these shifts into group-level preference distributions.

Across multiple model families and national election contexts, latent activation–based estimates often better match real-world survey outcomes than output probabilities alone, with strong variation by persona attribute categories.

## Repository contents

- `src/`: core code for persona prompting, activation extraction, and aggregation.
- `configs/`: experiment configurations.
- `scripts/`: batch and experiment launch scripts.
- `data/`: datasets used in experiments (see below).

## Data

This repo includes the **Wahl-O-Mat (Germany) dataset** via the open-source **qual-o-mat-data** project under `data/qual-o-mat-data/`.
That project provides machine-readable conversions of Wahl-O-Mat data from the Bundeszentrale für politische Bildung (bpb).
Please consult `data/qual-o-mat-data/README.md` for provenance and licensing notes.

Other datasets used in the paper are described in the paper and configuration files.

## Setup

A minimal conda environment is provided:

```bash
conda env create -f conda_env.yml
conda activate llm-o-mat
```

## Usage (high level)

The project is organized around running persona prompts, extracting internal activations, and aggregating them into preference distributions.
See `scripts/` and `configs/` for example runs and experiment definitions.

## Citation

If you use this code or data, please cite the ICML paper (citation to be added once finalized).

## License

See `LICENSE` for repository licensing. Note that datasets may have their own terms; the Wahl-O-Mat data is redistributed from `data/qual-o-mat-data/` as described in its README.
