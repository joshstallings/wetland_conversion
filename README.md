# wetland_conversion

This project aims to predict which tracts of land in Florida convert from wetland
to development, between the years of 2019 and 2024.

# Pipeline

Each stage reads the previous stage's output off disk, so any stage can be rerun
on its own once the one before it exists.

1. `wetland_sample_labels.ipynb` builds the label table from NLCD land cover.
2. `scripts/ingest/export_alphaearth_30m.py` and
   `scripts/ingest/join_alphaearth_samples.py` pull Google AlphaEarth
   embeddings and join them onto the labels.
3. `scripts/dataset/build_train_pool.py` builds a named, versioned dataset: a
   spatial fold assignment and a weighted training pool, under
   `data/processed/datasets/<name>/`. Its `features` subcommand also builds a
   shared cache of neighborhood features straight from the 2019 NLCD raster
   (local development density, edge density, direction to development), see
   `scripts/README.md` for what each one is and why.
4. `eda_dataset.ipynb` explores the joined table and any one built dataset.
5. `scripts/model/train.py` fits and persists models for a named experiment
   (`scripts/model/experiments.py`), under `results/models/<name>/`.
6. `model_analysis.ipynb` loads a trained experiment's models and predictions
   and produces diagnostic figures, no retraining.

# Directory structure

- `data/` : NLCD rasters and AlphaEarth tiles under `data/NLCD/` and
  `data/AlphaEarth/`, plus `data/processed/`, the joined per pixel table and
  `data/processed/datasets/<name>/`, one versioned training pool per named
  dataset. See `scripts/README.md` for what builds each file.
- `scripts/` : the data and modeling pipeline, a real Python package grouped
  by stage (`ingest/`, `dataset/`, `model/`, plus `data_constants.py` shared
  by all three), run from the repo root as
  `python -m scripts.<stage>.<name> ...`. See `scripts/README.md` for what
  each script does.
- `results/models/<name>/` : everything one trained experiment produced,
  fold models, metrics, figures, and hyperparameter grid search runs.
- `wetland_sample_labels.ipynb` : builds the label set for wetlands that
  convert to development, remain wetland, or convert to something else, from
  the NLCD land cover data. Also generates plots, maps, and summary
  statistics of that transition.
- `eda_dataset.ipynb` : exploration and figures for the joined AlphaEarth
  table (label balance, embedding ranges, spatial distribution) and for one
  generated training pool, chosen by setting `DATASET_NAME` at the top.
- `model_analysis.ipynb` : diagnostics for one trained experiment, decile
  precision and recall by distance to development, spatial error maps, all
  loaded from disk with no retraining. Set `EXPERIMENT_NAME` at the top.

# Requirements

See `requirements.yaml` for a recommended conda / mamba installation for this
project.
