"""Named experiment configs for scripts/model/train.py.

What used to be a copy-pasted "Experiment N" section in gradient_boosting.ipynb
is one ExperimentConfig entry here. Add a new entry to try a different feature
set, dataset, or grid; don't edit an existing one once it has a
results/models/<experiment_name>/ directory on disk, since the whole point of
naming a run is that the name always means the same thing.
"""

from dataclasses import dataclass, field

from scripts.model.model_common import (
    DIST_ONLY_FEATURE_COLS,
    DIST_PLUS_NEIGHBORHOOD_FEATURE_COLS,
    EXTENDED_FEATURE_COLS,
    FEATURE_COLS,
)

# Same 32 combination grid gradient_boosting.ipynb screened, kept as a shared
# constant so experiments meant to be compared use the exact same grid rather
# than a retyped copy that could quietly drift.
_STANDARD_GRID = {
    "n_estimators": [100, 300],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "feature_fraction": [0.7, 1.0],
    "early_stopping_rounds": [10, 30],
}

# Aimed at the imbalance itself rather than general capacity. scale_pos_weight
# and is_unbalance are deliberately left out: sample_weight already encodes
# the class balance the pool was built with, and stacking LightGBM's own
# rebalancing on top of that would just double count it.
#   - min_child_samples: LightGBM's leaf size floor counts raw rows, not
#     weighted mass, so it's what actually limits how finely a tree can
#     isolate a small cluster of converters, independent of how much weight
#     they carry. 20 is the library default; 5 tests relaxing it.
#   - boosting_type "goss": keeps the rows with the largest gradients (the
#     ones the model is currently getting wrong) and subsamples the rest,
#     which for a rare positive class means it's the hard, informative rows
#     that stick around, not a random slice of easy ones.
#   - reg_lambda: leaf value L2 penalty, checked against the smaller
#     min_child_samples in case relaxing the leaf floor just trades cleaner
#     recall for overfit noise in tiny leaves.
# n_estimators is fixed high with early stopping doing the real search, so
# the grid's combinations go toward the imbalance-relevant knobs instead.
_IMBALANCE_GRID = {
    "n_estimators": [500],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "min_child_samples": [5, 20],
    "boosting_type": ["gbdt", "goss"],
    "reg_lambda": [0.0, 1.0],
    "early_stopping_rounds": [30],
}


@dataclass
class ExperimentConfig:
    experiment_name: str  # output goes to results/models/<experiment_name>/
    dataset_name: str  # trained against data/processed/datasets/<dataset_name>/
    feature_cols: list[str]  # columns the model actually sees
    model_type: str = "lightgbm"
    model_params: dict = field(default_factory=lambda: {"random_state": 0})
    grid: dict | None = None  # None means no grid search configured

    @property
    def results_dir(self) -> str:
        return f"results/models/{self.experiment_name}"


EXPERIMENTS = {
    "Experiment5-BaselineV2-Neighborhood": ExperimentConfig(
        # Experiment4-BaselineV2 plus the six neighborhood columns, against
        # neighborhood-v2: same derived pos_weight_mult (~356) as baseline-v2,
        # built with --with-neighborhood-features instead
        experiment_name="Experiment5-BaselineV2-Neighborhood",
        dataset_name="neighborhood-v2",
        feature_cols=EXTENDED_FEATURE_COLS,
        grid=_IMBALANCE_GRID,
    ),
    "Experiment5b-BaselineV2": ExperimentConfig(
        # the Experiment5 counterpart without the six neighborhood columns,
        # same relationship 3b has to 3: same dataset (baseline-v2, so same
        # derived pos_weight_mult) and feature set as Experiment4-BaselineV2,
        # but screened against _IMBALANCE_GRID instead of left at defaults,
        # so it's a fair grid-search comparison against Experiment5
        experiment_name="Experiment5b-BaselineV2",
        dataset_name="baseline-v2",
        feature_cols=FEATURE_COLS,
        grid=None,
    ),
    "Experiment6-SLMInitialEvaluation": ExperimentConfig(
        # Running a simple linear model on Baseline-v2 with the AE embeddings and distance features.
        experiment_name="Experiment6-SLMInitialEvaluation",
        dataset_name="baseline-v2",
        feature_cols=FEATURE_COLS,
        model_type="slm",
        grid=None
    ),
    "Experiment7-SLMCluster0": ExperimentConfig(
        # Same SLM and feature set as Experiment6, but trained against
        # cluster0-v1, the near-development block_distance_profiles regime
        # only, built with --no-thinning so every natural pixel is kept.
        experiment_name="Experiment7-SLMCluster0",
        dataset_name="cluster0-v1",
        feature_cols=FEATURE_COLS,
        model_type="slm",
        grid=None
    ),
}


def get(experiment_name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[experiment_name]
    except KeyError:
        raise SystemExit(
            f"no experiment '{experiment_name}' in scripts/model/experiments.py, "
            f"known experiments: {', '.join(EXPERIMENTS)}"
        )
