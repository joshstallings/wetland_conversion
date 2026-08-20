"""Named experiment configs for scripts/gb_train.py.

What used to be a copy-pasted "Experiment N" section in gradient_boosting.ipynb
is one ExperimentConfig entry here. Add a new entry to try a different feature
set, dataset, or grid; don't edit an existing one once it has a
results/models/<experiment_name>/ directory on disk, since the whole point of
naming a run is that the name always means the same thing.
"""

from dataclasses import dataclass, field

from gb_common import DIST_ONLY_FEATURE_COLS, DIST_PLUS_NEIGHBORHOOD_FEATURE_COLS, EXTENDED_FEATURE_COLS, FEATURE_COLS

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
    # only "lightgbm" is implemented in gb_train.py's make_model(); the field
    # exists so a second model type is a new branch there, not a restructure
    model_type: str = "lightgbm"
    model_params: dict = field(default_factory=lambda: {"random_state": 0})
    grid: dict | None = None  # None means no grid search configured

    @property
    def results_dir(self) -> str:
        return f"results/models/{self.experiment_name}"


EXPERIMENTS = {
    "full-features": ExperimentConfig(
        experiment_name="full-features",
        dataset_name="baseline-v1",
        feature_cols=FEATURE_COLS,
        grid=_STANDARD_GRID,
    ),
    "distance-only": ExperimentConfig(
        experiment_name="distance-only",
        dataset_name="baseline-v1",
        feature_cols=DIST_ONLY_FEATURE_COLS,
        # feature_fraction is structurally inert with a single feature, kept
        # anyway so this experiment's grid results line up column for column
        # against full-features
        grid=_STANDARD_GRID,
    ),
    "neighborhood-features": ExperimentConfig(
        # results/models/Experiment3-neighborhood-features/, renamed by hand
        # to match the ExperimentN-Name directories the original notebook
        # left behind (Experiment0-Initial etc); the dict key below is the
        # short name gb_train.py --experiment still takes on the CLI
        experiment_name="Experiment3-neighborhood-features",
        # built with `folds`/`pool --with-neighborhood-features` under this
        # name, see PLAN.md for what the six extra columns are and how
        # they're built
        dataset_name="neighborhood-v1",
        feature_cols=EXTENDED_FEATURE_COLS,
        # baseline only, no grid configured
    ),
    "neighborhood-features-distance-only": ExperimentConfig(
        # results/models/Experiment3b-neighborhood-features-distance-only/,
        # 3b since it is a variant on Experiment3, same dataset and neighborhood
        # features, with the 192 AlphaEarth bands dropped, to see what the
        # distance and neighborhood columns alone are worth without embeddings
        # to lean on
        experiment_name="Experiment3b-neighborhood-features-distance-only",
        dataset_name="neighborhood-v1",
        feature_cols=DIST_PLUS_NEIGHBORHOOD_FEATURE_COLS,
        # baseline only, no grid configured
    ),
    "Experiment4-BaselineV2": ExperimentConfig(
        # same feature set as full-features, but against baseline-v2, which
        # drops the hand picked pos_weight_mult=10 for one derived from the
        # natural class balance (n_neg_natural / n_pos_natural, ~356 here),
        # so label=1 and label!=1 carry equal total weight in the loss
        experiment_name="Experiment4-BaselineV2",
        dataset_name="baseline-v2",
        feature_cols=FEATURE_COLS,
        # baseline only, no grid configured
    ),
    "Experiment5-BaselineV2-Neighborhood": ExperimentConfig(
        # Experiment4-BaselineV2 plus the six neighborhood columns, against
        # neighborhood-v2: same derived pos_weight_mult (~356) as baseline-v2,
        # built with --with-neighborhood-features instead
        experiment_name="Experiment5-BaselineV2-Neighborhood",
        dataset_name="neighborhood-v2",
        feature_cols=EXTENDED_FEATURE_COLS,
        grid=_IMBALANCE_GRID,
    ),
    "Experiment5b-BaselineV2-NoNeighborhood": ExperimentConfig(
        # the Experiment5 counterpart without the six neighborhood columns,
        # same relationship 3b has to 3: same dataset (baseline-v2, so same
        # derived pos_weight_mult) and feature set as Experiment4-BaselineV2,
        # but screened against _IMBALANCE_GRID instead of left at defaults,
        # so it's a fair grid-search comparison against Experiment5
        experiment_name="Experiment5b-BaselineV2-NoNeighborhood",
        dataset_name="baseline-v2",
        feature_cols=FEATURE_COLS,
        grid=_IMBALANCE_GRID,
    ),
}


def get(experiment_name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[experiment_name]
    except KeyError:
        raise SystemExit(
            f"no experiment '{experiment_name}' in scripts/experiments.py, "
            f"known experiments: {', '.join(EXPERIMENTS)}"
        )
