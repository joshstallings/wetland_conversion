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
}


def get(experiment_name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[experiment_name]
    except KeyError:
        raise SystemExit(
            f"no experiment '{experiment_name}' in scripts/experiments.py, "
            f"known experiments: {', '.join(EXPERIMENTS)}"
        )
