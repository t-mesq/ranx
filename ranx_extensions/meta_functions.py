from collections import defaultdict
from typing import List, Union

import numpy as np

from ranx.frozenset_dict import FrozensetDict
from ranx.meta_functions import format_metrics, evaluate
from ranx.qrels import Qrels
from ranx.run import Run
from ranx_extensions.statistical_testing import trials_randomization_test
from .report import TrialsReport


def compute_trials_statistical_significance(
    control_metric_scores,
    treatment_metric_scores,
    n_permutations: int = 1000,
    max_p: float = 0.01,
    random_seed: int = 42,
):
    metric_p_values = {}

    for m in list(control_metric_scores):
        (
            control_mean,
            treatment_mean,
            p_value,
            significant,
        ) = trials_randomization_test(
            control_metric_scores[m],
            treatment_metric_scores[m],
            n_permutations,
            max_p,
            random_seed,
        )

        metric_p_values[m] = {
            "p_value": p_value,
            "significant": significant,
        }

    return metric_p_values

def compare_trials(
    qrels: Qrels,
    runs: List[List[Run]],
    metrics: Union[List[str], str],
    n_permutations: int = 1000,
    max_p: float = 0.01,
    random_seed: int = 42,
    threads: int = 0,
    t_mode: str = 'full'
):
    metrics = format_metrics(metrics)
    assert all(type(m) == str for m in metrics), "Metrics error"

    model_names = []
    results = defaultdict(dict)
    standart_dev = defaultdict(dict)
    comparisons = FrozensetDict()

    metric_scores = {}

    # Compute scores for each run for each trial for each query -------------------------------
    for run in runs:
        model_names.append(run[0].name)
        metric_scores[run[0].name] = [evaluate(
            qrels=qrels,
            run=trial,
            metrics=metrics,
            return_mean=False,
            threads=threads,
        ) for trial in run]
        metric_scores[run[0].name] = {m: np.stack([trial_scores[m] for trial_scores in metric_scores[run[0].name]]) for m in metrics}
        for m in metrics:
            results[run[0].name][m] = np.mean(metric_scores[run[0].name][m])
            standart_dev[run[0].name][m] = np.std(np.mean(metric_scores[run[0].name][m], axis=1))

    # Run statistical testing --------------------------------------------------
    for i, control in enumerate(runs):
        control_metric_scores = metric_scores[control[0].name]
        for j, treatment in enumerate(runs):
            if i < j:
                treatment_metric_scores = metric_scores[treatment[0].name]

                # Compute statistical significance
                comparisons[
                    frozenset([control[0].name, treatment[0].name])
                ] = compute_trials_statistical_significance(
                    control_metric_scores,
                    treatment_metric_scores,
                    n_permutations,
                    max_p,
                    random_seed,
                )

    # Compute win / tie / lose -------------------------------------------------
    win_tie_loss = defaultdict(dict)

    for control in runs:
        for treatment in runs:
            for m in metrics:
                control_scores = metric_scores[control[0].name][m]
                treatment_scores = metric_scores[treatment[0].name][m]
                win_tie_loss[(control[0].name, treatment[0].name)][m] = {
                    "W": sum(control_scores > treatment_scores),
                    "T": sum(control_scores == treatment_scores),
                    "L": sum(control_scores < treatment_scores),
                }

    return TrialsReport(
        model_names=model_names,
        results=dict(results),
        comparisons=comparisons,
        metrics=metrics,
        max_p=max_p,
        win_tie_loss=dict(win_tie_loss),
        standart_dev=standart_dev
    )
