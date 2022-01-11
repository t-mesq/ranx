import json
from typing import Dict, List, Tuple

from ranx.frozenset_dict import FrozensetDict
from tabulate import tabulate

from ranx.report import super_chars, chars, Report


class TrialsReport(Report):
    def __init__(
        self,
        model_names: List[str],
        results: Dict,
        comparisons: FrozensetDict,
        metrics: List[str],
        max_p: float,
        win_tie_loss: Dict[Tuple[str], Dict[str, Dict[str, int]]],
        standart_dev: Dict,
    ):
        super().__init__(
            model_names=model_names,
            results=results,
            comparisons=comparisons,
            metrics=metrics,
            max_p=max_p,
            win_tie_loss=win_tie_loss,
        )
        self.standart_dev = standart_dev

    def get_superscript_for_table(self, model, metric):
        """Fixes significancy comparison to support varying max-p"""
        return ("").join(
            [
                super_chars[j]
                for j, _model in enumerate(self.model_names)
                if model != _model
                and (self.comparisons[model, _model][metric]["p_value"] <= self.max_p)
                and (self.results[model][metric] > self.results[_model][metric])
            ]
        )

    def to_table(self):
        """Used internally."""
        return tabulate(
            [
                [chars[i], run]
                + [
                    f"{score:.4f}±{self.standart_dev[run][metric]:.4f}{self.get_superscript_for_table(run, metric)}"
                    for metric, score in v.items()
                ]
                for i, (run, v) in enumerate(self.results.items())
            ],
            headers=["#", "Model"]
            + [
                self.get_metric_label(x)
                for x in list(list(self.results.values())[0].keys())
            ],
        )