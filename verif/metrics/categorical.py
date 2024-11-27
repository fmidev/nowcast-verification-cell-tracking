import numpy as np
from pysteps import verification
import xarray as xr

from verif.metrics import Metric


class CategoricalMetric(Metric):
    def __init__(self, leadtimes, cat_metrics, thresh, tables: dict = None, **kwargs) -> None:
        self.name_template = "CAT_t_{thresh}_l_{leadtime}"
        self.leadtimes = leadtimes
        self.thresholds = thresh
        self.cat_metrics = cat_metrics
        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                for thr in thresh:
                    name = self.name_template.format(leadtime=lt, thresh=thr)
                    self.tables[name] = verification.det_cat_fct_init(thr=thr)
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            for thr in self.thresholds:
                name = self.name_template.format(leadtime=lt, thresh=thr)
                verification.det_cat_fct_accum(contab=self.tables[name], pred=x_pred[i], obs=x_obs[i])
            self.is_empty = False

    def compute(self):
        values = np.empty((len(self.cat_metrics), len(self.thresholds), len(self.leadtimes)))
        n_pixels = np.empty((len(self.cat_metrics), len(self.thresholds), len(self.leadtimes)))
        n_obs = np.empty((len(self.cat_metrics), len(self.thresholds), len(self.leadtimes)))
        for i, metric in enumerate(self.cat_metrics):
            for j, thr in enumerate(self.thresholds):
                for k, lt in enumerate(self.leadtimes):
                    in_name = self.name_template.format(thresh=thr, leadtime=lt)
                    n_pixels[i, j, k] = (
                        self.tables[in_name]["hits"]
                        # + self.tables[in_name]["misses"]
                        + self.tables[in_name]["false_alarms"]
                        # + self.tables[in_name]["correct_negatives"]
                    )
                    n_obs[i, j, k] = (
                        self.tables[in_name]["hits"]
                        + self.tables[in_name]["misses"]
                        # + self.tables[in_name]["false_alarms"]
                        # + self.tables[in_name]["correct_negatives"]
                    )
                    values[i, j, k] = verification.det_cat_fct_compute(self.tables[in_name], scores=metric)[metric]
        return xr.Dataset(
            data_vars={
                "metric": (["cat_metric", "threshold", "leadtime"], values),
                "n_pixels": (["cat_metric", "threshold", "leadtime"], n_pixels),
                "n_obs": (["cat_metric", "threshold", "leadtime"], n_obs),
            },
            # dims=["cat_metric", "threshold", "leadtime"],
            coords={
                "cat_metric": self.cat_metrics,
                "threshold": self.thresholds,
                "leadtime": self.leadtimes,
            },
            attrs={"metric": "CAT"},
        )

    def merge(self, categorical_other):
        self.tables = {
            name: verification.det_cat_fct_merge(table, categorical_other.tables[name])
            for name, table in self.tables.items()
        }

    @staticmethod
    def plot(
        scores: dict,
        method: str,
        lt: np.array,
        exp_id: str,
        path_save: str,
        method_plot_params: dict = {},
        subplot_kwargs: dict = {},
        plot_kwargs: dict = {},
        kwargs: dict = {},
    ):
        pass
