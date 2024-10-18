import numpy as np
from pysteps import verification
import xarray as xr

from verif.metrics import Metric


class ContinuousMetric(Metric):
    def __init__(self, leadtimes, cont_metrics, tables: dict = None, thresh=None, **kwargs) -> None:
        self.name_template = "CONT_l_{leadtime}"
        self.leadtimes = leadtimes
        self.cont_metrics = cont_metrics
        self.thresh = [-np.inf]

        if thresh is not None:
            # Add -inf to the thresholds to include all values
            self.thresh.extend(thresh)
            self.name_template = "CONT_t_{thresh}_l_{leadtime}"

        if tables is None:
            self.tables = {}
            for lt in leadtimes:
                for thr in self.thresh:
                    name = self.name_template.format(leadtime=lt, thresh=thr)
                    self.tables[name] = verification.det_cont_fct_init()
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs):
        if x_pred.ndim == 4:
            x_pred = x_pred.mean(axis=1)
        for i, lt in enumerate(self.leadtimes):
            for thr in self.thresh:
                name = self.name_template.format(leadtime=lt, thresh=thr)
                mask = (x_obs[i] > thr) | (x_pred[i] > thr)
                verification.det_cont_fct_accum(err=self.tables[name], pred=x_pred[i][mask], obs=x_obs[i][mask])
        self.is_empty = False

    def compute(self):
        values = []
        for metric in self.cont_metrics:
            metric_values = np.zeros((len(self.thresh), len(self.leadtimes)))
            for i, thr in enumerate(self.thresh):
                for j, lt in enumerate(self.leadtimes):
                    in_name = self.name_template.format(leadtime=lt, thresh=thr)
                    metric_values[i, j] = verification.det_cont_fct_compute(self.tables[in_name], scores=metric)[metric]
            values.append(metric_values)
        return xr.DataArray(
            data=np.array(values),
            dims=["cont_metric", "threshold", "leadtime"],
            coords={"cont_metric": self.cont_metrics, "leadtime": self.leadtimes, "threshold": self.thresh},
            attrs={"metric": "CONT"},
        )

    def merge(self, continuous_other):
        self.tables = {
            name: verification.det_cont_fct_merge(table, continuous_other.tables[name])
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
