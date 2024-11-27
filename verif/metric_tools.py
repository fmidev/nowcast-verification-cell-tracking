"""
Utility functions for working with metrics
"""

from verif import metrics
from functools import reduce
import pandas as pd
import numpy as np


def get_metric_class(metric_name: str):
    "match metric name to appropriate class"
    metric_name = metric_name.upper()
    if metric_name == "OBJECTS":
        return metrics.ObjectMetrics
    elif metric_name == "CONT":
        return metrics.ContinuousMetric
    elif metric_name == "CAT":
        return metrics.CategoricalMetric
    else:
        raise NotImplementedError(f"Metric {metric_name.upper()} not implemented.")


def get_metric(metric_name: str, metric_params: dict, tables: dict = None):
    """Match metric names to object instantiations, containing i.e.
    1. A contingency table under the "table" attribute.
    2. An "accumulate(x_pred, x_obs)" method.
    3. A "compute() method.

    To add a new metric, create it under the metrics folder and add
    the matching choice here.

    Args:
        metric_name (str): unique string identifier for the metric, mapped via 'get_metric_class'
        tables (dict): existing table holding persistent metric state
        metric_params (dict): dictionnary of metric initiation keyword arguments.

    Returns:
        verif.metrics.Metric: Metric object of the wanted subclass.
    Raises:
        NotImplementedError: if metric class is not defined in the 'get_metric_class' function
    """
    return get_metric_class(metric_name=metric_name)(tables=tables, **metric_params)


def merge_metrics(metric_instance_1, metric_instance_2):
    "Wrapper function for merging two metrics of the same type"

    if (
        isinstance(metric_instance_1, float)
        and isinstance(metric_instance_2, float)
        and np.isnan(metric_instance_1)
        and np.isnan(metric_instance_2)
    ):
        return np.nan

    if (not hasattr(metric_instance_1, "merge")) or (not hasattr(metric_instance_2, "merge")):
        raise AttributeError(f"Either {metric_instance_1} or {metric_instance_2} does not have a 'merge' attribute.")
    if (not hasattr(metric_instance_1, "is_empty")) or (not hasattr(metric_instance_2, "is_empty")):
        raise AttributeError(f"Either {metric_instance_1} or {metric_instance_2} does not have a 'is_empty' attribute.")
    if not type(metric_instance_1) is type(metric_instance_2):
        raise TypeError(f"{metric_instance_1} and {metric_instance_2} are not of the same type.")

    if metric_instance_1.is_empty and metric_instance_2.is_empty:
        return metric_instance_1
    elif metric_instance_1.is_empty:
        return metric_instance_2
    elif metric_instance_2.is_empty:
        return metric_instance_1
    else:
        metric_instance_1.merge(metric_instance_2)
        return metric_instance_1


def merge_metrics_df(a, b, path=None):
    """
    merges output metric dictionnary b into output metric dictionnary a.
    Starts at path {path}.
    Works recursively through pairs of matching metrics present in the dictionary.

    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], pd.Series) and isinstance(b[key], pd.Series):
                merge_metrics_df(a[key], b[key], path + [str(key)])
            else:
                a[key] = merge_metrics(
                    metric_instance_1=a[key], metric_instance_2=b[key]
                )
        else:
            raise Exception("Outdicts do not match at %s" % ".".join(path + [str(key)]))
    return a
    """
    return pd.DataFrame(
        np.vectorize(merge_metrics)(a.dropna(), b.dropna()),
        index=a.dropna().index,
        columns=a.dropna().columns,
    )


def merge_metrics_df_list(outdict_list: list):
    "Merge list of dict of metrics through reduce"
    return reduce(merge_metrics_df, outdict_list)


def merge_boolean_df_list(done_df_list: list):
    "Merge list of boolean dataframes through reduce"
    return reduce(lambda ddf1, ddf2: ddf1 | ddf2, done_df_list)
