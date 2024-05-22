import numpy as np
from pysteps import verification
import xarray as xr
from collections import defaultdict
import pandas as pd

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

from pysteps.feature import tstorm as tstorm_detect
from pysteps.tracking import tdating as tstorm_dating

from verif.metrics import Metric
from verif import conversion_tools


class ObjectMetrics(Metric):
    """Object-based metrics."""

    def __init__(
        self,
        leadtimes,
        obj_metrics,
        tables: dict = None,
        prev_obs_times=4,
        tdating_kwargs: dict = {},
        **kwargs,
    ) -> None:
        self.name_template = "OBJ_cont_l_{leadtime}"
        self.leadtimes = leadtimes
        self.obj_metrics = obj_metrics
        self.prev_obs_times = prev_obs_times
        self.tdating_kwargs = tdating_kwargs
        self.dist_limit_matching = kwargs.get("dist_limit_matching", 10.0)

        self.prev_objects_cache = dict()
        self.next_objects_cache = dict()

        self.zr_a = kwargs.get("zr_a", 223.0)
        self.zr_b = kwargs.get("zr_b", 1.53)

        if tables is None:
            self.tables = {}
            self.tables["cont_track_df"] = None
            self.is_empty = True
        else:
            self.tables = tables
            self.is_empty = False

    def accumulate(self, x_pred, x_obs, sample=None, additional_measurements=None):
        prev_obs = x_obs[: self.prev_obs_times, ...]
        next_obs = x_obs[self.prev_obs_times :, ...]

        methods = list(x_pred.keys())

        pred_times = list(range(1, x_pred[methods[0]].shape[0] + 1))
        prev_obs_times = list(range(-prev_obs.shape[0] + 1, 1))
        timelist = prev_obs_times + pred_times

        # We assume data is in mm/h, so transaform to dBZ
        prev_obs_dbz = conversion_tools.rainrate_to_dbz(prev_obs, self.zr_a, self.zr_b)
        next_obs_dbz = conversion_tools.rainrate_to_dbz(next_obs, self.zr_a, self.zr_b)

        if sample is not None and sample in self.prev_objects_cache:
            prev_track_list, prev_cell_list, prev_label_list = self.prev_objects_cache[sample]
        else:
            # Track objects in observations
            prev_track_list, prev_cell_list, prev_label_list = tstorm_dating.dating(
                input_video=prev_obs_dbz,
                timelist=prev_obs_times,
                **self.tdating_kwargs,
            )
            self.prev_objects_cache[sample] = (
                prev_track_list,
                prev_cell_list,
                prev_label_list,
            )

        # Track objects in next observations
        if sample is not None and sample in self.next_objects_cache:
            next_track_list, next_cell_list, next_label_list = self.next_objects_cache[sample]
        else:
            next_track_list, next_cell_list, next_label_list = tstorm_dating.dating(
                input_video=np.concatenate([prev_obs_dbz[-2:, ...], next_obs_dbz], axis=0),
                timelist=prev_obs_times[-2:] + pred_times,
                start=2,
                cell_list=prev_cell_list.copy(),
                label_list=prev_label_list.copy(),
                **self.tdating_kwargs,
            )
            self.next_objects_cache[sample] = (
                next_track_list,
                next_cell_list,
                next_label_list,
            )

        next_track_dict = {df.ID.unique().item(): df for df in next_track_list}
        prev_track_dict = {df.ID.unique().item(): df for df in prev_track_list}

        # Clear cache from other samples than current
        samples = set(self.prev_objects_cache.keys()) | set(self.next_objects_cache.keys())
        for other_sample in samples:
            if other_sample != sample:
                try:
                    del self.prev_objects_cache[other_sample]
                    del self.next_objects_cache[other_sample]
                except KeyError:
                    pass

        pred_track_lists = {}
        pred_cell_lists = {}
        pred_label_lists = {}
        pred_track_dicts = {}
        x_pred_rr = {}
        for method, pred_arr in x_pred.items():
            pred_arr_dbz = conversion_tools.rainrate_to_dbz(pred_arr, self.zr_a, self.zr_b)
            # Track objects in nowcasts
            (
                pred_track_lists[method],
                pred_cell_lists[method],
                pred_label_lists[method],
            ) = tstorm_dating.dating(
                input_video=np.concatenate([prev_obs_dbz[-2:, ...], pred_arr_dbz], axis=0),
                timelist=prev_obs_times[-2:] + pred_times,
                start=2,
                cell_list=prev_cell_list.copy(),
                label_list=prev_label_list.copy(),
                **self.tdating_kwargs,
            )

            pred_track_dicts[method] = {df.ID.unique().item(): df for df in pred_track_lists[method]}
            x_pred_rr[method] = pred_arr

        # Transform nowcasts and observations to rainrate
        x_obs_rr = x_obs
        prev_obs_rr = x_obs_rr[: self.prev_obs_times, ...]
        # next_obs_rr = x_obs_rr[self.prev_obs_times :, ...]

        # Process tracks that continue from previous observations
        prev_track_ids = [df.ID.unique().item() for df in prev_track_list]

        max_num_cells_obs = max([len(df) for df in next_cell_list])
        max_num_cells_pred = max([len(df) for cl in pred_cell_lists.values() for df in cl])
        max_num_cells = max(max_num_cells_obs, max_num_cells_pred)

        cell_match_hits = np.ones((len(self.leadtimes), len(methods))) * np.nan
        cell_match_misses = np.ones((len(self.leadtimes), len(methods))) * np.nan
        cell_match_false_alarms = np.ones((len(self.leadtimes), len(methods))) * np.nan

        # Features for matche cells
        cell_match_dist = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_obs_sum_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_pred_sum_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_obs_max_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_pred_max_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_obs_mean_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_pred_mean_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_obs_area = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        cell_match_pred_area = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan

        # Features for unmatched cells
        unmatched_obs_sum_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        unmatched_pred_sum_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        unmatched_obs_max_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        unmatched_pred_max_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        unmatched_obs_mean_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        unmatched_pred_mean_rr = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        unmatched_obs_area = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan
        unmatched_pred_area = np.ones((len(self.leadtimes), len(methods), max_num_cells)) * np.nan

        # match cells with hungarian algorithm to target observations
        for ti, obs_cells in enumerate(next_cell_list[self.prev_obs_times :]):
            if len(obs_cells) == 0:
                continue

            obs_cells_xy = np.array([obs_cells.cen_x, obs_cells.cen_y]).T.astype(float)

            for method, pred_cells_timeseries in pred_cell_lists.items():
                pred_cells = pred_cells_timeseries[self.prev_obs_times :][ti]
                if len(pred_cells) == 0:
                    continue
                # match cells with hungarian algorithm
                pred_cells_xy = np.array([pred_cells.cen_x, pred_cells.cen_y]).T.astype(float)

                dist_matrix = distance_matrix(obs_cells_xy, pred_cells_xy, p=2)
                opt_i, opt_j = linear_sum_assignment(dist_matrix)

                matched_distances = dist_matrix[opt_i, opt_j]
                matched_distances[matched_distances > self.dist_limit_matching] = np.nan

                valid_opt_i = opt_i[np.isfinite(matched_distances)]
                valid_opt_j = opt_j[np.isfinite(matched_distances)]

                # hits and misses for contingency table
                n_hits = np.count_nonzero(np.isfinite(matched_distances))
                n_misses = len(obs_cells) - n_hits
                n_false_alarms = len(pred_cells) - n_hits

                cell_match_hits[ti, methods.index(method)] = n_hits
                cell_match_misses[ti, methods.index(method)] = n_misses
                cell_match_false_alarms[ti, methods.index(method)] = n_false_alarms

                # Get matched cells and their features
                for i, (obs_cell_id, pred_cell_id) in enumerate(zip(valid_opt_i, valid_opt_j)):
                    o_cell = obs_cells.iloc[obs_cell_id]
                    p_cell = pred_cells.iloc[pred_cell_id]

                    cell_match_dist[ti, methods.index(method), i] = dist_matrix[obs_cell_id, pred_cell_id]
                    # volume rain rate
                    cell_match_obs_sum_rr[ti, methods.index(method), i] = np.nansum(
                        x_obs_rr[self.prev_obs_times + ti, o_cell.y, o_cell.x]
                    )
                    cell_match_pred_sum_rr[ti, methods.index(method), i] = np.nansum(
                        x_pred_rr[method][ti, p_cell.y, p_cell.x]
                    )
                    # maximum rain rate
                    cell_match_obs_max_rr[ti, methods.index(method), i] = np.nanmax(
                        x_obs_rr[self.prev_obs_times + ti, o_cell.y, o_cell.x]
                    )
                    cell_match_pred_max_rr[ti, methods.index(method), i] = np.nanmax(
                        x_pred_rr[method][ti, p_cell.y, p_cell.x]
                    )
                    # mean rain rate
                    cell_match_obs_mean_rr[ti, methods.index(method), i] = np.nanmean(
                        x_obs_rr[self.prev_obs_times + ti, o_cell.y, o_cell.x]
                    )
                    cell_match_pred_mean_rr[ti, methods.index(method), i] = np.nanmean(
                        x_pred_rr[method][ti, p_cell.y, p_cell.x]
                    )
                    # cell area
                    cell_match_obs_area[ti, methods.index(method), i] = o_cell.area
                    cell_match_pred_area[ti, methods.index(method), i] = p_cell.area

                # Get unmatched cells and their features
                for i, cell_id in enumerate(list(set(obs_cells.index.values) - set(valid_opt_i))):
                    o_cell = obs_cells.loc[cell_id]

                    unmatched_obs_sum_rr[ti, methods.index(method), i] = np.nansum(
                        x_obs_rr[self.prev_obs_times + ti, o_cell.y, o_cell.x]
                    )
                    unmatched_obs_max_rr[ti, methods.index(method), i] = np.nanmax(
                        x_obs_rr[self.prev_obs_times + ti, o_cell.y, o_cell.x]
                    )
                    unmatched_obs_mean_rr[ti, methods.index(method), i] = np.nanmean(
                        x_obs_rr[self.prev_obs_times + ti, o_cell.y, o_cell.x]
                    )
                    unmatched_obs_area[ti, methods.index(method), i] = o_cell.area

                for i, cell_id in enumerate(list(set(pred_cells.index.values) - set(valid_opt_j))):
                    p_cell = pred_cells.loc[cell_id]

                    unmatched_pred_sum_rr[ti, methods.index(method), i] = np.nansum(
                        x_pred_rr[method][ti, p_cell.y, p_cell.x]
                    )
                    unmatched_pred_max_rr[ti, methods.index(method), i] = np.nanmax(
                        x_pred_rr[method][ti, p_cell.y, p_cell.x]
                    )
                    unmatched_pred_mean_rr[ti, methods.index(method), i] = np.nanmean(
                        x_pred_rr[method][ti, p_cell.y, p_cell.x]
                    )
                    unmatched_pred_area[ti, methods.index(method), i] = p_cell.area

        # Process through tracks that continue from previous observations
        obs_lifetimes = np.ones((len(prev_track_ids),)) * np.nan
        obs_area = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan
        obs_max_rr = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan
        obs_mean_rr = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan
        obs_sum_rr = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan
        prev_area = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
        prev_max_rr = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
        prev_mean_rr = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
        prev_sum_rr = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
        prev_merged = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
        prev_from_split = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
        prev_will_merge = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
        obs_merged = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan
        obs_from_split = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan
        obs_will_merge = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan

        pred_lifetimes = np.ones((len(prev_track_ids), len(methods))) * np.nan
        pred_area = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan
        pred_max_rr = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan
        pred_mean_rr = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan
        pred_sum_rr = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan
        pred_dist = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan
        pred_merged = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan
        pred_from_split = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan
        pred_will_merge = np.ones((len(prev_track_ids), len(self.leadtimes), len(methods))) * np.nan

        extra_stats = {}
        if additional_measurements is not None:
            for name, _ in additional_measurements.items():
                extra_stats[name] = {}
                extra_stats[name]["prev_obs_mean"] = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
                extra_stats[name]["prev_obs_max"] = np.ones((len(prev_track_ids), self.prev_obs_times)) * np.nan
                extra_stats[name]["next_obs_mean"] = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan
                extra_stats[name]["next_obs_max"] = np.ones((len(prev_track_ids), len(self.leadtimes))) * np.nan

        for i, track_id in enumerate(prev_track_ids):
            # Get max and mean of previous observations
            track_prev_obs = prev_track_dict[track_id].set_index("time")
            prev_obs_sel = track_prev_obs.index <= 0

            for idx, cell in track_prev_obs.loc[prev_obs_sel].iterrows():
                prev_area[i, idx - 1] = cell.area
                prev_mean_rr[i, idx - 1] = np.nanmean(prev_obs_rr[idx - 1, cell.y, cell.x])
                prev_max_rr[i, idx - 1] = np.nanmax(prev_obs_rr[idx - 1, cell.y, cell.x])
                prev_sum_rr[i, idx - 1] = np.nansum(prev_obs_rr[idx - 1, cell.y, cell.x])

                # Save status of merge
                prev_merged[i, idx - 1] = cell.merged
                prev_from_split[i, idx - 1] = cell.results_from_split
                prev_will_merge[i, idx - 1] = cell.will_merge

                if additional_measurements is not None:
                    for name, meas in additional_measurements.items():
                        extra_stats[name]["prev_obs_mean"][i, idx - 1] = np.nanmean(meas[idx - 1, cell.y, cell.x])
                        extra_stats[name]["prev_obs_max"][i, idx - 1] = np.nanmax(meas[idx - 1, cell.y, cell.x])

            try:
                track_obs = next_track_dict[track_id].set_index("time")
            except KeyError:
                # track not present in target observations
                continue
            obs_sel = track_obs.index > 0
            obs_lifetimes[i] = (obs_sel).sum()

            for idx, cell in track_obs.loc[obs_sel].iterrows():
                obs_i = timelist.index(idx)

                obs_area[i, idx - 1] = cell.area
                obs_max_rr[i, idx - 1] = np.nanmax(x_obs_rr[obs_i, cell.y, cell.x])
                obs_mean_rr[i, idx - 1] = np.nanmean(x_obs_rr[obs_i, cell.y, cell.x])
                obs_sum_rr[i, idx - 1] = np.nansum(x_obs_rr[obs_i, cell.y, cell.x])

                obs_merged[i, idx - 1] = cell.merged
                obs_from_split[i, idx - 1] = cell.results_from_split
                obs_will_merge[i, idx - 1] = cell.will_merge

                for name, meas in additional_measurements.items():
                    extra_stats[name]["next_obs_mean"][i, idx - 1] = np.nanmean(meas[obs_i, cell.y, cell.x])
                    extra_stats[name]["next_obs_max"][i, idx - 1] = np.nanmax(meas[obs_i, cell.y, cell.x])

            for mi, method in enumerate(methods):
                try:
                    track_pred = pred_track_dicts[method][track_id].set_index("time")
                except KeyError:
                    continue
                pred_sel = track_pred.index > 0
                pred_lifetimes[i, mi] = (pred_sel).sum()

                for idx, cell in track_pred.loc[pred_sel].iterrows():
                    pred_i = pred_times.index(idx)

                    # Calculate distance error
                    try:
                        obs_cell = track_obs.loc[idx]
                        pred_dist[i, idx - 1, mi] = np.sqrt(
                            (cell.cen_x - obs_cell.cen_x) ** 2 + (cell.cen_y - obs_cell.cen_y) ** 2
                        )
                    except KeyError:
                        # corresponding cell doesnt exist in target observations
                        pred_dist[i, idx - 1, mi] = np.nan

                    # Cell features
                    pred_area[i, idx - 1, mi] = cell.area
                    pred_max_rr[i, idx - 1, mi] = np.nanmax(x_pred_rr[method][pred_i, cell.y, cell.x])
                    pred_mean_rr[i, idx - 1, mi] = np.nanmean(x_pred_rr[method][pred_i, cell.y, cell.x])
                    pred_sum_rr[i, idx - 1, mi] = np.nansum(x_pred_rr[method][pred_i, cell.y, cell.x])

                    pred_merged[i, idx - 1, mi] = cell.merged
                    pred_from_split[i, idx - 1, mi] = cell.results_from_split
                    pred_will_merge[i, idx - 1, mi] = cell.will_merge

        # Gather arrays from extra_stats
        extra_stats_arrays = {}
        for name, d in extra_stats.items():
            extra_stats_arrays[f"{name}_prev_obs_mean"] = (
                ["sample", "track", "prev_time"],
                d["prev_obs_mean"][np.newaxis, ...],
            )
            extra_stats_arrays[f"{name}_prev_obs_max"] = (
                ["sample", "track", "prev_time"],
                d["prev_obs_max"][np.newaxis, ...],
            )
            extra_stats_arrays[f"{name}_next_obs_mean"] = (
                ["sample", "track", "leadtime"],
                d["next_obs_mean"][np.newaxis, ...],
            )
            extra_stats_arrays[f"{name}_next_obs_max"] = (
                ["sample", "track", "leadtime"],
                d["next_obs_max"][np.newaxis, ...],
            )

        ds = xr.Dataset(
            data_vars={
                **extra_stats_arrays,
                # Features for cells matched with hungarian algorithm per timestep
                "cell_match_hits": (["sample", "leadtime", "method"], cell_match_hits[np.newaxis, ...]),
                "cell_match_misses": (["sample", "leadtime", "method"], cell_match_misses[np.newaxis, ...]),
                "cell_match_false_alarms": (["sample", "leadtime", "method"], cell_match_false_alarms[np.newaxis, ...]),
                "cell_match_dist": (["sample", "leadtime", "method", "cell"], cell_match_dist[np.newaxis, ...]),
                "cell_match_obs_sum_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_obs_sum_rr[np.newaxis, ...],
                ),
                "cell_match_pred_sum_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_pred_sum_rr[np.newaxis, ...],
                ),
                "cell_match_obs_max_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_obs_max_rr[np.newaxis, ...],
                ),
                "cell_match_pred_max_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_pred_max_rr[np.newaxis, ...],
                ),
                "cell_match_obs_mean_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_obs_mean_rr[np.newaxis, ...],
                ),
                "cell_match_pred_mean_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_pred_mean_rr[np.newaxis, ...],
                ),
                "cell_match_obs_area": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_obs_area[np.newaxis, ...],
                ),
                "cell_match_pred_area": (
                    ["sample", "leadtime", "method", "cell"],
                    cell_match_pred_area[np.newaxis, ...],
                ),
                "unmatched_obs_sum_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    unmatched_obs_sum_rr[np.newaxis, ...],
                ),
                "unmatched_pred_sum_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    unmatched_pred_sum_rr[np.newaxis, ...],
                ),
                "unmatched_obs_max_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    unmatched_obs_max_rr[np.newaxis, ...],
                ),
                "unmatched_pred_max_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    unmatched_pred_max_rr[np.newaxis, ...],
                ),
                "unmatched_obs_mean_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    unmatched_obs_mean_rr[np.newaxis, ...],
                ),
                "unmatched_pred_mean_rr": (
                    ["sample", "leadtime", "method", "cell"],
                    unmatched_pred_mean_rr[np.newaxis, ...],
                ),
                "unmatched_obs_area": (["sample", "leadtime", "method", "cell"], unmatched_obs_area[np.newaxis, ...]),
                "unmatched_pred_area": (
                    ["sample", "leadtime", "method", "cell"],
                    unmatched_pred_area[np.newaxis, ...],
                ),
                # features for cell tracks starting from observations
                "obs_lifetime": (["sample", "track"], obs_lifetimes[np.newaxis, :]),
                "obs_area": (
                    ["sample", "track", "leadtime"],
                    obs_area[np.newaxis, ...],
                ),
                "obs_max_rr": (
                    ["sample", "track", "leadtime"],
                    obs_max_rr[np.newaxis, ...],
                ),
                "obs_mean_rr": (
                    ["sample", "track", "leadtime"],
                    obs_mean_rr[np.newaxis, ...],
                ),
                "obs_sum_rr": (
                    ["sample", "track", "leadtime"],
                    obs_sum_rr[np.newaxis, ...],
                ),
                "prev_area": (
                    ["sample", "track", "prev_time"],
                    prev_area[np.newaxis, ...],
                ),
                "prev_max_rr": (
                    ["sample", "track", "prev_time"],
                    prev_max_rr[np.newaxis, ...],
                ),
                "prev_mean_rr": (
                    ["sample", "track", "prev_time"],
                    prev_mean_rr[np.newaxis, ...],
                ),
                "prev_sum_rr": (
                    ["sample", "track", "prev_time"],
                    prev_sum_rr[np.newaxis, ...],
                ),
                "pred_lifetime": (
                    ["sample", "track", "method"],
                    pred_lifetimes[np.newaxis, ...],
                ),
                "pred_area": (
                    ["sample", "track", "leadtime", "method"],
                    pred_area[np.newaxis, ...],
                ),
                "pred_max_rr": (
                    ["sample", "track", "leadtime", "method"],
                    pred_max_rr[np.newaxis, ...],
                ),
                "pred_mean_rr": (
                    ["sample", "track", "leadtime", "method"],
                    pred_mean_rr[np.newaxis, ...],
                ),
                "pred_sum_rr": (
                    ["sample", "track", "leadtime", "method"],
                    pred_sum_rr[np.newaxis, ...],
                ),
                "pred_dist": (
                    ["sample", "track", "leadtime", "method"],
                    pred_dist[np.newaxis, ...],
                ),
                # Splits and merges
                "prev_merged": (["sample", "track", "prev_time"], prev_merged[np.newaxis, ...]),
                "prev_from_split": (["sample", "track", "prev_time"], prev_from_split[np.newaxis, ...]),
                "prev_will_merge": (["sample", "track", "prev_time"], prev_will_merge[np.newaxis, ...]),
                "obs_merged": (["sample", "track", "leadtime"], obs_merged[np.newaxis, ...]),
                "obs_from_split": (["sample", "track", "leadtime"], obs_from_split[np.newaxis, ...]),
                "obs_will_merge": (["sample", "track", "leadtime"], obs_will_merge[np.newaxis, ...]),
                "pred_merged": (["sample", "track", "leadtime", "method"], pred_merged[np.newaxis, ...]),
                "pred_from_split": (["sample", "track", "leadtime", "method"], pred_from_split[np.newaxis, ...]),
                "pred_will_merge": (["sample", "track", "leadtime", "method"], pred_will_merge[np.newaxis, ...]),
            },
            coords={
                "method": methods,
                "leadtime": self.leadtimes,
                "prev_time": prev_obs_times,
                "sample": [sample],
                "track": prev_track_ids,
                "cell": list(range(max_num_cells)),
            },
        )

        if self.tables["cont_track_df"] is None:
            self.tables["cont_track_df"] = ds
            self.is_empty = False
        else:
            self.tables["cont_track_df"] = xr.concat([self.tables["cont_track_df"], ds], dim="sample")

    def compute(self):
        # No actual computation here, just return the data
        return self.tables["cont_track_df"]

    def merge(self, other):
        self.tables["cont_track_df"] = xr.concat(
            [self.tables["cont_track_df"], other.tables["cont_track_df"]], dim="sample"
        )

    @staticmethod
    def plot(**kwargs):
        raise NotImplementedError(f"Plotting not implemented for {__class__.__name__}")
