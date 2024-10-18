"""Plot example nowcasts.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
from copy import copy
from datetime import datetime
from pathlib import Path
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyart
import xarray as xr
from matplotlib import colors
from matplotlib.collections import LineCollection
from tqdm import tqdm

import palettable

from pysteps.tracking import tdating as tstorm_dating

from utils.config_utils import load_config
from utils.plot_utils import plot_array

from utils.data_utils import (
    load_data,
)
from utils.io_tools import load_predictions_dict
from utils.conversion_tools import dbz_to_rainrate, rainrate_to_dbz
from utils.lagrangian_transform import read_advection_fields_from_nc

pyart.load_config(os.environ.get("PYART_CONFIG"))

bad_times_file = Path("bad_times_nowcast_plotting.csv")


BBOX_LW = 1.5
BBOX_COL = "tab:red"
RADAR_COL = "tab:orange"
RADAR_EDGECOL = "none"
RADAR_SIZE = 50
RADAR_MARKER = "X"

OBJECT_LW = 0.8
# OBJECT_CMAP = palettable.colorbrewer.qualitative.Paired_12.mpl_colormap
# OBJECT_CMAP = palettable.cubehelix.perceptual_rainbow_16.mpl_colormap
OBJECT_CMAP = palettable.tableau.Tableau_20.mpl_colormap
OBJECT_CENTROID_MARKER = "o"
OBJECT_CENTROID_SIZE = 0.5
OBJECT_CENTROID_TRACK_COLOR = "k"

UNUSED_OBJECT_LW = 0.5
UNUSED_OBJECT_COLOR = "k"

RASTERIZE_OBJECTS = False


def plot_objects(
    ax,
    time,
    analysis_track_ids,
    cell_tracks,
    all_cells,
    object_norm,
    method,
    ncst_xx,
    ncst_yy,
):
    """
    Plot object tracks on the given axis.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis on which to plot the object tracks.
    - time (datetime.datetime): The time at which to plot the object tracks.
    - analysis_track_ids (list): List of track IDs included in the analysis (plotted with color).
    - unused_cell_track_ids (list): List of track IDs that are not included in the analysis (plotted with black).
    - cell_tracks (dict): Dictionary containing cell tracks indexed by track ID.
    - object_norm (matplotlib.colors.Normalize): Normalization for the object track colors.
    """
    # Plot object tracks
    num_plotted_cells = 0
    plotted_tracks_ids = []
    for track_id in analysis_track_ids:
        try:
            track = cell_tracks[track_id].set_index("time")
        except IndexError:
            import ipdb

            ipdb.set_trace()
            print("IndexError")
        except KeyError:
            continue
        track.index = track.index.to_pydatetime()
        if time not in track.index:
            logging.debug(f"Track {track_id} not found in {method}!")
            continue

        # Pick current timestep from track
        cell = track.loc[time]

        # Plot track from cell centerpoints
        cells_at_prev_times = track[track.index <= time]

        cen_x = ncst_xx[cells_at_prev_times.cen_x.values]
        cen_y = ncst_yy[cells_at_prev_times.cen_y.values]

        obj_col = OBJECT_CMAP(object_norm(analysis_track_ids.index(track_id)))
        ax.plot(
            cen_x,
            cen_y,
            c=OBJECT_CENTROID_TRACK_COLOR,
            mfc=obj_col,
            mec=obj_col,
            lw=OBJECT_LW,
            zorder=20,
            marker=OBJECT_CENTROID_MARKER,
            markersize=OBJECT_CENTROID_SIZE,
            rasterized=RASTERIZE_OBJECTS,
        )

        for cont in cell.cont:
            xx_ = ncst_xx[np.rint(cont[:, 1]).astype(int)]
            yy_ = ncst_yy[np.rint(cont[:, 0]).astype(int)]

            ax.plot(
                xx_,
                yy_,
                color=obj_col,
                lw=OBJECT_LW,
                zorder=20,
                rasterized=RASTERIZE_OBJECTS,
            )
        plotted_tracks_ids.append(track_id)
        num_plotted_cells += 1
    # Plot objects that are not included in the analysis since
    # they are not matched to tracks at time t0
    plot_unused_ids = list(set(all_cells.ID) - set(plotted_tracks_ids))
    for track_id in plot_unused_ids:
        cell = all_cells.loc[all_cells.ID == track_id]
        for cont in cell.cont.item():
            xx_ = ncst_xx[np.rint(cont[:, 1]).astype(int)]
            yy_ = ncst_yy[np.rint(cont[:, 0]).astype(int)]

            ax.plot(
                xx_,
                yy_,
                color=UNUSED_OBJECT_COLOR,
                lw=UNUSED_OBJECT_LW,
                zorder=20,
                rasterized=RASTERIZE_OBJECTS,
            )
        num_plotted_cells += 1
    print(f"Plotted {num_plotted_cells} cells at {time}")


def track_cells(conf, prev_rr_obs, next_rr_obs, obs_times, load_ncst_times, nowcasts):
    # Transform to dbz
    prev_rr_obs = rainrate_to_dbz(
        prev_rr_obs,
        zr_a=conf.rz_transform_params.zr_a,
        zr_b=conf.rz_transform_params.zr_b,
        thresh=conf.rz_transform_params.thresh,
        zerovalue=conf.rz_transform_params.zerovalue,
    )
    next_rr_obs = rainrate_to_dbz(
        next_rr_obs,
        zr_a=conf.rz_transform_params.zr_a,
        zr_b=conf.rz_transform_params.zr_b,
        thresh=conf.rz_transform_params.thresh,
        zerovalue=conf.rz_transform_params.zerovalue,
    )

    # Track objects in prev obs
    prev_track_list, prev_cell_list, prev_label_list = tstorm_dating.dating(
        input_video=prev_rr_obs,
        timelist=obs_times,
        **conf.object_params.tdating_kwargs,
    )
    # Track objects in next obs
    next_track_list, next_cell_list, next_label_list = tstorm_dating.dating(
        input_video=np.concatenate([prev_rr_obs[-2:, ...], next_rr_obs], axis=0),
        timelist=obs_times[-2:] + load_ncst_times,
        start=2,
        cell_list=prev_cell_list.copy(),
        label_list=prev_label_list.copy(),
        **conf.object_params.tdating_kwargs,
    )

    prev_track_ids = [df.ID.unique().item() for df in prev_track_list]
    prev_tracks = {df.ID.unique().item(): df for df in prev_track_list}
    next_tracks = {df.ID.unique().item(): df for df in next_track_list}

    ncst_object_tracks = {}
    ncst_cells = {}
    # Nowcasts are still DBZH at this point, so we don't need to transform them
    # Track objects in nowcasts
    for j, method in enumerate(conf.nowcasts.keys()):
        pred_arr = rainrate_to_dbz(
            np.flip(nowcasts[method], axis=1),
            zr_a=conf.rz_transform_params.zr_a,
            zr_b=conf.rz_transform_params.zr_b,
            thresh=conf.rz_transform_params.thresh,
            zerovalue=conf.rz_transform_params.zerovalue,
        )
        pred_track_list, pred_cell_list, pred_label_list = tstorm_dating.dating(
            input_video=np.concatenate([prev_rr_obs[-2:, ...], pred_arr], axis=0),
            timelist=obs_times[-2:] + load_ncst_times,
            start=2,
            cell_list=prev_cell_list.copy(),
            label_list=prev_label_list.copy(),
            **conf.object_params.tdating_kwargs,
        )
        ncst_object_tracks[method] = {df.ID.unique().item(): df for df in pred_track_list}
        ncst_cells[method] = pred_cell_list

    if conf.object_params.remove_from_track_splits_merges:
        # Go through tracks and cut them at first time cell is merged or split
        new_prev_track_list = []
        for track in prev_track_list:
            ok_cells_condition = (track.merged | track.results_from_split).cumsum() == 0
            new_prev_track_list.append(track[ok_cells_condition])

            if ok_cells_condition.sum() < len(track):
                # removed some cells before t0, so need to discard the whole track
                prev_track_ids.remove(track.ID.unique().item())
        prev_track_list = new_prev_track_list

    if (
        conf.object_params.remove_from_track_splits_merges_after_t0
        or conf.object_params.remove_from_track_splits_merges
    ):
        # Go through tracks and cut them at first time cell is merged or split
        new_next_track_list = []
        for track in next_track_list:
            ok_cells_condition = (track.merged | track.results_from_split).cumsum() == 0
            new_next_track_list.append(track[ok_cells_condition])
        next_track_list = new_next_track_list

        new_ncst_object_tracks = {}
        for j, method in enumerate(conf.nowcasts.keys()):
            new_ncst_object_tracks[method] = {}
            for tid, track in ncst_object_tracks[method].items():
                ok_cells_condition = (track.merged | track.results_from_split).cumsum() == 0
                new_ncst_object_tracks[method][tid] = track[ok_cells_condition]
        ncst_object_tracks = new_ncst_object_tracks

    # Set norm for determining object track colors since this depends on the number of tracks
    OBJECT_NORM = colors.Normalize(vmin=0, vmax=len(prev_track_ids))
    return (
        prev_track_list,
        prev_cell_list,
        prev_label_list,
        next_track_list,
        next_cell_list,
        next_label_list,
        prev_track_ids,
        prev_tracks,
        next_tracks,
        ncst_object_tracks,
        ncst_cells,
        OBJECT_NORM,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("configpath", type=str, help="Configuration file path")
    argparser.add_argument("date", type=str, help="date to be plotted (YYYYmmddHHMM")
    args = argparser.parse_args()

    date = datetime.strptime(args.date, "%Y%m%d%H%M")
    sample = date.strftime("%Y-%m-%d %H:%M:%S")

    confpath = Path(args.configpath)
    conf = load_config(confpath)
    plt.style.use(conf.stylefile)

    outdir = Path(conf.outdir.format(year=date.year, month=date.month, day=date.day))
    outdir.mkdir(parents=True, exist_ok=True)

    # how many nowcasts to plot
    nrows = len(conf.input_data.keys()) + len(conf.nowcasts.keys()) + 1
    ncols = max(len(conf.leadtimes), conf.n_input_images)

    # Map parameters
    if conf.plot_map:
        # Borders
        border = gpd.read_file(conf.map_params.border_shapefile)
        border_proj = border.to_crs(conf.map_params.proj)

        segments = [np.array(linestring.coords)[:, :2] for linestring in border_proj["geometry"]]
        border_collection = LineCollection(segments, color="gray", linewidth=1, zorder=10, rasterized=True)

        # Radar locations
        if conf.map_params.radar_shapefile is not None:
            radar_locations = gpd.read_file(conf.map_params.radar_shapefile)
            radar_locations_proj = radar_locations.to_crs(conf.map_params.proj)
            xy = radar_locations_proj["geometry"].map(lambda point: point.xy)
            radar_locations_proj = list(zip(*xy))
        else:
            radar_locations_proj = None

    if conf.figsize is not None:
        figsize = conf.figsize
    else:
        figsize = (ncols * conf.col_width + 1, nrows * conf.row_height)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )

    # Load observations
    obs_times = pd.date_range(end=date, periods=conf.n_input_images, freq=conf.freq).to_pydatetime().tolist()
    ncst_times = [date + pd.Timedelta(minutes=5 * lt) for lt in conf.leadtimes]

    if not conf.get("plot_objects"):
        load_ncst_times = ncst_times
        load_ncst_steps = conf.leadtimes
    else:
        # We need to load every time step for object tracking
        load_ncst_times = [date + pd.Timedelta(minutes=5 * lt) for lt in range(1, max(conf.leadtimes) + 1)]
        load_ncst_steps = list(range(1, max(conf.leadtimes) + 1))

    get_times = [*obs_times, *load_ncst_times]

    missing_times_vars = []
    try:
        dataset = load_data(
            conf.input_data,
            date,
            get_times,
            len(get_times),
            None,
        )
    except (FileNotFoundError, ValueError) as e:
        with open(bad_times_file, "a") as f:
            f.write(f"{pd.Timestamp(date)},{e},\n")
        raise FileNotFoundError(f"Loading data failed for {date}") from e

    # Check if we have any missing data
    for var, da in dataset.data_vars.items():
        for tt in dataset.time.values:
            if ("input_files" in da.attrs.keys()) and np.all(np.isnan(da.sel(time=tt).values)):
                missing_times_vars.append((tt, var))

    if len(missing_times_vars):
        problem_times = []
        min_times = sorted(dataset.time.values)[: conf.n_images]
        for tt, var in missing_times_vars:
            if tt in min_times:
                with open(bad_times_file, "a") as f:
                    f.write(f"{pd.Timestamp(tt)},{var},\n")
                problem_times.append((pd.Timestamp(tt), var))
        raise FileNotFoundError(f"Files missing for {problem_times}")

    # Load nowcasts from HDF5 files
    nowcasts = load_predictions_dict(conf.nowcasts, date.strftime("%Y-%m-%d %H:%M:%S"), load_ncst_steps)
    if not isinstance(nowcasts, dict):
        raise ValueError(f"Nowcasts for {nowcasts} for {date} not found!")

    # Calculate nowcast X and Y based on bbox
    ncst_xx = dataset.x.values
    ncst_yy = dataset.y.values
    if conf.nowcast_bbox is not None:
        ncst_xx = ncst_xx[conf.nowcast_bbox[2] : conf.nowcast_bbox[3]]
        ncst_yy = ncst_yy[conf.nowcast_bbox[0] : conf.nowcast_bbox[1]]

    # Read advection field
    if conf.advection_field_path is not None:
        adv_path = datetime.strftime(date, conf.advection_field_path)
        adv_fields = read_advection_fields_from_nc(adv_path)

        bbox_x_slice = slice(conf.adv_field_bbox[0], conf.adv_field_bbox[1])
        bbox_y_slice = slice(conf.adv_field_bbox[2], conf.adv_field_bbox[3])

        # TODO implement picking correct field if multiple exist
        adv_field = adv_fields[next(iter(adv_fields))][:, bbox_x_slice, bbox_y_slice]
        quiver_thin = 25
        adv_field_x, adv_field_y = np.meshgrid(dataset.x.values, dataset.y.values)
        adv_field_alpha = 0.5
        adv_field_lw = 0.7
        adv_field_color = "k"
    else:
        adv_field = None

    if conf.get("plot_objects"):
        # Track object tracks in obs and nowcasts
        prev_rr_obs = dataset["RATE"].sel(time=obs_times).values
        next_rr_obs = dataset["RATE"].sel(time=load_ncst_times).values

        # Apply nowcasting bbox to arrrays
        if conf.nowcast_bbox is not None:
            prev_rr_obs = prev_rr_obs[
                :,
                conf.nowcast_bbox[0] : conf.nowcast_bbox[1],
                conf.nowcast_bbox[2] : conf.nowcast_bbox[3],
            ]
            next_rr_obs = next_rr_obs[
                :,
                conf.nowcast_bbox[0] : conf.nowcast_bbox[1],
                conf.nowcast_bbox[2] : conf.nowcast_bbox[3],
            ]

        (
            prev_track_list,
            prev_cell_list,
            prev_label_list,
            next_track_list,
            next_cell_list,
            next_label_list,
            prev_track_ids,
            prev_tracks,
            next_tracks,
            ncst_object_tracks,
            ncst_cells,
            OBJECT_NORM,
        ) = track_cells(conf, prev_rr_obs, next_rr_obs, obs_times, load_ncst_times, nowcasts)

    # Plot first input variables
    # Plot input
    row = 0
    for j, name in enumerate(conf.input_data_order):
        row = j
        var_name = conf.input_data[name].variable

        for i, time in enumerate(obs_times):
            arr = dataset.sel(time=time)
            im = arr[var_name].to_numpy().squeeze()

            nan_mask = arr[f"{var_name}_nan_mask"].values.squeeze()
            zero_mask = np.isclose(im, 0)

            im[zero_mask] = np.nan
            im[nan_mask] = np.nan

            # Plot input data
            plot_array(
                axes[row, i],
                im.copy(),
                x=dataset.x.values,
                y=dataset.y.values,
                qty=conf.input_data[name].cmap_qty,
                colorbar=(i == (conf.n_input_images - 1)),
                zorder=16,
            )
            axes[row, i].set_title(f"{pd.Timestamp(time):%Y-%m-%d %H:%M:%S}")

            if conf.get("plot_objects"):
                print(f"Plotting objects for input at {time}, total {len(prev_cell_list[i])} cells")
                plot_objects(
                    axes[row, i],
                    time,
                    prev_track_ids,
                    prev_tracks,
                    prev_cell_list[i],
                    OBJECT_NORM,
                    "observed",
                    ncst_xx,
                    ncst_yy,
                )

        # axes[row, 0].set_ylabel(f"{conf.input_data[name]['title']}")
        axes[row, 0].text(
            0.02,
            0.978,
            f"Input",  # \n{conf.input_data[name]['title']}",
            ha="left",
            va="top",
            color=plt.rcParams.get("axes.labelcolor"),
            fontsize=plt.rcParams.get("axes.labelsize"),
            fontweight=plt.rcParams.get("axes.labelweight"),
            transform=axes[row, 0].transAxes,
            bbox=dict(facecolor="white", alpha=1.0, edgecolor="black", boxstyle="square,pad=0.5"),
            zorder=20,
        )

        # Remove empty axes
        for j in range(i + 1, ncols):
            axes[row, j].axis("off")

    for name in conf.target_data_order:
        row += 1
        var_name = conf.input_data[name].variable
        # Plot target RATE
        for i, time in enumerate(ncst_times):
            arr = dataset.sel(time=time)
            im = arr[var_name].to_numpy().squeeze()

            nan_mask = arr[f"{var_name}_nan_mask"].values.squeeze()
            zero_mask = np.isclose(im, 0)

            im[zero_mask] = np.nan
            im[nan_mask] = np.nan

            # Plot input data
            plot_array(
                axes[row, i],
                im.copy(),
                x=dataset.x.values,
                y=dataset.y.values,
                qty=conf.input_data[name].cmap_qty,
                colorbar=(i == (ncols - 1)),
                zorder=16,
            )
            axes[row, i].set_title(f"{pd.Timestamp(time):%Y-%m-%d %H:%M:%S}")

            if conf.get("plot_objects"):
                print(
                    f"Plotting objects for target at {time}, total {len(next_cell_list[len(obs_times) + load_ncst_times.index(time)])} cells"
                )
                plot_objects(
                    axes[row, i],
                    time,
                    prev_track_ids,
                    next_tracks,
                    next_cell_list[len(obs_times) + load_ncst_times.index(time)],
                    OBJECT_NORM,
                    "target",
                    ncst_xx,
                    ncst_yy,
                )

        # axes[row, 0].set_ylabel(f"Target\n{conf.input_data[name]['title']}")
        # Add title to the first column
        axes[row, 0].text(
            0.02,
            0.978,
            f"Target",  # \n{conf.input_data[name]['title']}",
            ha="left",
            va="top",
            color=plt.rcParams.get("axes.labelcolor"),
            fontsize=plt.rcParams.get("axes.labelsize"),
            fontweight=plt.rcParams.get("axes.labelweight"),
            transform=axes[row, 0].transAxes,
            bbox=dict(facecolor="white", alpha=1.0, edgecolor="black", boxstyle="square,pad=0.5"),
            zorder=20,
        )

    # Plot nowcasts
    for j, method in enumerate(conf.nowcasts.keys()):
        row += 1
        for i in range(len(conf.leadtimes)):
            nan_mask = np.isnan(nowcasts[method][i]).astype(float)
            nan_mask[nan_mask == 0] = np.nan
            time = ncst_times[i]

            data_idx = load_ncst_times.index(time)

            if conf.plot_diff_from_obs:
                obs = dataset.sel(time=time)["RATE"].values

                if conf.nowcast_bbox is not None:
                    obs = obs[
                        conf.nowcast_bbox[0] : conf.nowcast_bbox[1],
                        conf.nowcast_bbox[2] : conf.nowcast_bbox[3],
                    ]

                arr = nowcasts[method][data_idx] - obs
                plot_array(
                    axes[row, i],
                    arr,
                    x=ncst_xx,
                    y=ncst_yy,
                    qty="RR_diff",
                    colorbar=(i == ncols - 1),
                    extend="both",
                    flip=True,
                    zorder=16,
                )
            else:
                plot_array(
                    axes[row, i],
                    nowcasts[method][data_idx],
                    x=ncst_xx,
                    y=ncst_yy,
                    qty=conf.nowcasts[method].cmap_qty,
                    colorbar=(i == ncols - 1),
                    extend="max",
                    flip=True,
                    zorder=16,
                )

            # Plot nan mask
            axes[row, i].pcolormesh(
                ncst_xx,
                ncst_yy,
                np.flipud(nan_mask),
                cmap=colors.ListedColormap(
                    [
                        "white",
                        "tab:gray",
                    ]
                ),
                zorder=9,
                rasterized=True,
                vmin=0,
                vmax=1,
                alpha=0.5,
            )
            # axes[row, i].set_title(times[conf.n_input_images + i])
            axes[row, i].set_title(f"{date:%Y-%m-%d %H:%M} + {conf.leadtimes[i] * 5:>3} min ")

            # Plot advection field
            if adv_field is not None:
                axes[row, i].quiver(
                    adv_field_x[::quiver_thin, ::quiver_thin],
                    np.flipud(adv_field_y)[::quiver_thin, ::quiver_thin],
                    adv_field[0, ...][::quiver_thin, ::quiver_thin],
                    np.flipud(adv_field[1, ...])[::quiver_thin, ::quiver_thin],
                    linewidth=adv_field_lw,
                    color=adv_field_color,
                    alpha=adv_field_alpha,
                    zorder=12,
                    rasterized=RASTERIZE_OBJECTS,
                )

            if conf.get("plot_objects"):
                print(
                    f"Plotting objects for {method} at {time}, total {len(ncst_cells[method][len(obs_times) + data_idx])} cells"
                )
                plot_objects(
                    axes[row, i],
                    time,
                    prev_track_ids,
                    ncst_object_tracks[method],
                    ncst_cells[method][len(obs_times) + data_idx],
                    OBJECT_NORM,
                    method,
                    ncst_xx,
                    ncst_yy,
                )

        # axes[row, 0].set_ylabel(conf.nowcasts[method]["title"])
        axes[row, 0].text(
            0.02,
            0.978,
            f"{conf.nowcasts[method]['title']}",  # \n{conf.input_data[name]['title']}",
            ha="left",
            va="top",
            color=plt.rcParams.get("axes.labelcolor"),
            fontsize=plt.rcParams.get("axes.labelsize"),
            fontweight=plt.rcParams.get("axes.labelweight"),
            transform=axes[row, 0].transAxes,
            bbox=dict(facecolor="white", alpha=1.0, edgecolor="black", boxstyle="square,pad=0.5"),
            zorder=20,
        )

    # if conf.plot_map:
    #     COPYRIGHT_TEXT = "Map tiles by Stamen Design, under CC BY 3.0. Map data by OpenStreetMap, under ODbL."
    #     fig.text(0.99, -0.005, COPYRIGHT_TEXT, fontsize=4, zorder=10, ha="right")

    for ax in axes.flat:
        if not ax.axison:
            # Skip axis, was turned off earlier
            continue

        ax.set_xticks(np.arange(dataset.x.values.min(), dataset.x.values.max(), conf.tick_spacing * 1e3))
        ax.set_yticks(np.arange(dataset.y.values.min(), dataset.y.values.max(), conf.tick_spacing * 1e3))
        ax.set_aspect(1)

        if conf.get("plot_nowcast_bbox", False):
            # Plot a box around the nowcast area
            ax.plot(
                ncst_xx[[0, -1, -1, 0, 0]],
                ncst_yy[[0, 0, -1, -1, 0]],
                color=BBOX_COL,
                lw=BBOX_LW,
                zorder=15,
                rasterized=RASTERIZE_OBJECTS,
            )

        if conf.plot_map:
            ax.add_collection(copy(border_collection))

            if radar_locations_proj is not None:
                ax.scatter(
                    *radar_locations_proj,
                    color=RADAR_COL,
                    edgecolor=RADAR_EDGECOL,
                    s=RADAR_SIZE,
                    marker=RADAR_MARKER,
                    zorder=10,
                    rasterized=RASTERIZE_OBJECTS,
                )

        if conf.zoom_bbox is not None:
            im_width = dataset.x.values.max() - dataset.x.values.min()
            im_height = dataset.y.values.max() - dataset.y.values.min()

            ax.set_xlim(
                (
                    dataset.x.values.min() + im_width * conf.zoom_bbox[0],
                    dataset.x.values.min() + im_width * conf.zoom_bbox[1],
                )
            )
            ax.set_ylim(
                (
                    dataset.y.values.min() + im_height * conf.zoom_bbox[2],
                    dataset.y.values.min() + im_height * conf.zoom_bbox[3],
                )
            )

        else:
            ax.set_xlim((dataset.x.values.min(), dataset.x.values.max()))
            ax.set_ylim((dataset.y.values.min(), dataset.y.values.max()))

        ax.grid(lw=0.5, color="tab:gray", ls=":", zorder=11)
        for axis in ax._axis_map.values():
            axis.set_zorder(15)

        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(True)

    # fig.subplots_adjust()
    fig.savefig(
        outdir / date.strftime(conf.filename),
        bbox_inches="tight",
        dpi=conf.dpi,
    )
