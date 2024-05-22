"""Plot nowcast gifs in separate figures.

Author: Jenna Ritvanen <jenna.ritvanen@fmi.fi>

"""

import argparse
import os
from copy import copy
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyart
from matplotlib.collections import LineCollection
from matplotlib import colors
from tqdm import tqdm

import palettable

from utils.config_utils import load_config
from utils.data_utils import load_data
from utils.plot_utils import plot_array

from utils.io_tools import load_predictions_dict
from utils.conversion_tools import dbz_to_rainrate
from utils.lagrangian_transform import read_advection_fields_from_nc

from plot_nowcast_figures import plot_objects, track_cells

pyart.load_config(os.environ.get("PYART_CONFIG"))

bad_times_file = Path("bad_times_input_data_plotting.csv")

BBOX_LW = 1.5
BBOX_COL = "tab:red"
RADAR_COL = "tab:orange"
RADAR_EDGECOL = "none"
RADAR_SIZE = 10
RADAR_MARKER = "X"

OBJECT_LW = 0.8
OBJECT_CMAP = palettable.colorbrewer.qualitative.Paired_12.mpl_colormap
OBJECT_CENTROID_MARKER = "o"
OBJECT_CENTROID_SIZE = 0.5
OBJECT_CENTROID_TRACK_COLOR = "k"

UNUSED_OBJECT_LW = 0.5
UNUSED_OBJECT_COLOR = "k"

RASTERIZE_OBJECTS = False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("configpath", type=str, help="Configuration file path")
    argparser.add_argument("date", type=str, help="date to be plotted (YYYYmmddHHMM)")
    args = argparser.parse_args()

    date = datetime.strptime(args.date, "%Y%m%d%H%M")
    sample = date.strftime("%Y-%m-%d %H:%M:%S")

    confpath = Path(args.configpath)
    conf = load_config(confpath)
    plt.style.use(conf.stylefile)

    outdir = Path(conf.outdir.format(year=date.year, month=date.month, day=date.day))
    outdir.mkdir(parents=True, exist_ok=True)

    # Set up figure
    m, n = conf.im_size

    if conf.plot_map:
        # Borders
        border = gpd.read_file(conf.map_params.border_shapefile)
        border_proj = border.to_crs(conf.map_params.proj)

        segments = [np.array(linestring.coords)[:, :2] for linestring in border_proj["geometry"]]
        border_collection = LineCollection(segments, color="k", linewidth=1, zorder=0)

        # Radar locations
        if conf.map_params.radar_shapefile is not None:
            radar_locations = gpd.read_file(conf.map_params.radar_shapefile)
            radar_locations_proj = radar_locations.to_crs(conf.map_params.proj)
            xy = radar_locations_proj["geometry"].map(lambda point: point.xy)
            radar_locations_proj = list(zip(*xy))
        else:
            radar_locations_proj = None

    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(conf.col_width + 1, conf.row_height),
        sharex="col",
        sharey="row",
        squeeze=True,
        constrained_layout=True,
    )

    obs_out_file = "obs_{name}_{time:%Y%m%d%H%M}.png"
    ncst_out_file = "ncst_{method}_{time:%Y%m%d%H%M}.png"
    target_out_file = "target_{name}_{time:%Y%m%d%H%M}.png"

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
    nowcasts = load_predictions_dict(conf.nowcasts, date.strftime("%Y-%m-%d %H:%M:%S"), conf.leadtimes)
    # Calculate nowcast X and Y based on bbox
    ncst_xx = dataset.x.values
    ncst_yy = dataset.y.values
    if conf.nowcast_bbox is not None:
        ncst_xx = ncst_xx[conf.nowcast_bbox[2] : conf.nowcast_bbox[3]]
        ncst_yy = ncst_yy[conf.nowcast_bbox[0] : conf.nowcast_bbox[1]]

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

    def set_ax(axs, title=None):
        if conf.get("plot_nowcast_bbox", False):
            # Plot a box around the nowcast area
            axs.plot(
                ncst_xx[[0, -1, -1, 0, 0]],
                ncst_yy[[0, 0, -1, -1, 0]],
                color=BBOX_COL,
                lw=BBOX_LW,
                zorder=15,
            )

        if conf.plot_map:
            axs.add_collection(copy(border_collection))

            if radar_locations_proj is not None:
                axs.scatter(
                    *radar_locations_proj,
                    color=RADAR_COL,
                    edgecolor=RADAR_EDGECOL,
                    s=RADAR_SIZE,
                    marker=RADAR_MARKER,
                    zorder=10,
                    rasterized=RASTERIZE_OBJECTS,
                )

        axs.set_xticks(
            np.arange(
                dataset.x.values.min(),
                dataset.x.values.max(),
                conf.tick_spacing * 1e3,
            )
        )
        axs.set_yticks(
            np.arange(
                dataset.y.values.min(),
                dataset.y.values.max(),
                conf.tick_spacing * 1e3,
            )
        )
        axs.set_aspect(1)

        if conf.extent is not None:
            axs.set_xlim(
                (
                    dataset.x.values.min() + conf.extent[2] * 1e3,
                    dataset.x.values.min() + conf.extent[3] * 1e3,
                )
            )
            axs.set_ylim(
                (
                    dataset.y.values.min() + conf.extent[0] * 1e3,
                    dataset.y.values.min() + conf.extent[1] * 1e3,
                )
            )
        else:
            axs.set_xlim((dataset.x.values.min(), dataset.x.values.max()))
            axs.set_ylim((dataset.y.values.min(), dataset.y.values.max()))

        axs.grid(lw=0.8, color="tab:gray", ls=":", zorder=11)
        for axis in axs._axis_map.values():
            axis.set_zorder(15)

        for tick in axs.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in axs.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        for spine in ["top", "right"]:
            axs.spines[spine].set_visible(True)

        if title is not None:
            axs.text(
                0.02,
                0.978,
                title,  # \n{conf.input_data[name]['title']}",
                ha="left",
                va="top",
                color=plt.rcParams.get("axes.titlecolor"),
                fontsize=plt.rcParams.get("axes.titlesize"),
                fontweight=plt.rcParams.get("axes.titleweight"),
                transform=axs.transAxes,
                bbox=dict(facecolor="white", alpha=1.0, edgecolor="black", boxstyle="square,pad=0.5"),
                zorder=20,
            )

    # Plot observations
    for j, name in enumerate(conf.input_data_order):
        var_name = conf.input_data[name].variable

        files = []
        for i, time in enumerate(obs_times):
            arr = dataset.sel(time=time)
            im = arr[var_name].to_numpy().squeeze()

            nan_mask = arr[f"{var_name}_nan_mask"].values.squeeze()
            zero_mask = np.isclose(im, 0)

            im[zero_mask] = np.nan
            im[nan_mask] = np.nan

            # Plot data
            cbar = plot_array(
                axs,
                im.copy(),
                x=dataset.x.values,
                y=dataset.y.values,
                qty=conf.input_data[name].cmap_qty,
                colorbar=True,
                zorder=1,
            )
            axs.set_title(f"{pd.Timestamp(time):%Y-%m-%d %H:%M:%S}", ha="center")
            # axs.set_ylabel(f"{conf.input_data[name]['title']}")

            if conf.get("plot_objects"):
                print(f"Plotting objects for input at {time}, total {len(prev_cell_list[i])} cells")
                plot_objects(
                    axs,
                    time,
                    prev_track_ids,
                    prev_tracks,
                    prev_cell_list[i],
                    OBJECT_NORM,
                    "observed",
                    ncst_xx,
                    ncst_yy,
                )

            set_ax(axs, title=f"Input")

            imfile = outdir / obs_out_file.format(name=name, time=pd.Timestamp(time))
            files.append(imfile)
            fig.savefig(
                imfile,
                # bbox_inches="tight",
                dpi=conf.dpi,
            )
            cbar.remove()
            axs.clear()

        if conf.make_gif:
            frames = np.stack(
                [iio.imread(filename) for filename in sorted(files)],
                axis=0,
            )

            iio.imwrite(
                outdir / f"obs_{name}_{date:%Y%m%d%H%M}.gif",
                frames,
                duration=conf.duration_per_frame * 1000,
                loop=0,
            )

    # Plot targets
    for j, name in enumerate(conf.target_data_order):
        var_name = conf.input_data[name].variable

        files = []
        for i, time in enumerate(ncst_times):
            arr = dataset.sel(time=time)
            im = arr[var_name].to_numpy().squeeze()

            nan_mask = arr[f"{var_name}_nan_mask"].values.squeeze()
            zero_mask = np.isclose(im, 0)

            im[zero_mask] = np.nan
            im[nan_mask] = np.nan

            # Plot data
            cbar = plot_array(
                axs,
                im.copy(),
                x=dataset.x.values,
                y=dataset.y.values,
                qty=conf.input_data[name].cmap_qty,
                colorbar=True,
                zorder=1,
            )
            axs.set_title(f"{pd.Timestamp(time):%Y-%m-%d %H:%M:%S}", ha="center")
            # axs.set_ylabel(f"{conf.input_data[name]['title']}")

            if conf.get("plot_objects"):
                print(f"Plotting objects for target at {time}, total {len(next_cell_list[len(obs_times) + i])} cells")
                plot_objects(
                    axs,
                    time,
                    prev_track_ids,
                    next_tracks,
                    next_cell_list[len(obs_times) + i],
                    OBJECT_NORM,
                    "target",
                    ncst_xx,
                    ncst_yy,
                )

            set_ax(axs, title="Target")

            imfile = outdir / target_out_file.format(name=name, time=pd.Timestamp(time))
            files.append(imfile)
            fig.savefig(
                imfile,
                # bbox_inches="tight",
                dpi=conf.dpi,
            )
            cbar.remove()
            axs.clear()

        if conf.make_gif:
            frames = np.stack(
                [iio.imread(filename) for filename in sorted(files)],
                axis=0,
            )

            iio.imwrite(
                outdir / f"target_{name}_{date:%Y%m%d%H%M}.gif",
                frames,
                duration=conf.duration_per_frame * 1000,
                loop=0,
            )

    # Plot predictions
    for j, method in enumerate(conf.nowcasts.keys()):
        files = []

        for i in range(len(conf.leadtimes)):
            nan_mask = np.isnan(nowcasts[method][i]).astype(float)
            nan_mask[nan_mask == 0] = np.nan

            if conf.plot_diff_from_obs:
                obs = dataset.sel(time=ncst_times[i])["RATE"].values

                if conf.nowcast_bbox is not None:
                    obs = obs[
                        conf.nowcast_bbox[0] : conf.nowcast_bbox[1],
                        conf.nowcast_bbox[2] : conf.nowcast_bbox[3],
                    ]

                arr = nowcasts[method][i] - obs
                cbar = plot_array(
                    axs,
                    arr,
                    x=ncst_xx,
                    y=ncst_yy,
                    qty="RR_diff",
                    colorbar=True,
                    extend="both",
                    zorder=1,
                    flip=True,
                )
            else:
                cbar = plot_array(
                    axs,
                    nowcasts[method][i],
                    x=ncst_xx,
                    y=ncst_yy,
                    qty=conf.nowcasts[method].cmap_qty,
                    colorbar=True,
                    extend="max",
                    zorder=1,
                    flip=True,
                )

            # Plot nan mask
            axs.pcolormesh(
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

            axs.set_title(f"{date:%Y-%m-%d %H:%M} + {conf.leadtimes[i] * 5:>3} min ", ha="center")
            # axs.set_ylabel(f"{conf.nowcasts[method]['title']}")

            if conf.get("plot_objects"):
                print(
                    f"Plotting objects for {method} at {ncst_times[i]}, total {len(ncst_cells[method][len(obs_times) + i])} cells"
                )
                plot_objects(
                    axs,
                    ncst_times[i],
                    prev_track_ids,
                    ncst_object_tracks[method],
                    ncst_cells[method][len(obs_times) + i],
                    OBJECT_NORM,
                    method,
                    ncst_xx,
                    ncst_yy,
                )

            set_ax(axs, title=f"{conf.nowcasts[method]['title']}")

            imfile = outdir / ncst_out_file.format(method=method, time=pd.Timestamp(ncst_times[i]))
            files.append(imfile)
            fig.savefig(
                imfile,
                # bbox_inches="tight",
                dpi=conf.dpi,
            )
            cbar.remove()
            axs.clear()

        if conf.make_gif:
            frames = np.stack(
                [iio.imread(filename) for filename in sorted(files)],
                axis=0,
            )

            iio.imwrite(
                outdir / f"ncst_{method}_{date:%Y%m%d%H%M}.gif",
                frames,
                duration=conf.duration_per_frame * 1000,
                loop=0,
            )
