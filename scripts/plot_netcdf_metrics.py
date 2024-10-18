"""Plot metrics from netcdf files."""

import argparse
from pathlib import Path
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pysteps.visualization.spectral import plot_spectrum1d
import geopandas as gpd
from matplotlib.collections import LineCollection
from matplotlib import colors, cm, gridspec, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from copy import copy
import cmcrameri  # noqa
import string

from verif.io_tools import load_metrics, load_yaml_config

UNIT_STRINGS = {
    "mmh": r"$\mathrm{mm\,h}^{-1}$",
    "dbz": r"$\mathrm{dBZ}$",
    "meters": r"$\mathrm{m}$",
}

alphabet = string.ascii_lowercase


def nested_list_to_tuple(lst):
    return tuple(nested_list_to_tuple(i) if isinstance(i, list) else i for i in lst)


def set_ax(ax, score_conf, leadtime_limits, leadtime_locator_multiples=[15, 5]):
    """Set axis limits and ticks."""
    if score_conf["limits"] is not None:
        ax.set_ylim(*score_conf["limits"])
    else:
        ax.autoscale(enable=True, axis="y", tight=True)
    if score_conf["ticks"] and len(score_conf["ticks"]) == 3:
        ax.set_yticks(np.arange(*score_conf["ticks"]))
    elif score_conf["ticks"] and len(score_conf["ticks"]) == 2:
        ax.yaxis.set_major_locator(plt.MultipleLocator(score_conf["ticks"][0]))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(score_conf["ticks"][1]))

    if score_conf.get("log_scale"):
        if score_conf["limits"] is not None:
            ax.set_ylim([10 ** score_conf["limits"][0], 10 ** score_conf["limits"][1]])
        else:
            ax.autoscale(enable=True, axis="y", tight=True)

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=15))
        ax.yaxis.set_minor_locator(plt.NullLocator())

    ax.xaxis.set_major_locator(plt.MultipleLocator(leadtime_locator_multiples[0]))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(leadtime_locator_multiples[1]))

    # Add first and last leadtime tick labels
    ax.set_xticks(list(ax.get_xticks()) + leadtime_limits)

    ax.set_xlim(*leadtime_limits)
    ax.set_xlabel("Leadtime [min]")


def plot_categorical_metrics(ds, config, save_dir):
    """Plot categorical metrics.

    Plots metrics in a plot with thresholds in columns and metrics in rows.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the metrics.
    config : addict.Dict
        Configuration dictionary.
    save_dir : pathlib.Path
        Directory to save the plot to.

    """
    thresholds = set(ds.threshold.values).intersection(set([float(t) for t in config.thresholds]))
    # Plot categorical metrics
    ncols = len(thresholds)
    nrows = len(config.categorical_metrics) + int(config.get("plot_count", 0))
    fig = plt.figure(
        figsize=(config.figures.col_width * ncols, config.figures.row_height * nrows),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=True)

    if len(config.categorical_metrics) == 1:
        subfigs = [subfigs]

    if config.legend_order is None:
        legend_order = config.methods
    legend_label_order = [config.methods[model]["label"] for model in legend_order if model in ds]

    for mi, metric in enumerate(config.categorical_metrics):
        axs = subfigs[mi].subplots(nrows=1, ncols=ncols, squeeze=True, sharey=False)
        for ti, thr in enumerate(thresholds):
            for model in config.methods.keys():
                ds[model].sel(cat_metric=metric, threshold=thr).plot.line(
                    ax=axs[ti],
                    c=config.methods[model]["color"],
                    label=config.methods[model]["label"],
                    linestyle=config.methods[model]["linestyle"],
                )
            set_ax(
                axs[ti],
                config.metric_conf[metric],
                config.leadtime_limits,
                config.leadtime_locator_multiples,
            )
            axs[ti].set_ylabel(config.metric_conf[metric]["label"])

            axs[ti].legend()
            axs[ti].grid(which="both", axis="both")
            if config.write_panel_labels:
                label = f"({alphabet[(mi * ncols + ti) % len(alphabet)]}) "
            else:
                label = ""
            axs[ti].set_title(f"{label}$\mathrm{{R}}_\mathrm{{thr}} = {thr:.1f}~${UNIT_STRINGS[config.unit]}")

            handles, labels = axs[ti].get_legend_handles_labels()
            order = [labels.index(label) for label in legend_label_order]
            axs[ti].legend(
                [handles[idx] for idx in order],
                [labels[idx] for idx in order],
                # bbox_to_anchor=(0.0, 0.7, 1.0, 0.3),
            )

        axs[0].set_ylabel(config.metric_conf[metric]["label"])
        subfigs[mi].suptitle(
            config.metric_conf[metric]["full_name"],
            color=plt.rcParams["axes.titlecolor"],
        )

    if config.get("plot_count", 0):
        metric = "COUNT"
        axs = subfigs[-1].subplots(nrows=1, ncols=ncols, squeeze=True, sharey=False)
        for ti, thr in enumerate(config.thresholds):
            for model in config.methods.keys():
                ds[f"{model}_n_pixels"].sel(cat_metric=config.categorical_metrics[0], threshold=thr).plot.line(
                    ax=axs[ti],
                    c=config.methods[model]["color"],
                    label=config.methods[model]["label"],
                    linestyle=config.methods[model]["linestyle"],
                )

            ds[f"{model}_n_obs"].sel(cat_metric=config.categorical_metrics[0], threshold=thr).plot.line(
                ax=axs[ti],
                c="k",
                label="Observation",
                linestyle="solid",
            )

            set_ax(
                axs[ti],
                config.metric_conf[metric],
                config.leadtime_limits,
                config.leadtime_locator_multiples,
            )
            axs[ti].set_ylabel(config.metric_conf[metric]["label"])
            axs[ti].legend()
            axs[ti].grid(which="both", axis="both")
            axs[ti].set_title(f"$\mathrm{{R}}_\mathrm{{thr}} = {thr:.1f}~${UNIT_STRINGS[config.unit]}")

            handles, labels = axs[ti].get_legend_handles_labels()
            order = [labels.index(label) for label in legend_label_order + ["Observation"]]
            axs[ti].legend([handles[idx] for idx in order], [labels[idx] for idx in order])

        axs[0].set_ylabel(config.metric_conf[metric]["label"])
        subfigs[-1].suptitle(
            config.metric_conf[metric]["full_name"],
            color=plt.rcParams["axes.titlecolor"],
        )

    fig.set_constrained_layout_pads(hspace=0.1)
    for ext in config.output_formats:
        fig.savefig(save_dir / f"CATEGORICAL_METRICS.{ext}")
    plt.close(fig)


def plot_continuous_metrics(ds, config, save_dir):
    """Plot continuous metrics.

    Plots metrics in a 1xN plot with metrics in columns.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the metrics.
    config : addict.Dict
        Configuration dictionary.
    save_dir : pathlib.Path
        Directory to save the plot to.

    """
    thresholds = set(ds.threshold.values).intersection(set([float(t) for t in config.thresholds]))
    ncols = len(thresholds)
    nrows = len(config.continuous_metrics)

    fig = plt.figure(
        figsize=(config.figures.col_width * ncols, config.figures.row_height * nrows),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(
        nrows=nrows,
        ncols=1,
        squeeze=True,
        # sharey=False,
        # width_ratios=[*[1 for _ in range(ncols - 1)], 1.1],
    )

    if config.legend_order is None:
        legend_order = config.methods

    legend_label_order = [config.methods[model]["label"] for model in legend_order if model in ds]

    for mi, metric in enumerate(config.continuous_metrics):
        axs = subfigs[mi].subplots(nrows=1, ncols=ncols, squeeze=True, sharey=False)
        for ti, thr in enumerate(thresholds):
            for model in config.methods.keys():
                ds[model].sel(cont_metric=metric, threshold=thr).plot.line(
                    ax=axs[ti],
                    c=config.methods[model]["color"],
                    label=config.methods[model]["label"],
                    linestyle=config.methods[model]["linestyle"],
                )
            set_ax(
                axs[ti],
                config.metric_conf[metric],
                config.leadtime_limits,
                config.leadtime_locator_multiples,
            )
            axs[ti].set_ylabel(config.metric_conf[metric]["label"])
            axs[ti].legend()
            axs[ti].grid(which="both", axis="both")

            if config.write_panel_labels:
                label = f"({alphabet[(mi * ncols + ti) % len(alphabet)]}) "
            else:
                label = ""
            if np.isfinite(thr):
                axs[ti].set_title(f"{label}$\mathrm{{R}}_\mathrm{{thr}} = {thr:.1f}~${UNIT_STRINGS[config.unit]}")
            else:
                axs[ti].set_title(f"{label}No threshold")

            handles, labels = axs[ti].get_legend_handles_labels()
            order = [labels.index(label) for label in legend_label_order]
            axs[ti].legend(
                [handles[idx] for idx in order],
                [labels[idx] for idx in order],
                bbox_to_anchor=(0.0, 0.7, 1.0, 0.3),
            )

        subfigs[mi].suptitle(f'{config.metric_conf[metric]["full_name"]}')

    fig.set_constrained_layout_pads(hspace=0.1)
    for ext in config.output_formats:
        fig.savefig(save_dir / f"CONTINUOUS_METRICS.{ext}")
    plt.close(fig)


def plot_fss_metrics(ds, config, save_dir):
    """Plot FSS metrics.

    Plots metrics in a plot with thresholds in columns and scales in rows.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the metrics.
    config : addict.Dict
        Configuration dictionary.
    save_dir : pathlib.Path
        Directory to save the plot to.

    """
    # Plot FSS
    # scales on rows, thresholds in columns
    nrows = len(config.scales)
    ncols = len(config.thresholds)
    metric = "FSS"

    if config.legend_order is None:
        legend_order = config.methods
    legend_label_order = [config.methods[model]["label"] for model in legend_order if model in ds]

    ds["FSS_u"] = 0.5 + ds["f0"] / 2

    fig = plt.figure(
        figsize=(
            (config.figures.col_width + 0.3) * ncols,
            config.figures.row_height * nrows,
        ),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=True)

    for si, scale in enumerate(config.scales):
        axs = subfigs[si].subplots(nrows=1, ncols=ncols, squeeze=True, sharey=False)
        for ti, thr in enumerate(config.thresholds):
            for model in config.methods.keys():
                ds[model].sel(scale=scale, threshold=thr).plot.line(
                    ax=axs[ti],
                    c=config.methods[model]["color"],
                    label=config.methods[model]["label"],
                    linestyle=config.methods[model]["linestyle"],
                )
            ds["FSS_u"].sel(scale=scale, threshold=thr).plot.line(
                ax=axs[ti],
                c="k",
                label=None,
                linestyle="--",
                lw=1,
            )
            set_ax(
                axs[ti],
                config.metric_conf[metric],
                config.leadtime_limits,
                config.leadtime_locator_multiples,
            )
            axs[ti].set_ylabel(config.metric_conf[metric]["label"])
            axs[ti].legend()
            axs[ti].grid(which="both", axis="both")

            if config.write_panel_labels:
                label = f"({alphabet[(si*ncols + ti) % len(alphabet)]}) "
            else:
                label = ""

            axs[ti].set_title(
                f"{label}scale = {scale} km $\mathrm{{R}}_\mathrm{{thr}} " f"= {thr:.1f}~${UNIT_STRINGS[config.unit]}"
            )

            handles, labels = axs[ti].get_legend_handles_labels()
            order = [labels.index(label) for label in legend_label_order]
            axs[ti].legend(
                [handles[idx] for idx in order],
                [labels[idx] for idx in order],
                bbox_to_anchor=(0.0, 0.7, 1.0, 0.3),
            )

        axs[0].set_ylabel(config.metric_conf[metric]["label"])
    fig.suptitle(config.metric_conf[metric]["full_name"], color=plt.rcParams["axes.titlecolor"])

    fig.set_constrained_layout_pads(hspace=0.1)
    for ext in config.output_formats:
        fig.savefig(save_dir / f"FSS_METRICS.{ext}")
    plt.close(fig)


def plot_rapsd_metrics(ds, config, save_dir, nrows=2):
    """Plot RAPSD metrics.

    Plots RAPSD metrics. The number of columns is determined by the number of
    existing leadtimes and the number of rows is determined by the nrows argument.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the metrics.
    config : addict.Dict
        Configuration dictionary.
    save_dir : pathlib.Path
        Directory to save the plot to.
    nrows : int, optional
        Number of rows in the plot, by default 2

    """
    rapsd_leadtimes = ds.leadtime.values

    if config.legend_order is None:
        legend_order = config.methods
    legend_label_order = [config.methods[model]["label"] for model in legend_order if model in ds]
    legend_label_order.append("Observation")

    ncols = (len(rapsd_leadtimes) + 1) // nrows

    ticks = np.array([2**n for n in range(1, 10)])
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(
            (config.figures.col_width + 0.2) * ncols,
            config.figures.row_height * nrows,
        ),
        constrained_layout=True,
    )

    unitstr = f"{{({UNIT_STRINGS[config.unit][1:-1]})}}"

    for i, lt in enumerate(rapsd_leadtimes):
        ax = axs.flat[i]

        for model in config.methods.keys():
            # Each model line
            plot_spectrum1d(
                fft_freq=ds.freq,
                fft_power=ds[model].sel(leadtime=lt, type="prediction"),
                color=config.methods[model]["color"],
                linestyle=config.methods[model].get("linestyle", "solid"),
                x_units="km",
                y_units=unitstr,
                wavelength_ticks=ticks,
                ax=ax,
                label=config.methods[model]["label"],
            )
        # Observation line
        plot_spectrum1d(
            fft_freq=ds.freq,
            fft_power=ds[model].sel(leadtime=lt, type="observation"),
            color="k",
            linestyle="-",
            x_units="km",
            y_units=unitstr,
            wavelength_ticks=ticks,
            ax=ax,
            label="Observation",
        )
        ax.legend(loc="lower left")
        ax.set_title(f"{lt} min")

        ax.grid(which="both", axis="both")

        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(label) for label in legend_label_order]
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            bbox_to_anchor=(0.0, 0.7, 1.0, 0.3),
        )

    # Remove empty axes
    for ax in axs.flat[i + 1 :]:
        ax.remove()

    fig.suptitle(
        "Radially averaged power spectral density",
        color=plt.rcParams["axes.titlecolor"],
    )

    fig.set_constrained_layout_pads(hspace=0.05, wspace=0.05)
    for ext in config.output_formats:
        fig.savefig(save_dir / f"RAPSD_METRICS.{ext}")
    plt.close(fig)


def get_sym_cmap_norm(cmap, lower, upper, n_shades):
    """Get a discrete norm and colormap with a center point."""
    if lower != -upper:
        msg = f"upper and lower bounds must be symmetric around zero"
        raise ValueError(msg)

    # n_colors always odd for a white band around zero
    n_colors = 2 * n_shades

    cmap = mpl.cm.get_cmap(cmap, n_colors)
    bounds = np.linspace(lower, upper, n_colors + 1)
    norm = mpl.colors.BoundaryNorm(bounds, n_colors)

    return cmap, norm


def plot_error_fields(ds, config, save_dir):
    """Plot error fields.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the metrics.
    config : addict.Dict
        Configuration dictionary.
    save_dir : pathlib.Path
        Directory to save the plot to.
    nrows : int, optional
        Number of rows in the plot, by default 2

    """

    mconf = config.metric_conf["ERRORFIELDS"]

    # Define geometries to plot in the background
    # Borders
    if mconf["border_shapefile"] is not None:
        border = gpd.read_file(mconf["border_shapefile"])
        border_proj = border.to_crs(mconf["proj4str"])

        segments = [np.array(linestring.coords)[:, :2] for linestring in border_proj["geometry"]]
        border_collection = LineCollection(segments, color="k", linewidth=2, zorder=10)
    else:
        border_collection = None

    # Radar locations
    if mconf["radar_shapefile"] is not None:
        radar_locations = gpd.read_file(mconf["radar_shapefile"])
        radar_locations_proj = radar_locations.to_crs(mconf["proj4str"])
        xy = radar_locations_proj["geometry"].map(lambda point: point.xy)
        radar_locations_proj = list(zip(*xy))
    else:
        radar_locations_proj = None

    # Example data to get x and y coordinates
    exds = xr.open_dataset(mconf["example_data_file"])

    # Copy x and y coordinates from example data in kilometers
    if mconf["bbox"] is not None:
        bbox = mconf["bbox"]
        ds["x"] = exds.x[bbox[2] : bbox[3]]
        ds["y"] = exds.y[bbox[0] : bbox[1]][::-1]
    else:
        ds["x"] = exds.x
        ds["y"] = exds.y[::-1]

    im_size = (ds.y.size, ds.x.size)

    plot_models = mconf.plot_models

    plot_leadtimes = mconf.plot_leadtimes
    ncols = len(plot_leadtimes)
    nrows = len(plot_models) * len(config.continuous_metrics)

    height = config.figures.row_height
    width = im_size[1] / im_size[0] * height

    fig = plt.figure(figsize=((width) * ncols + 0.1, height * nrows), constrained_layout=True)
    subfigs = fig.subfigures(
        nrows=len(config.continuous_metrics),
        ncols=1,
        wspace=0.07,
        hspace=0.05,
        squeeze=True,
    )

    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.05, 0.0, 1, 1),
        "borderpad": 0,
    }

    for mi, metric in enumerate(config.continuous_metrics):
        axs = subfigs[mi].subplots(
            nrows=len(plot_models),
            ncols=len(plot_leadtimes),
            squeeze=False,
            sharey=True,
            sharex=True,
            gridspec_kw=dict(wspace=0.005, hspace=0.05),
        )

        # Create cmap and norm for each metric
        if config.metric_conf[metric].cmap_type == "diverging":
            cmap, norm = get_sym_cmap_norm(
                config.metric_conf[metric].cmap,
                *config.metric_conf[metric].cmap_intervals,
            )
        elif config.metric_conf[metric].cmap_type == "sequential":
            bounds = np.linspace(*config.metric_conf[metric].cmap_intervals)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
            cmap = plt.get_cmap(config.metric_conf[metric].cmap, len(bounds))
        else:
            raise ValueError(f"Unknown cmap_type {config.metric_conf[metric].cmap_type} " f"for metric {metric}")

        for pi, model in enumerate(plot_models):
            for li, lt in enumerate(plot_leadtimes):
                ds[model].sel(cont_metric=metric, leadtime=lt).plot(
                    ax=axs[pi, li],
                    rasterized=True,
                    norm=norm,
                    cmap=cmap,
                    add_colorbar=False,
                    zorder=0,
                )
                axs[pi, li].set_aspect("equal")
                axs[pi, li].grid(which="both", axis="both")
                axs[pi, li].set_title(f"{config.methods[model].label} at {lt} min")
                axs[pi, li].set_xlabel(mconf.xlabel)
                axs[pi, li].set_ylabel(mconf.ylabel)
                axs[pi, li].label_outer()

                # Plot borders and radars on top
                if border_collection is not None:
                    axs[pi, li].add_collection(copy(border_collection))

                if radar_locations_proj is not None:
                    axs[pi, li].scatter(
                        *radar_locations_proj,
                        color="tab:red",
                        s=8,
                        marker="X",
                        zorder=10,
                    )

                axs[pi, li].xaxis.set_major_locator(ticker.MultipleLocator(mconf.ax_tick_locator_multiple))
                axs[pi, li].yaxis.set_major_locator(ticker.MultipleLocator(mconf.ax_tick_locator_multiple))
                axs[pi, li].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x / 1e3:.0f}"))
                axs[pi, li].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x / 1e3:.0f}"))
                # axs[pi, li].xaxis.set_minor_locator(ticker.MultipleLocator(1e3))

            # Add colorbar
            cax = inset_axes(axs[pi, -1], bbox_transform=axs[pi, -1].transAxes, **cbar_ax_kws)
            cbar = plt.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                # format=formatter,
                orientation="vertical",
                cax=cax,
                ax=None,
                pad=0.1,
                label=config.metric_conf[metric].label,
                **config.metric_conf[metric].cbar_kwargs,
            )
            cbar.locator = ticker.MultipleLocator(config.metric_conf[metric].cmap_locator_multiple)
            cbar.update_ticks()
            cbar.solids.set_edgecolor("face")

            subfigs[mi].suptitle(
                config.metric_conf[metric]["full_name"],
                color=plt.rcParams["axes.titlecolor"],
            )

    fig.set_constrained_layout_pads(hspace=0.05, wspace=0.05)
    for ext in config.output_formats:
        fig.savefig(save_dir / f"ERRORFIELDS.{ext}")
    plt.close(fig)


def run(config):
    # Convert linestyle from list to tuple if given in list format
    for method in config.methods.keys():
        if isinstance(config.methods[method].linestyle, list):
            config.methods[method].linestyle = nested_list_to_tuple(config.methods[method].linestyle)

    exp_id = config.exp_id
    result_dir = config.path.result_dir.format(id=exp_id)
    save_dir = Path(config.path.save_dir.format(id=exp_id))
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric in config.metrics:
        ds = load_metrics(result_dir, metric)
        if metric == "CAT":
            plot_categorical_metrics(ds, config, save_dir)
        elif metric == "CONT":
            plot_continuous_metrics(ds, config, save_dir)
        elif metric == "FSS":
            plot_fss_metrics(ds, config, save_dir)
        elif metric == "RAPSD":
            plot_rapsd_metrics(ds, config, save_dir)
        elif metric == "ERRORFIELDS":
            plot_error_fields(ds, config, save_dir)
        else:
            raise ValueError(f"Unknown metric {metric}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config_path", type=str, help="Configuration file path")
    args = argparser.parse_args()

    config = load_yaml_config(args.config_path)

    if config.stylefile is not None:
        plt.style.use(config.stylefile)

    run(config)
