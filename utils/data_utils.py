"""Utils for working with data."""
import logging
import re
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
import gzip

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from scipy.optimize import NonlinearConstraint, minimize
from pyproj import CRS, Transformer
import h5py
import rasterio as rio
import contextlib

try:
    import radlib

    RADLIB_AVAILABLE = True
except ImportError:
    RADLIB_AVAILABLE = False


PROJ4STR = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"

LL_lon = 3.169
LL_lat = 43.6301
UL_lon = 2.6896
UL_lat = 49.3767
LR_lon = 11.9566
LR_lat = 43.6201
UR_lon = 12.4634
UR_lat = 49.3654
XSIZE = 710
YSIZE = 640

# Get projection
crs = CRS.from_proj4(PROJ4STR)
crs_4326 = CRS.from_epsg(4326)
transformer = Transformer.from_crs(crs_4326, crs, always_xy=True)
x1, y1 = transformer.transform(
    LL_lon,
    LL_lat,
)
x2, y2 = transformer.transform(
    UR_lon,
    UR_lat,
)

XCOORDS = np.linspace(
    x1,
    x2,
    XSIZE,
)
YCOORDS = np.linspace(
    y1,
    y2,
    YSIZE,
)


def get_reader_func(name):
    if name == "h5_to_dataset":
        return h5_to_dataset
    elif name == "npy_to_dataset":
        return npy_to_dataset
    elif name == "metranet_to_dataset":
        return metranet_to_dataset
    else:
        raise NotImplementedError(f"Reader function {name} not implemented.")


def get_preproc_func(name):
    if name == "set_time_and_swap_xy":
        return set_time_and_swap_xy
    elif name == "floor_xy":
        return floor_xy
    else:
        raise NotImplementedError(f"Preprocessing function {name} not implemented.")


def set_time_and_swap_xy(xda):
    xda = xda.assign_coords(
        dict(
            time=[datetime.strptime(str(xda.time.values[0].astype(int)), "%Y%m%d%H%M")]
        )
    )
    xda = xda.rename_dims({"x": "y1", "y": "x1"}).rename_dims({"y1": "y", "x1": "x"})
    return xda


def floor_xy(xda):
    xda = xda.assign_coords(
        dict(
            x=xda.x.astype(int),
            y=xda.y.astype(int),
        )
    )
    return xda


def metranet_to_dataset(fn, var_name=None):
    """Read data from MeteoSwiss Metranet format.

    Parameters
    ----------
    fn : pathlib.Path
        Filepath to read.
    var_name : str, optional
        Name of the variable to use in the xarray dataset. If None, the name
        is inferred from the file type.

    Returns
    -------
    xr.Dataset
        The data in an xarray dataset.

    Raises
    ------
    ImportError
        If radblib is not available.
    NotImplementedError
        If the file is of unknown type.

    """
    if not RADLIB_AVAILABLE:
        raise ImportError(f"radlib needed to read file {fn} but not available!")

    # Read data
    radar_data = radlib.read_file(str(fn), physic_value=True)

    data = np.flipud(radar_data.data)

    quantity = radar_data.header["product"]
    if quantity == "NWP_HZEROCL":
        timestamp = datetime.strptime(
            re.findall(r"\d+", fn.name.split(".")[0])[0], "%y%j%H%M0"
        )
    else:
        timestamp = datetime.strptime(
            re.findall(r"\d+", fn.name.split(".")[0])[0], "%y%j%H%M"
        )

    if "ECHOTOP" in quantity:
        ds_attrs = {
            "long_name": "Echo top height",
            "units": "m",
            "threshold": float(quantity.split("_")[1]),
        }
        # Transform to meters
        data *= 1000
    elif quantity == "Rain_Rate":
        ds_attrs = {"long_name": "Rain rate", "units": "mm/h"}
        if var_name is None:
            var_name = "RATE"

        # In hdf5 files, values with 0.1 mm/h seem to be encoded as 0, so let's do
        # the same here
        # TODO: this feels a little sketchy, so find out if this is valid
        data[np.isclose(data, 0.1)] = 0
    elif quantity == "NWP_HZEROCL":
        # Freezing level height
        ds_attrs = {"long_name": "Freezing level height", "units": "m"}
        if var_name is None:
            var_name = "HGHT"
    else:
        raise NotImplementedError(f"Quantity {quantity} not implemented.")

    if var_name is None:
        var_name = quantity.lower()

    # Create xarray dataset
    ds = xr.Dataset(
        {
            var_name: xr.DataArray(
                data=data[np.newaxis, ...],  # enter data here
                dims=["time", "y", "x"],
                coords={
                    "time": [timestamp],
                    "y": YCOORDS,
                    "x": XCOORDS,
                },
                attrs=ds_attrs,
            ),
        },
        attrs={},
    )
    # Set projection
    ds = ds.rio.write_crs(rio.crs.CRS.from_proj4(crs.to_proj4()))
    ds.x.attrs["axis"] = "X"  # Optional
    ds.x.attrs["standard_name"] = "projection_x_coordinate"
    ds.x.attrs["long_name"] = "x-coordinate in projected coordinate system"
    ds.x.attrs["unit"] = "meters"

    ds.y.attrs["axis"] = "Y"  # Optional
    ds.y.attrs["standard_name"] = "projection_y_coordinate"
    ds.y.attrs["long_name"] = "y-coordinate in projected coordinate system"
    ds.y.attrs["unit"] = "meters"

    return ds


def npy_to_dataset(fn, var_name="zdr_column"):
    """Read dataset from NPY GZ and return as xarray dataset.

    Usage:
    ds = npy_to_dataset("file.npy.gz")


    Parameters
    ----------
    fn : function
        Input file name.
    var_name : str, optional
        Name of the variable to use in the xarray dataset. The default is "zdr_column".

    Returns
    -------
    xarray.Dataset
    """

    with gzip.GzipFile(fn, "rb") as f:
        data = np.load(f, allow_pickle=True)
        data = np.flipud(data)
    timestamp = datetime.strptime(
        fn.name.split("_")[0],
        "%Y%m%d%H%M%S",
    )

    ds_attrs = {
        "long_name": "ZDR column",
        "units": "m",
        "author": "Martin Aregger",
        "fileprefix": "zdr_column",
    }

    # Create xarray dataset
    ds = xr.Dataset(
        {
            var_name: xr.DataArray(
                data=data[np.newaxis, ...],  # enter data here
                dims=["time", "y", "x"],
                coords={
                    "time": [
                        timestamp,
                    ],
                    "y": YCOORDS,
                    "x": XCOORDS,
                },
                attrs=ds_attrs,
            ),
        },
        attrs={},
    )
    # Set projection
    ds = ds.rio.write_crs(rio.crs.CRS.from_proj4(crs.to_proj4()))
    ds.x.attrs["axis"] = "X"  # Optional
    ds.x.attrs["standard_name"] = "projection_x_coordinate"
    ds.x.attrs["long_name"] = "x-coordinate in projected coordinate system"
    ds.x.attrs["unit"] = "meters"

    ds.y.attrs["axis"] = "Y"  # Optional
    ds.y.attrs["standard_name"] = "projection_y_coordinate"
    ds.y.attrs["long_name"] = "y-coordinate in projected coordinate system"
    ds.y.attrs["unit"] = "meters"

    return ds


def h5_to_dataset(fn, var_name=None):
    """Read dataset from ODIM HDF5 and return as xarray dataset.

    Usage:
    ds = h5_to_dataset("file.h5")


    Parameters
    ----------
    fn : function
        Input file name.
    var_name : str, optional
        Name of the variable to use in the xarray dataset. If None, the name
        is inferred from the file type.

    Returns
    -------
    xarray.Dataset
    """
    with h5py.File(fn, "r") as f:
        data_ = f["dataset1/data1"]["data"]
        gain = f["dataset1/data1"]["what"].attrs["gain"]
        offset = f["dataset1/data1"]["what"].attrs["offset"]
        nodata = f["dataset1/data1"]["what"].attrs["nodata"]
        undetect = f["dataset1/data1"]["what"].attrs["undetect"]

        # Unpack data
        # We need to flip the data array
        data_ = np.flipud(data_)
        data = data_ * gain + offset
        data[data_ == nodata] = np.nan
        data[data_ == undetect] = np.nan

        timestamp = datetime.strptime(
            f["/what"].attrs["date"].decode() + f["/what"].attrs["time"].decode(),
            "%Y%m%d%H%M%S",
        )

        # Get projection
        crs = CRS.from_proj4(f["/where"].attrs["projdef"].decode())

        # Lat-lon corner points to projection
        crs_4326 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_4326, crs, always_xy=True)
        x1, y1 = transformer.transform(
            f["/where"].attrs["LL_lon"],
            f["/where"].attrs["LL_lat"],
        )
        x2, y2 = transformer.transform(
            f["/where"].attrs["UR_lon"],
            f["/where"].attrs["UR_lat"],
        )

        xcoords = np.linspace(
            x1,
            x2,
            f["/where"].attrs["xsize"],
        )
        ycoords = np.linspace(
            y1,
            y2,
            f["/where"].attrs["ysize"],
        )

        ds_attrs = {}
        quantity = f["/dataset1/data1/what"].attrs["quantity"].decode("utf-8")
        if quantity == "RATE":
            ds_attrs = {"long_name": "Rain rate", "units": "mm/h", "fileprefix": "rate"}
        elif quantity == "HGHT":
            ds_attrs = {
                "long_name": "Echo top height",
                "units": "m",
                "threshold": f["/dataset1/what"].attrs["prodpar"],
                "fileprefix": f"echotop_{f['/dataset1/what'].attrs['prodpar']}",
            }
            # Convert to meters
            data *= 1000
        if var_name is None:
            var_name = quantity

        # Create xarray dataset
        ds = xr.Dataset(
            {
                var_name: xr.DataArray(
                    data=data[np.newaxis, ...],  # enter data here
                    dims=["time", "y", "x"],
                    coords={
                        "time": [timestamp],
                        "y": ycoords,
                        "x": xcoords,
                    },
                    attrs=ds_attrs,
                ),
            },
            attrs={},
        )
        # Set projection
        crs = CRS.from_proj4(f["/where"].attrs["projdef"].decode())
        ds = ds.rio.write_crs(rio.crs.CRS.from_proj4(crs.to_proj4()))

        ds.x.attrs["axis"] = "X"  # Optional
        ds.x.attrs["standard_name"] = "projection_x_coordinate"
        ds.x.attrs["long_name"] = "x-coordinate in projected coordinate system"
        ds.x.attrs["unit"] = "meters"

        ds.y.attrs["axis"] = "Y"  # Optional
        ds.y.attrs["standard_name"] = "projection_y_coordinate"
        ds.y.attrs["long_name"] = "y-coordinate in projected coordinate system"
        ds.y.attrs["unit"] = "meters"

    return ds


def load_data(
    ds_conf: dict, curdate: datetime, obstimes: list, num_obs: int, radar: str
):
    """Load data for the VAR model.

    Parameters
    ----------
    ds_conf : addict.Dict
        Configuration for the input data. Example configuration:

        .. code-block:: yaml
            RATE:
            file: "{radar}_DPR_%Y%m%d%H%M_RATE.nc"
            path: "/input/path/%Y/%m/%d/{radar}"
            variable: "RATE"
            coord_system: "euler"
            normalize: true
            file_timestamp: "exact"
            theoretical_max: null
            theoretical_min: 0.1

        `file_timestamp` can be one of the following:
            - "exact": file timestamps must match `obstimes` exactly.
            - "closest": the files with timestamps closest to `obstimes` are read
                (NOTE: can result in duplicate files).
            - "n_next_files": `num_obs` files with timestamps after `curdate` are read.
            - "n_prev_files": `num_obs` files with timestamps before `curdate` are read.

    curdate : datetime.datetime
        The current date.
    obstimes : list of datetime.datetime
        Observation times that are fetched, if using the "exact" or "closest"
        file_timestamp
    num_obs : int
        Number of fetched observations, if using the "n_next_files" or
        "n_prev_files" file_timestamp
    radar : str
        Radar name.

    Returns
    -------
    xarray.Dataset
        The dataset containing the data.

    Raises
    ------
    ValueError
        Raised if opening files with xr.open_mfdataset fails for some dataset.

    """
    datasets = {}
    for name in ds_conf.keys():
        # Read data
        pathformat = str(
            Path(ds_conf[name].path.format(radar=radar))
            / Path(ds_conf[name].file.format(radar=radar))
        )
        filename_pattern = ds_conf[name].file.format(radar=radar)
        rootpath = Path(curdate.strftime(ds_conf[name].path.format(radar=radar)))

        filename_glob = re.sub(
            "(%[%YyjmdHMS])+",
            "*",
            filename_pattern,
        )
        filetimes = pd.DataFrame(
            {
                "time": [
                    datetime.strptime(
                        p.name.split(".")[0], filename_pattern.split(".")[0]
                    ).replace(second=0)
                    for p in rootpath.glob(filename_glob)
                ],
                "filename": list(rootpath.glob(filename_glob)),
            }
        ).sort_values("time")

        if ds_conf[name].file_timestamp == "exact":
            files = [Path(t.strftime(pathformat)) for t in obstimes]
        elif ds_conf[name].file_timestamp == "closest":
            plottimes = pd.DataFrame(
                obstimes,
                columns=[
                    "time",
                ],
            )
            try:
                files = (
                    pd.merge_asof(
                        plottimes,
                        filetimes,
                        tolerance=pd.Timedelta("5T"),
                        direction="backward",
                        on="time",
                        allow_exact_matches=True,
                    )
                    .dropna()
                    .filename.to_list()
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"No files found for {name} at {rootpath}!"
                ) from e

        elif ds_conf[name].file_timestamp == "n_next_files":
            # Find the N latest files after timestamp
            filetimes = filetimes[filetimes.time >= curdate]
            try:
                files = filetimes.iloc[
                    (filetimes["time"] - curdate).abs().argsort().iloc[:num_obs]
                ].filename.to_list()[::-1]
            except TypeError as e:
                raise FileNotFoundError(
                    f"No files found for {name} at {rootpath}!"
                ) from e

        elif ds_conf[name].file_timestamp == "n_prev_files":
            # Find the N latest files after timestamp
            filetimes = filetimes[filetimes.time <= curdate]
            try:
                files = filetimes.iloc[
                    (filetimes["time"] - curdate).abs().argsort().iloc[:num_obs]
                ].filename.to_list()[::-1]
            except TypeError as e:
                raise FileNotFoundError(
                    f"No files found for {name} at {rootpath}!"
                ) from e

        else:
            raise ValueError(
                f"Unknown file_timestamp method {ds_conf[name].file_timestamp} for{name}!"
            )

        if len(files) == 0:
            raise FileNotFoundError(f"No files found for {name} at {rootpath}!")

        var_name = ds_conf[name].variable
        try:
            if isinstance(ds_conf[name]["reader_func"], str):
                reader_func = get_reader_func(ds_conf[name]["reader_func"])
                ds_ = [reader_func(f, var_name=var_name) for f in files]
                datasets[name] = xr.concat(ds_, dim="time")
            else:
                if isinstance(ds_conf[name]["preprocess_func"], str):
                    preprocess_func = get_preproc_func(ds_conf[name]["preprocess_func"])
                else:
                    preprocess_func = None

                datasets[name] = xr.open_mfdataset(
                    files,
                    concat_dim="time",
                    combine="nested",
                    data_vars=[var_name],
                    compat="override",
                    coords="minimal",
                    join="override",
                    preprocess=preprocess_func,
                )
        except (ValueError, OSError) as e:
            raise ValueError(f"Couldn't read files {files} for variable {name}!") from e

        # Drop extra data
        variable_names = list(datasets[name].keys())
        variable_names.remove(var_name)
        # if "spatial_ref" in variable_names:
        #     variable_names.remove("spatial_ref")
        datasets[name] = datasets[name].drop(variable_names)

        # Set values above theoretical maximum to NaN
        # This is a ugly fix to bad TRENDSS values
        if isinstance(ds_conf[name]["theoretical_max"], float):
            datasets[name][var_name] = datasets[name][var_name].where(
                datasets[name][var_name] < ds_conf[name]["theoretical_max"]
            )
        if isinstance(ds_conf[name]["theoretical_min"], float):
            datasets[name][var_name] = datasets[name][var_name].where(
                datasets[name][var_name] > ds_conf[name]["theoretical_min"]
            )

        # Add mask for missing data
        datasets[name][f"{var_name}_nan_mask"] = np.isnan(datasets[name][var_name])
        datasets[name][var_name] = datasets[name][var_name].fillna(0)

        # Round time to nearest minute
        datasets[name] = datasets[name].assign_coords(
            {
                "time": datasets[name].time.dt.floor("min"),
            }
        )
        datasets[name] = datasets[name].sortby("time")
        datasets[name][var_name].attrs[f"input_files"] = [
            str(f.resolve()) for f in files
        ]

    dataset = xr.merge(
        [d for _, d in datasets.items()], compat="override", join="override"
    )
    # Drop z dimension for now if it exists
    if "z" in dataset.dims.keys():
        dataset = dataset.isel(z=0)

    return dataset


def run_differencing(dataset, ds_conf):
    """Run differencing on dataset for variables where that is requested.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to be differenced
    ds_conf : addict.Dict
        Dataset configuration

    Returns
    -------
    xr.Dataset
        Dataset with differenced variables

    """
    differenced_variable_names = []
    for _, conf in ds_conf.items():
        if conf["difference"]:
            var_name = conf["variable"]
            if conf["coord_system"] == "euler":
                var_name = f"{var_name}_lagr"
            dataset[f"{var_name}_diff"] = dataset[var_name].diff("time")
            dataset[f"{var_name}_diff"] = dataset[f"{var_name}_diff"].fillna(0.0)
            differenced_variable_names.append(f"{var_name}_diff")

    return dataset, differenced_variable_names


def get_continuous_time_intervals(
    dates: np.ndarray,
    timestep_variable_conf: dict,
    radar: str,
    min_length_items: int = 1,
    min_length_mins: int = 5,
    max_time_diff: int = 60,
    minute_divisible_by: int = None,
):
    """Get intervals of continuous input data.

    Parameters
    ----------
    dates : np.ndarray of datetime.datetime
        The input timesteps.
    timestep_variable_conf : addict.Dict
        The configuration for the timestep variable. Example configuration:

        .. code-block:: yaml

            file: "{radar}_DPR_%Y%m%d%H%M_RATE.nc"
            path: "/data/PINCAST/manuscript_2/NEXRAD_rainrate/%Y/%m/%d/{radar}"

    radar : str
        The radar name.
    min_length_items : int, optional
        The minimum number of items in a continuous interval. The default is 1.
    min_length_mins : int, optional
        The minimum length of a continuous interval in minutes. The default is 5.
    max_time_diff : int, optional
        The maximum time difference between two consecutive items in a continuous
        interval in minutes. The default is 60.
    minute_divisible_by : int, optional
        If provided, all times where the minute is not divisible by this number will
        be removed. This allows e.g. getting times every 5 minutes. The default is None.

    Returns
    -------
    list of lists of datetime.datetime
        The intervals of continuous input data.

    Raises
    ------
    ValueError
        Raised if no input files are found for a given day.

    """
    filename_pattern = timestep_variable_conf.file.format(radar=radar)
    rootpath_format = timestep_variable_conf.path.format(radar=radar)

    # Get days to iterate over
    days = pd.date_range(
        start=pd.Timestamp(dates.min()).floor("D"),
        end=pd.Timestamp(dates.max()).floor("D"),
        freq="D",
        inclusive="both",
    )

    intervals = []
    max_time_diff = timedelta(minutes=max_time_diff)

    filename_glob = re.sub(
        "(%[%YyjmdHMS])+",
        "*",
        filename_pattern,
    )
    for day in days:
        # Get all files for the day
        filetimes = pd.DataFrame(
            {
                "datatime": [
                    datetime.strptime(
                        p.name.split(".")[0], filename_pattern.split(".")[0]
                    ).replace(second=0)
                    for p in Path(day.strftime(rootpath_format)).glob(filename_glob)
                ],
                "filename": list(
                    Path(day.strftime(rootpath_format)).glob(filename_glob)
                ),
            }
        ).sort_values("datatime")

        if minute_divisible_by is not None:
            # Remove unwanted minutes
            filetimes = filetimes[
                filetimes["datatime"].dt.minute % minute_divisible_by == 0
            ]

        if len(filetimes) == 0:
            continue

        # Split lifetimesto multiple intervals if there are gaps in the data
        filetimes["datatime_diff"] = filetimes.datatime.diff()

        splits = filetimes["datatime_diff"] > max_time_diff

        if splits.any():
            idx = np.cumsum(
                np.in1d(
                    filetimes.index,
                    filetimes.where(splits == True).dropna().index.values,
                )
            )
            for _, group in filetimes.groupby(idx):
                intervals.append(group["datatime"].dt.to_pydatetime().tolist())
        else:
            intervals.append(filetimes["datatime"].dt.to_pydatetime().tolist())

    # Combine intervals that are close to each other
    # This should only happen at the start and end of days, so we should not have to
    # combine more than 2 intervals ever
    combined_intervals = []
    remove_idx = []
    for i in range(len(intervals) - 1):
        if (intervals[i + 1][0] - intervals[i][-1]) <= max_time_diff:
            combined_intervals.append([*intervals[i], *intervals[i + 1]])
            remove_idx.append(i)
            remove_idx.append(i + 1)

    intervals = [i for j, i in enumerate(intervals) if j not in remove_idx]
    # Add combined intervals
    intervals.extend(combined_intervals)

    # Sort by first value of list
    intervals = sorted(intervals, key=lambda x: x[0])

    # Remove intervals that are too short
    remove_idx = []
    for i, interval in enumerate(intervals):
        if len(interval) < min_length_items:
            remove_idx.append(i)
        if (max(interval) - min(interval)) < timedelta(minutes=min_length_mins):
            remove_idx.append(i)
    intervals = [i for j, i in enumerate(intervals) if j not in remove_idx]

    return intervals
