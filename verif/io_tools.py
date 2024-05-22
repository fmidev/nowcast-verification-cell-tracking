"""
Helper functions to read and write predictions, and
otherwise process the IO required for the verification of models.
"""

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import yaml
from addict import Dict

from .metranet_lookup import METRANET_RR_8BIT, METRANET_RR_8BIT_CENTERS


def load_metrics(path, metric_name, timestep=5):
    """Load metrics from netCDF files and return as xr.Dataset.

    Parameters
    ----------
    path : str
        Path to directory containing the netCDF files.
    metric_name : str
        Metric name that is used match the file with glob `*{metric_name}*.nc`.
    timestep : int, optional
        The leadtime timestep in minutes, by default 5. Applied to the `leadtime`
        coordinate.

    Returns
    -------
    xarray.Dataset
        Dataset containing the metrics.
    """
    files = sorted(Path(path).glob(f"*{metric_name}*.nc"))

    try:
        das = [xr.open_dataarray(p) for p in files]
        ds = xr.Dataset(data_vars={arr.name: arr for arr in das})
    except ValueError:
        ds = xr.open_mfdataset(files)

    # Change leadtime to minutes
    ds = ds.assign_coords(leadtime=(ds.leadtime) * timestep)

    return ds


def load_yaml_config(path: str):
    """
    Load a YAML config file as an attribute-dictionnary.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        Dict: Configuration loaded.
    """
    with open(path, "r") as f:
        config = Dict(yaml.safe_load(f))
    return config


def arr_compress_uint8(
    float_array: np.ndarray,
    nodata: np.uint8 = 255,
    gain: float = 0.5,
    offset: float = -32.0,
    max_value_uint8: np.uint8 = 254,
    use_metranet_lut: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Compress a float
    numpy array to uint8 with a scale/offset scheme.

    Args:
        float_array (np.ndarray): float array.
        missing_val (np.uint8, optional): uint8 value corresponding
            to np.nan missing data. Defaults to 255.
        gain (float, optional): gain of scheme.
        offset (float, optional): offset of scheme.
        max_value_uint8 (np.uint8, optional): max observation value in uint8

    Returns:
        np.ndarray: uint8 compressed array.
    """
    if use_metranet_lut:
        # find nearest value in lookup table
        float_arr = float_array.copy()
        float_arr[np.isnan(float_arr)] = 7000
        rounded = np.searchsorted(METRANET_RR_8BIT_CENTERS, float_arr)
        rounded[np.isnan(float_array)] = 255
        rounded[float_array < 0] = 0
        return rounded.astype(np.uint8)
    else:
        masked = np.ma.masked_where(~np.isfinite(float_array), float_array)
        max_value_float = offset + gain * max_value_uint8
        mask_big_values = float_array[...] >= max_value_float
        arr = (((1 / gain) * masked) - (offset / gain)).astype(np.uint8)
        arr[arr.mask] = nodata
        arr[mask_big_values] = max_value_uint8
        return arr.data


def arr_reconstruct_uint8(
    uint8_array: np.ndarray,
    gain: float = 0.5,
    offset: float = -32.0,
    missing_val: np.uint8 = 255,
    mask_val: float = np.nan,
    use_metranet_lut: bool = True,
    **kwargs,
):
    """
    Restores a float numpy array from uint8 using a
    scale/offset compression scheme.

    Args:
        uint8_array (np.ndarray): uint8 compressed array
        gain (float, optional): gain of scheme.
        offset (float, optional): offset of scheme.
        missing_val (np.uint8, optional): uint8 value corresponding to missing data. Defaults to 255.
        mask_val (float, optional): What the missing data value will be in float. Defaults to np.nan.

    Returns:
        np.ndarray: float dbz reflectivity array.
    """
    if use_metranet_lut:
        mask = uint8_array == 255
        uint8_array[mask] = 0
        arr = METRANET_RR_8BIT[uint8_array]
        arr[mask] = np.nan
    else:
        mask = uint8_array == missing_val
        arr = uint8_array.astype(np.float64)
        arr[mask] = mask_val
        arr = (arr * gain) + offset
    return arr


def write_data_to_h5_group(group: h5py.Group, ds_name: str, data: np.ndarray, what_attrs: dict = None):
    """
    Given a group in an HDF5 archive and float relfectivity data,
    write the data to the group.

    Args:
        group (h5py.Group): HDF5 file group.
        ds_name (str): name of the dataset to be created
        data (np.ndarray): float reflectivity data
        what_attrs (dict, optional): WHAT attributes corresponding to the ODIM format specification.
            Document parameters of the scale/offset compression scheme. Defaults to None.
    """
    data = arr_compress_uint8(data, **what_attrs)
    dataset = group.create_dataset(
        ds_name,
        data=data,
        dtype="uint8",
        compression="gzip",
        compression_opts=5,
    )
    dataset.attrs["CLASS"] = np.string_("IMAGE")
    dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")

    if what_attrs is None:
        what_attrs = {
            "quantity": "DBZH",
            "gain": 0.5,
            "offset": -32,
            "undetect": 255,
            "nodata": 255,
        }
    ds_what = group.require_group("what")

    for key, val in what_attrs.items():
        ds_what.attrs[key] = val


def read_data_from_h5_group(group: h5py.Group, mask_val: float = np.nan) -> np.ndarray:
    """
    Given an HDF5 group, read the (reflectivity) data saved in it.

    Args:
        group (h5py.Group): HDF5 group containing the data

    Returns:
        np.ndarray: float reflectivity data
    """
    img_uint8 = group["data"][...].squeeze()
    what_attrs = group["what"].attrs
    return arr_reconstruct_uint8(img_uint8, **what_attrs, mask_val=mask_val)


def get_neighboring_timestamp(time, distance: int, delta_min: int = 5):
    """
    Return neighboring timestamp at abs(distance)*delta_min
    minutes away.

    Args:
        time (str/datetime): reference timestamp
        distance (int): number of steps away for timestamp to get. Positive to the future, negative to the past.
        delta_min (int, optional): successive step distance in minutes. Defaults to 5.

    Raises:
        ValueError: if timestamp is not in a valid format (datetime or '%Y-%m-%d %H:%M:%S' if string.)

    Returns:
        str/datetime: neighboring timestamp
    """
    if isinstance(time, str):
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        neigh_time = time + timedelta(minutes=(delta_min * distance))
        return neigh_time.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(time, datetime):
        return time + timedelta(minutes=(delta_min * distance))

    else:
        raise ValueError("Time is not in a valid format (str/datetime)")


def get_observation_data_locs(time_0: str, leadtimes: list, method_name: str = "measurements") -> list:
    """
    Returns the location (group) in HDF5 file for given observation initial timestamp and lead times.

    Args:
        time_0 (str): first timestamp.
        leadtimes (list): lead times to get.
        method_name (str, optional): HDF5 group name for method. Defaults to "measurements".

    Returns:
        list: List of group locations for observations wanted in an HDF5 file.
    """
    obs_path_fmt = "{sample}/{method_name}/"
    timestamps = [get_neighboring_timestamp(time_0, lt) for lt in leadtimes]
    return [obs_path_fmt.format(sample=ts, method_name=method_name) for ts in timestamps]


def get_prediction_data_locs(time_0: str, leadtimes: list, method_name: str) -> list:
    """
    return location (group) in HDF5 file for given prediction initial timetamp and lead times.

    Args:
        ttime_0 (str): first timestamp.
        leadtimes (list): lead times to get.
        method_name (str, optional): HDF5 group name for prediction method.

    Returns:
        list: List of group locations for predictions wanted in an HDF5 file.
    """
    "."

    pred_path_fmt = "{sample}/{method_name}/{leadtime}/"
    return [pred_path_fmt.format(sample=time_0, method_name=method_name, leadtime=lt) for lt in leadtimes]


def _load_hdf5_data_at(archive_path: str, data_locations: list) -> np.ndarray:
    """
    return a 3D (t,w,h) numpy array sequence of reflectivity images at specified locations in the HDF5 file

    Args:
        archive_path (str): HDF5 archive path on disk.
        data_locations (list): list of group locations for the data to get from the HDF5 file.

    Returns:
        np.ndarray: Radar/nowcast data wanted.
    """
    with h5py.File(archive_path, "r") as db:
        return np.stack([read_data_from_h5_group(db[loc]) for loc in data_locations], axis=0)


def _write_hdf5_data_to(data: np.ndarray, archive_path: str, data_locations: list) -> None:
    """
    Save a 3D (t,w,h) numpy array sequence of reflectivity images to specified locations in the HDF5 file

    Args:
        data (np.ndarray): to write.
        archive_path (str): HDF5 archive path on disk.
        data_locations (list): list of group locations for the data to get from the HDF5 file.
    """
    with h5py.File(archive_path, "a") as db:
        for i, loc in enumerate(data_locations):
            grp = db.require_group(loc)
            write_data_to_h5_group(grp, ds_name="data", data=data[i])


def load_data(
    db_path: str,
    data_location_getter: callable,
    *data_location_getter_args,
    **data_location_getter_kwargs,
) -> np.ndarray:
    """
    return a (t,w,h) np array sequence of images
    using a location getter function for specifying locations in the HDF5 file

    Args:
        db_path (str): HDF5 archive path on disk.
        data_location_getter (callable): Function returning a list of the
            group locations for wanted data.

    Returns:
        np.ndarray: Radar/nowcast data wanted.
    """
    data_locations = data_location_getter(*data_location_getter_args, **data_location_getter_kwargs)
    return _load_hdf5_data_at(archive_path=db_path, data_locations=data_locations)


def get_hdf5_metadata(db_path: str):
    """
    Returns useful metadata from the HDF5 archive at the specified location.
    Includes available timestamps, method name for data retrieval, and if
    applicable prediction lead times available.

    Args:
        db_path (str): HDF5 archive path on disk.

    Returns:
        dict: metadata dictionnary with timestamps, method name, leadtimes
    """
    "works with method name embedded under one layer"
    metadata = {}
    with h5py.File(db_path, "r") as f:
        metadata.update({"timestamps": list(f.keys())})

        ts0 = metadata["timestamps"][0]
        metadata.update({"method_name": list(f[ts0].keys())[0]})

        lt_keys = list(f[ts0 + "/" + metadata["method_name"]].keys())
        try:
            lts = [int(k) for k in lt_keys]
            lts.sort()
            metadata.update({"leadtimes": lts})
        except ValueError:
            metadata.update({"leadtimes": None})
    return metadata


def load_observations(
    db_path: str,
    time_0: str,
    leadtimes: list,
    method_name: str = "measurements",
) -> np.ndarray:
    """
    load a (t,w,h) np array of observations
    using given information from the HDF5 database.

    Args:
        db_path (str): HDF5 archive path on disk.
        time_0 (str): current timestamp (corresponding to lead time = 0)
        leadtimes (list): list of integers corresponding to lead times to load.
        method_name (str, optional): HDF5 method group name. Defaults to "measurements".

    Returns:
        np.ndarray: observation data wanted.
    """
    try:
        return load_data(db_path, get_observation_data_locs, time_0, leadtimes, method_name)
    except:
        return None


def load_predictions(
    db_path: str,
    time_0: str,
    leadtimes: list,
    method_name: str,
) -> np.ndarray:
    """
    load a (t,w,h) np array of predictions
    using given information from the HDF5 database.

    Args:
        db_path (str): HDF5 archive path on disk.
        time_0 (str): current timestamp (corresponding to lead time = 0)
        leadtimes (list): list of integers corresponding to lead times to load.
        method_name (str, optional): HDF5 method group name.

    Returns:
        np.ndarray: prediction data wanted.
    """
    # TODO: better handling of missing data than returning method name...
    try:
        return load_data(db_path, get_prediction_data_locs, time_0, leadtimes, method_name)
    except:
        return method_name


def load_predictions_dict(prediction_methods: dict, time_0: str, leadtimes: list) -> dict:
    """
    load dictionnary of predictions of {sample} timestamp for {leadtimes},
    using input nested dictionnary of {method_name : dict containing 'path' key}.

    Args:
        prediction_methods (dict): dict mapping method_name to HDF5 archive path
        time_0 (str): current timestamp (corresponding to lead time = 0)
        leadtimes (list): list of integers corresponding to lead times to load.

    Returns:
        dict: method_name -> prediction array mapping
    """
    pred_dict = dict()
    for method in prediction_methods:
        try:
            pred_dict[method] = load_data(
                db_path=prediction_methods[method]["path"],
                data_location_getter=get_prediction_data_locs,
                time_0=time_0,
                leadtimes=leadtimes,
                method_name=method,
            )
        except:
            return method
    return pred_dict


def load_at_timestamp(path: str, timestamp: str, prediction: bool = True, leadtimes: list = None) -> np.ndarray:
    """
    High-level utility for loading predictions

    Args:
        path (str): Path to the HDF5 archive
        timestamp (str): ISO-8601 datetime string representing timestamp of last obs
        prediction (bool): True is prediction, false is observation
        leadtimes (list, optional): Iterable of lead times to load. Has to be specified for
        loading observations. With predictions if left to None, will be infered from the data.

    Returns:
        np.ndarray: spatiotemporal time-series -> predictions / observations

    """

    if prediction == False and leadtimes is None:
        raise ValueError("lead times have to be specified when loading observations.")

    meta = get_hdf5_metadata(path)

    if prediction:
        if leadtimes is None:
            leadtimes = meta["leadtimes"]
        load_function = load_predictions
    else:
        load_function = load_observations

    return load_function(
        db_path=path,
        time_0=timestamp,
        leadtimes=leadtimes,
        method_name=meta["method_name"],
    )


def read_timestamp_txt_file(rainy_days_path: str, start_idx: int = 0):
    """
    Read a TXT file containing timestamps from disk into a list of strings.

    Args:
        rainy_days_path (str): Path to the TXT file containing timestamps.
        start_idx (int, optional): Number of rows/lines to ignore at
            the start of the file. Defaults to 0.

    Returns:
        list: timestamps as strings
    """
    with open(rainy_days_path, "r") as f:
        rain_days = f.readlines()
        rain_days = [r.rstrip() for r in rain_days]
        rain_days = rain_days[start_idx:]
    return rain_days
