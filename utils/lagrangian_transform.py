"""Utility functions for Lagrangian transformation."""
import logging
import re
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import dask
from pysteps import motion, extrapolation
from scipy.ndimage import uniform_filter


def transform_to_lagrangian_with_motion_field(
    input_fields,
    motion_field,
    t0,
    compute_mask=True,
    extrap_kwargs={},
):
    """Transform the input time series to Lagrangian coordinates using the given advection field.

    Parameters
    ----------
    input_fields : np.ndarray
        Input precipitation fields, with shape (N, height, width).
    motion_field : np.ndarray
        The motion field as (u, v) with shape (2, height, width).
    t0 : int
        Number of the "common time" that all other fields are extrapolated to
        (1-based; from 1 ... N).
    compute_mask : boolean
        Whether to compute advection mask.
    extrap_kwargs : dict
        Keyword arguments passed to pysteps.extrapolation.semilagragian.extrapolate.

    Returns
    -------
    np.ndarray
        Output precipitation fields in Lagrangian coordinates.
    np.ndarray
        Advection mask.

    """
    n_fields = input_fields.shape[0]
    output_fields = np.empty(input_fields.shape)

    if compute_mask:
        mask_adv = np.isfinite(input_fields[-1])
    else:
        mask_adv = None

    for i in range(0, t0 - 1):
        # Extrapolate forwards
        output_fields[i, :] = extrapolation.semilagrangian.extrapolate(
            input_fields[i, :],
            motion_field,
            t0 - i - 1,
            **extrap_kwargs,
        )[-1]
        if compute_mask:
            mask_adv = np.logical_and(mask_adv, np.isfinite(output_fields[i, :]))

    # Field at t0 is not transformed
    output_fields[t0 - 1, :] = input_fields[t0 - 1, :]
    # To extrapolate backwards, we need to invert motion field
    motion_field *= -1

    for i in range(t0, n_fields):
        # Extrapolate backwards
        output_fields[i, :] = extrapolation.semilagrangian.extrapolate(
            input_fields[i, :],
            motion_field,
            abs(t0 - i - 1),
            **extrap_kwargs,
        )[-1]
        if compute_mask:
            mask_adv = np.logical_and(mask_adv, np.isfinite(output_fields[i, :]))

    return output_fields, mask_adv


def get_motionfield(
    dataset, date, oflow_conf, common_time_index=-1, read_from_file: bool = False
):
    """Calculate motion field using optical flow.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset returned by `get_var_dataset`.
    date : datetime.datetime
        The current date.
    oflow_conf : addict.Dict
        The configuration for the optical flow. Example configuration:

        .. code-block:: yaml
            # Read existing advection field from file
            read_from_file: false
            # Save advection fields to file
            save_to_file: true
            # Output path
            path: "/data/PINCAST/manuscript_2/advection_fields/{year}/{month:02d}/{day:02d}"
            filename: "motion_{year}{month:02d}{day:02d}{hour:02d}{minute:02d}.nc"
            # Quantity used to calculate motion field (needs to match an entry in datasources.yaml)
            oflow_quantity: RATE
            # Method name from pysteps
            oflow_method: "lucaskanade"
            # How many fields to use for optical flow
            oflow_history_length: 4
            # Whether advection field should be updated
            update_advfield: false
            # Parameters for different methods
            lucaskanade:
                fd_method: "shitomasi"
                # fd_kwargs:
                #   min_sigma: 2.
                #   max_sigma: 10.
                #   threshold: 0.1
                #   overlap: 0.5
                # lk_kwargs:
                #   winsize: [15, 15]
                # decl_scale: 10
                # interp_kwargs:
                #   epsilon: 5.0

    common_time_index : int, optional
        The index of the "common time" that all other fields are extrapolated
        to, by default -1 corresponding to last timestep.
        The timesteps for optical flow are picked backwards from this index.

    read_from_file : bool, optional
        Whether to read optical flow from file instead of calculating it,
        by default False

    Returns
    -------
    np.array
        Array of size (2, x, y) containing the motion field.

    Raises
    ------
    NotImplementedError
        Raised if `read_from_file` is True.

    """
    fn = Path(
        oflow_conf.path.format(
            year=date.year,
            month=date.month,
            day=date.day,
        )
    ) / Path(
        oflow_conf.filename.format(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=date.hour,
            minute=date.minute,
        )
    )
    if read_from_file and fn.exists():
        ds = xr.open_dataset(fn)
        motion_field = np.stack([ds["U"].values, ds["V"].values]).squeeze()
        return motion_field

    # Calculate motion field from optical flow
    # Pick data used to calculate motion field
    # timeslice = np.arange(-oflow_conf.oflow_history_length, 0, 1)
    timeslice = np.arange(
        common_time_index - oflow_conf.oflow_history_length + 1,
        common_time_index + 1,
        1,
    )

    oflow_data = dataset[oflow_conf.oflow_quantity].isel(time=timeslice).to_numpy()

    # Optical flow method
    calc_oflow = partial(
        motion.get_method(oflow_conf.oflow_method),
        **oflow_conf[oflow_conf.oflow_method],
    )
    motion_field = calc_oflow(oflow_data)

    # Save to file
    if oflow_conf.save_to_file:
        # Save to netcdf files
        u_ = xr.DataArray(
            motion_field[np.newaxis, np.newaxis, 0, :].astype(np.float32),
            coords=[
                [dataset.time.isel(time=timeslice[0]).values],
                [dataset.time.isel(time=timeslice[-1]).values],
                dataset.coords["y"].astype(np.float32),
                dataset.coords["x"].astype(np.float32),
            ],
            dims=["starttime", "endtime", "y", "x"],
            name="U",
        )
        v_ = xr.DataArray(
            motion_field[np.newaxis, np.newaxis, 1, :].astype(np.float32),
            coords=[
                [dataset.time.isel(time=timeslice[0]).values],
                [dataset.time.isel(time=timeslice[-1]).values],
                dataset.coords["y"].astype(np.float32),
                dataset.coords["x"].astype(np.float32),
            ],
            dims=["starttime", "endtime", "y", "x"],
            name="V",
        )
        # Add metadata
        data_vars = {"U": u_, "V": v_}
        attrs = {}
        attrs["oflow_method"] = oflow_conf.oflow_method
        for k, v in oflow_conf[oflow_conf.oflow_method].items():
            attrs["oflow_kwargs_" + k] = v

        attrs["input_files"] = [
            dataset[oflow_conf.oflow_quantity].attrs["input_files"][i_]
            for i_ in timeslice
        ]
        attrs["input_times"] = [
            str(pd.to_datetime(dataset.time.data[i_])) for i_ in timeslice
        ]

        # Get attributes from original data
        for k, v in dataset.attrs.items():
            attrs[k] = v

        motion_dataset = xr.Dataset(data_vars=data_vars, attrs=attrs)

        comp = dict(zlib=True, complevel=9, dtype="float32", least_significant_digit=4)
        for var in motion_dataset.data_vars:
            motion_dataset[var].encoding.update(comp)

        # Write to netcdf file
        fn.parents[0].mkdir(parents=True, exist_ok=True)
        motion_dataset.to_netcdf(fn, "w")
    return motion_field


def run_lagrangian_transform(
    date: datetime,
    dataset: xr.Dataset,
    motion_field: np.ndarray,
    ds_conf: dict,
    model_conf: dict,
    common_timestep_index: int = None,
    fill_rounds: int = 1,
):
    """Transform dataset to Lagrangian coordinates.

    Parameters
    ----------
    date : datetime
        Datetime of the input dataset
    dataset : xr.Dataset
        Input dataset
    motion_field : np.ndarray
        Motion field of size (2, H, W)
    ds_conf : addict.Dict
        Dataset configuration
    model_conf : addict.Dict
        Model configuration

    Returns
    -------
    xr.Dataset
        Dataset with Lagrangian-transformed variables
    list
        List of variable names of Lagrangian-transformed variables

    Raises
    ------
    ValueError
        Raised if the transformation fails.

    """
    if common_timestep_index is None:
        common_timestep_index = dataset.time.size - 1

    lagrangian_variable_names = []
    for dataname, conf in ds_conf.items():
        var_name = conf["variable"]
        if conf["coord_system"] != "euler":
            continue
        in_ = dataset[var_name].to_numpy()
        nan_mask = np.isnan(in_)
        in_[nan_mask] = 0.0
        try:
            out_, mask_ = transform_to_lagrangian_with_motion_field(
                in_,
                motion_field.copy(),
                common_timestep_index + 1,
                compute_mask=True,
                extrap_kwargs=model_conf.oflow.extrap_kwargs,
            )
        except ValueError as e:
            raise ValueError(
                f"Error transforming {dataname} to Lagrangian coordinates at {date}!"
            ) from e

        # Set advection mask to nan
        out_[:, ~mask_] = np.nan

        if dataname == "RATE":
            out_ = fill_negative_rain_rate(out_, iterations=fill_rounds)

        dataset[f"{var_name}_lagr"] = (("time", "y", "x"), out_)
        dataset[f"{var_name}_advmask"] = (("y", "x"), mask_)
        lagrangian_variable_names.append(f"{var_name}_lagr")

    return dataset, lagrangian_variable_names


def fill_negative_rain_rate(R, iterations=1, kernel_size=3):
    # Fill possible negative values with moving mean, with a 3x3 kernel
    if np.any(R < 0):
        for it in range(iterations):
            for i in range(R.shape[0]):
                mean_ = uniform_filter(R[i], size=kernel_size, mode="constant")
                R[i][np.where(R[i] < 0)] = mean_[np.where(R[i] < 0)]

        # If any negative remain, set to 0
        R[R < 0] = 0
    return R


def read_advection_fields_from_nc(filename):
    """Read advection fields from NetCDF file.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    dict
        Dictionary of advection fields, with keys being the start and end times

    """
    advfields = {}
    ds = xr.open_dataset(filename)
    for endtime, ds_ in ds.groupby("endtime"):
        endtime = pd.Timestamp(endtime).to_pydatetime()
        for starttime, ds__ in ds_.groupby("starttime"):
            starttime = pd.Timestamp(starttime).to_pydatetime()
            motion_field = np.stack([ds__.U, ds__.V])
            advfields[(starttime, endtime)] = motion_field
    return advfields


def transform_to_eulerian(
    input_fields,
    t0,
    dates,
    advfields,
    extrap_kwargs={},
    prediction_mode=False,
    n_workers=1,
):
    """Transform the input time series to Eulerian coordinates using the given advection field.

    Parameters
    ----------
    input_fields : np.ndarray
        Input precipitation fields, with shape (N, height, width).
    t0 : int
        Number of the "common time" that all other fields were extrapolated to
        (1-based; from 1 ... N).
    dates : list-like
        List of dates corresponding to each field.
    advfields : dict
        Advection fields with structure (startdate, enddate): advfield.
    extrap_kwargs : dict
        Keyword arguments passed to pysteps.extrapolation.semilagragian.extrapolate.
    prediction_mode : bool
        If we're predicting, observed motion fields from timesteps after common_time
        are not used.

    Returns
    -------
    np.ndarray
        Output precipitation fields in Lagrangian coordinates.
    np.ndarray
        Advection mask.
    dict
        Dictionary of advection fields with structure (startdate, enddate): advfield

    """
    n_fields = input_fields.shape[0]
    output_fields = np.empty(input_fields.shape)
    # common_time = dates[t0 - 1]
    input_fields[~np.isfinite(input_fields)] = 0

    def extrapolate_backwards(i, field, advfields):
        # Extrapolate backwards
        # Get advection field
        if len(advfields) == 1:
            advfield = (-1) * list(advfields.values())[0].copy()
        else:
            # Here, get correct advection field based on times
            # if advection field changes
            try:
                # We need advection field that starts with t (since t < t0)
                field_k = [k for k in advfields.keys() if k[0] == dates[i]][0]
                advfield = advfields[field_k].copy()
            except KeyError:
                raise KeyError(
                    f"Correct advection field for time {dates[i]} doesn't exist!"
                )

        output_field = extrapolation.semilagrangian.extrapolate(
            field,
            advfield,
            t0 - i - 1,
            **extrap_kwargs,
        )[-1]
        return i, output_field

    def extrapolate_forwards(i, field, advfields):
        # Extrapolate forwards
        if len(advfields) == 1:
            advfield = list(advfields.values())[0].copy()
        elif prediction_mode:
            # Look for the longest motion field interval ending at common_time
            possible_keys = [k for k in advfields.keys() if k[1] == dates[t0 - 1]]
            if len(possible_keys) == 1:
                key = possible_keys[0]
            else:
                i_key = np.argmax(np.diff([*sorted(possible_keys)], axis=0))
                key = sorted(advfields.keys())[i_key]
            advfield = advfields[key]
        else:
            # Here, get correct advection field based on times
            try:
                # Here we need adv.field that ends at t (since t > t0)
                field_k = [k for k in advfields.keys() if k[1] == dates[i]][0]
                advfield = advfields[field_k].copy()
            except (IndexError, KeyError):
                raise KeyError(
                    f"Correct advection field for time {dates[i]} doesn't exist!"
                )

        output_field = extrapolation.semilagrangian.extrapolate(
            field,
            advfield,
            abs(t0 - i - 1),
            **extrap_kwargs,
        )[-1]
        return i, output_field

    delayed = []

    if t0 == 0:
        # No previous timesteps, just extrapolate forwards
        for i in range(t0, n_fields):
            # Extrapolate forwards in time from common time to original time
            # This can also be predicting!
            delayed.append(
                dask.delayed(extrapolate_forwards)(i, input_fields[i, :], advfields)
            )
    else:
        for i in range(t0 - 1):
            # Extrapolate backwards in time from common time to original time
            delayed.append(
                dask.delayed(extrapolate_backwards)(i, input_fields[i, :], advfields)
            )

        # Field at t0 is not transformed
        output_fields[t0 - 1, :] = input_fields[t0 - 1, :]

        for i in range(t0, n_fields):
            # Extrapolate forwards in time from common time to original time
            # This can also be predicting!
            delayed.append(
                dask.delayed(extrapolate_forwards)(i, input_fields[i, :], advfields)
            )
    scheduler = "processes" if n_workers > 1 else "single-threaded"
    res = dask.compute(*delayed, num_workers=n_workers, scheduler=scheduler)

    for r in res:
        i, field = r
        output_fields[i, :] = field

    # return output_fields, mask_adv, advfields
    return output_fields
