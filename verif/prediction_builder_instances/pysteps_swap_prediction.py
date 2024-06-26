"""
    Class for computing PYSTEPS nowcasts based on advection extrapolation

    OPTICAL FLOW CALCULATED BEFORE BBOX
"""

from addict import Dict
import numpy as np
import h5py
from pysteps import motion, nowcasts
from pysteps.utils import conversion, transformation, dimension

from .. import conversion_tools
from ..prediction_builder import PredictionBuilder


class PystepsSwapPrediction(PredictionBuilder):
    def __init__(self, config: Dict):
        super().__init__(config)

    def read_input(self, timestamp: str, num_next_files: int = 0):
        return super().read_input(timestamp, num_next_files)

    def save(self, nowcast: np.ndarray, group: h5py.Group, save_parameters: Dict):
        return super().save(nowcast, group, save_parameters)

    def run(self, timestamp: str):
        "Overwrite to give metadata to nowcast method argument"
        data, metadata = self.read_input(timestamp=timestamp, **self.input_params)
        data, metadata = self.preprocessing(data, metadata, self.preprocessing_params)
        nowcast = self.nowcast(data, metadata, self.nowcast_params)
        nowcast = self.postprocessing(nowcast=nowcast, params=self.postprocessing_params)
        return nowcast

    def preprocessing(self, data: np.ndarray, metadata: dict, params: Dict = None):
        """
        All the processing of data before nowcasting
        in : data, metadata, params
        out: data, metadata
        """
        if params is None:
            params = Dict(
                {
                    "threshold": 0.1,
                    "zerovalue": 15.0,
                    "nan_to_zero": True,
                    "downscaling": 1.0,
                    "db_transform": False,
                    "convert": True,
                }
            )

        if params.convert:
            data, metadata = conversion.to_rainrate(data, metadata)
        if params.db_transform:
            data, metadata = transformation.dB_transform(
                R=data,
                metadata=metadata,
                threshold=params.threshold,
                zerovalue=params.zerovalue,
            )
        else:
            data[data < params.threshold] = metadata["zerovalue"]
        if not params.convert and not params.db_transform:
            data[data < params.threshold] = metadata["zerovalue"]

        if params.nan_to_zero:
            data[~np.isfinite(data)] = metadata["zerovalue"]

        if params.downscaling != 1.0:
            metadata["xpixelsize"] = metadata["ypixelsize"]
            data, metadata = dimension.aggregate_fields_space(
                data, metadata, metadata["xpixelsize"] * params.downscaling
            )

        return data, metadata

    def nowcast(self, data: np.ndarray, metadata: dict, params: Dict = None):
        "Advection extrapolation, S-PROG, LINDA feasible"

        if params is None:
            params = Dict(
                {
                    "bbox": [125, 637, 604, 1116],
                    "nowcast_method": "advection",
                    "sample_slice": [None, -1, None],
                    "oflow_slice": [0, -1, 1],
                    "n_leadtimes": 36,
                    "oflow_params": {"oflow_method": "lucaskanade"},
                    "nowcast_params": {},
                }
            )

        oflow_name = params.oflow_method
        oflow_fun = motion.get_method(oflow_name)
        nowcast_name = params.nowcast_method
        nowcast_fun = nowcasts.get_method(nowcast_name)
        sample_slice = slice(*params.sample_slice)

        # optical flow
        oflow = oflow_fun(data, **params.oflow_params)

        if params.get("convert", False):
            data, metadata = conversion.to_rainrate(data, metadata)
        if params.get("db_transform", False):
            data, metadata = transformation.dB_transform(
                R=data,
                metadata=metadata,
                threshold=params.threshold,
                zerovalue=params.zerovalue,
            )
        else:
            data[data < params.threshold] = metadata["zerovalue"]
        if not params.get("convert", False) and not params.get("db_transform", False):
            data[data < params.threshold] = metadata["zerovalue"]

        if params.nan_to_zero:
            data[~np.isfinite(data)] = metadata["zerovalue"]

        if params.bbox_type == "data":
            # Clip domain
            bbox = params.bbox
            bbox = (
                bbox[0] * metadata["xpixelsize"],
                bbox[1] * metadata["xpixelsize"],
                bbox[2] * metadata["ypixelsize"],
                bbox[3] * metadata["ypixelsize"],
            )
            data = data.squeeze()
            metadata["yorigin"] = "lower"
            data, _ = dimension.clip_domain(R=data, metadata=metadata, extent=bbox)
            oflow, _ = dimension.clip_domain(R=oflow, metadata=metadata, extent=bbox)

        elif params.bbox_type == "pixels":
            # Clip image
            bbox = params.bbox
            data = data.squeeze()
            data = data[..., bbox[2] : bbox[3], bbox[0] : bbox[1]]
            oflow = oflow[..., bbox[2] : bbox[3], bbox[0] : bbox[1]]

        # nowcast itself
        nowcast = nowcast_fun(data[sample_slice, ...].squeeze(), oflow, params.n_leadtimes, **params.nowcast_params)
        return nowcast

    def postprocessing(self, nowcast, params):
        if params is None:
            params = Dict(
                {
                    "threshold": -10,
                    "zerovalue": 0,
                    "nan_to_zero": True,
                    "db_transform": False,
                    "convert": True,
                }
            )

        if params.convert:
            if params.db_transform:
                nowcast, _ = transformation.dB_transform(
                    nowcast,
                    threshold=params.threshold,
                    zerovalue=params.zerovalue,
                    inverse=True,
                )
            else:
                nowcast[nowcast < params.threshold] = params.zerovalue

            if params.get("store_as_reflectivity", True):
                nowcast = conversion_tools.rainrate_to_dbz(nowcast, zr_a=params.get("zr_a"), zr_b=params.get("zr_b"))
        else:
            nowcast[nowcast < params.threshold] = params.zerovalue

        if params.nan_to_zero:
            nowcast[~np.isfinite(nowcast)] = params.zerovalue

        if nowcast.ndim == 4:  # S,T,W,H case
            nowcast = nowcast.transpose(1, 0, 2, 3)
            # putting T axis first for saving 1 lt S,W,H preds together

        return nowcast
