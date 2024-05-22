"""
    This script will run nowcasting predictions
    for advection based deterministic methods implemented in pysteps, with multiple different configurations

    Working (tested) prediction types:
    - extrapolation
    - S-PROG
    - LINDA
    - ANVIL
    - STEPS

    Usage requires:
    1) Having a workable pysteps installation with
    .pystepsrc configured.
    2) (Optionally) modifying your AdvectionPrediction class to
    satisfy requirements.
    3) Setting configuration files for each prediction experiment
    to be run, putting them in the folder passed as an argument
"""
import argparse
import sys
import os
from typing import Sequence
import argparse
from pathlib import Path
import yaml
import logging
import logging.config

import h5py
from tqdm import tqdm

from verif.prediction_builder_instances import PystepsSwapPrediction
from verif import io_tools

# temp for except handling
import dask

# Setup logging
with open("logconf.yaml", "rt") as f:
    log_config = yaml.safe_load(f.read())
    f.close()
logging.config.dictConfig(log_config)
logging.captureWarnings(True)
logger = logging.getLogger(Path(__file__).stem)


def run(builders: Sequence[PystepsSwapPrediction]) -> None:
    date_paths = [builder.date_path for builder in builders]
    if any(path != date_paths[0] for path in date_paths):
        raise ValueError(
            "The datelists used must be the same for all runs,\
                        Please check that the paths given match."
        )

    logger.info(f"Using datelists {date_paths}")
    timesteps = io_tools.read_timestamp_txt_file(date_paths[0])
    output_dbs = [h5py.File(builder.hdf5_path, "a") for builder in builders]

    for t in tqdm(timesteps):
        logger.info(f"sample {t} ongoing...")
        for i, builder in enumerate(builders):
            group_name = builder.save_params.group_format.format(
                timestamp=io_tools.get_neighboring_timestamp(
                    time=t, distance=builder.input_params.num_next_files
                ),
                method=builder.nowcast_params.nowcast_method,
            )
            group = output_dbs[i].require_group(group_name)
            if len(group.keys()) == builder.nowcast_params.n_leadtimes:
                logger.info(
                    f"Skipping {builder.nowcast_params.nowcast_method} for {t}, already computed."
                )
                continue

            logger.info(
                f"Running predictions for {builder.nowcast_params.nowcast_method} for {t} method."
            )
            # sys.stdout = open(os.devnull, "w")
            try:
                nowcast = builder.run(t)
            except (ValueError, OSError) as e:
                logger.exception(
                    f"Error in run method for time {t} and method {builder.nowcast_params.nowcast_method}: {e}"
                )
                sys.stdout = sys.__stdout__
                continue

            sys.stdout = sys.__stdout__
            try:
                builder.save(
                    nowcast=nowcast, group=group, save_parameters=builder.save_params
                )
            except (ValueError, OSError) as e:
                logger.exception(
                    f"Error in save method for time {t} and method {builder.nowcast_params.nowcast_method}: {e}"
                )
                continue

    for db in output_dbs:
        db.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument(
        "config",
        type=str,
        help="Configuration folder path, contains \
                           one YAML configuration file per forecast \
                           type that is to be computed.",
    )
    args = argparser.parse_args()

    config_dir = Path("config") / args.config
    config_filenames = list(config_dir.glob("*.yaml"))
    configurations = [
        io_tools.load_yaml_config(filename) for filename in config_filenames
    ]
    predictor_builders = [
        PystepsSwapPrediction(config=config) for config in configurations
    ]
    run(builders=predictor_builders)
