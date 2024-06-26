"""
"""
import argparse
from typing import Sequence
import argparse
from datetime import datetime, timedelta

import h5py
from tqdm import tqdm

from verif.prediction_builder_instances import MeasurementBuilder
from verif import io_tools


def run(builders: Sequence[MeasurementBuilder]) -> None:
    date_paths = [builder.date_path for builder in builders]
    if any(path != date_paths[0] for path in date_paths):
        raise ValueError(
            "The datelists used must be the same for all runs,\
                        Please check that the paths given match."
        )

    days = io_tools.read_timestamp_txt_file(date_paths[0], start_idx=0)
    timesteps = []
    for d in days:
        for i in range(288):
            tnow = datetime.strptime(d, "%Y-%m-%d") + timedelta(minutes=i * 5)
            timesteps.append(datetime.strftime(tnow, "%Y-%m-%d %H:%M:%S"))
    output_dbs = [h5py.File(builder.hdf5_path, "w") for builder in builders]

    for t in tqdm(timesteps):
        for i, builder in enumerate(builders):
            group_name = builder.save_params.group_format.format(
                timestamp=t, method=builder.nowcast_params.nowcast_method
            )
            group = output_dbs[i].require_group(group_name)
            try:
                nowcast = builder.run(t)
                builder.save(
                    nowcast=nowcast, group=group, save_parameters=builder.save_params
                )
            except IOError:
                print("IO error")
                continue
            # except ValueError:
            #    print("Value error")
            #    continue

    for db in output_dbs:
        db.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration path")
    args = argparser.parse_args()

    config_path = args.config
    config = io_tools.load_yaml_config(config_path)
    predictor_builders = [MeasurementBuilder(config=config)]
    run(builders=predictor_builders)
