# Code for the manuscript "Cell tracking -based framework for assessing nowcasting model skill in reproducing growth and decay of convective rainfall" by Ritvanen et al.

This repository contains code for the manuscript

Ritvanen, J., Pulkkinen, S., Moisseev, D., Nerini, D. (2024): _Cell tracking -based framework for assessing nowcasting model skill in reproducing growth and decay of convective rainfall_

submitted to the Geoscientific Model Development journal.

## Python environment

For installing the conda environment for running the pysteps nowcasts and cell tracking analysis, run

```bash
conda env create -f environment_proc.yml
```

For installing the conda environment for plotting the figures with the notebooks, run

```bash
conda env create -f environment_jupyter.yml
```

## Replicating pysteps nowcasts

1. Create a section in the `pystepsrc` file for your data source. For more information, refer to the existing sections or the [pysteps documentation](https://pysteps.readthedocs.io/en/stable/user_guide/pystepsrc_example.html).
1. Update configuration files in `config/swiss-data/pysteps` directory
   - `hdf5_path` to the output filepath
   - `datelist_path` to the file containing the list of dates to process
   - `data_source_name` to the name of the data source in the `pystepsrc` file
   - as required, the attributes that are written to output file under `save:what_attrs`
1. Run the nowcasting script with the configuration file folder as an argument

```bash
conda activate proc
export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=1

python scripts/run_pysteps_swap_predictions.py swiss-data/pysteps-predictions
```

The script will run the nowcasts for all configuration files in the provided folder. If necessary, move the configuration files to another folder to run only a subset of the configurations.

## Storing the measurement data

1. Create a section in the `pystepsrc` file for your data source. For more information, refer to the existing sections or the [pysteps documentation](https://pysteps.readthedocs.io/en/stable/user_guide/pystepsrc_example.html).
2. Update the paths in the `config/swiss-data/save_measurements.yaml` file
   - `hdf5_path` to the output filepath
   - `datelist_path` to the file containing the list of dates to process
   - `data_source_name` to the name of the data source in the `pystepsrc` file
   - as required, the attributes that are written to output file under `save:what_attrs`
3. Run the script to save the measurements

```bash
conda activate proc

export PYTHONPATH=$PYTHONPATH:.
export OMP_NUM_THREADS=1

python scripts/save_measurements.py config/swiss-data/save_measurements.yaml
```

For both the nowcasts and measurements, the values are packed into an 8-bit format using a lookup table. The lookup table is stored in the `verif/metranet_lookup.py` file. If you wish to disable this behaviour and instead pack using a specified `gain` and `offset` values, set `use_metranet_lookup` to `False` in the configuration file and instead provide the `gain` and `offset` values.

## Running the cell tracking verification framework

The cell tracking verification expects input data in the HDF5 format. The data should have the following structure for the observation files:

```txt
├YYYY-mm-dd HH:MM:SS
│ └measurements
│   ├data [uint8: HEIGHT × WIDTH]
│   │ └2 attributes:
│   │   ├CLASS: b'IMAGE'
│   │   └IMAGE_VERSION: b'1.2'
│   └what
│     └N attributes:
│       ...
```

where the `<YYYY-mm-dd HH:MM:SS>/data` is the dataset containing the rainrate values in mm/h.
The data should be stored as an 8-bit integer with a lookup table to convert the values to the original scale. If the data is instead stored in 8-bit format with `gain` and `offset`, make sure that the `gain` and `offset` values are provided as attributes in `<YYYY-mm-dd HH:MM:SS>/data/what` and that `<YYYY-mm-dd HH:MM:SS>/data/what/use_metranet_lookup` attribute is false.

The nowcasts are expected to be stored in the following structure:

```txt
├YYYY-mm-dd HH:MM:SS
<model-name>
  ├<leadtime-index>
  │ ├data [uint8: WIDTH × HEIGHT]
  │ │ └2 attributes:
  │ │   ├CLASS: b'IMAGE'
  │ │   └IMAGE_VERSION: b'1.2'
  │ └what
  │   N attributes:
  │     ....
  ├<leadtime-index>
  │ ├data [uint8: WIDTH × HEIGHT]
  │ │ └2 attributes:
  │ │   ├CLASS: b'IMAGE'
  │ │   └IMAGE_VERSION: b'1.2'
  │ └what
  │   N attributes:
  ...
```

where the `<YYYY-mm-dd HH:MM:SS>/data/<i>` is the dataset containing the nowcast for leadtime step `<i>` values in mm/h.
The data should be stored as an 8-bit integer with a lookup table to convert the values to the original scale. If the data is instead stored in 8-bit format with `gain` and `offset`, make sure that the `gain` and `offset` values are provided as attributes in `<YYYY-mm-dd HH:MM:SS>/data/<i>/what` and that `<YYYY-mm-dd HH:MM:SS>/data/<i>/what/use_metranet_lookup` attribute is false.

Next, update the configuration file `config/swiss-data/calculate_metrics_objects.yaml`:

- `exp_id` to the experiment identifier
- `path:root` to the output directory root
- `path:metrics`: to the output file location for the metrics
- `path:states`: to the output file location for the verification state
- `path:timestamps`: to the list of timestamps to process
- `path:done`: the path to the CSV file recording which metrics have been calculated
- `path:config_copy`: the path to the configuration file copy
- `path:logging`: the path to the log file
- `methods` items for each nowcast method with the structure

```yaml
<model-name>:
  path: <path-to-hdf5-file>
```

- `measurements` with the following structure

```yaml
measurements:
  name: measurements
  path: <path-to-hdf5-file>
```

- if required, update the cell tracking parameters in the `metrics:OBJECTS:init_kwargs` section:
  - `leadtimes` to the list of leadtimes to process
  - `prev_obs_times`: the number of previous timesteps to load (i.e, t-1, t-2, ...)
  - `zr_a` and `zr_b` to the Z-R relationship parameters used to transform to reflectivity values
  - `dist_limit_matching` to the maximum distance for matching cells in the Hungarian algorithm, in pixel units
  - `tdating_kwargs` to the parameters for the t-dating algorithm. For more information, refer to [pysteps documentation](https://pysteps.readthedocs.io/en/stable/generated/pysteps.tracking.tdating.dating.html)

After that, run the script to calculate the metrics

```bash
conda activate proc

export PYTHONPATH=$PYTHONPATH:.

python scripts/calculate_metrics.py config/swiss-data/calculate_metrics_objects.yaml
```

## Plotting the figures

To plot the results, first set the configuration file `config/swiss-data/plot_metrics_objects.yaml`:

- `exp_id` to the experiment identifier
- `path:result_dir` to the output directory root
- `stylefile` path to the stylefile for the plots
- `legend_order` a list specifying the order of the models in figure legends
- `metric_conf` for each metric specify e.g. the y axis limits and label names
- `methods` each model to be plotted in the following structure

```yaml
<model-name>:
  color: "<model-color-name-or-hex-code"
  label: "<model-label>"
  linestyle: "<model-linestyle>"
```

For running the notebooks, use the `jupyter` conda environment. The notebooks are located in the `notebooks` directory. It is recommended to start the jupyter server in the root directory of the repository, otherwise some paths to stylefiles and configuration files need to be updated in the notebooks.

```bash
conda activate jupyter
export PYTHONPATH=$PYTHONPATH:.

jupyter lab
```

The figures can be plotted with the following notebooks:

- `notebooks/cell_verification_article_figures.ipynb` for all result figures in the article, except Figure 12 (number of splits and merges)
- `notebooks/cell_verification_article_splits_merges.ipynb` for Figure 12 (number of splits and merges)
- `notebooks/cell_verification_article_supplementary_figures.ipynb` for supplementary figures
- `notebooks/plot_domain_figure.ipynb` for plotting the domain figure. Update the path to the topography data in the 4th cell in the notebook.
- `notebooks/plot_pixel_csi_rmse.ipynb` for plotting the pixel-wise CSI and RMSE figures. Note that this notebook uses the

The nowcast figure can be plotted with the script `scripts/plot_nowcast_figures.py` that uses the configuration file `config/swiss-data/plot_nowcast_figs.yaml`. To update the configuration file, refer to the comments in the file. The script can be run with

```bash
conda activate proc

export PYTHONPATH=$PYTHONPATH:.

python scripts/plot_nowcast_figures.py config/swiss-data/plot_nowcast_figs.yaml <YYYYMMMDDHHMM>
```

where `<YYYYMMMDDHHMM>` is the timestamp to plot.

Note that, as the configuration file is provided here, the observations are read from the original metranet data file, so plotting the nowcasts requires the original metranet data file to be available and the `py-radlib` package to be installed.
Also ODIM compliant HDF5 files are supported; in this case, change the `input_data:RATE:reader_func` to `h5_to_dataset`.

## License

The code is licensed under the MIT license. See the LICENSE file for more information.
