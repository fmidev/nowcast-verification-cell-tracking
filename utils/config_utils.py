import yaml
from addict import Dict


def load_config(file):
    """Load configuration from YAML file.

    Parameters
    ----------
    file : str
        Path to configuration file.

    Returns
    -------
    conf_dict : attrdict.AttrDict
        AttrDict containing configurations.

    """
    # read configuration files
    with open(file, "r") as f:
        conf_dict = Dict(yaml.safe_load(f))
    return conf_dict