import os
import toml
from easydict import EasyDict


this_path = os.path.abspath(os.path.dirname(__file__))


def get_config():
    config_path = os.path.join(this_path, 'detector_config.toml')

    cfg = toml.load(config_path)
    cfg = EasyDict(cfg)  # Convert to EasyDict for simplicity
    cfg = _check_and_correct_config(cfg)

    return cfg


def _check_and_correct_config(cfg):
    assert cfg.mode == 'deploy'

    cfg.model_cfg = os.path.join(this_path, cfg.model_cfg)  # Convert to absolute path
    cfg.trained_weights = os.path.join(this_path, cfg.trained_weights)  # Convert to absolute path

    assert os.path.exists(cfg.model_cfg)  # Model must exist
    assert os.path.exists(cfg.trained_weights)  # Model must exist

    return cfg

get_config()