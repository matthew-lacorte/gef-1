import os
import sys
import yaml
import argparse
import datetime
from pathlib import Path
import shutil
import uuid
import random
import numpy as np

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)  # if using PyTorch, etc.
    # ...add any others as needed

def load_config(script_dir, config_name=None, cli_overrides=None):
    """
    Load YAML config from configs/ directory, merge with CLI overrides.
    Returns the config dict and the full config path used.
    """
    configs_dir = Path(script_dir) / "configs"
    script_base = Path(script_dir).name
    config_file = configs_dir / (config_name or f"default_{script_base}.yml")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Merge CLI overrides (assume dot notation: key1.key2=value)
    if cli_overrides:
        for override in cli_overrides:
            key, val = override.split("=")
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            # Very simple parsing—customize as needed
            try:
                val = eval(val)  # Use with caution!
            except Exception:
                pass
            d[keys[-1]] = val

    return config, str(config_file)

def make_output_dir(script_name, base_output_dir=None):
    """
    Creates a timestamped output directory for the script run.
    Returns the path to the created output directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    out_base = Path(base_output_dir or "outputs") / script_name
    out_dir = out_base / f"{timestamp}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir
