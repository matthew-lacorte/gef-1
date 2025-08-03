def load_script_config(script_path: Path, config_name: str = None) -> dict:
    """Standard config loading for all GEF scripts."""
    # Unified logic for finding configs/ relative to script location