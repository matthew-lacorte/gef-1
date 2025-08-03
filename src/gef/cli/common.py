"""
Common CLI options and utilities shared across all GEF commands.
"""
import typer
from pathlib import Path
from typing import Optional, Union
import random
import time
from dataclasses import dataclass

@dataclass
class CommonOptions:
    """Standard options available to all GEF scripts."""
    config: Optional[Path] = None
    output: Optional[Path] = None
    seed: Optional[int] = None
    verbose: bool = False
    dry_run: bool = False

def setup_common_options():
    """
    Returns a function that adds common options to any command.
    Use as a decorator or call directly.
    """
    def decorator(func):
        # Add options in reverse order since decorators are applied bottom-up
        func = typer.Option(help="Configuration file path (default: auto-detect)")(func)
        func = typer.Option(help="Output directory (default: script-specific location)")(func) 
        func = typer.Option(help="Random seed (integer or 'random' for time-based)")(func)
        func = typer.Option(help="Show what would be done without executing")(func)
        return func
    return decorator

def parse_seed(seed_str: Optional[str]) -> Optional[int]:
    """Parse seed string into integer, handling 'random' case."""
    if seed_str is None:
        return None
    if seed_str.lower() == 'random':
        return int(time.time() * 1000) % (2**31)  # Keep it within int32 range
    try:
        return int(seed_str)
    except ValueError:
        raise typer.BadParameter(f"Seed must be an integer or 'random', got: {seed_str}")

def resolve_config_path(config: Optional[Path], script_name: str, search_dirs: list = None) -> Path:
    """
    Resolve configuration file path with smart defaults.
    
    Args:
        config: Explicit config path from user
        script_name: Name of the calling script/command
        search_dirs: Additional directories to search (default: common locations)
    
    Returns:
        Path to configuration file
        
    Raises:
        typer.BadParameter: If config file not found
    """
    if config and config.exists():
        return config.resolve()
    
    # Default search locations
    if search_dirs is None:
        search_dirs = [
            Path.cwd() / "configs",
            Path(__file__).parent.parent.parent.parent / "configs",  # Project root configs
            Path(__file__).parent.parent / "configs",  # Package configs
        ]
    
    # Try default names
    default_names = [
        f"default_{script_name}.yml",
        f"default_{script_name}.yaml", 
        f"{script_name}.yml",
        f"{script_name}.yaml"
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for name in default_names:
            candidate = search_dir / name
            if candidate.exists():
                return candidate.resolve()
    
    # If explicit config given but not found
    if config:
        raise typer.BadParameter(f"Configuration file not found: {config}")
    
    # No config found anywhere
    raise typer.BadParameter(
        f"No configuration file found for '{script_name}'. "
        f"Searched: {search_dirs} for files like: {default_names}"
    )

def resolve_output_path(output: Optional[Path], script_name: str, base_name: str = None) -> Path:
    """
    Resolve output directory with smart defaults.
    
    Args:
        output: Explicit output path from user  
        script_name: Name of the calling script/command
        base_name: Base name for auto-generated directory
        
    Returns:
        Path to output directory (created if necessary)
    """
    if output:
        output.mkdir(parents=True, exist_ok=True)
        return output.resolve()
    
    # Auto-generate output directory
    from gef.core.utils import make_output_dir
    return make_output_dir(script_name, base_output_dir=Path.cwd() / "outputs")
