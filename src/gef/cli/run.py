import typer
from pathlib import Path
import importlib.util
import sys

run_app = typer.Typer(help="Run GEF analysis scripts")

@run_app.command("list")
def list_scripts():
    """List available analysis scripts."""
    # Scan for runnable scripts and show them

@run_app.command("execute") 
def run_script(
    script_name: str = typer.Argument(..., help="Script to run"),
    config: Path = typer.Option(None, help="Config file path")
):
    """Execute a specific analysis script."""
    # Load and run the script with proper path handling