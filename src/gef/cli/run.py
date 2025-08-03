"""
Script execution commands for GEF analysis workflows.
"""
import typer
from pathlib import Path
from typing import Optional
import importlib.util
import sys
from rich.console import Console
from rich.table import Table

from gef.cli.common import CommonOptions, parse_seed, resolve_config_path, resolve_output_path
from gef.core.logging import logger

console = Console()
run_app = typer.Typer(help="Execute GEF analysis scripts")

@run_app.command("list")
def list_scripts():
    """List all available analysis scripts."""
    # Find all runnable scripts
    script_dirs = [
        Path(__file__).parent.parent.parent.parent / "scripts",  # Project scripts
        Path(__file__).parent.parent / "scripts",  # Package scripts  
    ]
    
    table = Table(title="Available GEF Scripts")
    table.add_column("Script", style="cyan")
    table.add_column("Location", style="dim")
    table.add_column("Description", style="green")
    
    found_scripts = []
    for script_dir in script_dirs:
        if not script_dir.exists():
            continue
        
        # Look for Python files
        for script_file in script_dir.rglob("*.py"):
            if script_file.name.startswith("__"):
                continue
            rel_path = script_file.relative_to(script_dir)
            script_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
            
            # Try to extract docstring for description
            description = "No description"
            try:
                with open(script_file) as f:
                    content = f.read()
                    if '"""' in content:
                        start = content.find('"""') + 3
                        end = content.find('"""', start)
                        if end > start:
                            description = content[start:end].strip().split('\n')[0]
            except:
                pass
            
            found_scripts.append((script_name, str(rel_path), description))
    
    for script_name, location, description in sorted(found_scripts):
        table.add_row(script_name, location, description)
    
    if found_scripts:
        console.print(table)
    else:
        console.print("[yellow]No scripts found in search directories[/yellow]")

@run_app.command("fractal-spelunker")
def run_fractal_spelunker(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path (default: auto-detect)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: script-specific location)"), 
    seed: Optional[str] = typer.Option(None, "--seed", help="Random seed (integer or 'random' for time-based)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """Run the fractal spelunker analysis."""
    _run_script_with_common_options(
        script_name="fractal_spelunker",
        script_path="scripts/fractals/fractal_spelunker/fractal_spelunker.py",
        config=config,
        output=output,
        seed=seed,
        dry_run=dry_run
    )

@run_app.command("planck-particle")  
def run_planck_particle(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path (default: auto-detect)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: script-specific location)"),
    seed: Optional[str] = typer.Option(None, "--seed", help="Random seed (integer or 'random' for time-based)"), 
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """Run Planck particle geometric radius calculations."""
    _run_script_with_common_options(
        script_name="planck_particle", 
        script_path="scripts/core/geometric_speed_of_light/planck_particle.py",
        config=config,
        output=output,
        seed=seed,
        dry_run=dry_run
    )

@run_app.command("borwein")
def run_borwein_stabilization(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path (default: auto-detect)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: script-specific location)"),
    seed: Optional[str] = typer.Option(None, "--seed", help="Random seed (integer or 'random' for time-based)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")  
):
    """Run Borwein integral stabilization experiment."""
    _run_script_with_common_options(
        script_name="borwein_stabilization",
        script_path="scripts/fractals/borwein_stabilization/borwein_stabilization.py", 
        config=config,
        output=output,
        seed=seed,
        dry_run=dry_run
    )

def _run_script_with_common_options(
    script_name: str,
    script_path: str, 
    config: Optional[Path],
    output: Optional[Path],
    seed: Optional[str],
    dry_run: bool
):
    """
    Internal helper to run a script with standardized option handling.
    """
    try:
        # Resolve paths
        config_path = resolve_config_path(config, script_name)
        output_path = resolve_output_path(output, script_name)
        parsed_seed = parse_seed(seed)
        
        console.print(f"[bold green]Running {script_name}[/bold green]")
        console.print(f"Config: {config_path}")
        console.print(f"Output: {output_path}")
        if parsed_seed is not None:
            console.print(f"Seed: {parsed_seed}")
        
        if dry_run:
            console.print("[yellow]DRY RUN - not executing[/yellow]")
            return
            
        # Set up environment
        if parsed_seed is not None:
            from gef.core.utils import set_global_seed
            set_global_seed(parsed_seed)
            
        # Load and execute the script
        script_full_path = Path(__file__).parent.parent.parent.parent / script_path
        if not script_full_path.exists():
            raise typer.BadParameter(f"Script not found: {script_full_path}")
            
        # Import and run the script's main function
        spec = importlib.util.spec_from_file_location(script_name, script_full_path)
        module = importlib.util.module_from_spec(spec)
        
        # Prepare arguments that the script expects
        script_args = [
            "--config", str(config_path),
            "--output-dir", str(output_path)
        ]
        
        # Temporarily modify sys.argv so the script sees our arguments
        original_argv = sys.argv.copy()
        sys.argv = [str(script_full_path)] + script_args
        
        try:
            spec.loader.exec_module(module)
            # Most scripts have a main() function
            if hasattr(module, 'main'):
                module.main()
            else:
                console.print("[yellow]Script executed but no main() function found[/yellow]")
        finally:
            sys.argv = original_argv
            
        console.print(f"[bold green]✓ {script_name} completed successfully[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error running {script_name}: {e}[/bold red]")
        raise typer.Exit(1)