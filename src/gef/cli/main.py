# src/gef/cli/main.py
import typer
from gef.cli.constants import constants_app
from gef.cli.run import run_app

app = typer.Typer(
    help="GEF: General Euclidean Flow CLI",
    context_settings={"help_option_names": ["-h", "--help"]}
)

# Add sub-commands
app.add_typer(constants_app, name="constants")
app.add_typer(run_app, name="run")

@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    version: bool = typer.Option(False, "--version", help="Show version and exit")
):
    """
    GEF: General Euclidean Flow analysis toolkit.
    
    Use 'gef COMMAND --help' to see options for specific commands.
    """
    if version:
        from gef.core.version import __version__
        typer.echo(f"GEF version {__version__}")
        raise typer.Exit()
    
    # Store global options in context for sub-commands to access
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

if __name__ == "__main__":
    app()