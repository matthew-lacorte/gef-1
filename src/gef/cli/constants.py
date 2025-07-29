# gef/cli/constants.py
import typer
from gef.constants import CONSTANTS_DICT
from rich import print  # Optional: nice formatting

constants_app = typer.Typer(help="View and manage model/derived constants.")

@constants_app.command("list")
def list_constants():
    """List all model/derived constants (names and descriptions)."""
    for name, const in CONSTANTS_DICT.items():
        print(f"[bold cyan]{name}[/bold cyan]: {const.description}")

@constants_app.command("show")
def show_constant(
    name: str = typer.Argument(..., help="Constant name"),
    format: str = typer.Option("plain", help="Output format: plain|json|md")
):
    """Show all metadata for a constant."""
    const = CONSTANTS_DICT.get(name)
    if not const:
        print(f"[red]Constant not found:[/red] {name}")
        raise typer.Exit(1)
    if format == "json":
        import json
        data = const.model_dump() if hasattr(const, "model_dump") else const.dict()
        print(json.dumps(data, indent=2))
    elif format == "md":
        print(
            f"## {const.name}\n\n"
            f"{const.description}\n\n"
            f"- **Symbol:** `{const.symbol}`\n"
            f"- **Units:** {const.units}\n"
            f"- **Category:** {const.category}\n"
            f"- **Value:** {const.value}\n"
            f"- **Sidecar:** {const.sidecar_path}\n"
        )
    else:
        print(str(const))
