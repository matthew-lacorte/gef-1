# gef/cli/main.py
import typer
from gef.cli.constants import constants_app

app = typer.Typer(help="GEF: General Euclidean Flow CLI")
app.add_typer(constants_app, name="constants")

if __name__ == "__main__":
    app()
