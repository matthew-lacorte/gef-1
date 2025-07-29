# gef/cli/main.py
import typer
from gef.cli.main import app
from gef.cli.constants import constants_app
# In future: from gef.cli.run import run_app, etc.

app = typer.Typer(help="GEF: General Euclidean Flow CLI")

app.add_typer(constants_app, name="constants")
# app.add_typer(run_app, name="run")

if __name__ == "__main__":
    app()
