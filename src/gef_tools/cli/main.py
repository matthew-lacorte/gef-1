# gef_tools/cli/main.py

# REFERENCE, set up an actual CLI registry

import click

@click.group()
def gef():
    """GEF Research Framework CLI"""
    pass

@gef.command()
def vault_processor():
    """Run vault-wide linting and generation"""
