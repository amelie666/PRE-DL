"""Console script for pre_dl."""
import sys
import click


@click.group()
def cli():
    pass


@click.command()
@click.argument('datetime')
@click.option('--out', default='out', help='output directory')
def download_pre(datetime, out):
    click.echo('Initialized the database')


@click.command()
@click.argument('datetime')
@click.option('--out', default='out', help='output directory')
def download_bt(datetime, out):
    click.echo('Dropped the database')


cli.add_command(download_pre)
cli.add_command(download_bt)

if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
