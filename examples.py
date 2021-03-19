import click
from examples.xor_neat_example import neat_xor_example
from examples.cart_pole_NEAT import neat_cart_pole
from examples.bip_walker_NEAT import neat_bipedal_walker


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command()
def xor():
    neat_xor_example()


@cli.command()
def cart_pole():
    neat_cart_pole()


@cli.command()
def bipedal_walker():
    neat_bipedal_walker()


if __name__ == "__main__":
    cli()
