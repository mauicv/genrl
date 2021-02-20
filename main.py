import click
import numpy as np
import os
import shutil
from time import time, sleep
from gym import Env

def setup_env(var=0, vis=True):
    env = Env('quadruped', var=var, vis=vis)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    state_norm_array = env.observation_space.high
    min_action = env.action_space.low.min()
    max_action = env.action_space.high.max()
    if np.any(np.isinf(state_norm_array)):
        state_norm_array = np.ones_like(state_norm_array)
    return env, state_space_dim, action_space_dim, state_norm_array, \
        min_action, max_action


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    print('here')
    if ctx.invoked_subcommand is None:
        pass


# @cli.command()
# @click.pass_context
# @click.option('--episodes', '-e', default=5000, type=int,
#               help='Number of epsiodes of training')
# @click.option('--steps', '-s', default=400, type=int,
#               help='Max number of steps per episode')
# @click.option('--dir', '-d', default='default',
#               help='Save file location')
# def train(ctx, episodes, steps, noise_type, dir):
#     Path(f"save/{dir}").mkdir(parents=True, exist_ok=True)
#     logger = Logging(['episode',
#                       'rewards',
#                       'episode_length',
#                       'epsiode_run_time',
#                       'average_step_run_time',
#                       'q_loss',
#                       'p_loss'],
#                       params={
#                         "EPISODES": episodes,
#                         "STEPS":    steps
#                       },
#                       save_loc=dir)
#
#     env, state_space_dim, action_space_dim, state_norm_array, min_action, \
#         max_action = setup_env()

@cli.command()
@click.pass_context
@click.option('--steps', '-s', default=400, type=int,
              help='Max number of steps per episode')
@click.option('--dir', '-nt', default='default',
              help='save file location')
def play(ctx, steps, dir):
    env, state_space_dim, action_space_dim, state_norm_array, min_action, \
        max_action = setup_env(var=0, vis=True)
    r = 0
    for i in range(steps):
        state, reward, done, _ = env \
            .step([])

# @cli.command()
# @click.pass_context
# @click.option('--dir', '-nt', default='default',
#               help='Save file location')
# @click.option('--all', '-a', is_flag=True,
#               help='Delete all files')
# def clean(ctx, dir, all):
#     if all:
#         for dir in os.listdir('save'):
#             shutil.rmtree(f'save/{dir}')
#     else:
#         shutil.rmtree(f'save/{dir}')

if __name__ == '__main__':
    cli()
