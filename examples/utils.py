import numpy as np
from gerel.model.model import Model
import gym


def make_counter_fn(lim=5):
    def counter_fn():
        if not hasattr(counter_fn, 'count'):
            counter_fn.count = 0
        counter_fn.count += 1
        if counter_fn.count == lim:
            counter_fn.count = 0
            return True
        return False
    return counter_fn


def build_simple_env(target_mu):
    """ Testing environment

    Creates function that takes a Population subclass instance and
    computes the distance each genome is from target_mu.

    :param target_mu: Genomes score higher fitness dependent on how close
        there weights are to target_mu.
    :return: compute_population_fitness function that takes a population
        and assigns a fitness to each member.
    """
    def compute_population_fitness(population):
        for genome in population.genomes:
            # fitness is symmetrically allocated.
            _, ws = genome.values()
            ws = np.array(ws)
            fitness = 1 / np.power(target_mu - ws, 2).sum()
            genome.fitness = fitness
    return compute_population_fitness


def build_env(
        env_name='CartPole-v0',
        target=200,
        num_steps=1000,
        repetition=3):
    """Builds cart pole environment for calling on population of genomes.

    :param target: Target score for success.
    :param env_name: Environment to run.
    :param num_steps: Number of environment steps
    :param repetition: Number of environment reruns
    :return: Function that accepts population and assigns genome fitness.
        Returns True if a solution found.
    """
    def compute_population_fitness(population):
        success = False
        for genome in population.genomes:
            fitness = n_run_env(
                repetition,
                genome.to_reduced_repr,
                env_name=env_name,
                num_steps=num_steps)
            genome.fitness = fitness
            if fitness == target:
                success = True
        return success
    return compute_population_fitness


def run_env(
        genome,
        render=True,
        env_name="CartPole-v0",
        num_steps=1000
            ):
    model = Model(genome)
    env = gym.make(env_name)
    state = env.reset()
    fitness = 0

    if env_name == 'CartPole-v0':
        action_map = lambda a: 0 if a[0] <= 0 else 1
    elif env_name == 'BipedalWalker-v3':
        action_map = lambda x: np.tanh(np.array(x))
    else:
        action_map = lambda x: x

    for _ in range(num_steps):
        action = model(state)
        action = action_map(action)
        state, reward, done, _ = env.step(action)
        fitness += reward
        if render:
            env.render()

        if done:
            break

    return fitness


def n_run_env(n, genome, env_name="CartPole-v0", num_steps=1000):
    fitness = 0
    for i in range(n):
        fitness += run_env(
            genome,
            render=False,
            env_name=env_name,
            num_steps=num_steps)
    return fitness/n
