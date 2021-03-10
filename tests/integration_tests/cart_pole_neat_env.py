"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src import Genome
from src import Population
from src import Mutator
from src import Model
from src import generate_neat_metric
from tqdm import tqdm
import gym
import numpy as np
from src.batch import BatchJob
from src.util import print_population
import matplotlib.pyplot as plt


# batch_job = BatchJob(num_processes=3)

# @batch_job
def compute_fitness(genome, render=False):
    # print(genome)
    model = Model(genome)
    env = gym.make("CartPole-v0")
    state = env.reset()
    fitness = 0
    action_map = lambda a: 0 if a[0] <= 0 else 1
    for _ in range(1000):
        action = model(state)
        action = action_map(action)
        state, reward, done, _ = env.step(action)
        fitness += reward
        if render:
            env.render()

        if done:
            break

    return fitness


def compute_n_fitness(n, genome):
    fitness = 0
    for i in range(n):
        fitness += compute_fitness(genome)
    return fitness/n


def test_cart_pole():
    pop_size = 1000
    m = Mutator(
        new_edge_probability=0.1,
        new_node_probability=0.05
    )
    g = Genome.default(
        input_size=4,
        output_size=1,
        depth=5
    )
    p = Population(
        population_size=pop_size,
        seed_genomes=[g],
        mutator=m,
        delta=4
    )
    d = generate_neat_metric(
        c_1=1,
        c_2=1,
        c_3=3
    )

    # for i in tqdm(range(10)):
    #     genomes = [g.to_reduced_repr for g in p.genomes]
    #     rewards = compute_fitness(genomes)
    #     print(rewards)

    train_data = []
    for i in range(10):
        best_fitness = 0
        fitnesses = []
        for g in p.genomes:
            reward = compute_n_fitness(5, g.to_reduced_repr)
            g.fitness = reward
            fitnesses.append(reward)
            if reward > best_fitness:
                best_fitness = reward
                best_genome = g.to_reduced_repr
        compute_fitness(best_genome)
        p.step(metric=d)
        fitnesses = np.array(fitnesses)
        print('run:', i, 'best:', fitnesses.max(), 'worst: ', fitnesses.min(), 'avg:', fitnesses.mean())
        train_data.append(fitnesses.mean())
    plt.plot(np.array(train_data))
    plt.show()
    print_population(p)
    compute_fitness(best_genome, True)
    return True
