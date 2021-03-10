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
from src.util import print_population, save
import matplotlib.pyplot as plt


# batch_job = BatchJob(num_processes=3)

# @batch_job
def compute_fitness(genome, render=False):
    model = Model(genome)
    env = gym.make("BipedalWalker-v3")
    state = env.reset()
    fitness = 0
    for _ in range(200):
        action = model(state)
        action = np.tanh(np.array(action))
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


def test_bip_walker():
    pop_size = 400
    m = Mutator(
        new_edge_probability=0.1,
        new_node_probability=0.05
    )
    g = Genome.default(
        input_size=24,
        output_size=4,
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

    means = []
    mins = []
    maxs = []
    for i in range(100):
        best_fitness = 0
        fitnesses = []
        best_genome = p.genomes[0].to_reduced_repr
        for g in p.genomes:
            reward = compute_n_fitness(1, g.to_reduced_repr)
            g.fitness = reward
            fitnesses.append(reward)
            if reward > best_fitness:
                best_fitness = reward
                best_genome = g.to_reduced_repr
        compute_fitness(best_genome)
        p.step(metric=d)
        fitnesses = np.array(fitnesses)
        print('run:', i, 'best:', fitnesses.max(), 'worst:', fitnesses.min(), 'avg:', fitnesses.mean())
        means.append(fitnesses.mean())
        mins.append(fitnesses.min())
        maxs.append(fitnesses.max())
    plt.plot(np.array(maxs))
    plt.show()
    print_population(p)
    compute_fitness(best_genome, True)
    save(best_genome, 'save.txt')
    return True
