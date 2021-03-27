from gerel.model.model import Model
import gym
import numpy as np
import matplotlib.pyplot as plt
import csv
import json


def compute_fitness(genome, render=False):
    model = Model(genome)
    env = gym.make("BipedalWalker-v3")
    state = env.reset()
    fitness = 0
    for _ in range(1500):
        action = model(state)
        action = np.tanh(np.array(action))
        state, reward, done, _ = env.step(action)
        fitness += reward
        if render:
            env.render()

        if done:
            break

    return fitness


def run_trained_RES_bip_walker_example():
    with open('../assets/RES_bip_walk_train_logs.csv', 'r') as train_logs:
        reader = csv.reader(train_logs)
        next(reader)
        plt.plot([float(best) for _, best, _, _ in reader])
        plt.show()

    with open('../assets/RES_bip_walk_soln.json', 'r') as soln_file:
        best_genome = json.loads(soln_file.read())
        compute_fitness(best_genome, render=True)


if __name__ == "__main__":
    run_trained_RES_bip_walker_example()
