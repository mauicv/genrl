## pygrl

pygrl is a simple framework for running genetic aglorithms for 
reinforcement learning.

__NOTE__: pygrl is currently in development.

### Example:

The following uses [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) to solve openai 
[cartpole environment](https://gym.openai.com/envs/CartPole-v1/)

```python
from src import NEATPopulation
from src import NEATMutator
from src import generate_neat_metric
from src import Model


def compute_fitness(genome):
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
        if done:
            break

    return fitness


if __name__ == '__main__':
    mutator = NEATMutator()
    population = NEATPopulation()
    for i in range(10):
        for genome in population.genomes:
            genome.fitness = compute_fitness(genome.to_reduced_repr)
        population.speciate()
        mutator(population)
```

To see other examples using NEAT, REINFORCE-ES or other algorithms applied to some openai
gym environments see examples.

___

### Tests:

To run all unittests:

```shell
python -m unittest discover tests/unit_tests; pyclean .
```

___



