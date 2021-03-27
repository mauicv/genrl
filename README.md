## GeReL

GeReL is a simple library for genetic algorithms applied to
reinforcement learning.

_NOTE_: GeReL is in development.

### Example:

The following uses [REINFORCE-ES](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) to solve openai 
[cartpole environment](https://gym.openai.com/envs/CartPole-v1/)

```python
from gerel.genome.factories import dense
from gerel.algorithms.RES.population import RESPopulation
from gerel.algorithms.RES.mutator import RESMutator
from gerel.model.model import Model
import gym
import numpy as np
from gerel.populations.genome_seeders import curry_genome_seeder
from string import Template


def compute_fitness(genome):
    model = Model(genome)
    env = gym.make("CartPole-v0")
    state = env.reset()
    fitness = 0
    action_map = lambda a: 0 if a[0] <= 0 else 1  # noqa
    for _ in range(1000):
        action = model(state)
        action = action_map(action)
        state, reward, done, _ = env.step(action)
        fitness += reward
        if done:
            break

    return fitness


if __name__ == '__main__':
    genome = dense(
        input_size=4,
        output_size=1,
        layer_dims=[2, 2, 2]
    )

    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.random.uniform(-1, 1, weights_len)

    mutator = RESMutator(
        initial_mu=init_mu,
        std_dev=0.1,
        alpha=0.05
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=50,
        genome_seeder=seeder
    )

    report_temp = Template('generation: $generation, mean: $mean, best: $best')
    for generation in range(100):
        for genome in population.genomes:
            genome.fitness = compute_fitness(genome.to_reduced_repr)
        population.speciate()
        data = population.to_dict()
        mutator(population)
        report = report_temp.substitute(
            generation=generation,
            mean=data['mean_fitness'],
            best=data['best_fitness'])
        print(report)
```

___

### Tests:

To run all unittests:

```shell
python -m unittest discover tests/unit_tests; pyclean .
```

___



