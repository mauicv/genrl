## PGY DOCUMENTATION:

The library exposes a set of objects that can be combined to implement evolutionary 
algorithms. Specific object methods can be overwritten in order to extend 
them for different use cases.

#### Simple NEAT example:

The default behaviour implements a NEAT algorithm. The user is left to define a 
function that computes the fitness of the each genome in the population.

```python
from src import NEATPopulation
from src import NEATMutator
from src import generate_neat_metric
from src import Model


def compute_fitness(genome):
    model = Model(genome)
    # compute fitness of model here
    return fitness


mutator = Mutator()
population = Population(mutator=mutator)
metric = generate_neat_metric()
for i in range(10):
    for genome in population.genomes:
        genome.fitness = compute_fitness(genome.to_reduced_repr)
    population.step(metric=metric)
    # want to replace the above with: 
    # population.speciate(metric=metric)
    # mutator(population)

```

___

### Objects:

There are four main objects

| Param | Description |
| --- | ----------- |
| Population | Holds a collection of genomes. Can be initialized or speciated with custom functionality |
| Mutator | Defines the mutation step. Applied to a population to get the next generation. |
| Genome | Individual genome structure. |
| Model | Neural Network Expression of a Genome |


___


## Population

```python
import src

population = src.Population(
    population_size=150, 
    delta=3.0,
    mutation_without_crossover_rate=0.25,
    interspecies_mating_rate=0.001,
    species_member_survival_rate=0.2,
    seed_genomes=None,
    mutator=src.Mutator(),
    speciation_active=True)
```
| Param | Description |
| --- | ----------- |
| population_size | ... |
| mutation_without_crossover_rate | ... |
| interspecies_mating_rate | ... |
| species_member_survival_rate | ... |
| seed_genomes | ... |
| mutator | ... |
| speciation_active | ... |

TODO:
- Implement as interface
- Add population initialisation function.
- Speciate method should be overwriteable
- Evolve should be the mutator `__call__` method.


___


## Mutator

```python
import src

mutator = src.Mutator(
    weight_mutation_likelihood=0.8,
    weight_mutation_rate_random=0.1,
    weight_mutation_rate_uniform=0.9,
    weight_mutation_variance=0.1,
    gene_disable_rate=0.75,
    new_node_probability=0.03,
    new_edge_probability=0.05)
```

| Param | Description |
| --- | ----------- |
| weight_mutation_likelihood | ... |
| weight_mutation_rate_random | ... |
| weight_mutation_rate_uniform | ... |
| weight_mutation_variance | ... |
| gene_disable_rate | ... |
| new_node_probability | ... |
| new_edge_probability | ... |

Mutations:

1. **Node Mutation**:

  *Def* (**Admissible edge**): an edge such that the from_node and to_node occupy layers more than a distance 1 apart.

  When a node is added we randomly sample an admissible edge and then randomly sample an layer index in the range of layers the edge spans. After disabling the sampled edge we add a new node in the selected layer and then connect it with two new edges.

2. **Edge Mutation**:

  *Def* (**Admissible node pair**): A pair of nodes such that the index of the layer that the first node is in is less than the index of the layer the second node occupies.

  When a edge is added we sample two layers without replacement and then order them. We then sample a node from each and add a new edge between them.

___

TODO:
- Implement as subclass of interface.
- Implement a Topological mutator and Value mutator.
___

## Genome

```python
import src

genome = src.Genome(
    input_size=2,
    output_size=2,
    weight_low=-2,
    weight_high=2,
    depth=3)
```
| Param | Description |
| --- | ----------- |
| input_size | ... |
| output_size | ... |
| weight_low | ... |
| weight_high | ... |
| depth | ... |
