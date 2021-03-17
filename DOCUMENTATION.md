## PGY DOCUMENTATION:

The intent of this library is to expose a set of objects that can be easily combined 
to implement evolutionary algorithms.

#### Simple NEAT example:

The following implements a NEAT algorithm. The user is left to define a function that 
computes the fitness of each genome in the population. The model is a graph object
corresponding to the expression of The genome. `model(inputs)` will run the computation
across all the nodes and edges and return an array. In this case the genome defaults to
assuming input and output arrays of size 2. 

```python
from src import NEATPopulation
from src import NEATMutator
from src import Model


def compute_fitness(soln):
    model = Model(soln)
    # compute fitness of model here
    return fitness


mutator = NEATMutator()
population = NEATPopulation()
for i in range(10):
    for genome in population.genomes:
        genome.fitness = compute_fitness(genome.to_reduced_repr)
    population.speciate()
    mutator(population)

```

___

### API:

The core API is made up of 4 main objects:

| Param | Description |
| --- | ----------- |
| Population | Holds a collection of genomes. Supports speciation if needed.|
| Mutator | Defines the mutation step. Applied to a population to get the next generation. `call_on_genome` and `call_on_population` must be overriden|
| Genome | Individual genome structure. Holds a record of nodes and edges making up the network and the order in which innovation there evolutionary innovation occured. |
| Model | Neural Network Expression of a Genome. Genomes can't run computation themselves but can be mapped to a simpler representation which passed to a Model will generate the expression of that Genome. |

___


#### src.Population

Holds a population of genomes. If `metric` and `delta` parameters are defined then the method
`speciate` will evolve the collection of genomes into seperate species within `delta` of each 
other in the defined `metric` function. 

```python
import src

population = src.Population(
    population_size=None,
    metric=None,
    delta=None,
    genome_seeder=None
)
```
| Param | Description |
| --- | ----------- |
| population_size | An integer number giving the size of the total Population. |
| metric | Function that takes two genomes and returns the distance between them. The metric is used to partition the genomes into species. If None then no speciation will take place. |
| delta | A float number that defines the minimum distance two genomes require to be apart before there sorted into different species. Set to None if single species is desired (no speciation). |
| genome_seeder | An iterable of genomes that will be used to generate the initial population. |

___

#### src.curry_genome_seeder

Returns a generator function that takes a parameter `n` and returns `n` Genomes. The genomes
are cyclically drawn from the list of `seed_genomes` and mutated by the `mutator`. 

```python
import src

src.curry_genome_seeder(
    mutator=None, 
    seed_genomes=None)
```

| Param | Description |
| --- | ----------- |
| seed_genomes | List of genomes that will be copied and mutated. |
| mutator | Defined mutator object that acts on genomes. |

__Returns__: function _(Genome generator function)_

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
