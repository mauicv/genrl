## Genetic Algorithms

Aim is to use genetic algorithms to do the bulk of the heavy lifting and traditional reinforcement learning methods to optimize at a later date. The process should be periodically human driven and from the data gathered from human selection process we can further derive the value function that the automated RL and Evolutionary Strategies use.

___

#### Mutator Class:

## use:

```py
m = Mutator(config)
new_genome = m.mutate(genome)
```

Config:
- c_1
- c_2
- c_3
- delta
- weight_mutation_likelyhood
- weight_mutation_rate_random
- weight_mutation_rate_uniform
- gene_disable_rate
- mutation_without_crossover_rate
- insterspecies_mating_rate
- new_node_probability
- new_link_probability

#### Genome class:

Properties:
- inputs: array of input nodes
- outputs: array of output nodes
- layers: number of available layers in which nodes can exist
- input_edges:
- layer_edges

Mutations:

1. **Node Mutation**:

  *Def* (**Admissible edge**): an edge such that the from_node and to_node occupy layers more than a distance 1 apart.

  When a node is added we randomly sample an admissible edge and then randomly sample an layer index in the range of layers the edge spans. After disabling the sampled edge we add a new node in the selected layer and then connect it with two new edges.

2. **Edge Mutation**:

  *Def* (**Admissible node pair**): A pair of nodes such that the index of the layer that the first node is in is less than the index of the layer the second node occupies.

  When a edge is added we sample two layers without replacement and then order them. We then sample a node from each and add a new edge between them.

___

## Resources:

1. [TensorFlow Tutorials](https://www.tensorflow.org/guide/intro_to_graphs)
2. [NEAT in TensorFlow](https://github.com/crisbodnar/TensorFlow-NEAT/blob/master/tf_neat/adaptive_net.py)

___

## Tests:

To run all tests:

```sh
python -m unittest discover tests; pyclean .
```
