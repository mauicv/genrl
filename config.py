"""From NEAT paper:

- All experiments except DPNV, which had a population of 1,000, used a
population of 150 NEAT networks.
- The coefficients for measuring compatibility were c1 = 1.0, c2 = 1.0,
and c3 = 0.4.
- In all experiments, δt = 3.0, except in DPNV where it was 4.0, to make
room for the larger weight significance coefficient c3.
- There was an 80% chance of a genome having its connection weights mutated,
in which case each weight had a 90% chance of being uniformly perturbed and
a 10% chance of being assigned a new random value.
- There was a 75% chance that an inherited gene was disabled if it was disabled
in either parent
- In each generation, 25% of offspring resulted from mutation without
crossover.
- The interspecies mating rate was 0.001.
- In smaller populations, the probability of adding a new node was 0.03 and the
probability of a new link mutation was 0.05. In the larger population,
the probability of adding a new link was 0.3, because a larger population can
tolerate a larger number of prospective species and greater topological
diversity.
- We used a modified sigmoidal transfer function, ϕ(x) = 1/(1+e−4.9x) , at
all nodes.

- If the maximum fitness of a species did not improve in 15 generations,
the networks in the stagnant species were not allowed to reproduce.
- The champion of each species with more than five networks was copied into
the next generation unchanged.
"""


config = {
    "POPULATION": 150,
    "C_1": 1.0,
    "C_2": 1.0,
    "C_3": 0.4,
    "DELTA": 3.0,
    "WEIGHT_MUTATION_LIKELYHOOD": 0.8,
    "WEIGHT_MUTATION_RATE_UNIFORM": 0.9,
    "WEIGHT_MUTATION_RATE_RANDOM": 0.1,
    "WEIGHT_MUTATION_VARIANCE": 0.1,
    "GENE_DISABLE_RATE": 0.75,
    "MUTATION_WITHOUT_CROSSOVER_RATE": 0.25,
    "INSTERSPECIES_MATING_RATE": 0.001,
    "NEW_NODE_PROBABILITY": 0.03,
    "NEW_LINK_PROBABILITY": 0.05
}
