def validate_population(population, *args, **kwargs):
    for key, item in population.species.items():
        if len(set(item['group'])) != len(item['group']):
            raise ValueError('Non unique genomes elements in species group.')
