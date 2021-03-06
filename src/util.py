import functools
import random


def sample_weight(low, high):
    weight_range = high - low
    return random.random() * weight_range + low


def get_random():
    return random.random() * 2 - 1


def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def verify_genome_consistency(genome):
    edge_innov = set(e.innov for e in genome.edges)
    node_innov = set(n.innov for n in genome.nodes)
    truth = len(edge_innov) == len(genome.edges)
    truth = truth and len(node_innov) == len(genome.nodes)
    for e in genome.edges:
        truth = truth and (e.from_node.innov, e.to_node.innov) in genome.edge_innovs
    for e_1, e_2 in zip(genome.edges, genome.edges[1:]):
        truth = truth and e_1.innov < e_2.innov
    return truth


def print_genome(genome):
    edge_innov = set(e.innov for e in genome.edges)
    node_innov = set(n.innov for n in genome.nodes)
    print('')
    print(genome)
    msg = "" if len(node_innov) == len(genome.nodes) else "(ERROR)"
    nodes, edges = genome.to_reduced_repr
    print(f' -------- nodes {msg} -------- ')
    for (a, b, i, w) in nodes:
        print((a, b), "innov:", i, "weight:", round(w, 2))
    msg = "" if len(edge_innov) == len(genome.edges) else "(ERROR, inconsistent innov nums)"
    truth = True
    for e_1, e_2 in zip(genome.edges, genome.edges[1:]):
        truth = truth and e_1.innov < e_2.innov
    msg = msg + "" if truth else msg + "(ERROR, wrong innov order)"
    print(f' -------- edges {msg} -------- ')
    for (_, _, i_1, _), (_, _, i_2, _), w, i in edges:
        print(i_1, "->", i_2, "innov:", i, "weight:", round(w, 2))
    truth = True
    for e in genome.edges:
        truth = truth and (e.from_node.innov, e.to_node.innov) in genome.edge_innovs
    msg = "" if truth else "(ERROR)"
    print(f' -------- edge_innovs {msg} -------- ')
    print('edge_innovs:', genome.edge_innovs)


def catch(genome):
    if not verify_genome_consistency(genome):
        print_genome(genome)
        raise ValueError('Inconsistent genome found')


def print_population(population):
    for k, v in population.species.items():
        print('')
        print(' --- ', k, ' --- ')
        print('representative: ', v['repr'])
        print('size: ', len(v['group']))
