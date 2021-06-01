import random


def sample_weight(low, high):
    weight_range = high - low
    return random.random() * weight_range + low


def get_random():
    return random.random() * 2 - 1


def equal_genome(g1, g2):
    same_num_edges = len(g1.edges) == len(g2.edges)
    same_num_nodes = len(g1.nodes) == len(g2.nodes)
    same_edges = True
    for g1_edge, g2_edge in zip(g1.edges, g2.edges):
        same_edges = same_edges and g1_edge.weight == g2_edge.weight
        same_edges = same_edges and g1_edge.from_node.layer_num == g2_edge.from_node.layer_num
        same_edges = same_edges and g1_edge.from_node.layer_ind == g2_edge.from_node.layer_ind
        same_edges = same_edges and g1_edge.to_node.layer_num == g2_edge.to_node.layer_num
        same_edges = same_edges and g1_edge.to_node.layer_ind == g2_edge.to_node.layer_ind
        same_edges = same_edges and g1_edge.innov == g2_edge.innov
    same_nodes = True
    for g1_node, g2_node in zip(g1.nodes, g2.nodes):
        same_nodes = same_nodes and g1_node.weight == g2_node.weight
        same_nodes = same_nodes and g1_node.layer_num == g2_node.layer_num
        same_nodes = same_nodes and g1_node.layer_num == g2_node.layer_num
        same_nodes = same_nodes and g1_node.innov == g2_node.innov
    return same_num_nodes and same_num_edges and same_nodes and same_edges


def print_genome(genome):
    edge_innov = set(e.innov for e in genome.edges)
    node_innov = set(n.innov for n in genome.nodes)
    print('')
    print(genome)
    msg = "" if len(node_innov) == len(genome.nodes) else "(ERROR)"
    nodes, edges = genome.to_reduced_repr
    print(f' -------- nodes {msg} -------- ')
    for (a, b, i, w, type) in nodes:
        print('type: ', type, (a, b), "innov:", i, "weight:", round(w, 2))
    msg = "" if len(edge_innov) == len(genome.edges) else "(ERROR, inconsistent innov nums)"
    truth = True
    for e_1, e_2 in zip(genome.edges, genome.edges[1:]):
        truth = truth and e_1.innov < e_2.innov
    msg = msg + "" if truth else msg + "(ERROR, wrong innov order)"
    print(f' -------- edges {msg} -------- ')
    for (_, _, i_1, _, _), (_, _, i_2, _, _), w, i, active in edges:
        print(i_1, "->", i_2, "innov:", i, "active:", active, "weight:", round(w, 2))
    truth = True
    for e in genome.edges:
        truth = truth and (e.from_node.innov, e.to_node.innov) in genome.edge_innovs
    msg = "" if truth else "(ERROR)"
    print(f' -------- edge_innovs {msg} -------- ')
    print('edge_innovs:', genome.edge_innovs)


def print_population(population):
    for k, v in population.species.items():
        print('')
        print(' --- species ', k, ' --- ')
        print('representative: ', v['repr'])
        print('size: ', len(v['group']))

