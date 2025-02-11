import copy
import functools
import time
import random
import subprocess
from ast import literal_eval
import math
from matplotlib import pyplot as plt
import networkx as nx
from .graph import Graph
# from graph_generator import generate_random_graph
from .partial_symmetries import PartialSymmetries

# https://oeis.org/A002720
# number of permutations of , i-th index is the number of partial permutations of i-element set
n_partial_permutations = [1, 2, 7, 34, 209, 1546, 13327, 130922, 1441729, 17572114, 234662231, 3405357682, 53334454417, 896324308634,
              16083557845279, 306827170866106, 6199668952527617, 132240988644215842, 2968971263911288999,
              69974827707903049154, 1727194482044146637521, 44552237162692939114282]

# https://www.sciencedirect.com/science/article/pii/S0095895617300539
asymmetric_graphs = {'X1': nx.Graph({1: {2, 3}, 2: {1, 3, 4}, 3: {1, 2, 5}, 4: {2, 6}, 5: {3}, 6: {4}}),
                     'X8': nx.Graph({1: {2, 3}, 2: {1, 3, 4, 5}, 3: {1, 2, 4, 6}, 4: {2, 3, 6}, 5: {2, 6}, 6: {3, 4, 5}}),
                     'X2': nx.Graph({1: {2, 3}, 2: {1, 3, 4}, 3: {1, 2, 5}, 4: {2, 5, 6}, 5: {3, 4}, 6: {4}}),
                     'X7': nx.Graph({1: {2, 3}, 2: {1, 3, 4}, 3: {1, 2, 4, 5}, 4: {2, 3, 6}, 5: {3, 6}, 6: {4, 5}}),
                     'X3': nx.Graph({1: {2, 3}, 2: {1, 4}, 3: {1, 4, 5}, 4: {2, 3, 5, 6}, 5: {3, 4}, 6: {4}}),
                     'X6': nx.Graph({1: {2, 3}, 2: {1, 3, 4}, 3: {1, 2, 4, 5}, 4: {2, 3, 5}, 5: {3, 4, 6}, 6: {5}}),
                     'X4': nx.Graph({1: {2, 3}, 2: {1, 3, 4}, 3: {1, 2, 4, 5}, 4: {2, 3, 6}, 5: {3}, 6: {4}}),
                     'X5': nx.Graph({1: {2, 3}, 2: {1, 3, 4}, 3: {1, 2, 4, 5}, 4: {2, 3, 5, 6}, 5: {3, 4}, 6: {4}}),
                     'X9': nx.Graph({1: {2, 3}, 2: {1}, 3: {1, 4, 5}, 4: {3}, 5: {3, 6}, 6: {5, 7}, 7: {6}}),
                     'X14':nx.Graph({1: {2, 3, 4, 5, 6}, 2: {1, 3, 5, 7}, 3: {1, 2, 4}, 4: {1, 3, 5, 6, 7}, 5: {1, 2, 4, 6, 7}, 6: {1, 4, 5, 7}, 7: {2, 4, 5, 6}}),
                     'X10':nx.Graph({1: {2}, 2: {1, 3, 4}, 3: {2, 5}, 4: {2, 5, 6}, 5: {3, 4}, 6: {4, 7}, 7: {6}}),
                     'X13':nx.Graph({1: {2, 3, 6}, 2: {1, 3, 4, 7}, 3: {1, 2, 4, 6}, 4: {2, 3, 5, 6, 7}, 5: {4, 6, 7}, 6: {1, 3, 4, 5, 7}, 7: {2, 4, 5, 6}}),
                     'X11':nx.Graph({1: {2, 3}, 2: {1, 4}, 3: {1, 4, 5}, 4: {2, 3, 6}, 5: {3, 6, 7}, 6: {4, 5}, 7: {5}}),
                     'X12':nx.Graph({1: {2, 3, 6}, 2: {1, 3, 4, 7}, 3: {1, 2, 4}, 4: {2, 3, 5, 6, 7}, 5: {4, 6, 7}, 6: {1, 4, 5, 7}, 7: {2, 4, 5, 6}}),
                     'X15':nx.Graph({1: {2, 5}, 2: {1, 3}, 3: {2, 4, 6}, 4: {3, 5}, 5: {1, 4, 6, 7}, 6: {3, 5, 8}, 7: {5}, 8: {6}}),
                     'X18':nx.Graph({1: {2, 4, 5, 6, 7}, 2: {1, 3, 4, 5, 6}, 3: {2, 6, 8}, 4: {1, 2, 5, 7}, 5: {1, 2, 4, 6, 7, 8}, 6: {1, 2, 3, 5, 7, 8}, 7: {1, 4, 5, 6, 8}, 8: {3, 5, 6, 7}}),
                     'X16':nx.Graph({1: {2, 3, 6}, 2: {1, 4}, 3: {1, 5}, 4: {2, 5, 7}, 5: {3, 4, 6}, 6: {1, 5, 7, 8}, 7: {4, 6}, 8: {6}}),
                     'X17':nx.Graph({1: {2, 4, 5, 6, 7}, 2: {1, 3, 6}, 3: {2, 5, 6, 8}, 4: {1, 5, 7, 8}, 5: {1, 3, 4, 6, 7, 8}, 6: {1, 2, 3, 5, 7}, 7: {1, 4, 5, 6, 8}, 8: {3, 4, 5, 7}}),
                     None: nx.Graph({})}

for key in asymmetric_graphs.keys():
    asymmetric_graphs[key] = Graph(asymmetric_graphs[key])



def read_graph_from_file():
    graphs = []
    with open('symmetries_refactor/tests/inputs.txt', 'r') as file:
        file = file.read().split('\n')[:-1]
        d = dict()
        for row in file:
            if row == '':
                graphs.append(Graph(nx.from_dict_of_lists(d)))
                d = dict()
                continue
            key, values = row.split('->')
            values = set(map(int, filter(lambda x: len(x), values.split(' '))))
            d[int(key)] = values
    return graphs


def partial_symmetries_info(vertices, include_complete=True):
    with open('symmetries_refactor/all_graphs/data/all_graphs_' + str(vertices) + 'n.txt') as file:
            file = file.read().split('\n')
            file = file[4:-5] if not include_complete else file[:-1]
            info = dict()
            for i in range(0, len(file), 4):
                data = file[i].strip().encode()
                g = Graph(nx.from_graph6_bytes(data))
                p = PartialSymmetries(g, use_timer=False)
                edges = int(file[i + 1].split(':')[1].strip())
                p.total_partial_symmetries = int(file[i + 2].split(':')[1].strip())
                p.total_induced_subgraphs = int(file[i + 3].split(':')[1].strip())
                info.setdefault(edges, []).append(p)
    return info

def partial_symmetries_average(n):
    total = 0
    graph_count = 0
    for val in partial_symmetries_info(n).values():
        graph_count += len(val)
        for graph in val:
            total += graph.total_partial_symmetries
    print(n, round(total / graph_count, 2), round((total / graph_count) / n_partial_permutations[n], 5))
    return round(total / graph_count, 2), round((total / graph_count) / n_partial_permutations[n], 5)

def run_tests(runs=3, clear_output_file=False):
    """
    function that runs tests for all graphs entered in 'tests/inputs.txt' file,
    output is appended to 'tests/output.txt' file
    Parameters
    ----------
    runs - how many times should partial symmetries be calculated for each graph
    clear_output_file - if contents of output file should first be deleted
    """
    if clear_output_file:
        with open('symmetries_refactor/tests/output.txt', 'w') as file:
            file.write('')
    for graph in read_graph_from_file():
        runtime = []
        p = PartialSymmetries(graph, use_timer=False, only_full_symmetries=False)
        for i in range(runs):
            for _ in p.get_partial_symmetries():
                pass
            runtime.append(p.runtime)
            if i != runs - 1:
                p.clear()
        with open('symmetries_refactor/tests/output.txt', 'a') as file:
            file.write(nx.readwrite.to_graph6_bytes(graph.to_networkx()).decode())
            for t in runtime:
                file.write(str(t) + '\n')
            file.write(str(p.total_partial_symmetries) + '\n')
            file.write(str(p.total_induced_subgraphs) + '\n')


def run_asymmetric():
    for i in range(9):
        print(list(asymmetric_graphs.keys())[i])
        graph = asymmetric_graphs[list(asymmetric_graphs.keys())[i]]
        for j in range(2):
            graph.checks = 0
            p = PartialSymmetries(graph)
            for _ in p.get_partial_symmetries():
                pass
            print(p.runtime)
            print(p.total_partial_symmetries)
            print(p.total_induced_subgraphs)
            print(p.graph.checks)
            print()


def all_partial_symmetries(n, include_complement=False):
    def write_graph_data(file, g, p):
        file.write(nx.readwrite.to_graph6_bytes(g.to_networkx()).decode())
        file.write(f'Edges: {len(g.edges())}\n')
        file.write(f'Partial symmetries: {int(p.get_number_of_partial_symmetries())}\n')
        file.write(f'Non-isomorphic induced subgraphs: {p.total_induced_subgraphs}\n')

    found = False
    for g in read_gap_file(n):
        if not found:
            if g.data == {1: {6, 8, 9}, 2: {7, 9, 10}, 3: {7}, 4: {8, 9, 10}, 5: set(), 6: {1, 10}, 7: {2, 3, 8, 10}, 8: {1, 4, 7, 9, 10}, 9: {1, 2, 4, 8}, 10: {2, 4, 6, 7, 8}}:
                found = True
            continue
        with open('symmetries_refactor/data/all_graphs_' + str(n) + 'n.txt', 'a') as file:
            if not include_complement:
                if len(g.edges()) <= (n * (n - 1) / 2) // 2 + 1: 
                    write_graph_data(file, g, PartialSymmetries(g, use_timer=False))
            else:
                write_graph_data(file, g, PartialSymmetries(g, use_timer=False))

def single_edge_partial_symmetries(n):
    """
    calculates the number of partial symmetries for a graphs with 'n' vertices and 1 edge, all other vertices are
    isolated, this is the second highest number of partial symmetries for any 'n' vertex graph after K_n
    Parameters
    ----------
    n - number of vertices of the graph
    Returns
    -------
    total - number of partial symmetries for graph with 'n' vertices and 1 edge
    """
    total = 0
    for i in range(n + 1):
        if i == 0:
            total += 1
        elif i == 1:
            total += n ** 2
        elif i == n:
            total += math.factorial(n - 2) * 2
        else:
            subgraphs = math.comb(n - 2, i - 2)
            total += pow(subgraphs, 2) * (math.factorial(i - 2) * 2)
            subgraphs = 2 * math.comb(n - 2, i - 1) + math.comb(n - 2, i)
            total += pow(subgraphs, 2) * math.factorial(i)
    return total


def n_partial_perm_n_set(n):
    """
    counts the number of partial permutations of an 'n' element set, https://oeis.org/A002720
    Parameters
    ----------
    n - number of elements in a set
    Returns
    -------
    total - number of partial permutations of an 'n'-set
    """
    # total = 0
    # for i in range(n + 1):
    #     total += math.factorial(i) * math.comb(n, i) ** 2
    return sum(math.factorial(k)*math.comb(n, k)**2 for k in range(n+1))


def read_gap_file(n):
    """
    yields all graphs with 'n' vertices, without calculating their symmetries, only parses data from GAP files
    """
    data = dict()
    # filename = 'all_graphs/gap/graph' + str(n) + '.g6'
    # popen = subprocess.Popen(['nauty-showg', filename], stdout=subprocess.PIPE, universal_newlines=True)
    # for std_out in iter(popen.stdout.readline, ''):
    #     std_out = std_out.strip()
    #     if 'Graph' in std_out:
    #         if len(data):
    #             yield Graph(data)
    #             data = dict()
    #     elif std_out != '':
    #         key, value = std_out[:-1].split(':')
    #         key = int(key.strip()) + 1
    #         value = value.strip().split(' ')
    #         if value[0] != '':
    #             value = set(map(lambda x: x + 1, map(int, value)))
    #         else:
    #             value = set()
    #         data[key] = value
    # yield Graph(data, aut_group=set())
    yield from nx.read_graph6(f'all_graphs/graphs{n}.g6') 


if __name__ == '__main__':
    # run_tests(clear_output_file=True)
    # subgraphs_runtime_test()
    # nx.draw(asymmetric_graphs['X1'])
    # plt.show()
    # # nx.draw(read_graph_from_file()[0])
    # # plt.show()
    
    # nx.draw(read_gap_file(5).__next__())
    # plt.show()
    
    nx.draw(read_graph_from_file()[0])
    plt.show()