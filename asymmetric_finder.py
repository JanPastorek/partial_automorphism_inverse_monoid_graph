import random

from typing import Dict

import copy

import itertools
from ast import literal_eval

import networkx as nx
from sage.combinat.combination import Combinations
# from utils import read_gap_file
from typing import Dict
import copy
from symmetries_refactor import Graph
from ast import literal_eval
import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt
import multiprocessing
import sqlite3
import logging

from sage.parallel.decorate import Parallel, parallel, fork, Fork


# https://www.sciencedirect.com/science/article/pii/S0095895617300539
MINIMAL_ASYMMETRIC_GRAPHS =  {'X1': Graph(nx.Graph({1:  [2, 3], 2:  [1, 3, 4], 3:  [1, 2, 5], 4:  [2, 6], 5:  [3], 6:  [4]})),
                     'X8': Graph(nx.Graph( {1:  [2, 3], 2:  [1, 3, 4, 5], 3:  [1, 2, 4, 6], 4:  [2, 3, 6], 5:  [2, 6], 6:  [3, 4, 5]})),
                     'X2': Graph(nx.Graph( {1:  [2, 3], 2:  [1, 3, 4], 3:  [1, 2, 5], 4:  [2, 5, 6], 5:  [3, 4], 6:  [4]})),
                     'X7': Graph(nx.Graph( {1:  [2, 3], 2:  [1, 3, 4], 3:  [1, 2, 4, 5], 4:  [2, 3, 6], 5:  [3, 6], 6:  [4, 5]})),
                     'X3': Graph(nx.Graph( {1:  [2, 3], 2:  [1, 4], 3:  [1, 4, 5], 4:  [2, 3, 5, 6], 5:  [3, 4], 6:  [4]})),
                     'X6': Graph(nx.Graph( {1:  [2, 3], 2:  [1, 3, 4], 3:  [1, 2, 4, 5], 4:  [2, 3, 5], 5:  [3, 4, 6], 6:  [5]})),
                     'X4': Graph(nx.Graph( {1:  [2, 3], 2:  [1, 3, 4], 3:  [1, 2, 4, 5], 4:  [2, 3, 6], 5:  [3], 6:  [4]})),
                     'X5': Graph(nx.Graph( {1:  [2, 3], 2:  [1, 3, 4], 3:  [1, 2, 4, 5], 4:  [2, 3, 5, 6], 5:  [3, 4], 6:  [4]})),
                     'X9': Graph(nx.Graph( {1:  [2, 3], 2:  [1], 3:  [1, 4, 5], 4:  [3], 5:  [3, 6], 6:  [5, 7], 7:  [6]})),
                     'X14':Graph(nx. Graph( {1:  [2, 3, 4, 5, 6], 2:  [1, 3, 5, 7], 3:  [1, 2, 4], 4:  [1, 3, 5, 6, 7], 5:  [1, 2, 4, 6, 7], 6:  [1, 4, 5, 7], 7:  [2, 4, 5, 6]})),
                     'X10':Graph(nx. Graph( {1:  [2], 2:  [1, 3, 4], 3:  [2, 5], 4:  [2, 5, 6], 5:  [3, 4], 6:  [4, 7], 7:  [6]})),
                     'X13':Graph(nx. Graph( {1:  [2, 3, 6], 2:  [1, 3, 4, 7], 3:  [1, 2, 4, 6], 4:  [2, 3, 5, 6, 7], 5:  [4, 6, 7], 6:  [1, 3, 4, 5, 7], 7:  [2, 4, 5, 6]})),
                     'X11':Graph(nx. Graph( {1:  [2, 3], 2:  [1, 4], 3:  [1, 4, 5], 4:  [2, 3, 6], 5:  [3, 6, 7], 6:  [4, 5], 7:  [5]})),
                     'X12':Graph(nx. Graph( {1:  [2, 3, 6], 2:  [1, 3, 4, 7], 3:  [1, 2, 4], 4:  [2, 3, 5, 6, 7], 5:  [4, 6, 7], 6:  [1, 4, 5, 7], 7:  [2, 4, 5, 6]})),
                     'X15':Graph(nx. Graph( {1:  [2, 5], 2:  [1, 3], 3:  [2, 4, 6], 4:  [3, 5], 5:  [1, 4, 6, 7], 6:  [3, 5, 8], 7:  [5], 8:  [6]})),
                     'X18':Graph(nx. Graph( {1:  [2, 4, 5, 6, 7], 2:  [1, 3, 4, 5, 6], 3:  [2, 6, 8], 4:  [1, 2, 5, 7], 5:  [1, 2, 4, 6, 7, 8], 6:  [1, 2, 3, 5, 7, 8], 7:  [1, 4, 5, 6, 8], 8:  [3, 5, 6, 7]})),
                     'X16':Graph(nx. Graph( {1:  [2, 3, 6], 2:  [1, 4], 3:  [1, 5], 4:  [2, 5, 7], 5:  [3, 4, 6], 6:  [1, 5, 7, 8], 7:  [4, 6], 8:  [6]})),
                     'X17':Graph(nx. Graph( {1:  [2, 4, 5, 6, 7], 2:  [1, 3, 6], 3:  [2, 5, 6, 8], 4:  [1, 5, 7, 8], 5:  [1, 3, 4, 6, 7, 8], 6:  [1, 2, 3, 5, 7], 7:  [1, 4, 5, 6, 8], 8:  [3, 4, 5, 7]})),
                     }


# ASYM_G_DB_PATH = 'C:\local_repos\symmetries_final\symmetries_refactor/all_graphs/asymmetric/asym_graphs.db'
# ASYM_G_DB_PATH = '/home/sage/repository/partial_symmetries/symmetries_refactor/all_graphs/asymmetric/asym_graphs.db'
ASYM_G_DB_PATH = 'projects/partial_symmetries/partial_symmetries/symmetries_refactor/all_graphs/asymmetric/asym_graphs.db'
ASYM_G_DB_PATH = '/home/p/pastorek20/projects/partial_symmetries/partial_symmetries/symmetries_refactor/all_graphs/asymmetric/asym_graphs.db'


@parallel()
def induced_asymmetric(g, size, verbose = False):
    """
    generates all induced subgraphs with 'size' vertices
    Parameters
    ----------
    g - instance of 'Graph'
    size - number of vertices of a subgraph

    Returns
    -------
    are_asymmetric; subgraphs - True if all induced subgraphs with 'size' vertices are asymmetric; a dictionary of all
    induced subgraphs with 'size' vertices, filtered by their degree sequences, this only matters when 'are_asymmetric' is True
    """
    subgraphs = dict() 
    for vertices in itertools.combinations(g.vertices(), size):
    # for vertices in itertools.combinations(g.vs, size):
        # subgraph = Graph(nx.subgraph(g, vertices))
        # print(size)
        # subgraph = Graph(g.subgraph(vertices))
        subgraph = g.subgraph(vertices) 
        if not subgraph.automorphism_group(return_group=False, order=True) == 1:
            print(list(subgraph.automorphism_group())) if verbose else None
            print(set(g.vertices()) - set(vertices)) if verbose else None
            return False, dict()
        
        subgraphs.setdefault(subgraph.degree_sequence(), []).append(subgraph)
    return True, subgraphs

@parallel()
def asymmetric_non_isomorphic(graphs, verbose = False):
    """
    returns True if all asymmetric graphs in the list 'graphs' are pairwise non-isomorphic
    """
    for key in graphs:
        for a, b in itertools.combinations(graphs[key], 2):
            if a.is_isomorphic(b):
                print(nx.vf2pp_isomorphism(a.networkx_graph(), b.networkx_graph())) if verbose else None
                return False
    return True

@parallel()
def has_asymmetricity_level(g, depth):
    """
    returns True if graph 'g' has asymmetricity level 'depth'
    """
    for i in reversed(range(depth + 1)):
        asym, subgraphs = induced_asymmetric(g, len(g) - i)
        if not asym or not asymmetric_non_isomorphic(subgraphs):
            return False
    return True

@parallel()
def find_asym_d_(g,return_asymmetric_subgraphs = False, verbose = False):
    """
    returns the asymmetric depth for graph 'g'
    """
    depth = -1
    while True:
        print(len(g.vertices()) - (depth + 1), depth) if verbose else None
        asym, subgraphs = induced_asymmetric(g, len(g.vertices()) - (depth + 1), verbose=verbose)
        if not asym:
            if not return_asymmetric_subgraphs:
                return (depth, dict()) 
            else:
                return (depth, subgraphs) 
        depth += 1

@parallel()
def find_asym_d(g,return_asymmetric_subgraphs = False, verbose = False):
    """
    returns the asymmetric depth for graph 'g'
    """
    depth = -1
    while True:
        print(len(g.vertices()) - (depth + 1), depth) if verbose else None
        asym, subgraphs = induced_asymmetric(g, len(g.vertices()) - (depth + 1), verbose=verbose)
        if not asym or not asymmetric_non_isomorphic(subgraphs, verbose=verbose):
            if not return_asymmetric_subgraphs:
                return (depth, dict()) 
            else:
                return (depth, subgraphs) 
        depth += 1
        
        

@parallel()
def find_asym_a(g,return_asymmetric_subgraphs = False, verbose = False):
    """
    returns the asymmetric depth for graph 'g'
    """
    depth = -1
    while True:
        print(len(g.edges()) - (depth + 1), depth) if verbose else None
        asym, subgraphs = induced_asymmetric_a(g, len(g.edges()) - (depth + 1), verbose=verbose)
        if not asym or not asymmetric_non_isomorphic(subgraphs):
            return depth if not return_asymmetric_subgraphs else depth, subgraphs 
        depth += 1

def induced_asymmetric_a(g, size, verbose=False):
    subgraphs = dict() 
    for es in Combinations(g.edges(labels=False), size):
        subgraph = Graph()
        subgraph.add_vertices(g.nodes())
        subgraph.add_edges(es)
        if not subgraph.is_asymmetric():
            subgraph.show() if verbose else None
            print(list(subgraph.automorphism_group())) if verbose else None
            print(set(g.edges(labels=False)) - set(es)) if verbose else None
            return False, dict()
        subgraphs.setdefault(subgraph.degree_sequence(), []).append(subgraph)
    return True, subgraphs



def asym_depth_for_n_vertices(n):
    """
    returns the asymmetric depth any graph with 'n' vertices can have
    Parameters
    ----------
    n - number of vertices of a graph

    Returns
    -------
    cur_max; graphs_with_max_depth - asymmetric depth; a list of graphs with asymmetric depth 'cur_max'
    """
    cur_max = 0
    graphs_with_max_depth = []
    for g in read_gap_file(n):
        max_depth = find_asym_d(g)
        if max_depth == cur_max:
            graphs_with_max_depth.append(g)
        elif max_depth > cur_max:
            graphs_with_max_depth = [g]
            cur_max = max_depth
    return cur_max, graphs_with_max_depth


def add_vertex_to_asymmetric(add_to_vertices, from_depth, to_depth):
    with open('symmetries_refactor/all_graphs/asymmetric/asym_d_' + str(from_depth) + '_' + str(add_to_vertices) + 'v.txt') as file:
        file = file.read().split('\n')[:-1]
        for g in file:
            data = literal_eval(g)
            new_vertex = len(data) + 1
            for no_neighbours in range(len(data) + 1):
                for combs in itertools.combinations(data.keys(), no_neighbours):
                    data_copy: Dict = copy.deepcopy(data)
                    if len(combs) == 0:
                        data_copy[new_vertex] = set()
                    for vertex in combs:
                        data_copy.setdefault(new_vertex, set()).add(vertex)
                        data_copy[vertex].add(new_vertex)
                    # if has_asymmetricity_level(Graph(nx.from_dict_of_lists(data_copy)), to_depth):
                    if has_asymmetricity_level(Graph(nx.from_dict_of_lists(data_copy)), to_depth):
                        with open('symmetries_refactor/all_graphs/asymmetric/asym_d_' + str(to_depth) + '_' + str(add_to_vertices + 1) + 'v.txt', 'a') as output:
                            print(str(to_depth) + '_' + str(add_to_vertices + 1))
                            output.write(str(data_copy) + '\n')


def find_asym_d_random(n, only_save_depth_gte=4, lock=None):
    """
    finds the maximal asymmetric depth for randomly generated graphs,
    only saves graphs with maximal asymmetric depth >= 'only_save_depth_gte'
    """
    while True:
        # print(n)
        # g = Graph(nx.gnm_random_graph(n, random.randint(2, (n * (n - 1)) // 2)))
        g = Graph(nx.gnm_random_graph(n, random.randint(2, (n * (n - 1)) // 2)))
        # print(len(g.edges))
        asym_d_max = find_asym_d(g)
        if asym_d_max >= only_save_depth_gte:
            if lock: lock.acquire()  # Acquire the lock
            try:
                with open('symmetries_refactor/all_graphs/asymmetric/random_maximal_asym_d.txt', 'a') as file:
                    g_repr = str(nx.to_dict_of_lists(g))
                    print(str(asym_d_max) + ' -> ' + g_repr)
                    file.write(str(asym_d_max) + ' -> ' + g_repr + '\n')
            finally:
                if lock: lock.release()  # Release the lock
                
import multiprocessing
    
def run_parallel_find_asym_d_random(n,  num_processes):
        # Use a manager to create a shared lock
        with multiprocessing.Manager() as manager:
            lock = manager.Lock()

            # Create a pool of worker processes
            with multiprocessing.Pool(num_processes) as pool:
                # Run the function in parallel
                tasks = [(m, 4, lock) for _ in range(num_processes) for m in n]
                pool.starmap(find_asym_d_random, tasks)


import sqlite3

def create_asym_db():
    # Connect to the SQLite database
    conn = sqlite3.connect(ASYM_G_DB_PATH)

    # Create a cursor object
    c = conn.cursor()

    # Execute an SQL statement to create a table
    c.execute('''
        CREATE TABLE graphs (
            id INTEGER PRIMARY KEY,
            number_of_vertices INTEGER,
            asymmetric_depth INTEGER,
            degree_sequence TEXT,
            graph_representation TEXT
        )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def upsert_graph(asymmetric_depth, G):
    number_of_vertices = len(G.nodes())
    degree_sequence = G.degree_sequence()
    if type(G) == Graph:
        G = G.to_networkx()
    
    # Convert the degree sequence to a string
    degree_sequence_str = str(degree_sequence)
    # print(degree_sequence_str)

    
    # Connect to the SQLite database
    conn = sqlite3.connect(ASYM_G_DB_PATH)

    # Create a cursor object
    c = conn.cursor()

    # Retrieve all graphs with the same degree sequence
    c.execute('''
        SELECT graph_representation
        FROM graphs
        WHERE degree_sequence = ?
    ''', (degree_sequence_str,))

    # Fetch all rows
    rows = c.fetchall()

    isomorphic = False
    # Check for isomorphism
    for row in rows:
        # print(row[0])
        H = nx.from_dict_of_lists(literal_eval(row[0]))
        # H = Graph(nx.from_dict_of_lists(literal_eval(row[0])))
        if nx.is_isomorphic(G, H):
        # if G.isomorphic(H):
            # The graph is already in the database, so return without upserting
            conn.close()
            isomorphic = True
            break
    
    if not isomorphic:
        print(number_of_vertices, asymmetric_depth)
        # Execute an SQL statement to upsert a graph
        c.execute('''
            INSERT INTO graphs (number_of_vertices, asymmetric_depth, degree_sequence, graph_representation)
            VALUES (?, ?, ?, ?)
        ''', (number_of_vertices, asymmetric_depth, degree_sequence_str, str(nx.to_dict_of_lists(G))))
        # Commit the changes and close the connection
        conn.commit()
        conn.close()


def update_graph(G, new_asymmetric_depth):
    # Connect to the SQLite database
    conn = sqlite3.connect(ASYM_G_DB_PATH)
    c = conn.cursor()
    
    canonical_label = G.canonical_label().graph6_string()

    # Execute an SQL statement to update the graph
    c.execute('''
        UPDATE graphs
        SET asymmetric_depth = ?
        WHERE canonical_label = ?
    ''', (new_asymmetric_depth, canonical_label))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    
def update_graph_regularity_girth(G, regular, girth):
    # Connect to the SQLite database
    conn = sqlite3.connect(ASYM_G_DB_PATH)
    c = conn.cursor()

    canonical_label = G.canonical_label().graph6_string()
    # Execute an SQL statement to update the graph
    c.execute('''
        UPDATE graphs
        SET regular = ?, girth = ?
        WHERE canonical_label = ?
    ''', (regular, girth, canonical_label))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def get_graphs_n_d(number_of_vertices, asymmetric_depth):
    # Connect to the SQLite database
    conn = sqlite3.connect(ASYM_G_DB_PATH)

    # Create a cursor object
    c = conn.cursor()

    # Execute an SQL statement to retrieve graphs
    c.execute('''
        SELECT graph_representation
        FROM graphs
        WHERE number_of_vertices = ? AND asymmetric_depth = ?
    ''', (int(number_of_vertices), int(asymmetric_depth)))

    # Fetch all rows
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Return the graph representations
    return [Graph(nx.from_dict_of_lists(literal_eval(row[0]))) for row in rows]

def get_graphs_d(asymmetric_depth):
    # Connect to the SQLite database
    print(ASYM_G_DB_PATH)
    conn = sqlite3.connect(ASYM_G_DB_PATH)

    # Create a cursor object
    c = conn.cursor()

    # Execute an SQL statement to retrieve graphs
    c.execute('''
        SELECT graph_representation
        FROM graphs
        WHERE asymmetric_depth = ?
    ''', (asymmetric_depth,))

    # Fetch all rows
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Return the graph representations
    return [Graph(nx.from_dict_of_lists(literal_eval(row[0]))) for row in rows]

# # Shared set to store the IDs of the processed graphs
# manager = multiprocessing.Manager()
# processed_graphs = manager.dict()

# Function to get the graph IDs from the database
def get_graph_ids_from_d(asymmetric_depth):
    # Connect to the SQLite database
    conn = sqlite3.connect(ASYM_G_DB_PATH)
    c = conn.cursor()

    # Execute an SQL statement to get the graph IDs
    c.execute('SELECT id FROM graphs WHERE asymmetric_depth = ?', (asymmetric_depth,))

    # Fetch all the graph IDs and put them in the queue
    graph_ids = [row[0] for row in c.fetchall()]

    # Close the connection
    conn.close()
    return graph_ids

def get_graph_by_id(graph_id):
    # Connect to the SQLite database
    conn = sqlite3.connect(ASYM_G_DB_PATH)
    c = conn.cursor()

    # Execute an SQL statement to get the graph
    c.execute('SELECT graph_representation FROM graphs WHERE id = ?', (graph_id,))

    # Fetch the graph
    graph = c.fetchall()

    # Close the connection
    conn.close()
    
    G = Graph(nx.from_dict_of_lists(literal_eval(graph[0][0])))
    return G


def deeper_around_middle(G, asymmetric_depth):
    new_node_1 = G.number_of_nodes() + 1
    G.add_node(new_node_1)	
    
    # join this node to all other nodes
    for node in G.nodes:
        if node != new_node_1 and random.random() > 0.5:
            G.add_edge(new_node_1, node)
    
    print(G.number_of_nodes())
    d_union = find_asym_d(G)
    print(d_union, asymmetric_depth)
    if d_union >= asymmetric_depth:
        print(d_union, asymmetric_depth)
        degree_sequence = G.degree_sequence()
        upsert_graph(len(G.nodes()), d_union, degree_sequence, G)
        
def vertex_induced(G, asymmetric_depth):
    for n in G.nodes():
        G_copy = copy.deepcopy(G)
        # create n-vertex induced subgraph
        G_copy.remove_node(n)
        if len(G_copy.nodes()) == len(G.nodes()):
            print("error")
        d_G_v_i = find_asym_d(G_copy)
        if d_G_v_i >= asymmetric_depth:
            print(d_G_v_i, asymmetric_depth)
            degree_sequence = G.degree_sequence()
            upsert_graph(len(G.nodes()), d_G_v_i, degree_sequence, G)
    


# Worker function to process a graph
def process_graph(graph_id, asymmetric_depth, fun=deeper_around_middle):
    try:
        logging.info(f"Processing graph {graph_id}")
        # Check if the graph has already been processed
        
        # if graph_id in processed_graphs:
        #     return

        print(graph_id)
        # Connect to the SQLite database
        conn = sqlite3.connect(ASYM_G_DB_PATH, check_same_thread=False)
        c = conn.cursor()
        # Execute an SQL statement to get the graph
        c.execute('SELECT * FROM graphs WHERE id = ?', (graph_id,))
        # Close the connection
        graph = c.fetchall()
        conn.close()
        # Fetch the graph
        
        
        graph = graph[0][4]
        
        print(graph)
        
        G = Graph(nx.from_dict_of_lists(literal_eval(graph)))

        # Process the graph using the two_new_nodes function
        fun(G, asymmetric_depth)

        # Add the graph ID to the set of processed graphs
        # processed_graphs[graph_id] = True
    except Exception as e:
        logging.error(f"Error processing graph {graph_id}: {e}")
    

if __name__ == '__main__':
    # depth, graphs = max_depth_for_n_vertices(9)
    # print(depth, len(graphs))
    # add_vertex_to_asymmetric(11, 1, 2)
    # create_asym_db()
    # Get the graph IDs from the database
    
    # deeper around middle
    # asymmetric_depth = 6

    # graph_ids = get_graph_ids_from_d(asymmetric_depth=asymmetric_depth)

    # print(graph_ids)

    # # Configure the logging module to write logs to a file
    # logging.basicConfig(filename='process_graph.log', level=logging.INFO, format='%(asctime)s %(message)s')


    # # with multiprocessing.Manager() as manager:
    # # Create a pool of worker processes
    # with multiprocessing.Pool(processes=3) as pool:
    #     # Run the function in parallel
    #     # result = pool.starmap(process_graph, [(graph_id, asymmetric_depth) for graph_id in graph_ids])
    #     # pool.close() 
    #     # pool.join()
    #             # Run the function in parallel
    #     results = [pool.apply_async(process_graph, (graph_id, asymmetric_depth)) for graph_id in graph_ids]
    #     # Get the results when they are ready
    #     for result in results:
    #         result.get()
    
    # ------------------------------------        
    # deeper from deg. sequence
    # asymmetric_depth = 5

    # graph_ids = get_graph_ids_from_d(asymmetric_depth=asymmetric_depth)

    # print(graph_ids)

    # # Configure the logging module to write logs to a file
    # logging.basicConfig(filename='process_graph.log', level=logging.INFO, format='%(asctime)s %(message)s')


    # # with multiprocessing.Manager() as manager:
    # # Create a pool of worker processes
    # with multiprocessing.Pool(processes=3) as pool:
    #     # Run the function in parallel
    #     # result = pool.starmap(process_graph, [(graph_id, asymmetric_depth) for graph_id in graph_ids])
    #     # pool.close() 
    #     # pool.join()
    #             # Run the function in parallel
    #     results = [pool.apply_async(process_graph, (graph_id, asymmetric_depth,vertex_induced)) for graph_id in graph_ids]
    #     # Get the results when they are ready
    #     for result in results:
    #         result.get()

    # ------------------------------------
    
    
    # print(get_graph_by_id(5536))
    
    # graphs = get_graphs_d(asymmetric_depth=5)
    # for g in graphs:
    #     print(g.is_asymmetric())
    #     print(g.number_of_nodes())
    #     print(g.number_of_edges())
    #     print(g.degree_sequence())
    #     print('------------------')
    
    
    @parallel
    def f(n): return n*n
    sorted(list(f([i for i in range(100000)])))


