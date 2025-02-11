import copy
import pprint
import itertools
import hashlib
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt

def reencode_colors_to_strings(colors):
    """
    Re-encode the colors to be strings of colors usable in NetworkX.
    
    Parameters:
    colors (dict): A dictionary mapping each node to its color.
    
    Returns:
    dict: A dictionary mapping each node to its new string color.
    """
    color_palette = [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "cyan",
        "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#8A2BE2", "#FF4500", "#FF69B4", "#A52A2A", "#708090", "#00CED1"
    ]
    
    color_to_str = {}
    str_color_counter = 0
    
    for node, color in colors.items():
        color_str = str(color)  # Ensure the color is treated as a string
        if color_str not in color_to_str:
            color_to_str[color_str] = color_palette[str_color_counter % len(color_palette)]
            str_color_counter += 1
    
    str_colors = {node: [color_to_str[str(color)]] for node, color in colors.items()}
    
    return str_colors

def one_WL(G, verbose=False, initial_colors=None, fixed_vertices=None):
    # 1. 
    if initial_colors is None:
        colors = {n: ["0"] for n in G.nodes()}
    else:
        colors = initial_colors
    old_colors = copy.deepcopy(colors)
    # 2.
    print(colors) if verbose else None
    for i in range(len(G.nodes())):
        # 3.
        for n in G.nodes():
            # if n not in fixed_vertices:
                neigh_colors = [old_colors[nb][0] for nb in G.neighbors(n)]
                colors[n].extend(neigh_colors)
                print(n, neigh_colors) if verbose else None
                colors[n].sort()
        # print number of different colors
        # if i >=4:
        #     print(nx.to_dict_of_lists(G), G)
        if verbose:
            print(f'Iteration {i}: {len(set([item for sublist in colors.values() for item in sublist]))} different colors')
            print(colors)
        # 4. Update with the hash
        # colors = {c: [hashlib.sha224("".join(colors[c]).encode('utf-8')).hexdigest()] if c not in fixed_vertices else colors[c] for c in colors}
        colors = {c: [hashlib.sha224("".join(colors[c]).encode('utf-8')).hexdigest()] for c in colors}

        # is it stable?
        if list(Counter([item for sublist in colors.values() for item in sublist]).values()) == list(Counter([item for sublist in old_colors.values() for item in sublist]).values()) and i != 0:
            if verbose:
                print(f'Converged at iteration {i}!')
            break
        old_colors = copy.deepcopy(colors)
    colors = reencode_colors_to_strings(colors)
            
    canonical_form = sorted(Counter([item for sublist in colors.values() for item in sublist]).items())
    if verbose:
        print(colors)
        print(f'Canonical Form Found: \n {canonical_form} \n')
    return colors, canonical_form
    
    
def base_WL(G_, k, verbose, n_set, initial_colors_func, find_neighbors_func):    
    def is_cyclic_permutation(a, b):
        return any(b == a[i:] + a[:i] for i in range(len(a)))
    
    G, n_k_sets, n_k_reversed, V_k_all = n_set(G_)
    # print(n_k_reversed)
    # print(n_k_sets)
    # 1. 
    colors = initial_colors_func(G,n_k_sets)
    
    if verbose:
        print(colors) 
    
    for n_k in V_k_all:
        sorted_n_k = sorted(n_k)
        colors[tuple(n_k)] = colors[tuple(sorted_n_k)]
    # for n_k in n_k_sets:
    #     colors[n_k_reversed[n_k]] = colors[n_k]
    
    
    old_colors = copy.deepcopy(colors)
    # 2.
    for i in range(len(n_k_sets)):
        
        # 3.
        for n_k in n_k_sets:
            
            # either this
            c_n_k = dict()
            for k_i in range(k):
                c_n_k[k_i] = [old_colors[nb][0] for nb in find_neighbors_func(G, V_k_all, n_k, k_i)]
                # c_n_k[k_i].sort() # TODOD: this sort should be inside the for loop above
                c_n_k[k_i] = sorted(c_n_k[k_i])
                c_n_k[k_i] = ["".join(c_n_k[k_i])]
                
            for k_i in range(k):
                colors[n_k].extend(c_n_k[k_i])
            
            # or this
            
            # neigh_colors = [old_colors[nb][0] for nb in find_neighbors_func(G, V_k_all, n_k, k)]
            # colors[n_k].extend(neigh_colors)
            
            colors[n_k] = sorted(colors[n_k]) # TODOD: this sort should be inside the for loop above
        
        # for n_k in n_k_sets:
        #     colors[n_k_reversed[n_k]] = colors[n_k]
        for n_k in V_k_all:
            sorted_n_k = sorted(n_k)  # TODO: this is problematic for k-WL k>3
            colors[tuple(n_k)] = colors[tuple(sorted_n_k)]
            
            # cyclic_n_k = next((key for key in n_k_sets if is_cyclic_permutation(key, n_k)), None)
            # if cyclic_n_k is not None:
            #     colors[tuple(n_k)] = colors[cyclic_n_k]
            # else:
            #     print("problem")
            
            
        # print number of different colors
        if verbose:
            print(f'Iteration {i}: {len(set([item for sublist in colors.values() for item in sublist]))} different colors')
        
        # 4. Update with the hash
        colors = {c: [hashlib.sha224("".join(colors[c]).encode('utf-8')).hexdigest()] for c in colors}
        # print(colors)
        
        # check which pairs changed color this iteration
        changed_colors = set()
        # for q in find_neighbors_func(G, V_k_all, n_k, k):
        for p,q in itertools.combinations(n_k_sets,2):
            if old_colors[p] == old_colors[q] and colors[p] != colors[q]:
                # changed_colors.add((p, q, old_colors[p][0], old_colors[q][0], colors[p][0], colors[q][0]))
                changed_colors.add((p, q))
        if verbose:
            print(f"{i}-th iteration: ", changed_colors, len(changed_colors)) 
        
        
        # draw_graph(G, colors, i)
        # is it stable?
        if list(Counter([item for sublist in colors.values() for item in sublist]).values()) == list(Counter([item for sublist in old_colors.values() for item in sublist]).values()) and i != 0:
            if verbose:
                print(f'Converged at iteration {i}!')
            break
        
        old_colors = copy.deepcopy(colors)
        
    colors = reencode_colors_to_strings(colors)
        
    # canonical_form = sorted(Counter([item for sublist in colors.values() for item in sublist]).items())
    canonical_form = sorted(Counter([item for sublist in (colors[n_k] for n_k in n_k_sets) for item in sublist]).items())
    if verbose:
        # for node in n:
            # print(f'Node {node}, Neighbors: {list(find_neighbors_func(G, n, node))}')
        print(colors)
        print(f'Canonical Form Found: \n {canonical_form} \n')
        
    
    return colors, canonical_form

def kWL(G, k, verbose=False, initial_colors=None, fixed_vertices=None):
    def n_set(G):
        V = sorted(list(G.nodes()))
        V_k = [comb for comb in itertools.combinations(V, k)]
        V_k_reversed = {comb: tuple(reversed(comb)) for comb in V_k}
        V_k_all = [perm for perm in itertools.permutations(V, int(k))]
        return G, V_k, V_k_reversed, V_k_all
    def set_initial_colors(G, n):
        initial_colors = {i: [nx.weisfeiler_lehman_graph_hash(G.subgraph(i))] for i in n}
        print("initial colors: ", initial_colors, sep="\n") if verbose else None
        return initial_colors
    
    def find_neighbors(G, V_k, v_k, i):
        # find all nodes that are one edge away (so everything except ith position) from v_k in V_k,
        # print(v_k, i, sep=": ")
        # neighbors = []
        # for n in V_k:
        #     # if len(set(n) - set(v_k)) == 1 and (v_k[i] != n[i]): #or v_k[len(v_k)-i-1] != n[i]): 
        #     if  v_k[:i] == n[:i]  and v_k[i] != n[i] and (i+1 >= len(v_k) or v_k[i+1:] == n[i+1:]): #or v_k[len(v_k)-i-1] != n[i]): 
        #         # yield n
        #         neighbors.append(n)
        # print(neighbors)
        # return neighbors
        
        # either this
        # return [n for n in V_k if len(set(n) - set(V_k[V_k.index(v_k)])) == 1]
        # or this
        return [n for n in V_k if len(set(n) - set(V_k[V_k.index(v_k)])) == 1 and (v_k[i] != n[i])]
    
    return one_WL(G, verbose, initial_colors=initial_colors, fixed_vertices=fixed_vertices) if k == 1 else base_WL(G, k, verbose, n_set, set_initial_colors, find_neighbors)
    

    # return base_WL(G, k, verbose, n_set, set_initial_colors, find_neighbors)
    
if __name__ == "__main__":
    G = nx.Graph({0: [4], 1: [], 2: [], 3: [4], 4: [0, 3]})
    nx.draw(G, with_labels=True)
    plt.show()
    for i in range(1,len(G.nodes())):
        kWL(G, i, verbose=True)

    G = nx.Graph({0: [1,2], 1: [2,3]})
    nx.draw(G, with_labels=True)
    plt.show()
    for i in range(1,len(G.nodes())):
        kWL(G, i, verbose=True)

    G = nx.Graph({0: [1], 1: [2], 2: [3], 3: [0] })
    nx.draw(G, with_labels=True)
    plt.show()
    for i in range(1,len(G.nodes())):
        kWL(G, i, verbose=True)

    G = nx.Graph({0: [1], 1: [2], 2: [3]})
    nx.draw(G, with_labels=True)
    plt.show()
    for i in range(1,len(G.nodes())):
        kWL(G, i, verbose=True)

    G = nx.Graph({0: [1,2,3], 1: [2,3], 2: [3]})
    nx.draw(G, with_labels=True)
    plt.show()
    for i in range(1,len(G.nodes())):
        kWL(G, i, verbose=True)

    G = nx.Graph({0: [1,2,3], 1: [2,3]})
    nx.draw(G, with_labels=True)
    plt.show()
    for i in range(1,len(G.nodes())):
        kWL(G, i, verbose=True)

