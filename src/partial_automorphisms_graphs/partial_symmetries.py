import json
import math
import multiprocessing
import time
import networkx as nx
import itertools
from .graph import Graph
import networkx as nx
import itertools
from typing import Generator, Dict, List
from .graph import node_link_data
from .partial_permutation import PartialPermutation
from .isomorphism_class import check_if_isomorphic, IsomorphismClass
import matplotlib.pyplot as plt

from networkx.readwrite import json_graph

    # This function should replace the `find_isomorphism_classes` method from the Graph class
    # and should return isomorphism classes of k-vertex induced subgraphs.


class PartialSymmetries:

    def __init__(self, G: Graph, only_full_symmetries=False, use_timer=False, timeout_after=60):
        self.graph = G
        self.chunk = 0
        self.only_full_symmetries = only_full_symmetries
        self.start_size = 0 if not only_full_symmetries else len(G)
        self.total_partial_symmetries = 0
        self.total_induced_subgraphs = 0
        self.runtime = 0
        self.start = 0
        self.use_timer = use_timer
        self.timer_process = multiprocessing.Process(target=timer, args=(timeout_after,))
        if self.use_timer:
            self.timer_process.start()
            
    def find_isomorphism_classes(self, size):
        induced_subgraphs = {}
        for comb in itertools.combinations(self.graph.nodes(), size):
            subgraph = Graph(self.graph.subgraph(comb))
            # Use the appropriate sequence for identification
            seq = subgraph.degree_sequence()

            if seq not in induced_subgraphs.keys():
                # Create a new IsomorphismClass with the subgraph
                induced_subgraphs[seq] = {IsomorphismClass(subgraph)}
            elif len(comb) <=5:
                (ic,) = induced_subgraphs[seq]
                ic.add_isomorphic(subgraph)
            else:
                # The sequence exists, so we must check for isomorphism
                isomorphic, ic, checks = check_if_isomorphic(subgraph, induced_subgraphs[seq])
                # self.checks += checks
                if isomorphic:
                    # new member of existing isomorphism class
                    ic.add_isomorphic(subgraph)
                else:
                    # add new isomorphism class since it does not belong to any
                    induced_subgraphs[seq].add(IsomorphismClass(subgraph)) 
        
        # Yield the isomorphism classes, sorted by the number of edges in their representative graphs
        # for isomorphism_class in sorted(*induced_subgraphs.values(), key=lambda ic: len(ic.rep.edges())):
        #     yield isomorphism_class
        yield from list(sorted(set().union(*(induced_subgraphs.values())), key=lambda ic: len(ic.rep.edges())))
            
    def get_data_for_k_vertex_subgraphs(self, k: int, json_output=True, string=False, structured=True, counting=False) -> Generator[str, None, None]:
        isomorphism_classes = self.find_isomorphism_classes(k)
        
        for iso_class in isomorphism_classes:
            # Instantiate an IsomorphismClass object with the representative graph
            # Now call the create_d_class method to create the D-class diagram
            iso_class.create_d_class(debug=False, string=string, counting=counting)
            self.total_induced_subgraphs += 1
            
            size = iso_class.size()
            self.total_partial_symmetries += size
            # Serialize the output as JSON, including the D-class diagram and other data
            # G = json_graph.node_link_data(iso_class.rep.to_networkx())
            # print(G)
            if counting:
                yield iso_class.count
                continue
            
            if structured:
                result = {
                        'chunk': self.chunk,  # Placeholder for chunk value
                        'result': iso_class.d_class,
                        'vertices': k,
                        # 'edges': nx.number_of_edges(iso_class.rep),  # Calculate the number of edges in the representative subgraph
                        'edges': len(iso_class.rep.edges()),  # Calculate the number of edges in the representative subgraph
                        'size': size,  # This should now call the size method from the IsomorphismClass
                        # 'graph': json_graph.node_link_data(iso_class.rep)  # Serialize the representative subgraph
                        'graph': node_link_data(iso_class.rep)  # Serialize the representative subgraph
                    }
            else:
                result = iso_class.pAs
            yield json.dumps(result) if json_output else result
            self.chunk += 1

    def get_partial_symmetries(self, json_output=True, string=False,structured=True) -> Generator[str, None, None]:
        self.start = time.time()
        for k in reversed(range(self.start_size, len(self.graph) + 1)):
            if k == 0:
                yield from self.empty_permutation(json_output, string, structured)
            else:
                yield from self.get_data_for_k_vertex_subgraphs(k, json_output, string, structured)
                if self.timed_out():
                    result = {'chunk': self.chunk, 'error': 'timeout'}
                    yield json.dumps(result) if json_output else result
                    return
        self.kill_timer()
        

    def empty_permutation(self, json_output=True, string=False, structured=True) -> Generator[str, None, None]:
        self.total_partial_symmetries += 1
        pAs = [PartialPermutation([], [])]
        result = {'chunk': self.chunk, 'result': [[{'data': pAs, 'is_group': False}]],
                          'vertices': 0, 'edges': 0, 'size': 1, 'graph': None}
        yield json.dumps(result) if json_output else result if structured else pAs 

    def get_number_of_partial_symmetries(self, json_output=True, debug=False) -> int:
        total_symmetries = 0  # Start with zero and accumulate the number of symmetries
        for k in range(1, len(self.graph) + 1):
            for d_class_data_json in self.get_data_for_k_vertex_subgraphs(k, json_output=json_output):
                if json_output:
                    d_class_data = json.loads(d_class_data_json)
                else:
                    d_class_data = d_class_data_json
                total_symmetries += d_class_data['size']
            if debug:
                print(f'k = {k}')
                print(total_symmetries)
        # Include the identity symmetry
        total_symmetries += 1
        return total_symmetries

    def timed_out(self) -> bool:
        if self.use_timer and not self.timer_process.is_alive():
            self.get_runtime()
            return True
        return False

    def kill_timer(self):
        self.get_runtime()
        if self.timer_process.is_alive():
            self.timer_process.terminate()

    def get_runtime(self):
        self.runtime = round(time.time() - self.start, 3)

    def clear(self):
        self.__init__(self.graph, self.only_full_symmetries, self.use_timer)
        
def draw_monoid_for_subgraph(graph, n, i, D_class, with_labels=True):
    plt.subplot(1, n, i)
    plt.subplots_adjust(wspace=0.5)  # Adjust the width space
    G_nx = graph.to_networkx()
    
    G_sage = Graph(G_nx)
    # back to networkx with positions
    G_sage.plot(save_pos=True, layout='spring')
    pos = G_sage.get_pos() 

    # nx.draw(graph, with_labels=False)
    nx.draw(G_nx, with_labels=with_labels, pos=pos, node_color='skyblue')
    # use spring layout
    

    D_class_extracted = [['\n'.join(item['data']) for item in row] for row in D_class]
    
    # count the maximum number of \n in any cell
    max_line_breaks = max([max([item.count('\n') for item in row]) for row in D_class_extracted])
        
    # print(D_class_extracted)

    plt.rcParams['text.color'] = 'black'
    
    # Display the table, color of text is black
    tbl = plt.table(cellText=D_class_extracted, loc='center', cellLoc='center')
    # Autosize the table columns to fit the content
    tbl.auto_set_font_size(False)
    
    if n == 1:
        tbl.set_fontsize(10)
    else: 
        tbl.set_fontsize(14)
    tbl.auto_set_column_width(col=list(range(len(D_class_extracted))))
    
    
    # Set fill color for each cell
    for i, cell in tbl.get_celld().items():
        if n == 1: cell.set_height(max([max_line_breaks * 0.1, 0.2]))
        else: cell.set_height(max([max_line_breaks * 0.05, 0.2]))

def draw_inverse_monoid(graph, depths=True):
    
    n = len(graph)
    if depths is True:
        depths = [l for l in range(0, n + 1)]
    # print(n)
    p = PartialSymmetries(graph)
    k_vertex_induced_subgraphs = dict()
    for k in depths:
        # print(k)
        # start = time.time()
        k_vertex_induced_subgraphs[k] = list([p.get_data_for_k_vertex_subgraphs(n - k, string=True, json_output=False)][0])
        # print(time.time() - start)
        n_k_vertex_subgraphs = len(k_vertex_induced_subgraphs[k])
        # print("n_k_vertex_subgraphs ", n_k_vertex_subgraphs)
        plt.figure(figsize=(n_k_vertex_subgraphs *  10 * (k%(n-1)+1), n_k_vertex_subgraphs * 3))
        for i, data_for_i_th_k_vertex_subgraph in enumerate(k_vertex_induced_subgraphs[k]):
            # print("i ", i)
            # import json data from data_for_i_vertex_induced_subgraph
            # data_for_i_th_k_vertex_subgraph = json.loads(data_for_i_th_k_vertex_subgraph)
            # print(data_for_i_th_k_vertex_subgraph)
            # print(data_for_i_th_k_vertex_subgraph["graph"])
            # # convert values to strings
            # data_for_i_th_k_vertex_subgraph["graph"] = {str(key): [str(v) for v in value] for key, value in data_for_i_th_k_vertex_subgraph["graph"].items()}
            # i_th_k_vertex_induced_subgraph = json_graph.node_link_graph(data_for_i_th_k_vertex_subgraph["graph"])
            i_th_k_vertex_induced_subgraph = Graph(json_graph.node_link_graph(data_for_i_th_k_vertex_subgraph["graph"]))
            
            if k == 0:
                draw_monoid_for_subgraph(i_th_k_vertex_induced_subgraph, n_k_vertex_subgraphs, i+1, data_for_i_th_k_vertex_subgraph["result"])
            else:
                draw_monoid_for_subgraph(i_th_k_vertex_induced_subgraph, n_k_vertex_subgraphs, i+1, data_for_i_th_k_vertex_subgraph["result"], with_labels=False)
        plt.show()
        
from collections import defaultdict
import copy
import itertools
import hashlib

def build_D_class(G, reduce_to_coloring=True, show_info={"extensions":False,"coloring":True}):
    # def adjacency_condition(G, pA, u_index, v_index, label_to_index):
    #     for w in pA.dom:
    #         w_index = label_to_index[w]
    #         pw = pA._to(w)
    #         pw_index = label_to_index[pw]
    #         u_w_adj = G.are_connected(u_index, w_index)
    #         v_pw_adj = G.are_connected(v_index, pw_index)
    #         # if (u_w_adj and v_pw_adj) or (not u_w_adj and not v_pw_adj):
    #         if u_w_adj == v_pw_adj:
    #             return True
    #     return False
    def complete_pA(pA, G):
        # pA is a partial isomorphism
        # return true if pA is complete
        # else return possible extension
        
        # take all remaining pairs of vertices, u,v, such that u is not in domain,
        # v is not in image, such that u is adjacent to w in domain, and v is adjacent to w in image
        # for all w belonging to domain of pA
        # if there are no such pairs, then pA is complete
        extension_pairs = set()
        # for u, v in itertools.combinations(G.nodes(), 2):
        
        # O(n^2) time
        
        for u, v in [(u, v) for u, v in perm_pairs]:
        # for u in G.nodes():
        #     for v in G.nodes():
                # if (u in pA.dom or v in pA.ran) or adjacency_condition(G, pA, u_index, v_index, label_to_index):
                if (u not in pA.dom and v not in pA.ran) and all([G.are_connected(u, w) == G.are_connected(v, pA._to(w)) for w in pA.dom]):
                    extension_pairs.add((u,v))
        return list(extension_pairs)     
    
      
    p = PartialSymmetries(G)
    # 0 case if no edges, then just run the original algorithm and return
    # 1 Add all empty partial isomorphism, and all partial isomorphism of rank 1
    pAs = defaultdict(set)
    for k in range(0, 2):
        # for iso_class in list(p.find_isomorphism_classes(k)):
        isomorphism_classes = p.find_isomorphism_classes(k)
        for iso_class in isomorphism_classes:
            iso_class.create_d_class()
            # print(iso_class.size())
            print(iso_class.pAs) if show_info["extensions"] else None
            print(set(iso_class.pAs)) if show_info["extensions"] else None
            pAs[k].update(iso_class.pAs)
                    # pAs[k].extend(p_)
        print(len(pAs[k])) if show_info["extensions"] else None
    print(pAs) if show_info["extensions"] else None
    
    combs_pairs = list(itertools.combinations(G.nodes(), 2))
    perm_pairs = combs_pairs + [(v, u) for u, v in itertools.combinations(G.nodes(), 2)]
    
    # 2 For each pair of adjacent vertices, construct all partial isomorphisms
    # O(n^4)
    for u, v in combs_pairs:  # O(n^2)
        for u_, v_ in perm_pairs: # O(n^2)
            u_v_adj = G.are_connected(u, v)
            u__v__adj = G.are_connected(u_, v_)
            # if (u_v_adj and u__v__adj) or (not u_v_adj and not u__v__adj):
            if (u_v_adj == u__v__adj):
                pA = PartialPermutation((u,v), (u_,v_))
                pAs[2].add(pA)

    print(len(pAs[2])) if show_info["extensions"] else None
   
    # 4 set k =2 
    k = 2
    
    if reduce_to_coloring:
        colors = {v: ['0'] for v in G.nodes()}
        old_colors = copy.deepcopy(colors)
        color_history = defaultdict(dict)
    
    # all_extensions = defaultdict(list)        
    
    while True:
        # 5 For each partial isomorphism pA of rank k, 
        #   if pA is complete, then add it to the list of complete partial isomorphism
        #   else, add all possible extensions of pA to the list of partial isomorphisms k = k+1
        complete = True
        n_ext = 0
        # > O(n!)
        if reduce_to_coloring:
            all_extensions_same_colors = set()
        
        
        for pA in pAs[k]:
            # if pA.is_identity():
            #     extensions = set(G.nodes()) - set(pA.dom)
            #     for u in extensions:
            #         pA_ext = PartialPermutation(tuple(list(pA.dom) + [u]), tuple(list(pA.ran) + [u]))
            #         pAs[k+1].add(pA_ext)
            #     continue
            
            extensions = complete_pA(pA, G)
            
            
            if extensions:
                print(f"{k+1}-extensions of {pA}: ", extensions) if show_info["extensions"] else None
                n_ext += len(extensions)
                complete = False
                for u,v in extensions:
                    if reduce_to_coloring: all_extensions_same_colors.add((u,v))
                    # if reduce_to_coloring: all_extensions_same_colors.add((v,u)) # does not change anything
                    # print(u,v)
                    pA_ext = PartialPermutation(tuple(list(pA.dom) + [u]), tuple(list(pA.ran) + [v]))    
                    pAs[k+1].add(pA_ext)
                    # all_extensions[k+1].append((str(pA), (u,v))) if show_info["extensions"] else None
            # else:
            #     print(all_extensions_same_colors)
        
        # 2 steps ahead
        # for pA in pAs[k-1]: 
        #     extensions_ = complete_pA_2(pA, G)
        #     if extensions_:
        #         print(f"{k+1}-2-extensions of {pA}: ", extensions_) if show_info["extensions"] else None
        #         for u,x,v,y in extensions_:
        #             print((u,v),(x,y)) if show_info["extensions"] else None
                    # pA_ext = PartialPermutation(tuple(list(pA.dom) + [u,x]), tuple(list(pA.ran) + [v,y]))    
                
                
        
        # extending - partial identities    
        for pA_id in itertools.combinations(G.nodes(), k+1):
            pA = PartialPermutation(pA_id, pA_id)
            pAs[k+1].add(pA)
            n_ext += 1

        if n_ext == 0: # not possible to extend, no new partial identities
            break
        
        if reduce_to_coloring:
            # the vertices u, v receive different colors at level k if no partial isomorphism of rank k can be extended by adding u -> v
            # contraposition: if there exists a partial isomorphism of rank k that can be extended by adding u -> v, then u and v receive the same color at level k
            
            all_extensions_same_colors_indices = set([(u, v) for u, v in all_extensions_same_colors]).union(set([(v, u) for u, v in all_extensions_same_colors]))
            diff_colors = set(perm_pairs) - all_extensions_same_colors_indices
            
            # delete all pairs that are just reordering of the same pair
            # Convert each pair to a frozenset before adding it to the set
            diff_colors = set(frozenset(pair) for pair in diff_colors) # TODO: tomuto sa viem vyhnut  kvoli cyklu nizsie
            
            # if n_ext ==1 and show_info["extensions"]:
            #     print("all extensions, indices: ", all_extensions_same_colors_indices) 
            #     print(diff_colors)
            
            colors = {v: ['0'] for v in G.nodes()}
            
            # determine color classes
            for u, v in diff_colors:
                # pair = sorted([u,v])
                # u,v=pair 
                colors[u].append(str(u)+","+str(v)) 
                colors[v].append(str(v)+","+str(u)) 
                
            for u in G.nodes():
                colors[u].sort()
                colors[u] = [hashlib.sha224("".join(colors[u]).encode('utf-8')).hexdigest()]
            
            buckets_colors = []
            for u, v in all_extensions_same_colors: # tu je asi chyba
                # colors[u] = colors[v]
                
                are_added = False
                for i, bucket in enumerate(buckets_colors):
                    if u in bucket or v in bucket:
                        buckets_colors[i].add(v)
                        buckets_colors[i].add(u)
                        are_added = True
                        break
                if not are_added:
                    buckets_colors.append(set([u, v]))
                    
            # if the bucket have intersection, then merge them
            for i, bucket_1 in enumerate(buckets_colors):
                for j, bucket_2 in enumerate(buckets_colors):
                    if i != j and len(bucket_1.intersection(bucket_2)) > 0:
                            buckets_colors[i] = buckets_colors[i].union(buckets_colors[j])
                            buckets_colors[j] = set()
                            
            buckets_colors = [bucket for bucket in buckets_colors if len(bucket) > 0]
            
            # merge all colors from the same bucket
                    
            for bucket in buckets_colors:
                b = sorted(list(bucket))
                c_b = [hashlib.sha224("".join([colors[u][0] for u in b]).encode('utf-8')).hexdigest()]
                for u in b:
                    colors[u] = c_b
        
            print("k: ", k , colors, sep=' ') if show_info["coloring"] else None
            color_history[k] = copy.deepcopy(colors)

        print("k: ", k, "|k|: ", len(pAs[k])) if show_info["extensions"] else None
        print("n_ext: ", n_ext) if show_info["extensions"] else None
        if complete:
            break
            
        k += 1
    
    # print(all_extensions) if show_info["extensions"] else None
    
    # return the colors dictionary with k that had the most colors (value classes)
    # Assuming colors_dict is a dictionary where the keys are values of k and the values are color dictionaries
    most_colors = max(color_history.items(), key=lambda item: len(set(value for values in item[1].values() for value in values)))[1]
    
    print(pAs) if show_info["extensions"] else None
    return pAs, colors if reduce_to_coloring else pAs
    # print(len(pAs))

def PAut_level_count(G,k):
    n = len(G.nodes())
    p_G = PartialSymmetries(G)
    k_partial_automorphisms_count = 0
    for p in p_G.get_data_for_k_vertex_subgraphs(n - k, string=False, json_output=False, structured=False,counting=True):
        k_partial_automorphisms_count += p
    return k_partial_automorphisms_count

def PAut_count(G):
    n = len(G.nodes())
    p_G = PartialSymmetries(G)
    k_partial_automorphisms_count = 0
    for k in range(0, n + 1):
        c = 0
        for p in p_G.get_data_for_k_vertex_subgraphs(n - k, string=False, json_output=False, structured=False,counting=True):
            print("p: ", p, "k: ", k)
            c += p
            k_partial_automorphisms_count += p
        print("c: ", c)
    return k_partial_automorphisms_count

def PAut_level(G,k):
    n = len(G.nodes())
    p_G = PartialSymmetries(G)
    all_pperms_G = set()
    for p in p_G.get_data_for_k_vertex_subgraphs(n - k, string=False, json_output=False, structured=False):
        all_pperms_G = all_pperms_G.union(p)
    return all_pperms_G

def PAut(G,verbose=False):
    n = len(G.nodes())
    p_G = PartialSymmetries(G)
    all_pperms_G = set()
    for k in range(0, n + 1):
        c = 0
        for p in p_G.get_data_for_k_vertex_subgraphs(n - k, string=False, json_output=False, structured=False):
            print("p: ", len(p), "k: ", k) if verbose else None
            c += len(p)
            all_pperms_G = all_pperms_G.union(p)
        print("c: ", c) if verbose else None
    return all_pperms_G

def PAut(G):
    n = len(G.nodes())
    p_G = PartialSymmetries(G)
    for k in range(0, n + 1):
        c = 0
        for p in p_G.get_data_for_k_vertex_subgraphs(n - k, string=False, json_output=False, structured=False):
            # print("p: ", len(p), "k: ", k)
            c += len(p)
            yield p
        # print("c: ", c)
        
def PAut_k(G, k = 0):
    n = len(G.nodes())
    p_G = PartialSymmetries(G)
    all_pperms_G_k = set()
    for p in p_G.get_data_for_k_vertex_subgraphs(k, string=False, json_output=False, structured=False):
        all_pperms_G_k = all_pperms_G_k.union(p)
    return all_pperms_G_k

def PAUT_up_to(G, k):
    for i in range(k+1):
        yield from PAut_k(G, i)


def timer(timeout_after=60):
    time.sleep(timeout_after)
