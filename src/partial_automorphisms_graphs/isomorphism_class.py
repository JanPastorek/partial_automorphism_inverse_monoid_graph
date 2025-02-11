import math
import networkx as nx
from typing import Tuple, List, Optional
from .graph import Graph
from .partial_permutation import PartialPermutation
from networkx.algorithms import isomorphism
import numpy

def check_if_isomorphic(G, isomorphism_classes) -> Tuple[bool, Optional[Graph], int]:
    """
    Checks if 'G' is isomorphic to any graph in the list 'other'
    Returns
    -------
    tuple (is_isomorphic, graph, checks) - (
        True if 'G' is isomorphic to any graph in 'other',
        if it is isomorphic, then the graph to which it is isomorphic (representative of that isomorphism class),
        how many times we called the 'is_isomorphic' function
    )
    """
    checks = 0
    if G is None:
        return False, None, checks
    
    for ic in isomorphism_classes:
        checks += 1
        # if isomorphism(G, g):
        # if nx.is_isomorphic(G, g):
        if ic.is_belonging(G):
            return True, ic, checks
    return False, None, checks


class IsomorphismClass:
    def __init__(self, rep: Graph):
        self.graphs = [rep]
        self.rep = rep  # representative of this isomorphism class
        self.d_class = []
        self.pAs = []
        self.count = 0
        self.finished = False

    def size(self) -> int:
        """
        Calculates the size of the D-class eggbox diagram
        """
        return math.pow(len(self.graphs), 2) * len(self.rep.automorphism_group(order=True))

    def add_isomorphic(self, item: Graph):
        self.graphs.append(item)
    
    def is_belonging(self, other):
        """
        check if the representative of this isomorphic class is isomorphic to 'other' graph
        """
        return self.rep.is_isomorphic(other)

    def create_d_class(self, debug=False, string=False, counting=False):
        """
        Creates the eggbox diagram for this isomorphism class
        """
        if self.finished:
            return
        
        count = 0
        vertices = [list(G.nodes()) for G in self.graphs]
        size = len(self.graphs)
        for i in range(size):
            self.d_class.append([])
            for j in range(size):
                h_class = []
                if i != j:  # two different graphs pairwise isomorphism
                    # for mapping in isomorphism.vf2pp_all_isomorphisms(self.graphs[j], self.graphs[i]):
                    for mapping in self.graphs[j].get_isomorphisms(self.graphs[i]):
                        # mapping of indices to names
                        # mapping_names = [self.graphs[i].vs[index]['name'] for index in mapping]
                        # print(self.graphs[j].nodes(),mapping_names)
                        # pp = PartialPermutation(list(mapping.keys()), list(mapping.values()))
                        
                        count += 1
                        if counting:
                            continue                            
                        
                        pp = PartialPermutation(tuple(mapping.keys()), tuple(mapping.values()))
                        self.pAs.append(pp)
                        h_class.append(str(pp)) if string else h_class.append(pp)
                        if debug:
                            print("diff graphs isomorphism")
                            print(list(mapping.keys()), list(mapping.values()))
                            print(str(pp))
                else:  # i == j, diagonal on the eggbox diagram
                    # dom = vertices[j]
                    # for ran in self.graphs[i].aut_group:
                    for mapping in self.graphs[i].get_automorphisms():
                        count += 1
                        if counting:
                            continue                            
                        pp = PartialPermutation(list(mapping.keys()), list(mapping.values()))
                        self.pAs.append(pp)
                        h_class.append(str(pp)) if string else h_class.append(pp)
                        if debug:
                            print("automorphism")
                            print(list(mapping.keys()), list(mapping.values()))
                            print(str(pp))
                self.d_class[-1].append({'data': h_class, 'is_group': i == j and len(h_class) > 1})
        
        if not counting:
            self.finished = True
        self.count = count