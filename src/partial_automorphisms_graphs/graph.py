import networkx as nx
from pyvis.network import Network
from sage.parallel.decorate import Parallel, parallel, fork, Fork
import itertools
from .partial_permutation import PartialPermutation

# class Graph(nx.Graph):
#     def __init__(self, incoming_graph_data=None, **attr):
#         super().__init__(incoming_graph_data, **attr)
        
#     def get_automorphisms(self, check_is_asymmetric=False):
#         aut_group = list()
#         for sym in nx.vf2pp_all_isomorphisms(self,self):
#             aut_group.append(tuple(sym[key] for key in sorted(sym)))
#             if check_is_asymmetric:
#                 if len(aut_group) > 1:
#                     return aut_group
#         return aut_group
    
#     def is_asymmetric(self):
#         """_summary_

#         Returns:
#             if the graph is asymmetric, i.e. it has only one automorphism - Identity
#         """        
#         if len(self.get_automorphisms(check_is_asymmetric=True)) == 1:
#             return True
#         return False
    
#     def degree_sequence(self):
#         return tuple(sorted([d for n, d in self.degree()], reverse=True))

from sage.graphs.graph import Graph as SageGraph

# def node_link_data(graph: IGraph):
#     nodes = [{'id': v.index} for v in graph.vs]
#     links = [{'source': e.source, 'target': e.target} for e in graph.es]
#     return {'nodes': nodes, 'links': links}

def node_link_data(graph: SageGraph):
    nodes = [{'id': int(v)} for v in graph.nodes()]
    links = [{'source': int(n1), 'target': int(n2)} for n1,n2 in graph.edges()]
    return {'nodes': nodes, 'links': links}


from sage.graphs.graph import Graph as SageGraph
import sage.all

class Graph(SageGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(data=incoming_graph_data, multiedges=False, loops=False, weighted=False)
        self.is_asym = None
        self.change = False
        self.canonical_string = None

    def add_vertex(self, vertex):
        super().add_vertex(vertex)
        self.change = True

    def add_edge(self, u, v):
        super().add_edge(u, v)
        self.change = True

    def delete_vertex(self, vertex):
        super().delete_vertex(vertex)
        self.change = True

    def delete_edge(self, u, v):
        super().delete_edge(u, v)
        self.change = True

    def get_automorphisms(self, as_permutations=False):
        for a in self.automorphism_group():
            if as_permutations:
                yield PartialPermutation(list(a.dict().keys()), list(a.dict().values()))
            else:
                yield a.dict()
                
    # i implemented is_isomorphic using canonical_label but it is not working faster
    # def is_isomorphic_can(self, other):
    #     if self.change or self.canonical_string is None:
    #         # exchange graph with the canonical label
    #         self.canonical_string = self.canonical_label(return_graph=False)
    #         self.change = False
        
    #     if other.change or other.canonical_string is None:
    #         other.canonical_string = other.canonical_label(return_graph=False)
    #         other.change = False
    #     return self.canonical_string == other.canonical_string
        
    def is_asymmetric(self):
        """_summary_

        Returns:
            if the graph is asymmetric, i.e. it has only one automorphism - Identity
        """
        if self.is_asym is None or self.change:
            self.is_asym = self.automorphism_group(return_group=False, order=True) == 1
            self.change = False
        return self.is_asym 
    
    @parallel
    def get_isomorphisms(self, other):
        is_iso, cert = self.is_isomorphic(other, certificate=True)
        all_isomorphisms = []
        if is_iso:
            
            # Generating all isomorphisms by applying automorphisms to the certificate
            all_isomorphisms = []
            for perm in self.get_automorphisms():
                # Apply the automorphism to the isomorphism certificate
                new_iso = {perm[v]: cert[v] for v in self.vertices()}
                all_isomorphisms.append(new_iso)
            
        return all_isomorphisms        
    
    def get_partial_automorphisms(self):
        from partial_symmetries import PartialSymmetries
        n = len(self.nodes())
        p_G = PartialSymmetries(self)
        all_pperms_G = set()
        for k in range(0, n + 1):
            for p in p_G.get_data_for_k_vertex_subgraphs(k, string=False, json_output=False, structured=False):
                all_pperms_G = all_pperms_G.union(p)
        return all_pperms_G
    
    def get_partial_automorphisms_of_rank(self, rank):
        from partial_symmetries import PartialSymmetries
        n = len(self.nodes())
        all_pperms_G_of_rank = set()
        p_G = PartialSymmetries(self)
        for p in p_G.get_data_for_k_vertex_subgraphs(rank, string=False, json_output=False, structured=False):
            all_pperms_G_of_rank = all_pperms_G_of_rank.union(p)
        return all_pperms_G_of_rank 
        
    
    def get_partial_automorphisms_of_rank(self, rank):
        from partial_symmetries import PartialSymmetries
        n = len(self.nodes())
        p_G = PartialSymmetries(self)
        for p in p_G.get_data_for_k_vertex_subgraphs(rank, string=False, json_output=False, structured=False):
            yield from p
            
            
    # def p_vertex_transitive(self):
    #     # Graph is p-vertex-transitive if for any vertices u,v there exists a partial automorphism of rank p taking u to v
    #     n = len(self.nodes())
    #     for p in reversed(range(2,n+1)):
    #         pas = self.get_partial_automorphisms_of_rank(p)
    #         for n1, n2 in itertools.permutations(self.nodes(),2):
    #             exists_pa_taking_n1_to_n2 = False 
    #             for pa in pas:
    #                 if pa.is_from_to(n1,n2)[0]:
    #                     exists_pa_taking_n1_to_n2 = True
    #                     break
    #             if not exists_pa_taking_n1_to_n2:
    #                 break
    #         if not exists_pa_taking_n1_to_n2:
    #             continue
    #         return p
    #     return 1                
    
    def p_vertex_transitive(self):
        # Graph is p-vertex-transitive if for any vertices u,v there exists a partial automorphism of rank p taking u to v
        n = len(self.nodes())
        for p in reversed(range(2,n+1)):
            
            perms = itertools.permutations(self.nodes(),2)
            perms_ = set(list(itertools.permutations(self.nodes(),2)))
            
            pas = list(self.get_partial_automorphisms_of_rank(p))
            
            for n1, n2 in perms:
                if not (n1,n2) in perms_: continue
                exists_pa_taking_n1_to_n2 = False 
                for pa in pas:
                    # print(pa)
                    n1_to_n2, n1_to_nx = pa.is_from_to(n1,n2)
                    if n1_to_n2:
                        exists_pa_taking_n1_to_n2 = True
                        n1_to_nx = [(n1,nx) for nx in n1_to_nx]
                        perms_.difference_update(n1_to_nx)
                        break
                if not exists_pa_taking_n1_to_n2:
                    break
            if not exists_pa_taking_n1_to_n2:
                continue
            return p
        return 1                
            
            
    
    def degree_sequence(self):
        return tuple(sorted(self.degree(), reverse=True))
    
    def nodes(self):
        return self.vertices()
    
    def neighbors(self, node):
        return super().neighbors(node)
    
    def edges(self,labels = False):
        # return edgelist of names of nodes
        return super().edges(labels=labels)
    
    def are_connected(self, n1, n2):
        return n1 in self.neighbors(n2)
    
    def to_networkx(self):
        return self.networkx_graph()
    
    def draw(self,pos=None):
        # Create a new PyVis network
        net = Network(height="750px", width="100%", bgcolor="white", font_color="black", select_menu=True, notebook=False, directed=False, filter_menu=True)
        # net.from_nx(self.to_networkx())
        
        # for node in net.nodes:
        #     # add all properties as node attributes
        #     node['title'] = ""
        #     for key, value in node.items():
        #         try:
        #             node["title"] += str(key) + " : " + str(value) + "\n"
        #         except:
        #             for k, v in value.items():
        #                 node["title"] += str(k) + ":" + str(v) + "\n"

        # Add nodes and edges to the network
        for node in self.nodes():
            net.add_node(str(node), label=str(node), size=10, color='black')
        for n1,n2 in self.edges():
            net.add_edge(str(n1), str(n2) , color='black', physics=False)

        # # Create a NetworkX graph to generate a circular layout
        # G = nx.Graph()
        # G.add_nodes_from(range(len(self.vs)))
        # G.add_edges_from([(node_mapping[self.vs[edge.source]['name']], node_mapping[self.vs[edge.target]['name']]) for edge in self.es])
        # # pos = nx.circular_layout(G)

        # # # Set the positions of the nodes in the PyVis network
        # # for node_id, position in pos.items():
        # #     net.nodes[node_id]['x'] = position[0] * 1000
        # #     net.nodes[node_id]['y'] = position[1] * 1000
        
        # for n in net.nodes:
        #     n['label']= n['id'] #concatenate label of the node with its attribute
            
        # Show the network
        net.show_buttons(filter_=['nodes', 'edges', 'physics', 'layout', 'interaction', 'manipulation',
                                       'selection', 'renderer'])   
        
        net.repulsion(0)
        net.set_edge_smooth('continuous')
        net.toggle_physics(False)
        net.show("graph.html")
    
    def shared_neighbors(self, v1, v2):
        return len(list(nx.common_neighbors(self.to_networkx(),v1, v2)))

    def symmetric_difference_neighbors(self, v1, v2):
        neighbors_v1 = set(self.neighbors(v1))
        neighbors_v2 = set(self.neighbors(v2))
        
        n1 = len(list(neighbors_v1))
        n2 = len(list(neighbors_v2))
        
        n12 = self.shared_neighbors(v1, v2)
        
        sigma = 0
        if v1 in neighbors_v2:
            sigma = 1
        
        return n1 + n2 - 2*n12 - 2*sigma

    def delta(self):
        n = len(self.nodes())
        sym_diff = n
        for v1 in self.nodes():
            for v2 in self.nodes():
                if v1 < v2:
                    delta_new = self.symmetric_difference_neighbors(v1, v2)
                    if delta_new < sym_diff:
                        sym_diff = delta_new
        return sym_diff


    def S_lower_bound(G):
        n = len(self.nodes())
        delta = n
        for v1 in self.nodes():
            for v2 in self.nodes():
                if v1 < v2:
                    delta_new = self.symmetric_difference_neighbors(v1, v2)
                    if delta_new < delta:
                        delta = delta_new
        
        print(delta)
        return (n - delta) / n
    
if __name__ == "__main__":
    # Create a graph
    # g = Graph([(0, 1), (1, 2), (2, 0)])
    
    # print(list(g.vs))
    # print(list(g.vs)[0])
    
    g = Graph(nx.Graph([(0, 1), (1, 2), (2, 0)]))
    
    print(list(g.get_automorphisms(as_permutations=True)))
    
    print(g.p_vertex_transitive())
    self = nx.from_dict_of_lists({1: [2,3,4,5], 2:[3,4,5], 3:[4]})
    self = Graph(self)
    
    from partial_symmetries import draw_inverse_monoid

    draw_inverse_monoid(self,depths=[0])
    
    print(self.p_vertex_transitive())
    
    # g.draw()

    # print(list(g.vs)[0]['_nx_name'])
    # print(list(g.vs))

    # Test the is_asymmetric method
    print(f"Is the graph asymmetric? {g.is_asymmetric()}")

    # Test the get_degree_sequence method
    print(f"Degree sequence of the graph: {g.degree_sequence()}")
    print(g.get_automorphisms())
    print(list(g.nodes()))
    
    print(g.to_dict_list())
    
    print(g.subgraph([0,1]))
    g = Graph([(0, 1), (1, 2), (2, 0)])
    print(g.degree_sequence())
    print(g.get_automorphisms())
    