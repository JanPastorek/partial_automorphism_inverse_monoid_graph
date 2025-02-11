from .cycle_notation import to_cycle_path_notation, find_start

class PartialPermutation:
    """
    class for representing partial permutations
    """
    def __init__(self, dom, ran):
        
        self.dom = tuple(dom)
        self.ran = tuple(ran)
        self.pairs = set()
        if len(self.dom) != 0 and len(self.ran) != 0:
            # Create pairs from domain and range
            pairs = list(zip(self.dom, self.ran))
            pairs.sort(key=lambda x: x[0])
            self.dom, self.ran = list(zip(*pairs))
            self.pairs = set(pairs)
            # Sort pairs based on the domain
            # Unzip pairs back into sorted domain and corresponding range
        self.cycle_path = to_cycle_path_notation(dom, ran)

    def __str__(self):
        if len(self.ran) == 0 and len(self.dom) == 0: return '∅'
        if len(self.cycle_path) == 0: return 'id'
        return ''.join(str(x) for x in self.cycle_path)
    
    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.dom[self.ran.index(item)]

    def _from(self, item):
        return self.dom[self.ran.index(item)]

    def _to(self, item):
        return self.ran[self.dom.index(item)]

    def __contains__(self, item):
        return item in self.dom or item in self.ran
    
    def __getitem__(self, index):
        if index in self.dom:
            return self._to(index)
        else:
            raise IndexError("Index out of range")
    
    def fixed_points(self):
        # return [x for x in self.dom if self.dom[x] == self.ran[x]]
        return {x for i, x in enumerate(self.dom) if x == self.ran[i]}
    
    def moved_points(self):
        return {x for i, x in enumerate(self.dom) if x != self.ran[i]}

    def is_from_to(self,_from, _to):
        result = False
        # finds the starting element 
        # - either the element that is in domain but not in range or min
        try:
            min_unchecked = self.dom.index(_from)
            curr_index = min_unchecked
        except ValueError: # nie je v domene
            return False, []
        bracket = [self.dom[curr_index]] # finds the domain of the first element in the cycle
        while True:
            current = self.ran[curr_index]
            if current == _to:
                result=True
                break
            if current in bracket[1:]: # if it is its own range and domain the cycle is complete
                result = False
                break
            bracket.append(current)
            curr_index = self.dom.index(current) if current in self.dom else -1
            if curr_index == -1: 
                # if the element has none in the domain,  
                # it is the last element in the cycle-path, ideme z prava do lava e.g. [4 5 6)
                result = False
                break
        return result, bracket

    def __hash__(self):
        return hash((tuple(self.dom), tuple(self.ran)))

    def __eq__(self, other):
        # mapping = list(zip(self.dom, self.ran))
        # other_mapping = list(zip(other.dom, other.ran))
        # for i in range(len(mapping)):
        #     if mapping[i] != other_mapping[i]:
        #         return False
        # return True
        return self.pairs == other.pairs
    
    def __len__(self):
        return len(self.dom)

    def __lt__(self, other):
        # natural partial ordering
        try:
            if len(self) < len(other):
                for v in self.dom:
                    if self[v] != other[v]:
                        return False
                return True
            else:
                return False
        except IndexError:
            return False
    

    def is_identity(self):
        """
        returns True if this partial permutation only contains identity, so domain = range
        """
        return all(self.dom[i] == self.ran[i] for i in range(len(self.dom)))

    def inverse(self):
        return PartialPermutation(self.ran, self.dom)
    
    def rank(self):
        return len(self)
    
    def union(self, other):
        """
        Create a union of two partial permutations.
        
        Args:
        - other (PartialPermutation): The other partial permutation to union with.
        
        Returns:
        - PartialPermutation: A new partial permutation that is the union of self and other.
        """
        # Combine domains and ranges
        combined_dom = list(frozenset(list(self.dom) + list(other.dom)))
        combined_ran = list(frozenset(list(self.ran) + list(other.ran)))
        
        return PartialPermutation(combined_dom, combined_ran)
    
    def partial_permutation_multiplication_dom_ran(self,Y1, Z1, Y2, Z2):
        # Amalgamate the elements of Y1 and Z1, and Y2 and Z2 into tuples
        phi1_mapping = dict(zip(Y1, Z1))
        phi2_mapping = dict(zip(Y2, Z2))

        # Determine the common elements in the range of phi1 (Z1) and the domain of phi2 (Y2)
        common_elements = frozenset(phi1_mapping.values()).intersection(frozenset(phi2_mapping.keys()))

        # Evaluate the domain and range for the composed partial permutation phi2_phi1
        dom_phi2_phi1 = [y for y, z in phi1_mapping.items() if z in common_elements]
        ran_phi2_phi1 = [phi2_mapping[z] for z in common_elements]

        return dom_phi2_phi1, ran_phi2_phi1
    
    def __mul__(self, other):
        phi1_mapping = dict(zip(self.dom, self.ran))
        phi2_mapping = dict(zip(other.dom, other.ran))
        dom_phi2_phi1, _ = self.partial_permutation_multiplication_dom_ran(self.dom, self.ran, other.dom, other.ran)
        ran_phi2_phi1 = []
        for d in dom_phi2_phi1:
            ran_phi2_phi1.append(phi2_mapping[phi1_mapping[d]])
        return PartialPermutation(dom_phi2_phi1, ran_phi2_phi1)
    
    def is_idempotent(self):
        if self * self == self:
            return True
        return False
    
    def is_compatible(self, other):
        i_1 = self.inverse() * other
        i_2 = self * other.inverse()
        if i_1.is_idempotent() and i_2.is_idempotent():
            return True
        return False
    
    def is_ortogonal(self, other):
        return set(self.dom).isdisjoint(set(other.dom)) and set(self.ran).isdisjoint(set(other.ran))
    
if __name__ == "__main__":
    
    
    print(PartialPermutation([3,2],[4,1]) == PartialPermutation([2,3],[1,4]))
    
    print(PartialPermutation([3,2],[4,1]))
    print(PartialPermutation([2,3],[1,4]))
    
    
    P1 = PartialPermutation([3,2],[4,1])
    P2 = PartialPermutation([3],[4])
    print(P1.is_compatible(P2))
    
    
    print(P2 < P1)
    print(PartialPermutation([1,2,3,4,5,6,7],[1,2,4,3,5,6,7]) < PartialPermutation([1,2,3,4,5,6,7,8],[1,2,4,3,5,6,7,8]))

    print(P1.union(P2))
    
    P1 = PartialPermutation([2,3],[3,4])
    P2 = PartialPermutation([5],[4])
    P1_P2 = P1 * P2
    print(P1_P2.dom, P1_P2.ran)

    print(P1.is_compatible(P2))
    
    P1 = PartialPermutation([1,2],[2,3])
    P2 = PartialPermutation([2,3],[1,2])
    P1_P2 = P1 * P2
    P2_P1 = P2 * P1
    print(P1_P2)
    print(P1_P2.dom, P1_P2.ran)
    print(P2_P1) 
    print(P2_P1.dom, P2_P1.ran) 
    
    
    print(P1.is_from_to(2,1))
    print(P1.is_from_to(1,3))
    P2 = PartialPermutation([2,3,4,5,6],[1,2,3,4,5])
    print(P2.is_from_to(5,5))