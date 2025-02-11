from typing import Set


class Cycle:

    def __init__(self, cycle):
        self.cycle_notation = cycle

    def __str__(self):
        values = list(self.cycle_notation[:-1])
        min_index = values.index(min(values))
        data = values[min_index:] + values[:min_index]
        values = tuple(data)
        # if len(values) == 1:
        #     values = '(' + str(values[0]) + ')'
        return str(values).replace(',', '')
        # return str(values).replace(' ', '')

    def __len__(self):
        return sum(len(x) for x in self.cycle_notation)

    def flatten(self) -> Set[int]:
        return set(item for set_ in self.cycle_notation for item in set_)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        # TODO: document this - multiplies with other cycle
        result = []
        remaining = set(range(len(self)))
        while remaining:
            current_item = list(remaining)[0]
            first_item = current_item
            result_set = [current_item]
            while True:
                next_item = other.get_next(current_item)
                next_item_other = self.get_next(next_item)
                current_item = next_item_other
                if current_item == first_item:
                    break
                result_set.append(next_item_other)
            result.append(tuple(result_set))
            remaining.difference_update(result_set)
        result = [set(x) for x in result]
        return Cycle(result)

    def get_set_containing(self, what):
        for set_ in self.cycle_notation:
            if what in set_:
                return set_

    def get_next(self, what):
        for set_ in self.cycle_notation:
            if what in set_:
                for i, item in enumerate(list(set_)):
                    if what == item:
                        if i == len(set_) - 1:
                            return list(set_)[0]
                        else:
                            return list(set_)[i + 1]

    def gap_repr(self):
        repr_str = ''
        for cycle in self.cycle_notation:
            if len(cycle) > 1:
                repr_str += str(cycle)
        return repr_str

class Path:

    def __init__(self, path):
        # this is called cycle notation just so we don't have to change the code to differentiate between
        # Path and Cycle instances
        self.cycle_notation = list(path)

    def __str__(self):
        # return '[' + ','.join(map(str, self.cycle_notation)) + ')'
        return '[' + ' '.join(map(str, self.cycle_notation)) + ')'


def find_start(dom, ran, unchecked):
    for i, item in enumerate(dom):
        if item not in ran and ran[i] in unchecked:
            return ran[i]
    return None


def to_cycle_path_notation(dom, ran, debug=False):
    # TODO: document this
    result = []
    unchecked = set(ran)
    while unchecked:
        # finds the starting element 
        # - either the element that is in domain but not in range or min
        min_unchecked = find_start(dom, ran, unchecked) or min(unchecked)
        if debug: print(min_unchecked)
        curr_index = ran.index(min_unchecked)
        bracket = [dom[curr_index]] # finds the image of the first element in the cycle
        while True:
            if debug: print(bracket)
            current = ran[curr_index]
            if current in bracket[1:]: # if it is its own range and domain the cycle is complete
                result.append(Cycle(bracket)) 
                break
            bracket.append(current)
            unchecked.remove(current)
            curr_index = dom.index(current) if current in dom else -1
            if curr_index == -1: 
                # if the element has noone in the domain,  
                # it is the last element in the cycle-path, ideme z prava do lava e.g. [4 5 6)
                result.append(Path(reversed(bracket)))
                break
    return result