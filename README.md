
The repositoty provides code for analyzing and processing partial automorphisms in graphs. 

## Requirements

- SageMath, version 10.1.
	- **Dependencies**  
	  Install the required Python packages:
	  ```sh
	  pip install -r requirements.txt
	  ```
	  To install the local module where the setup.py is located, use
	  ```sh
	  pip install -e .    
	  ```
- GAP, version 4.12.2
- Nauty 2.8.6
- Semigroups package for GAP

## Directory Structure
```
├── __init__.py
├── __pycache__/
├── all_graphs/  
├── data/
├── asymmetric_finder.py
├── cycle_notation.py
├── graph.py
├── main.ipynb
├── isomorphism_class.py
├── partial_permutation.py
├── partial_symmetries.py
├── utils.py
└── w_l.py
```

## Module Descriptions

- **asymmetric_finder.py** 
	- Provides functions to detect asymmetric structures within a graph.

- **cycle_notation.py**  
	- Implements functions to parse and generate cycle-path notation for partial permutations.

- **graph.py**  
	- Defines the graph structure and associated operations for building and processing graphs.

- **isomorphism_class.py**  
	- Groups graphs into isomorphism classes

- **partial_permutation.py**  
	- Manages creation and manipulation of partial permutations, a key step in handling partial symmetries.

- **partial_symmetries.py**  
	- The core module that aggregates functionality from other modules to analyze partial symmetries in graphs.

- **utils.py**  
	- Contains utility functions that are shared across modules to support common tasks.

- **w_l.py**  
	- Implements the Weisfeiler-Leman algorithm

- **gap_functions.g**
	- Contains functions that check whether the given monoid is partial automorphism inverse monoid of a graph.

- /all_graphs/
	- Contains graph databases, including record graphs for asymmetric depth

## Usage Example

To work with the modules in the folder, import the necessary functions or classes in your Python script. For example:

```python
from graph import Graph
from partial_symmetries import PAut

# Create a graph object
G = Graph()

# Analyze partial symmetries in the graph
PAUT = PAut(G)
```

More examples and code used is in main.ipynb