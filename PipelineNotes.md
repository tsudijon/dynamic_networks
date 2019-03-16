# Model Outline
Description of our pipeline and the model that we study. Documents the parameters that go into the construction. 

## Graph Distances
We consider graph edit distance, where the distance between two graphs is the minimum cost of an edit path - a sequence of transformations converting $G_1 \rightarrow G_2.$ 

Edits are of the form:

- Adding/deleting a node
- node substitution
- Adding/deleting an edge
- Edge substitution

We set the costs to be:

- Adding a node: the weight of the node
- Deleting a node: deleting the link of the node - the sum of the weights of the link. The maximum node weight  and the surrounding edge weights also works.
- Node substitution - absolute difference in values
- Similarly for the edges.


### Other ideas
Inspired by the sensor model on the globe example, essentially we can interpret the graphs as discrete representations of functions on a manifold (similar to the initial idea of using geodesic distance on the graphs). We want graphs that represent different functions on the same manifold to be farther away in terms of distance than graphs that represent the same function on a common manifold. Then we would want a stability result proving that this graph distance is not too far away from the persistence diagram distance as in our pipeline.

## Sensor Model on the Globe
- Begin with $N$ points on the sphere
- on average, in one time unit, $\lambda$ sensors drop off the grid and $\lambda$ come online. The exact is distribution is $\text{exponential}(\lambda).$
- At each time in this process, we sample a network by looking at the sensors that are alive at that time. Connect them via a Delaunay triangulation, for example. 
- Set the edge values to be spherical distance between the nodes
- Set the node values to be the value of the observation function at that point.


## Pipeline
Given a graph with values on the nodes and edges (representing the importance? This may be arbitrary).

Convert to a network in the sense of Memoli and Chowdhury 2017, by mapping the edge and nodes weights of the network by a parameter $\phi: \mathbb{R}^+ \rightarrow \mathbb{R}^+$ such that $\phi(0)$ is finite, $\phi$ decreasing, and $\phi'$ bounded.

Convert the network into a filtered simplicial complex via setting the birth time of a simplex as $$ B(\sigma) = \max_{v,e \in \sigma}(w_v,\lambda w_e)$$ where $w_v,w_e$ are the node and edge weights, and we can take $\lambda = 1$ for simplicity.

Convert the filtered simplicial complex into a Persistence diagram, looking at $H_0.$ (What does this mean)?

We compute the distance matrix of these persistence diagrams in order to do the sliding window embedding.







