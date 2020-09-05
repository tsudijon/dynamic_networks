# References
## Topological Time Series Analysis
A [review paper](https://arxiv.org/pdf/1812.05143.pdf)  by Jose. Recalls the Takens Embedding theorem. The ideas of the sliding window embedding begin in nonlinear time series analysis, see e.g. Nonlinear dynamics, delay times, and embedding windows - this paper in particular chooses the delay time $\tau.$

Some applications of the Topological Time Series methodology - see the review paper above.

- Wheeze detection.
- Periodicity & quasiperioidicty detection in video.
- Periodic gene expression quantification
- Chatter detection, classification in machining
- Segmentation of Dynamic Regimes
- Social Interaction Networks

Jose Perea's work revolves around the Sw1Pers method, which looks at the persistent homology of the sliding window embedding point cloud, and tries to characterize the shape using homology. In particular,

- If the system is periodic, the SW point cloud should be a circle/closed loop.
- If the system is quasiperiodic, it should be a Torus of some dimension.
- If we have recurrent behavior, like the Lorentz Attractor, should be represented by a wedge / bouquet of circles.

Another approach on topological recurrence of recurrent systems by Vin da Silva - utilizes persistent cohomology [see here](https://pdfs.semanticscholar.org/a1ae/796d306abe2e55beba70ee0516bffbe1e23e.pdf). Is there an advantage of using cohomology versus homology? 
> A well-known result in algebraic topology asserts that co-circles are classified (up to homotopy)  by 1-dimensional cohomology.
The paper then chooses a persistent cocyle, then recovers circular coordinates from it somehow such as harmonic smoothing




## Lozeve TDA on Networks
Approach doesn't seem to use sliding window embeddings. It seems to use clustering techniques on vectorized Persistence Diagrams, considering instead subnetworks defined by a sliding window construction. Applications to social network data. Project networks into metric spaces using either *spectral methods* (MDS, Laplacian eigenmaps, kernel PCA, Markov Diffusion maps) or *Latent State Methods* which embed using a physical analogy such as every point repels one another using a spring analogy. Using the graph embeddings, we get point clouds:
$$ Networks \rightarrow Point Cloud \rightarrow Simplicial Complex \rightarrow Persistence Diagram$$

A few options for the conversion of the point cloud to a simplicial complex:

- Cech complex
-Vietoris Rips complex
- weight rank-clique filtration: add edge with highest weights in order, then produce the clique network. Computing cliques is computationally expensive (Bron Kerbosch)

Another pipeline is to use the existing temporal sequence of the network to build the filtration, but then there is no one directional inclusion. The way to over come this is *Zigzag persistence* - compute persistent homology on filtrations that are not nested. (This construction is generalized by multiparameter persistence, but the fundamental theorem of persistent homology is no longer true in this case). The complexity of the zigzag algorithm is cubic in the maximum number of simplices in the complex [13], which is equivalent to the worst-case complexity of the standard algorithm for persistent homology

The thesis then seems to apply kernel techniques to the persistence diagram (or persistence landscape / persistence image) output of the dynamic network - using machine learning. Some kernels defined on the space of PDs include Sliced Wasserstein Kernel, PWGK (seen this before), persistence scale-space kernel

The timescale of the dynamic network can have large impacts on the outcome: see e.g.:

- Gautier Krings, Márton Karsai, Sebastian Bernhardsson, et al. “Effects of time window size and placement on the structure of an aggregated communication network”.
- Quantifying the effect of temporal resolution on time-varying networks
- Meaningful Selection of Temporal Resolution for Dynamic Networks

The thesis uses temporal partitioning to solve this, which might be relevant in our example: ![Algorithm](/Users/sianamuljadi/projects/Research/dynamic_networks/References/temporalpartitioning.png)

The dataset used is "infectious sociopatterns" - contact social interaction dataset.

# Parameter Choice
How is the choice of homological dimension, sliding window length, and delay made? Theoretical contributions can be found for certain simple periodic functions, see Jose's paper with John Harer [Sliding Windows and persistence](https://arxiv.org/abs/1307.6188) as well as for quasiperiodic functions (toroidal sliding window embeddings) [see here](http://www.mirlab.org/conference_papers/international_conference/ICASSP%202016/pdfs/0006435.pdf). 

Nonlinear dynamics, delay times, and embedding windows - this paper in particular chooses the delay time $\tau.$ Many other papers for choosing the delay coordinates, mainly in Physics papers. See Vin de Silva's paper for more references.

How to choose the proper dimension $d$? The method `false nearest neighbors` can help with this, given in [Kennel et al.](https://journals.aps.org/pra/pdf/10.1103/PhysRevA.45.3403), [Review of FNN](https://www.sciencedirect.com/science/article/pii/S0098135497876570/pdf?md5=53b2460bde34be51ae9d7287c1f81b2e&pid=1-s2.0-S0098135497876570-main.pdf). The idea is that if the delay embedding is smaller than the needed one, the topology is not faithfully reconstructed and so there will be neighbors that are in incorrect places.

- Consider the vectors $\phi_l(k) = (y(k-\tau),\dots,y(k - l\tau)).$
- Given $\phi_l(k)$ find $\phi_l(j)$ such that the distance between these two, Euclidean wise, is minimized.
- Check that \\[ \frac{dist(y(k),y(j))}{||\phi_l(k),\phi_l(j) ||_2} \leq R\\] for som threshold $R$ if so, the neighbor is labeled a *false neighbor.*
- Continue for all $k,$ and calculate the number of points with false neighbors.
- Continue raising $l$ until the number of points with false neighbors is small.

# Papers
## (Quasi)Periodicity Quantification in Video Data, using Topology

Taken's Embedding theorem: there exists an integer $D,$ so that for all $d \geq D$ and generic $\tau > 0$ the sliding window embedding $SW_{d,\tau}X$ reconstructs the state space of the underlying dynamics witnessed by the signal $X.$ To find the such a minimal $D$: use *false nearest neighbors scheme* - keep track of the $k$th nearest neighbors of each point in the delat embedding, and if they change as $d$ is increased, then $d$ is too low. 

What about picking the delay $\tau?$ The Sliding window embedding of periodic signals is roundest (so that periodicity score is mazimized) when the window length $d\tau$ satisfies $$ d\tau = \frac{\pi k}{L}\left( \frac{d}{d+1} \right)$$ where $L$ is the number of periods the signal has in $[0,2\pi]$ and $k \in \mathbb{N}.$ **This seems to coincide with our dynamic networks analysis.**


