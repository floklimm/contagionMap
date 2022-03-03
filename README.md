# contagionMap
Python library that allows the construction of contagion maps from network data.


This code accompanies the paper "Topological data analysis of truncated contagion maps" by Florian Klimm.

This code also allows the construction of non-truncated contagion maps as originally introduced in
Taylor, D., Klimm, F., Harrington, H. A., Kramár, M., Mischaikow, K., Porter, M. A., & Mucha, P. J. (2015). Topological data analysis of contagion maps for examining spreading processes on networks. Nature Communications, 6(1), 1-11.

![embedding example figure](./python/figures/Fig5-embeddingTruncatedContagionMaps.png)


## Prerequisites
- Python (tested for 3.9.7)
- Some Python standard libraries (numpy,networkx,...)
- [Ripser](https://github.com/Ripser/ripser)  for the topological data analysis (persistent homology)

## How-to
The code enables
1. Construction of noisy geoemtric ring lattice networks
2. Computation of (truncated) contagion maps
3. Quantification of barcodes of these contagion maps with persistent homology

The simplest use case is
```Python
import cmap as conmap

# network construction
noisyRL = conmap.constructNoisyRingLattice(numberNodes=400,geometricDegree=6,nongeometricDegree=2)

# contagion map
contagionMap_t03_truncated = conmap.runTruncatedContagionMap(noisyRL,threshold=0.3,numberSteps=20)

# compute ring stability with Ripser
ringStabilityTruncated = conmap.callRipser(contagionMap_t03_truncated)

```

There is a small tutorial Jupyter Notebook in `/python/tutorial.ipynb` that compares a truncated with a full contagion map.

To reproduce the figures in the manuscript, see Jupyter notebooks in the folder `/python`.
