# contagionMap
Python library that allows the construction of contagion maps from network data.


This code accompanies the paper "Topological data analysis of truncated contagion maps" by Florian Klimm.

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
contagionMap_t03_truncated = conmap.runTruncatedContagionMap(noisyRL,threshold=0.3,numberSteps=40)

# visualisation

# 

```


To reproduce the figures in the manuscript, see Jupyter notebooks in the folder `/python`.
