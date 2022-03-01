# library for contagion maps

import networkx as nx
from random import sample
import numpy as np
import copy
from scipy.spatial import distance
from scipy import stats 
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# not needed anymore (doesn't provide a proper speed up)
from numba import jit, prange



# A) Network construction

# check whether nodes are adjacet in a ring lattice
def adjacent_edges(nodes, halfk): 
    n = len(nodes) 
    for i, u in enumerate(nodes): 
        for j in range(i+1, i+halfk+1): 
            v = nodes[j % n] 
            yield u, v

# construct a regular ring lattice network
def make_ring_lattice(numberNodes, degree): 
    G = nx.Graph() 
    nodes = range(numberNodes)
    G.add_nodes_from(nodes)
    # compute the positions of the nodes on a circle
    nodeAngle = np.arange(0,2*np.pi,2*np.pi/numberNodes)
    x = np.sin(nodeAngle)
    y = np.cos(nodeAngle)
    # create a position dictionary
    pos = {}
    for i in np.arange(0,numberNodes):
        pos[i] = (x[i],y[i])
    nx.set_node_attributes(G, pos, 'coord')

    # construct edges
    G.add_edges_from(adjacent_edges(nodes, degree//2), type="geometric") 
    return G



# add random edges
def add_non_geometric_edges(G,edgesPerNode,type='semirandom'):
    H = G.copy()
    # add same number of edges outgoing each node
    if type == 'semirandom':
        for node in range(H.number_of_nodes()):
            edgesCreated = 0
            # try random nodes to connect to and break after succesfully created enough (this only works well for sparse graphs)
            random_node = sample(H.nodes(),1)[0]
            if H.has_edge(node, random_node) is False:
                H.add_edge(node,random_node, type="nongeometric")
                edgesCreated = edgesCreated + 1 # increase count
            if edgesCreated == (edgesPerNode*H.number_of_nodes())/2:
                break
    elif type == 'regular':
        # add the same number of edges for each node
        # the stubs are each node as often as edges per node
        stubList = [list(H.nodes()) for i in np.arange(edgesPerNode)]
        stubs = [item for sublist in stubList for item in sublist]

        edgesCreated = 0 
        failedConstructionAttempt = 0
        while edgesCreated<(edgesPerNode*G.number_of_nodes()/2):
            # get two random stubs
            random_stubs = sample(stubs,2)
            # check if you can create this edge 
            if (H.has_edge(random_stubs[0], random_stubs[1]) is False) & (random_stubs[0] != random_stubs[1]):
                # create it
                H.add_edge(random_stubs[0],random_stubs[1], type="nongeometric")
                # remove the nodes from the stub list
                stubs.remove(random_stubs[0])
                stubs.remove(random_stubs[1])
                edgesCreated = edgesCreated + 1 # increase count
                failedConstructionAttempt = 0
            else:
                # if failed 100 times repeatedly, restart
                failedConstructionAttempt = failedConstructionAttempt + 1
                if failedConstructionAttempt > 100:
                    print('failed adding of non-geometric edges: restart')
                    return(None)
    else:
        ## should implement a completely random procedure
        error('not implemented!')
    return(H)    

# construct a noisy geometric ring lattice network
def constructNoisyRingLattice(numberNodes=200,geometricDegree=6,nongeometricDegree=2,type='regular'):
    # 1) construct a ring lattice
    ringLatticeNetwork = make_ring_lattice(numberNodes, geometricDegree)
    # 2) add non-geometric edges
    ringLatticeNetworkWithNoise = None
    while ringLatticeNetworkWithNoise is None:
        ringLatticeNetworkWithNoise = add_non_geometric_edges(ringLatticeNetwork,nongeometricDegree,type=type)
    return(ringLatticeNetworkWithNoise)

# # add random edges
# def add_non_geometric_edges(G,edgesPerNode,type='semirandom'):
#     # add same number of edges outgoing each node
#     if type == 'semirandom':
#         for node in range(G.number_of_nodes()):
#             edgesCreated = 0
#             # try random nodes to connect to and break after succesfully created enough (this only works well for sparse graphs)
#             random_node = sample(G.nodes(),1)[0]
#             if G.has_edge(node, random_node) is False:
#                 G.add_edge(node,random_node, type="nongeometric")
#                 edgesCreated = edgesCreated + 1 # increase count
#             if edgesCreated == (edgesPerNode*G.number_of_nodes())/2:
#                 break
#     elif type == 'regular':
#         # add the same number of edges for each node
#         # the stubs are each node as often as edges per node
#         stubList = [list(G.nodes()) for i in np.arange(edgesPerNode)]
#         stubs = [item for sublist in stubList for item in sublist]

#         edgesCreated = 0 
#         failedConstructionAttempt = 0
#         while edgesCreated<(edgesPerNode*G.number_of_nodes()/2):
#             # get two random stubs
#             random_stubs = sample(stubs,2)
#             # check if you can create this edge 
#             if (G.has_edge(random_stubs[0], random_stubs[1]) is False) & (random_stubs[0] != random_stubs[1]):
#                 # create it
#                 G.add_edge(random_stubs[0],random_stubs[1], type="nongeometric")
#                 # remove the nodes from the stub list
#                 stubs.remove(random_stubs[0])
#                 stubs.remove(random_stubs[1])
#                 edgesCreated = edgesCreated + 1 # increase count
#                 failedConstructionAttempt = 0
#             else:
#                 # if failed 100 times repeatedly, restart
#                 failedConstructionAttempt = failedConstructionAttempt + 1
#                 if failedConstructionAttempt > 100:
#                     return()
#     else:
#         ## should implement a completely random procedure
#         error('not implemented!')
#     return(G)    

# # construct a noisy geometric ring lattice network
# def constructNoisyRingLattice(numberNodes=200,geometricDegree=6,nongeometricDegree=2,type='regular'):
#     # 1) construct a ring lattice
#     ringLatticeNetwork = make_ring_lattice(numberNodes, geometricDegree)
#     # 2) add non-geometric edges
#     add_non_geometric_edges(ringLatticeNetwork,nongeometricDegree,type=type)
#     return(ringLatticeNetwork)

# drawing geometrically
def drawGeometricNetwork(G,node_size=10):
    # get node positions
    posDict = nx.get_node_attributes(G, 'coord')
    # actual plotting
    # 1) draw geometric edges
    geometricEdges = [ (u,v) for u,v,d in G.edges(data=True) if d['type'] == 'geometric'] 
    nx.draw_networkx_edges(G, pos=posDict,edgelist = geometricEdges,edge_color = '#d64161')
    # 2) draw non-geometric edges
    nongeometricEdges = [ (u,v) for u,v,d in G.edges(data=True) if d['type'] == 'nongeometric'] 
    nx.draw_networkx_edges(G, pos=posDict,edgelist = nongeometricEdges,edge_color = '#6b5b95')
    # 3) afterwards nodes
    nx.draw_networkx_nodes(G, pos=posDict,node_color='#25a9ff',node_size=node_size)

# drawing geometrically with the activation times
def plotGraphWithActivationTimes(graph,activationTimes,vmax=None):
    if vmax is None:
        vmax = np.max(activationTimes)

    nodePositions = nx.get_node_attributes(graph,'coord')


    # 1) draw geometric edges
    geometricEdges = [ (u,v) for u,v,d in graph.edges(data=True) if d['type'] == 'geometric'] 
    nx.draw_networkx_edges(graph, pos=nodePositions,edgelist = geometricEdges,edge_color = '#696969',alpha=0.2)
    # 2) draw non-geometric edges
    nongeometricEdges = [ (u,v) for u,v,d in graph.edges(data=True) if d['type'] == 'nongeometric'] 
    nx.draw_networkx_edges(graph, pos=nodePositions,edgelist = nongeometricEdges,edge_color = '#696969',alpha=0.2)
    # 3) Draw edges between seed nodes
    seedNodes = np.where(activationTimes==0)[0]
    seedEdges=[]
    for i in seedNodes:
        for j in seedNodes:
            if graph.has_edge(i,j):
                seedEdges.append((i,j))
    nx.draw_networkx_edges(graph, pos=nodePositions,edgelist = seedEdges,edge_color = '#008000',alpha=0.8)

    # 1) plot seed nodes
    #seedNodes = np.where(activationTimes==0)[0]
    nodesSeedPlot = nx.draw_networkx_nodes(graph, pos=nodePositions,nodelist= seedNodes, node_size=1,node_color='#008000')
    # 2) Plot nodes that were never activated
    unctivatedNodes = np.array([i[0] for i in np.argwhere(np.isnan(activationTimes))])
    nodesUnactivatedPlot = nx.draw_networkx_nodes(graph, pos=nodePositions,nodelist= unctivatedNodes, node_size=1,node_color='#808080')
    # 3) Plot all other nodes
    otherNodes = list(set(np.arange(activationTimes.shape[0])).difference(set(unctivatedNodes)).difference(set(seedNodes)))
    nodes = nx.draw_networkx_nodes(graph, pos=nodePositions,nodelist= otherNodes,node_size=1,node_color=activationTimes[otherNodes],vmin=0,vmax=vmax)
    # Add edges?
    
def plotTwoDimensionalEmbedding(contagionMap,activationTimes,vmax=None):
    if vmax is None:
        vmax = np.max(activationTimes)

    # First two dimensions of the PCA
    pca = PCA(n_components=2)
    X_projected = pca.fit_transform(contagionMap)

    # plotting of the nodes
    
    #plt.scatter(x=X_projected[:,0],y=X_projected[:,1],s=10,c=activationTimes)

    # 1) plot seed nodes
    seedNodes = np.where(activationTimes==0)[0]
    plt.scatter(x=X_projected[seedNodes,0],y=X_projected[seedNodes,1],s=10,c='#008000')
    # 2) Plot nodes that were never activated
    unctivatedNodes = np.array([i[0] for i in np.argwhere(np.isnan(activationTimes))])
    #unctivatedNodes = np.where(activationTimes>vmax)[0]
    if unctivatedNodes.size > 0: # if not empty
        plt.scatter(x=X_projected[unctivatedNodes,0],y=X_projected[unctivatedNodes,1],s=10,c='#808080')
    
    # # 3) Plot all other nodes
    otherNodes = list(set(np.arange(contagionMap.shape[0])).difference(set(unctivatedNodes)).difference(set(seedNodes)))
    plt.scatter(x=X_projected[otherNodes,0],y=X_projected[otherNodes,1],s=10,c=activationTimes[otherNodes],vmin=0,vmax=vmax)

   


# dynamical processes
# def simulateWattsThresholdModel(network,initialCondition,threshold=0.1,numberSteps=np.Inf):
#     # simulates the Watts threshold model for a single starting condition
#     # initialCondition = index of initially activated nodes
#     # threshold =  at the moment only homogenous threshold implemented


#     # vector with activation time of each node
#     activationTime =np.empty((network.number_of_nodes()))
#     activationTime[:] = np.NaN
#     activationTime[initialCondition] = 0
#     # run of a certain number of steps
#     if numberSteps>0:
#         timeStep = 0
#         while timeStep<numberSteps:
#             activationTimeLast = copy.deepcopy(activationTime)
#             timeStep = timeStep + 1 # increase step
#             #### This will be possible to speed up by checking only a subset of the nodes (i.e., those that are neighbouring just activated ones ####
#             for j in np.arange(network.number_of_nodes()):
#                 # only check nodes that are not active yet
#                 if np.isnan(activationTimeLast[j]):
#                     # compute number of neighbours that are active
#                     neighbours_j = list(network.neighbors(j)) ## maybe to this outside of the loop once for all nodes
#                     degree = float(len(neighbours_j))
#                     numberActiveNeighbours = float(np.count_nonzero(~np.isnan(activationTimeLast[neighbours_j])))
#                     if ((numberActiveNeighbours/degree)>threshold):
#                         # activate the node
#                         activationTime[j] = timeStep
#             # break if steady state is reached 
#             if np.array_equal(activationTimeLast, activationTime):
#                 break
#     else:
#         # run until a steady state is reached
#         error('not implemented')
#     return(activationTime)

def simulateWattsThresholdModel(network,initialCondition,threshold=0.1,numberSteps=np.Inf):
    # simulates the Watts threshold model for a single starting condition
    # initialCondition = index of initially activated nodes
    # threshold =  at the moment only homogenous threshold implemented


    # vector with activation time of each node
    activationTime =np.empty((network.number_of_nodes()))
    activationTime[:] = np.NaN
    activationTime[initialCondition] = 0
    # run of a certain number of steps
    if numberSteps>0:
        timeStep = 0
        # in the first step we check all nodes (it would be possible to check only neighbours of seed nodes)
        nodesToCheck = np.arange(network.number_of_nodes())
        while timeStep<numberSteps:
            activationTimeLast = copy.deepcopy(activationTime)
            timeStep = timeStep + 1 # increase step
            #### This will be possible to speed up by checking only a subset of the nodes (i.e., those that are neighbouring just activated ones ####
            nodesToCheckNext = []
            for j in nodesToCheck:
                # only check nodes that are not active yet
                if np.isnan(activationTimeLast[j]):
                    # compute number of neighbours that are active
                    neighbours_j = list(network.neighbors(j)) ## maybe to this outside of the loop once for all nodes
                    degree = float(len(neighbours_j))
                    numberActiveNeighbours = float(np.count_nonzero(~np.isnan(activationTimeLast[neighbours_j])))
                    if ((numberActiveNeighbours/degree)>threshold):
                        # activate the node
                        activationTime[j] = timeStep
                        # add its neighbours to be checked next time
                        nodesToCheckNext.append(neighbours_j)

            # unlist this list of lists
            nodesToCheck = [item for sublist in nodesToCheckNext for item in sublist]
            nodesToCheck = set(nodesToCheck) # make unique and only check those that are not activated yet
            unactivatedNodes = set(np.where(np.isnan(activationTime))[0])
            nodesToCheck = list(nodesToCheck.intersection(unactivatedNodes))

            # break if steady state is reached 
            if np.array_equal(activationTimeLast, activationTime,equal_nan=True):
                break
    else:
        # run until a steady state is reached
        error('not implemented')
    return(activationTime)


# @jit(parallel=True)
# def runTruncatedContagionMap(network,threshold=0.1,numberSteps=np.Inf,symmetric=True):
#     # run the  truncated contagion map by simulating each of the Watts' threshold models
#     # intialise the output matrix
#     contagionMap = np.zeros((network.number_of_nodes(),network.number_of_nodes()))
#     # run for each node the Watts' threshold model
#     for i in prange(network.number_of_nodes()):
#         # find cluster seeding as node itself plus all neighbours
#         seeding = list(network.neighbors(i))
#         seeding.append(i)
#         # run the Watts' model
#         contagionMap[:,i] = simulateWattsThresholdModel(network,seeding,threshold=threshold,numberSteps=numberSteps)
#     # replace the never activated with 2(number of nodes)
#     contagionMap = np.nan_to_num(contagionMap,nan=2*network.number_of_nodes()+1)

#     # symmetrise the contagion map
#     if symmetric:
#         contagionMap = contagionMap + np.transpose(contagionMap)

#     return(contagionMap)

# def runTruncatedContagionMap(network,threshold=0.1,numberSteps=np.Inf,symmetric=True):
#     # run the  truncated contagion map by simulating each of the Watts' threshold models
#     # intialise the output matrix
#     contagionMap = np.zeros((network.number_of_nodes(),network.number_of_nodes()))
#     # run for each node the Watts' threshold model
#     for i in np.arange(network.number_of_nodes()):
#         # find cluster seeding as node itself plus all neighbours
#         seeding = list(network.neighbors(i))
#         seeding.append(i)
#         # run the Watts' model
#         contagionMap[:,i] = simulateWattsThresholdModel(network,seeding,threshold=threshold,numberSteps=numberSteps)
#     # replace the never activated with 2(number of nodes)
#     contagionMap = np.nan_to_num(contagionMap,nan=2*network.number_of_nodes()+1)

#     # symmetrise the contagion map
#     if symmetric:
#         contagionMap = contagionMap + np.transpose(contagionMap)

#     return(contagionMap)

def runTruncatedContagionMap(network,threshold=0.1,numberSteps=np.Inf,symmetric=True):
    # run the  truncated contagion map by simulating each of the Watts' threshold models
    # intialise the output matrix
    contagionMap = np.zeros((network.number_of_nodes(),network.number_of_nodes()))
    # run for each node the Watts' threshold model
    for i in np.arange(network.number_of_nodes()):
        # find cluster seeding as node itself plus all neighbours
        seeding = list(network.neighbors(i))
        seeding.append(i)
        # run the Watts' model
        contagionMap[:,i] = simulateWattsThresholdModel(network,seeding,threshold=threshold,numberSteps=numberSteps)
    # # replace the never activated with 2(number of nodes)
    # contagionMap = np.nan_to_num(contagionMap,nan=2*network.number_of_nodes()+1)
    if np.isinf(numberSteps):
        contagionMap = np.nan_to_num(contagionMap,nan=2*network.number_of_nodes())
    else:
        contagionMap = np.nan_to_num(contagionMap,nan=2*numberSteps)


    # symmetrise the contagion map
    if symmetric:
        contagionMap = contagionMap + np.transpose(contagionMap)

    return(contagionMap)

## Testing of contagion map
# correlation
def computeCorrelationDistances(network,contagionMap,type='Spearman'):

    posDict = nx.get_node_attributes(network, 'coord')
    # go pairwise over nodes and compute distances
    euclideanDistances = []
    contagionMapDistance = []
    for i in np.arange(network.number_of_nodes()):
        position_i = posDict[i]
        for j in np.arange(i+1,network.number_of_nodes()): # only upper part of the matrix
            # a) compute euclidean distances
            position_j = posDict[j]
            distanceOriginalEmbedding = distance.euclidean(position_i, position_j)
            euclideanDistances.append(distanceOriginalEmbedding)
            # b) compute contagion map distances
            distanceContagionEmbedding = distance.euclidean(contagionMap[i,:], contagionMap[j,:])
            contagionMapDistance.append(distanceContagionEmbedding)
    # compute the correlation
    if type=='Spearman':
        correlation = stats.spearmanr(euclideanDistances, contagionMapDistance)

    return(correlation[0])

# # TDA part


def plotPersistenceDiagram(contagionMap,dataIn='./temp/contagionMapPersistenceIntervals.csv'):
    intervals1D = readRipserOutput('./temp/contagionMapPersistenceIntervals.csv')
    # normalise them 
    distanceMatrix = np.linalg.norm(contagionMap - contagionMap[:,None], axis=-1)
    maxdistance = np.max(distanceMatrix)
    intervals1D_normalised = intervals1D/maxdistance
    # compute the length
    intervalsDF = pd.DataFrame(intervals1D, columns=['start','end'])
    intervalsDF['length'] = intervalsDF['end'] - intervalsDF['start']
    #intervalsDF = intervalsDF.sort_values(by='length')

    # find the two largest lengths
    largestLengths = intervalsDF['length'].nlargest(2).values
    print(largestLengths)

    # plotting
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)


    for index, row in intervalsDF.iterrows():

        if row['length'] == largestLengths[0]:
            plt.plot([row['start'],row['end']],[index,index],color='#009aff',linewidth=2)
        elif row['length'] == largestLengths[1]:
            plt.plot([row['start'],row['end']],[index,index],color='#ffa500',linewidth=2)
        else:
            plt.plot([row['start'],row['end']],[index,index],color='#696969')
    
    plt.xlabel('')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylabel('$H_1$')

    #return(fig)
            

def callRipser(contagionMap):
    # write the contagion maps as temporary file
    np.savetxt('./temp/contagionMap.csv', contagionMap, delimiter=',',fmt='%i')
    # call ripser on it
    os.system('./ripser ./temp/contagionMap.csv --format point-cloud   --dim 2 > ./temp/contagionMapPersistenceIntervals.csv')
    # read the ripser output
    # select the 1st dimension intervals, which is what we are interested in
    intervals1D = readRipserOutput('./temp/contagionMapPersistenceIntervals.csv')
    # normalise them 
    distanceMatrix = np.linalg.norm(contagionMap - contagionMap[:,None], axis=-1)
    maxdistance = np.max(distanceMatrix)
    intervals1D_normalised = intervals1D/maxdistance
    # compute the length
    intervalsDF = pd.DataFrame(intervals1D_normalised, columns=['start','end'])
    intervalsDF['length'] = intervalsDF['end'] - intervalsDF['start']
    # compute ring stability
    if len(intervalsDF)==0:
        ringStability = 0
    elif len(intervalsDF)==1:
        largestRingIntervals = intervalsDF['length'].nlargest(1).tolist() # get the longest interval
        ringStability = largestRingIntervals[0] # ring stbaility is just the longest, because it is the only
    else:
        largestRingIntervals = intervalsDF['length'].nlargest(2).tolist() # get the two longest ring itnervals
        ringStability = largestRingIntervals[0] - largestRingIntervals[1] # ring stbaility is their difference
    return(ringStability)


def readRipserOutput(filename):
    intervals=[]
    with open(filename) as f:
        saveLine = False
        lines=f.readlines()
        for line in lines:
            #print(line)
            if saveLine==True:
                # stop saving if next dimension is reached
                if  'persistence intervals in dim 2:' in line:
                    saveLine=False
                else:
                    interval=[float(line.split(',')[0][2:]), float(line.split(',')[1][0:-2]) ]
                    intervals.append(interval)
            # we start saving after this line
            if  'persistence intervals in dim 1:' in line:
                saveLine=True
    return(np.array(intervals))