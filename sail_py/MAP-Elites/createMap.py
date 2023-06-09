# createMap - Defines map struct and feature space cell divisions
# 
#  Syntax:  [map, edges] = createMap(featureResolution, genomeLength)
# 
#  Inputs:
#     featureResolution - [1XN] - Number of bins in N dimensions
#     featureResolution - [1X1] - Length of genome
#     featureResolution - cell  - Strings with name of additional value
# 
#  Outputs:
#     map  - struct with [M(1) X M(2)...X M(N)] matrices for fitness, etc
#        edges               - {1XN} cell of partitions for each dimension
#        fitness, drag, etc  - [M(1) X M(2) X M(N)] matrices of scalars
#        genes               - [M(1) X M(2) X M(N) X genomeLength]
# 
#  Example: 
#    map = createMap([10 5], 3); % 10 X 5 map of genomes with 3 parameters
#    OR
#    extraMapValues = {'cD','cL'};
#    map = createMap(d.featureRes, d.dof, extraMapValues)
# 


import numpy as np

def createMap(featureResolution, genomeLength, *extraMapValues):
    map = {}
    edges = []

    for res in featureResolution:
        edges.append(np.linspace(0, 1, res + 1))
    map['edges'] = edges

    blankMap = np.full(featureResolution, np.nan)
    map['fitness'] = blankMap
    map['genes'] = np.tile(blankMap[..., np.newaxis], (1, 1, genomeLength))

    if extraMapValues:
        for value in extraMapValues[0]:
            map[value] = blankMap.copy()

    return map, edges
