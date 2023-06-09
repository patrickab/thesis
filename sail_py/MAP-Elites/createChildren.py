# createChildren - produce new children through mutation of map elite
# 
#  Syntax:  children = createChildren(map,nChildren,p,d)
# 
#  Inputs:
#    map         - Population struct
#     .fitness
#     .genes
#    nChildren - number of children to create
#    p           - SAIL hyperparameter struct
#     .mutSigma      - sigma of gaussian mutation applied to children
#    d           - Domain description struct
#     .dof           - Degrees of freedom (genome length)
# 
#  Outputs:
#    children - [nChildren X genomeLength] - new solutions
# 
# 

import numpy as np

#------------- BEGIN CODE --------------  
    # Remove empty bins from parent pool
    parentPool = map['genes'].reshape((-1, d['dof']))
    parentPool = parentPool[~np.isnan(parentPool[:, 0]), :]
    
    # Choose parents and create mutation
    parents = parentPool[np.random.randint(0, parentPool.shape[0], nChildren), :]
    mutation = np.random.randn(nChildren, d['dof']) * p['mutSigma']
    
    # Apply mutation
    children = parents + mutation
    children = np.clip(children, 0, 1)

    return children
# ------------- END OF CODE --------------