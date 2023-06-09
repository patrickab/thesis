import botorch

def acquisition_mes(drag, lift, gpModel, candidate_solution):
    """
    python translation of velo_AcquisitionFunc()
    implementing MES instead of UCB
        (with minor modifications)

    Note: 
        simplified input values 
            (drag, lift):               // extract variables from generate_xfouil_output.py
                here:     2x1 matrices
                original: 2xN matrices
            (candidate_solution):
                here:     1xDim matrices
                original: NxDim matrices
                            
        renamed variables
            fitness   -> acq_fitness
            predValue -> acq_behavior  

    Inputs: 
        drag                : (mean, variance)   ->  (Luftwiderstand) ; found in 
        lift                : (mean, variance)   ->  (Auftrieb)
        gpModel             : result from train_gp()
        candidate_solutions : tensor struct of candidate solutions
                              (provided by MAP-Elites line 1 acquisition loop)
    
    Outputs:
        acq_fitness  : float
            Fitness value (lower drag is better)

        acq_behavior : 
            Predicted drag force (mean and variance)
    """

    acq_fitness  = botorch.acquisition.qMaxValueEntropy(gpModel, candidate_solution)
    acq_behavior = [drag[0], lift[0]]

    return acq_fitness, acq_behavior