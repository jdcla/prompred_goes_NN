# %load ../../src/data/data_utils.py
# %%writefile ../../src/data/data_utils.py

"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""

import numpy as np
import pandas as pd
from statsmodels import robust


def GetDataLocations(sigma):
    """ Helper function for quick access of ChIP-chip data
    sigma: string
        Sigma-factor for which data is loaded
        
    OUTPUT
    -------
    """
    experiments = {"RPOD":3, "RPOS":3, "RNAP":3, "SIGMA":2, "BETA":2}
    
    if sigma in experiments:
        for i in range(experiments[sigma]):
            data_ip = ["../data/processed/{}_EXP_{}_635.extr".format(sigma, u+1) for u in range(experiments[sigma])]
            data_mock_ip = ["../data/processed/{}_EXP_{}_532.extr".format(sigma, u+1) for u in range(experiments[sigma])]
    else:
        return [], []
        
    return data_ip, data_mock_ip

