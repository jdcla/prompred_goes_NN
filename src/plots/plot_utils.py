# %load ../../src/plots/plot_utils.py
# %%writefile ../../src/plots/plot_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def VisualizeSelection(Y, mask, start, stop, sequences, IDs ):
   TSS_seq = pd.read_csv("data/external/TSS_seq.csv")
   TSS_info = pd.read_csv("data/exteral/TSS_info.csv")
   mask_TSS = (TSS_seq["strand"] == "+") & (TSS_info["sigma_binding"].str.count("D") == 1) & (TSS_seq["conditions"].str.count("E") == 1)
   TSS = TSS_seq.loc[mask_TSS,"TSS_position"].values
   positions = [int(id[-8:]) for id in IDs]
   
   fig, ax = plt.subplots(figsize=((end-start)/250,7))
   ax.plot(positions[start:end], Smooth(Y, window_len=50)[start:end], 'b')
   ax.vlines(np.array(positions)[start:end][mask[start:end]],-2,5, 'g')
   for x_value in TSS:
       if x_value > positions[start] and (x_value< positions[end]):
           ax.vlines(x_value,-2,4, 'orange')
   
   return fig, ax
   