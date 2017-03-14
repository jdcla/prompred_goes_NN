# %load ../../src/log_utils.py
# %%writefile ../../src/log_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""

import json
import datetime as dt
import pandas as pd

def LogInit(function, model, arg_dict, hyp_string):
    time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
    LOGFILENAME = '{}_{}_{}'.format(time, function, model)
    RESULTLOG = '../logs/result_logger/'+LOGFILENAME
    
    MAINLOG = '../logs/log.txt'
    output = "\n\nSTARTED {}\n\thyper-parameters: {}\n\targuments: {}".format(LOGFILENAME, hyp_string, arg_dict)
    with open(MAINLOG, 'a') as f:   f.write(output)
    f.close()
    print(output)
    return LOGFILENAME, MAINLOG, RESULTLOG
    
def LogWrap(MAINLOG, RESULTLOG, results, repeat, repeats):

    results.to_csv("{}_{}.txt".format(RESULTLOG, repeat),index=False)
    
    if repeat+1 == repeats:
        outputWrap = '\n...FINISHED'
        with open(MAINLOG, 'a') as f:
            f.write(outputWrap)
        print(outputWrap)
        f.close()

                