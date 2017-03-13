# %load ../../src/log_utils.py
# %%writefile ../../src/log_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""

import json
import datetime as dt
import pandas as pd

def LogInit(function, model, localarg):
    time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M')            
    parString = ''.join([num for num in localarg])
    LOGFILENAME = '{}_{}_{}'.format(time, function, model)
    RESULTLOG = '../logs/result_logger/'+LOGFILENAME
    
    MAINLOG = '../logs/log.txt'
    output = '\n\nSTARTED  '+LOGFILENAME + '\n\targuments: '+str(localarg)
    with open(MAINLOG, 'a') as f:
        f.write(output)
    f.close()
    print(output)
    return LOGFILENAME, MAINLOG, RESULTLOG
    
def LogWrap(MAINLOG, RESULTLOG, repeat, results):

    results.to_csv("{}_{}.txt".format(RESULTLOG, repeat),index=False)

    outputWrap = '\n...FINISHED'
    with open(MAINLOG, 'a') as f:
        f.write(outputWrap)
    print(outputWrap)
    f.close()
    
                