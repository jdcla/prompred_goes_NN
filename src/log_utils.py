# %load ../../src/log_utils.py
# %%writefile ../../src/log_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""

import json
import datetime as dt
import pandas as pd
import random

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def LogInit(function, model, arg_dict, hyp_string):
    time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
    LOGFILENAME = '{}_{}_{}__({})'.format(time, function, model, random.randint(0,9))
    RESULTLOG = '../logs/result_logger/'+LOGFILENAME
    
    MAINLOG = '../logs/log.txt'
    output = "\n\nSTARTED {}\n\thyper-parameters: {}\n\targuments: {}".format(LOGFILENAME, hyp_string, arg_dict)
    with open(MAINLOG, 'a') as f:   f.write(output)
    f.close()
    print(output)
    return LOGFILENAME, MAINLOG, RESULTLOG, time
    
def LogWrap(MAINLOG, RESULTLOG, results, hyp_string, repeat, repeats):

    filename = "{}_{}.txt".format(RESULTLOG, repeat)
    results.to_csv(filename,index=False)
    
    line_prepender(filename, "##" + hyp_string)
    
        
    if repeat+1 == repeats:
        outputWrap = '\n...FINISHED'
        with open(MAINLOG, 'a') as f:
            f.write(outputWrap)
        print(outputWrap)
        f.close()

                
