# %load ../../src/feature/feature_utils.py
# %%writefile ../../src/features/feature_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels import robust

def AllocatePromoters(experiment, IDs):
    
    TSS_info = pd.read_csv("../data/external/TSS_info.csv")
    TSS_seq = pd.read_csv("../data/external/TSS_seq.csv")
    mask = (TSS_seq["strand"] == "+") & (TSS_info["sigma_binding"].str.count(experiment[-1]) == 1) & (TSS_seq["conditions"].str.count("E") == 1)
    TSS = TSS_seq.loc[mask,"TSS_position"].values
    positions = [int(id[-8:]) for id in IDs]

    mask_promoters = []
    for position in positions:
        mask_promoters.append(((position+35 <= TSS) & (position+60 > TSS)).any())
    
    return mask_promoters

def augment_sequences(X_batch, Y_batch):
    X_batch_aug = np.copy(X_batch)
    aug_rand = np.random.randint(15)
    X_batch_aug[:,:aug_rand,:] = 0.25
    X_batch_aug, Y_batch_aug = np.concatenate((X_batch,X_batch_aug)), np.concatenate((Y_batch,Y_batch))
    
    return X_batch_aug, Y_batch_aug
    
def BinaryOneHotEncoder(Y_bool):
    hot_array = np.zeros([len(Y_bool), 2], dtype=np.int8)
    for i in range(len(Y_bool)): 
        if Y_bool[i] == True:
            hot_array[i,1]=1 
        else:
            hot_array[i,0]=1
    
    return hot_array


def CreateBalancedTrainTest(X,Y, test_size=0.1):
    Y_0 = Y[Y[:,1]==0]
    Y_1 = Y[Y[:,1]==1]

    X_0 = X[Y[:,1]==0]
    X_1 = X[Y[:,1]==1]

    X_train_0, X_test_0, Y_train_0, Y_test_0= train_test_split(X_0, Y_0, test_size=test_size)
    X_train_1, X_test_1, Y_train_1, Y_test_1= train_test_split(X_1, Y_1, test_size=test_size)

    X_train = np.vstack((X_train_0, X_train_1))
    Y_train = np.vstack((Y_train_0, Y_train_1))
    X_test = np.vstack((X_test_0, X_test_1))
    Y_test = np.vstack((Y_test_0, Y_test_1))

    return X_train, X_test, Y_train, Y_test


def CreateImageFromSequences(sequences, length= 50):

    lib = np.zeros((len(sequences),length, 4))
    index = 0
    cut = 0
    for string in sequences:
        length_seq = len(sequences[index])
        diff = length - length_seq
        if diff<0:
            cut+=1
            pre_map = None
            string = string[-50:]
            length_seq = 50
            diff = 0
        pre_map = np.full(diff,0.25, dtype=np.float16)  
        Amap = np.hstack((pre_map, [(x==y)*1 for (x,y) in zip(string,"A"*length_seq)]))
        Tmap = np.hstack((pre_map, [(x==y)*1 for (x,y) in zip(string,"T"*length_seq)]))
        Cmap = np.hstack((pre_map, [(x==y)*1 for (x,y) in zip(string,"C"*length_seq)]))
        Gmap = np.hstack((pre_map, [(x==y)*1 for (x,y) in zip(string,"G"*length_seq)]))
        image = np.array([Amap,Tmap,Cmap,Gmap], dtype=np.float16)
        

        lib[index,:,:] = np.transpose(image)
        index+=1
    if cut>0:
        print("{} sequences have been cut".format(cut))
    
    return lib

def DetectPeaks(scores, cutoff, smoothing=True, window_len=50):
    peak_index = []
    index_list = []
    if smoothing is True:
        scores = Smooth(scores, window_len=window_len)
    for i,B in enumerate(scores>cutoff):
        if B == True:
            if (scores[i]<scores[i-1] and scores[i]<scores[i-2]) and scores[i-2]>scores[i-3]:
                peak_index.append(i-2)
                
    mask = np.full(len(scores),False)
    mask[peak_index] = True
    
    return peak_index, mask


def Smooth(x,window_len=50,window='hanning'):
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.{}(window_len)'.format(window))
        y=np.convolve(w/w.sum(),s,mode='same')
        
        return y[window_len:-window_len+1]

def LoadValidationData():

    A_raw = pd.read_csv("../data/external/anderson_NN.csv")
    A_X, A_Y = CreateImageFromSequences(A_raw["PROBE_SEQUENCE"]), A_raw["PM"]
    B_raw = pd.read_csv("../data/external/brewster_NN.csv")
    B_X, B_Y = CreateImageFromSequences(B_raw["PROBE_SEQUENCE"]), B_raw["PM"]
    R_raw = pd.read_csv("../data/external/rand_mut_NN.csv")
    R_X, R_Y = CreateImageFromSequences(R_raw["PROBE_SEQUENCE"]), R_raw["PM"]
    M_raw = pd.read_csv("../data/external/mod_mut_NN.csv")
    M_X, M_Y = CreateImageFromSequences(M_raw["PROBE_SEQUENCE"]), M_raw["PM"]
    D_raw = pd.read_csv("../data/external/davis_NN.csv")
    D_X, D_Y = CreateImageFromSequences(D_raw["PROBE_SEQUENCE"]), D_raw["PM"]

    return A_X, A_Y, B_X, B_Y, R_X, R_Y, M_X, M_Y, D_X, D_Y

def LoadDataTSS(path, experiment):
    data_extra = pd.read_csv(path)
    sequences_extra = data_extra["PROBE_SEQUENCE"]
    X_extra = CreateImageFromSequences(sequences_extra)
    Y_extra_raw = data_extra[experiment]
    Y_extra = BinaryOneHotEncoder(Y_extra_raw==1)
    
    return X_extra, Y_extra



def TransformDataSimple(data_ip, data_mock_ip):

    list_mock_ip = []
    list_ip = []

    for datafile in data_ip:
        list_ip.append(pd.read_csv(datafile)["PM"].values)
    for datafile in data_mock_ip:
        list_mock_ip.append(pd.read_csv(datafile)["PM"].values)
            
    datafile = pd.read_csv(data_ip[0])
    sequences = datafile["PROBE_SEQUENCE"].values
    IDs = datafile["PROBE_ID"].values
    
    list_ip = np.vstack(list_ip).T
    list_mock_ip = np.vstack(list_mock_ip).T
    log_list_ip = np.log2(list_ip)
    log_mock_list_ip = np.log2(list_mock_ip)

    median_ip = [np.median(log_list_ip[:,u]) for u in range(np.shape(list_ip)[1])]
    mad_ip = [robust.mad(log_list_ip[:,u]) for u in range(np.shape(list_ip)[1])]
    mock_median_ip = [np.median(log_mock_list_ip[:,u]) for u in range (np.shape(list_ip)[1])]
    mock_mad_ip = [robust.mad(log_mock_list_ip[:,u]) for u in range(np.shape(list_ip)[1])]
    
    ip_norm = np.array([(log_list_ip[:,u]-median_ip[u])/mad_ip[u] for u in range(len(mad_ip))]).T
    mock_ip_norm = np.array([(log_mock_list_ip[:,u]-mock_median_ip[u])/mock_mad_ip[u] for u in range(len(mad_ip))]).T
    
    fold = ip_norm-mock_ip_norm
    mean_fold = np.mean(fold,axis=1)
    
    ip_norm_mean = np.mean(ip_norm, axis=1)
    mock_ip_norm_mean = np.mean(mock_ip_norm, axis=1)
    fold_mean = ip_norm_mean - mock_ip_norm_mean

    sequences_img = CreateImageFromSequences(sequences)

    return  sequences_img, mean_fold, sequences, IDs
