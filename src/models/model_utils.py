# %load ../../src/models/model_utils.py
# %%writefile ../../src/models/model_utils.py
"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""

import numpy as np
import pandas as pd
import math
import tensorflow as tf
from sklearn import metrics


def CalculateAUC(pred, true):
    """ 
    Calculate AUC given both the true and predicted labels of a dataset
    
    INPUT
    ------   
    pred: np.array
        Predicted labels
        
    true: np.array
        True labels
        
    OUTPUT
    -------   
    auc: scalar
        AUC
    """
    
    fpr, tpr, tresholds = metrics.roc_curve(true, pred)
    auc = metrics.auc(fpr,tpr)
    
    return auc
    

def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = variable_on_cpu(name, shape, 
                           tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        w_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', w_decay)
    return var

def par_conv_split(x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
                   w_out_decay, num_classes=2, padding=False, extra_layer=False):
    
    x_image = tf.reshape(x, [-1,1,50,4])
    padding = motif_length//2 if padding is True else 0
    pool_list = [] 
    par_conv = math.ceil(50/motif_length)
    for conv in range(par_conv):
        i_start = (50-(conv+1)*motif_length)-padding
        i_end = (50-conv*motif_length)+padding
        if i_start < 0:
            i_start = 0
        x_sub_image_length = len(range(i_start,i_end))
        x_sub_image = x_image[:,:,i_start:i_end,:]
        with tf.variable_scope('conv{}'.format(conv)) as scope:    
            kernel = variable_with_w_decay('weights',
                                             shape=[1, motif_length, 4, motifs],
                                             stddev=stdev,
                                             wd=w_decay)
            conv_unit = tf.nn.conv2d(x_sub_image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = variable_on_cpu('biases', [motifs], tf.constant_initializer(0.001))
            pre_activation = tf.nn.bias_add(conv_unit, biases)
            conv_act = tf.nn.relu(pre_activation, name=scope.name)

        with tf.variable_scope('pool{}'.format(conv)) as scope:
            pool_unit = tf.nn.max_pool(conv_act, ksize=[1, 1, x_sub_image_length, 1], 
                        strides=[1, 1, x_sub_image_length, 1], padding='SAME')
            pool_flat = tf.reshape(pool_unit, [-1, motifs])
            pool_list.append(pool_flat)
            
    layer2 = tf.reshape(tf.concat(pool_list,1),[-1, motifs*par_conv])
    
    if extra_layer is True:
        with tf.variable_scope('fully_connected'):
            weights = variable_with_weight_decay('weights', shape=[motifs*par_conv, motifs*par_conv],
                                              stddev=stdev_out, wd=w_out_decay)
            biases = variable_on_cpu('biases', motifs*par_conv, tf.constant_initializer(0.1))
            layer2 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    with tf.variable_scope('out') as scope:
        weights = variable_with_w_decay('weights', shape=[motifs*par_conv, num_classes],
                                          stddev=stdev_out, wd=w_out_decay)
        biases = variable_on_cpu('biases', num_classes, tf.constant_initializer(0))
        softmax_linear = tf.nn.sigmoid(tf.matmul(layer2, weights) + biases)

        return softmax_linear

def conv_network(x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
                 w_out_decay, single_pool=True, num_classes=2, padding=False,
                 extra_layer=False):
    
    x_image = tf.reshape(x, [-1,1,50,4])
    padding = motif_length//2 if padding is True else 0
    pool_list = [] 
    par_conv = math.ceil(50/motif_length)
    flat_layer = (motifs*(math.ceil(50/motif_length)))

    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights',
                                             shape=[1, motif_length, 4, motifs],
                                             stddev=stdev,
                                             wd=w_decay)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [motifs], tf.constant_initializer(0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
    if single_pool == True:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 50, 1], strides=[1, 1, 50, 1],
                             padding='SAME', name='pool1')
        layer2 = tf.reshape(pool1, [-1, motifs])
        full_connect = motifs
    if single_pool == False:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, motif_length, 1], strides=[1, 1, motif_length, 1],
                             padding='SAME', name='pool1')
        layer2 = tf.reshape(pool1, [-1, flat_layer])
        full_connect = flat_layer
    
    if extra_layer is True:
        with tf.variable_scope('fully_connected'):
            weights = variable_with_weight_decay('weights', shape=[full_connect, full_connect],
                                              stddev=stdev_out, wd=w_out_decay)
            biases = variable_on_cpu('biases', full_connect, tf.constant_initializer(0.1))
            fully_connected = tf.nn.relu(tf.matmul(layer2, weights) + biases)
            layer2 = tf.nn.dropout(fully_connected, keep_prob)

    with tf.variable_scope('out') as scope:
        weights = variable_with_weight_decay('weights', [full_connect, num_classes],
                                              stddev=stdev_out, wd=w_out_decay)
        biases = variable_on_cpu('biases', [num_classes],
                                  tf.constant_initializer(0))
        softmax_linear = tf.sigmoid(tf.matmul(layer2, weights) + biases)


    return softmax_linear
    

def SelectModel(model_label, x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
                w_out_decay, num_classes=2, padding=False, extra_layer=False):
    
    if model_label == "MS":
        model = conv_network(x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay,
                             w_out_decay, False, num_classes, padding, extra_layer)
    if model_label == "MS1":
        model = conv_network(x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
                             w_out_decay, True, num_classes, padding, extra_layer)
    
    if model_label == "MS2":
        model = par_conv_split(x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay,
                               w_out_decay, num_classes, padding, extra_layer)
        
    return model
        