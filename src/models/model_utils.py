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

def par_conv_split_duo(x, keep_prob, motifs_1, motifs_2, motif_length_1, motif_length_2, stdev, stdev_out, w_decay, 
                   w_out_decay, pooling=1, num_classes=2, padding=False, extra_layer=False):
    
    x_image = tf.reshape(x, [-1,1,50,4])
    padding = motif_length//2 if padding is True else 0
    pool_list = []
    par_conv_1 = math.ceil(50/motif_length_1)
    par_conv_2 = math.ceil(50/motif_length_2)
    for conv in range(par_conv_1):
        i_start = (50-(conv+1)*motif_length_1)-padding
        i_end = (50-conv*motif_length_1)+padding
        if i_start < 0:
            i_start = 0
        x_sub_image_length = len(range(i_start,i_end))
        x_sub_image = x_image[:,:,i_start:i_end,:]
        with tf.variable_scope('conv{}_sh'.format(conv)) as scope:    
            kernel = variable_with_weight_decay('weights',
                                             shape=[1, motif_length_1, 4, motifs_1],
                                             stddev=stdev,
                                             wd=w_decay)
            conv_unit = tf.nn.conv2d(x_sub_image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = variable_on_cpu('biases', [motifs_1], tf.constant_initializer(0.001))
            pre_activation = tf.nn.bias_add(conv_unit, biases)
            conv_act = tf.nn.relu(pre_activation, name=scope.name)
            
        with tf.variable_scope('pool{}'.format(conv)) as scope:
            if pooling in [-1,2]:
                max_pool_unit = tf.nn.max_pool(conv_act, ksize=[1, 1, x_sub_image_length, 1], 
                            strides=[1, 1, x_sub_image_length, 1], padding='SAME')
                max_pool_flat = tf.reshape(max_pool_unit, [-1, motifs_1])
                pool_list.append(max_pool_flat)
            if pooling in [1,2]:
                avg_pool_unit = tf.nn.avg_pool(conv_act, ksize=[1, 1, x_sub_image_length, 1], 
                                               strides=[1, 1, x_sub_image_length, 1], padding='SAME')
                avg_pool_flat = tf.reshape(avg_pool_unit, [-1, motifs_1])
                pool_list.append(avg_pool_flat)
            
    for conv in range(par_conv_2):
        i_start = (50-(conv+1)*motif_length_2)-padding
        i_end = (50-conv*motif_length_2)+padding
        if i_start < 0:
            i_start = 0
        x_sub_image_length = len(range(i_start,i_end))
        x_sub_image = x_image[:,:,i_start:i_end,:]
        with tf.variable_scope('conv{}_lo'.format(conv)) as scope:    
            kernel = variable_with_weight_decay('weights',
                                             shape=[1, motif_length_2, 4, motifs_2],
                                             stddev=stdev,
                                             wd=w_decay)
            conv_unit = tf.nn.conv2d(x_sub_image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = variable_on_cpu('biases', [motifs_2], tf.constant_initializer(0.001))
            pre_activation = tf.nn.bias_add(conv_unit, biases)
            conv_act = tf.nn.relu(pre_activation, name=scope.name)
        

        with tf.variable_scope('pool{}'.format(conv)) as scope:
            if pooling in [-1,2]:
                max_pool_unit = tf.nn.max_pool(conv_act, ksize=[1, 1, x_sub_image_length, 1], 
                            strides=[1, 1, x_sub_image_length, 1], padding='SAME')
                max_pool_flat = tf.reshape(max_pool_unit, [-1, motifs_2])
                pool_list.append(max_pool_flat)
            if pooling in [1,2]:
                avg_pool_unit = tf.nn.avg_pool(conv_act, ksize=[1, 1, x_sub_image_length, 1], 
                                               strides=[1, 1, x_sub_image_length, 1], padding='SAME')
                avg_pool_flat = tf.reshape(avg_pool_unit, [-1, motifs_2])
                pool_list.append(avg_pool_flat)
            
    num_pool_values = ((motifs_1*par_conv_1)+(motifs_2*par_conv_2))*abs(pooling)
    layer2 = tf.reshape(tf.concat(pool_list, 1), [-1, num_pool_values])
    
    if extra_layer is True:
        with tf.variable_scope('fully_connected'):
            weights = variable_with_weight_decay('weights', shape=[num_pool_values, num_pool_values],
                                              stddev=stdev_out, wd=w_out_decay)
            biases = variable_on_cpu('biases', num_pool_values, tf.constant_initializer(0.1))
            layer2 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    with tf.variable_scope('out') as scope:
        weights = variable_with_weight_decay('weights', shape=[num_pool_values, num_classes],
                                          stddev=stdev_out, wd=w_out_decay)
        biases = variable_on_cpu('biases', num_classes, tf.constant_initializer(0))
        softmax_linear = tf.nn.sigmoid(tf.matmul(layer2, weights) + biases)

        return softmax_linear

def par_conv_split(x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
                   w_out_decay, pooling=1, num_classes=2, padding=False, extra_layer=False):
    
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
            kernel = variable_with_weight_decay('weights',
                                             shape=[1, motif_length, 4, motifs],
                                             stddev=stdev,
                                             wd=w_decay)
            conv_unit = tf.nn.conv2d(x_sub_image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = variable_on_cpu('biases', [motifs], tf.constant_initializer(0.001))
            pre_activation = tf.nn.bias_add(conv_unit, biases)
            conv_act = tf.nn.relu(pre_activation, name=scope.name)

        with tf.variable_scope('pool{}'.format(conv)) as scope:
            if pooling in [-1,2]:
                max_pool_unit = tf.nn.max_pool(conv_act, ksize=[1, 1, x_sub_image_length, 1], 
                            strides=[1, 1, x_sub_image_length, 1], padding='SAME')
                max_pool_flat = tf.reshape(max_pool_unit, [-1, motifs])
                pool_list.append(max_pool_flat)
            if pooling in [1,2]:
                avg_pool_unit = tf.nn.avg_pool(conv_act, ksize=[1, 1, x_sub_image_length, 1], 
                                               strides=[1, 1, x_sub_image_length, 1], padding='SAME')
                avg_pool_flat = tf.reshape(avg_pool_unit, [-1, motifs])
                pool_list.append(avg_pool_flat)
                
    num_pool_values = motifs*par_conv*abs(pooling)
    layer2 = tf.reshape(tf.concat(pool_list, 1), [-1, num_pool_values])
    
    if extra_layer is True:
        with tf.variable_scope('fully_connected'):
            weights = variable_with_weight_decay('weights', shape=[num_pool_values, num_pool_values],
                                              stddev=stdev_out, wd=w_out_decay)
            biases = variable_on_cpu('biases', motifs*par_conv, tf.constant_initializer(0.1))
            layer2 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    with tf.variable_scope('out') as scope:
        weights = variable_with_weight_decay('weights', shape=[num_pool_values, num_classes],
                                          stddev=stdev_out, wd=w_out_decay)
        biases = variable_on_cpu('biases', num_classes, tf.constant_initializer(0))
        softmax_linear = tf.nn.sigmoid(tf.matmul(layer2, weights) + biases)

        return softmax_linear

def conv_network(x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
                 w_out_decay, single_pool=True, pooling=1, num_classes=2, padding=False,
                 extra_layer=False):
    
    x_image = tf.reshape(x, [-1,1,50,4])
    padding = motif_length//2 if padding is True else 0
    motifs = math.ceil(50/motif_length)*motifs
    if single_pool:
        num_pool_values = motifs*abs()
        pool_stride = 50
    else:
        num_pool_values = (motifs*(math.ceil(50/motif_length)))
        pool_stride = motif_length

    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights',
                                             shape=[1, motif_length, 4, motifs],
                                             stddev=stdev,
                                             wd=w_decay)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [motifs], tf.constant_initializer(0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        
        if pooling in [-1,2]:
            max_pool_unit = tf.nn.max_pool(conv_act, ksize=[1, 1, pool_stride, 1], 
                        strides=[1, 1, pool_stride, 1], padding='SAME')
            max_pool_flat = tf.reshape(max_pool_unit, [-1, num_pool_values])
        if pooling in [1,2]:
            avg_pool_unit = tf.nn.avg_pool(conv_act, ksize=[1, 1, pool_stride, 1], 
                                           strides=[1, 1, pool_stride, 1], padding='SAME')
            avg_pool_flat = tf.reshape(avg_pool_unit, [-1, num_pool_values])
            
        num_pool_values = num_pool_values*abs(pooling)
        layer2 = tf.reshape(tf.concat([max_pool_flat, avg_pool_flat], axis=0), [-1, num_pool_values])
    
    if extra_layer is True:
        with tf.variable_scope('fully_connected'):
            weights = variable_with_weight_decay('weights', shape=[num_pool_values, num_pool_values],
                                              stddev=stdev_out, wd=w_out_decay)
            biases = variable_on_cpu('biases', num_pool_values, tf.constant_initializer(0.1))
            fully_connected = tf.nn.relu(tf.matmul(layer2, weights) + biases)
            layer2 = tf.nn.dropout(fully_connected, keep_prob)

    with tf.variable_scope('out') as scope:
        weights = variable_with_weight_decay('weights', [num_pool_values, num_classes],
                                              stddev=stdev_out, wd=w_out_decay)
        biases = variable_on_cpu('biases', [num_classes],
                                  tf.constant_initializer(0))
        softmax_linear = tf.sigmoid(tf.matmul(layer2, weights) + biases)


    return softmax_linear
    

def SelectModel(model_label, x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
                w_out_decay, pooling, num_classes=2, padding=False, extra_layer=False):
    
    if model_label == "MS1":
        model = conv_network(x, keep_prob, motifs[0], motif_length[0], stdev, stdev_out, w_decay,
                             w_out_decay, False, pooling, num_classes, padding, extra_layer)
    if model_label == "MS2":
        model = conv_network(x, keep_prob, motifs[0], motif_length[0], stdev, stdev_out, w_decay, 
                             w_out_decay, True, pooling, num_classes, padding, extra_layer)
    
    if model_label == "MS3":
        model = par_conv_split(x, keep_prob, motifs[0], motif_length[0], stdev, stdev_out, w_decay,
                               w_out_decay, pooling, num_classes, padding, extra_layer)
    if model_label == "MS4":
        model = par_conv_split_duo(x, keep_prob, motifs[0], motifs[1], motif_length[0], motif_length[1], stdev, stdev_out, w_decay,
                               w_out_decay, pooling, num_classes, padding, extra_layer)
        
    return model
        