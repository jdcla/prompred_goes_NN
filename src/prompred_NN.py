import matplotlib
matplotlib.use("Agg")
import sys
import json
import math
import argparse
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import data.data_utils as du
import features.feature_utils as fu
import models.model_utils as mu
import plots.plot_utils as pu
import log_utils as log
import datetime as dt 
from scipy import stats
from sklearn.utils import shuffle


def load_data(experiment, cutoff, lowess):
	path_TSS = "../data/external/promotor_list_exp_growth.csv"
	if experiment == 'all':
		with h5py.File('../data/processed/RPOD_low_Y.h5', 'r') as hf:
			Y_R = hf['chip'][:]
		with h5py.File('../data/processed/RPOD_low_X.h5', 'r') as hf:
			X_R = hf['chip'][:]*1
		IDs_R = pd.read_csv('../data/processed/RPOD_ID_list',header=None).values.ravel()
		X_TSS, Y_TSS = fu.LoadDataTSS(path_TSS, 'RPOD')
		mask_TSS = fu.AllocatePromoters(experiment, IDs_R)
		_, mask_peak_R = fu.DetectPeaks(Y_R,cutoff, smoothing=True)
		mask = np.any([mask_peak_R,mask_TSS],axis=0)
		Y_masked = fu.BinaryOneHotEncoder(mask)
		Y_full = np.vstack((Y_masked, Y_TSS))
		X_full = np.vstack((X_R, X_TSS))
		with h5py.File('../data/processed/SIGMA_low_Y.h5', 'r') as hf:
			Y_S = hf['chip'][:]
		with h5py.File('../data/processed/SIGMA_low_X.h5', 'r') as hf:
			X_S = hf['chip'][:]*1
		_, mask_peak_S = fu.DetectPeaks(Y_S,cutoff, smoothing=True)
		Y_masked_S = fu.BinaryOneHotEncoder(mask_peak_S)
		with h5py.File('../data/processed/BETA_low_Y.h5', 'r') as hf:
			Y_B = hf['chip'][:]
		with h5py.File('../data/processed/BETA_low_X.h5', 'r') as hf:
			X_B = hf['chip'][:]*1
		_, mask_peak_B = fu.DetectPeaks(Y_B,cutoff, smoothing=True)
		Y_masked_B = fu.BinaryOneHotEncoder(mask_peak_B)
		X_full = np.vstack((X_full, X_S, X_B))
		Y_full = np.vstack((Y_full, Y_masked_S, Y_masked_B))

	elif experiment == 'SIGMA':
		with h5py.File('../data/processed/SIGMA_low_Y.h5', 'r') as hf:
			Y_S = hf['chip'][:]
		with h5py.File('../data/processed/SIGMA_low_X.h5', 'r') as hf:
			X_S = hf['chip'][:]*1
		_, mask_peak_S = fu.DetectPeaks(Y_S,cutoff, smoothing=True)
		Y_masked_S = fu.BinaryOneHotEncoder(mask_peak_S)
		with h5py.File('../data/processed/BETA_low_Y.h5', 'r') as hf:
			Y_B = hf['chip'][:]
		with h5py.File('../data/processed/BETA_low_X.h5', 'r') as hf:
			X_B = hf['chip'][:]*1
		_, mask_peak_B = fu.DetectPeaks(Y_B,cutoff, smoothing=True)
		Y_masked_B = fu.BinaryOneHotEncoder(mask_peak_B)
		X_full = np.vstack((X_S, X_B))
		Y_full = np.vstack((Y_masked_S, Y_masked_B))

	else:
		if lowess:
			with h5py.File('../data/processed/RPOD_low_Y.h5', 'r') as hf:
				Y = hf['chip'][:]
			with h5py.File('../data/processed/RPOD_low_X.h5', 'r') as hf:
				X = hf['chip'][:]*1
			IDs = pd.read_csv('../data/processed/RPOD_ID_list',header=None).values.ravel()
		else:
			data_ip, data_mock_ip = du.GetDataLocations(experiment)
			X, Y, sequences, IDs = fu.TransformDataSimple(data_ip, data_mock_ip)
		X_TSS, Y_TSS = fu.LoadDataTSS(path_TSS, experiment)
		_, mask_peak = fu.DetectPeaks(Y,cutoff, smoothing=True)
		mask_TSS = fu.AllocatePromoters(experiment, IDs)
		mask = np.any([mask_peak,mask_TSS],axis=0)
		Y_masked = fu.BinaryOneHotEncoder(mask)
		Y_full = np.vstack((Y_masked, Y_TSS))
		X_full = np.vstack((X, X_TSS))
	
	return X_full, Y_full 

def load_model(model_label, ratio, motifs, motif_length, stdev, stdev_out, w_decay, 
				w_out_decay, pooling, train_step, padding, extra_layer, fc_nodes):

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, shape=[None, 50, 4], name="x")
	y_ = tf.placeholder(tf.float32, shape=[None, 2], name="y_")
	keep_prob = tf.placeholder(tf.float32)
	accuracy = tf.placeholder(tf.float32)
	class_weight = tf.constant([[ratio, 1.0 - ratio]])
	with tf.name_scope('Model'):
		softmax_linear = mu.SelectModel(model_label, x, keep_prob, motifs, motif_length, stdev, stdev_out, w_decay, 
										w_out_decay, pooling, num_classes=2, padding=padding, extra_layer=extra_layer, fc_nodes=fc_nodes)

	with tf.name_scope('Loss'):
		weight_per_label = tf.transpose( tf.matmul(y_, tf.transpose(class_weight)) )
		xent = tf.multiply(weight_per_label, 
					  tf.nn.softmax_cross_entropy_with_logits(logits=softmax_linear, labels=y_, name="xent_raw")) #shape [1, batch_size]
		loss = tf.reduce_mean(xent)

	with tf.name_scope('SGD'):
		step_op = tf.train.AdamOptimizer(train_step).minimize(loss)

	with tf.name_scope('Accuracy'):
		correct_prediction = tf.equal(tf.argmax(softmax_linear,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	var_init = tf.global_variables_initializer()

	tf.summary.scalar("accuracy", accuracy)
	tf.summary.scalar("loss", loss)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)
	summary_op = tf.summary.merge_all()
	
	return x, y_, keep_prob, var_init, summary_op, softmax_linear, accuracy, loss, step_op


def run_model(timestamp, par_dict, x, y_, keep_prob, var_init, summary_op, softmax_op, acc_op, loss_op, step_op,
				X, Y, model_label, epochs, motif_length, batch_size, test_size, verbose):

	X_train, X_test, Y_train, Y_test = fu.CreateBalancedTrainTest(X,Y, test_size)
	A_X, A_Y, B_X, B_Y, R_X, R_Y, M_X, M_Y, D_X, D_Y = fu.LoadValidationData()
	
	v_out = [v for v in tf.trainable_variables() if v.name == "out/weights:0"]
	if model_label == "MS2":
		par_conv = math.ceil(50/motif_length[0])
		par_conv_names = []
		for i in range(par_conv):
			par_conv_names.append("conv{}/weights:0".format(i))
		v = [v for v in tf.trainable_variables() if v.name in par_conv_names]
	else:
		v = [v for v in tf.trainable_variables() if v.name == "conv1/weights:0"]
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		
		filter_list = [] 
		filter_out = []
		sess.run(var_init)
		A_spear, B_spear, R_spear, M_spear, D_spear = [], [], [], [], []
		avg_spear, losses, AUCs = [], [], []
		runs = len(X_train)//batch_size
		summary_writer = tf.summary.FileWriter('../models/tf_logs/example_{:%m:%d:%H:%M}'.format(dt.datetime.now()), 
												graph=tf.get_default_graph())
		for epoch in range(epochs):
			X_train_S, Y_train_S = shuffle(X_train, Y_train)
			avg_loss = 0
			avg_acc = 0
			filter_list.append(sess.run(v))
			filter_out.append(np.array(sess.run(v_out))[0,:,:])
			for run in range(runs):
				X_batch = X_train_S[run*batch_size:(run+1)*batch_size]
				Y_batch = Y_train_S[run*batch_size:(run+1)*batch_size]
				X_batch_aug, Y_batch_aug = fu.augment_sequences(X_batch, Y_batch)
				_, model_pred, model_loss, model_acc, summary = sess.run([step_op, softmax_op, loss_op, acc_op, summary_op],
					feed_dict={x: X_batch_aug , y_: Y_batch_aug, keep_prob: 1})
				avg_loss += model_loss/runs
				avg_acc += model_acc/runs
			softmax_test = softmax_op.eval(feed_dict={x: X_test , y_: Y_test, keep_prob: 1})
			AUCs.append(mu.CalculateAUC(softmax_test[:,1], Y_test[:,1]))
			A_spear.append(stats.spearmanr(softmax_op.eval(feed_dict={x: A_X , y_: Y_train[:len(A_X)], keep_prob: 1})[:,1], A_Y)[0])
			B_spear.append(stats.spearmanr(softmax_op.eval(feed_dict={x: B_X , y_: Y_train[:len(B_X)], keep_prob: 1})[:,1], B_Y)[0])
			R_spear.append(stats.spearmanr(softmax_op.eval(feed_dict={x: R_X , y_: Y_train[:len(R_X)], keep_prob: 1})[:,1], R_Y)[0])
			M_spear.append(stats.spearmanr(softmax_op.eval(feed_dict={x: M_X , y_: Y_train[:len(M_X)], keep_prob: 1})[:,1], M_Y)[0])
			D_spear.append(stats.spearmanr(softmax_op.eval(feed_dict={x: D_X , y_: Y_train[:len(M_X)], keep_prob: 1})[:,1], D_Y)[0])
			avg_spear.append((A_spear[-1]+abs(B_spear[-1])+R_spear[-1]+M_spear[-1]+D_spear[-1])/5)
			print(avg_spear[-1], AUCs[-1])
			losses.append(avg_loss)
			if verbose:
				if (avg_spear[-1] > .70) and (np.all(avg_spear[:-1]<avg_spear[-1])):
					saver.save(sess,"../models/model_{}_{:2.03f}_epoch{}.ckpt".format(timestamp, avg_spear[-1], epoch))
					with open('../models/model_{}_{:2.03f}_epoch{}.json'.format(timestamp, avg_spear[-1], epoch), 'w') as outfile:
						json.dump(par_dict, outfile)
				summary_writer.add_summary(summary, epoch)
		if verbose:
			saver.save(sess,"../models/model_{}_{:.3f}_epoch{}.ckpt".format(timestamp, avg_spear[-1], epoch))
	results = pd.DataFrame({"A_spear": A_spear, "B_spear": B_spear, "D_spear": D_spear, "R_spear": R_spear, "M_spear": M_spear, "AVG_spear": avg_spear, "AUC":AUCs})
	
	return results

def ExecuteFunction(function, model_label, experiment, epochs, repeats, cutoff, motifs, motif_length, stdev,
					stdev_out, w_decay, w_out_decay, pooling=1, batch_size=40, train_step=1e-4, 
					test_size=0.1, fc_nodes=32, padding=False, extra_layer=False, verbose=False, lowess=False):
	if function == "rand":
		model_label = np.random.choice(['MS3','MS4'])
		cutoff = np.random.choice([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,2,2.5])
		#cutoff = np.random.uniform([0,2.5])
		if model_label != "MS4":
			motifs = [2**np.random.randint(6,9)]
			motif_length = [np.random.randint(6,14)]
		else:
			motifs = [2**i for i in np.random.randint(6,9,size=2)]
			motif_length = np.random.randint(6,14,size=2)
		pooling = np.random.choice([-1,1,2])
		stdev = 10**np.random.uniform(-14, -1)
		stdev_out = 10**np.random.uniform(-10, -1)
		w_decay = 10**np.random.uniform(-14, -1)
		w_out_decay = 10**np.random.uniform(-14, -1)
		extra_layer = np.random.choice([True, False])
	if function =='t_rand':
		fc_nodes = np.random.choice([32,64,128,256])
		padding = np.random.choice([True, False])
		cutoff = np.random.choice([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,2,2.5])
		motifs = [2**np.random.randint(6,9)]
		motif_length = [np.random.randint(6,14)]
		pooling = np.random.choice([-1,1,2])
		stdev = 10**np.random.uniform(-14, -1)
		stdev_out = 10**np.random.uniform(-10, -1)
		w_decay = 10**np.random.uniform(-14, -1)
		w_out_decay = 10**np.random.uniform(-14, -1)
		extra_layer = np.random.choice([True, False])
	hyp_string = "model_label:{} , cutoff_value:{} , motif_length:{} , motifs:{} , stdev:{} , stdev_out:{} , w_decay:{} , w_out_decay:{} , pooling:{} , fully_connect:{} , fc_nodes:{} , padding:{}".format(model_label,  
								cutoff, motif_length, motifs, stdev, stdev_out, w_decay, w_out_decay, pooling, extra_layer, fc_nodes, padding)
	localarg = locals()
	LOGFILENAME, MAINLOG, RESULTLOG, timestamp = log.LogInit(function, model_label, localarg, hyp_string)
	X, Y = load_data(experiment, cutoff, lowess)
	ratio = sum(Y[:,1]==1)/len(Y)
	par_dict = {"model_label": model_label, "ratio":ratio, "motifs":motifs, "motif_length":motif_length, "stdev":stdev, "stdev_out":stdev_out, 
	"w_decay":w_decay, "w_out_decay":w_out_decay, "pooling":pooling, "train_step":train_step, "padding":padding, "extra_layer":extra_layer, 
	"fc_nodes":fc_nodes}
	for repeat in range(repeats):
		x, y_, keep_prob, var_init, summary_op, softmax_op, acc_op, loss_op, step_op = load_model(model_label, ratio, 
									motifs, motif_length, stdev, stdev_out, w_decay, w_out_decay, pooling, train_step, padding, extra_layer, fc_nodes)
		results = run_model(timestamp, par_dict, x, y_, keep_prob, var_init, summary_op, softmax_op, acc_op, loss_op, step_op, 
							X, Y, model_label, epochs, motif_length, batch_size, test_size, verbose)
		log.LogWrap(MAINLOG, RESULTLOG, results, hyp_string, repeat, repeats)


def main():
	parser = argparse.ArgumentParser(description='high-end script function for prompred')
	parser.add_argument('function', type=str,choices=('eval','no', 'rand', 't_rand'), help="function to execute")
	parser.add_argument('-d', '--data', type=str, choices=('RPOD', 'RPOS', 'RPON', 'SIGMA', 'BETA', 'all'), help='chooses data experiment')
	parser.add_argument('-e', '--epochs', type=int, help='amount of epochs to train')
	parser.add_argument('-r', '--repeats', type=int, default=1, help='amount of repeats of the experiment')
	parser.add_argument('-m', '--model', type=str, choices=('MS1','MS2','MS3','MS4'), help=' type of architecture of the model')
	parser.add_argument('-c', '--cutoff', type=float, help="cutoff value to select datasets from ")
	parser.add_argument('-b', '--batch_size', type=int, default=40, help="determines batch size")
	parser.add_argument('-M', '--motifs', type=int, nargs='+', help="amount of motifs")
	parser.add_argument('-ML', '--motif_length', type=int, nargs='+', help="motif length")
	parser.add_argument('-S', '--stdev', type=float, help="stdev of the conv. layer")
	parser.add_argument('-SO', '--stdev_out', type=float, help="stdev out layer")
	parser.add_argument('-W', '--weight_dec', type=float, help="weight decay")
	parser.add_argument('-WO', '--weight_dec_out', type=float, help="weight decay out")
	parser.add_argument('-P', '--pooling', type=int, choices=(-1,1,2), default=1, help="-1: avg pooling only, 1: max pooling only, 2: both pooling methods (features x2!)")
	parser.add_argument('-LS', '--learning_step', type=float, default=1e-4, help="learning step of the model")
	parser.add_argument('-TS', '--test_size', type=float, default=0.1, help="fraction of the data used as a test set")
	parser.add_argument('-FC', '--fc_nodes', type=int, default=32, help="# nodes in fully connected layer")
	parser.add_argument('-p', '--padding', action="store_true", help="add_padding")
	parser.add_argument('-F', '--fully_connected', action="store_true", help="add fully connected layer behind main layer")
	parser.add_argument('-v', '--verbose', action="store_true", help="create tensorboard model summaries")
	parser.add_argument('-l', '--lowess', action="store_true")
	args = parser.parse_args()
	ExecuteFunction(args.function,  args.model, args.data, args.epochs, args.repeats, args.cutoff, args.motifs,
					args.motif_length, args.stdev, args.stdev_out, args.weight_dec, args.weight_dec_out, args.pooling, args.batch_size,
					args.learning_step, args.test_size, args.fc_nodes, args.padding, args.fully_connected, args.verbose, args.lowess)



if __name__ == "__main__":
    sys.exit(main())
