import matplotlib
matplotlib.use("Agg")
import sys
import json
import math
import argparse
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


def load_data(experiment, cutoff):
	path_TSS = "../data/external/promotor_list_exp_growth.csv"

	data_ip, data_mock_ip = du.GetDataLocations(experiment)
	X_TSS, Y_TSS = fu.LoadDataTSS(path_TSS, experiment)
	A_X, A_Y, B_X, B_Y, R_X, R_Y, M_X, M_Y, D_X, D_Y = fu.LoadValidationData()
	X, Y, sequences, IDs = fu.TransformDataSimple(data_ip, data_mock_ip)
	_, mask_peak = fu.DetectPeaks(Y,cutoff, smoothing=True)
	mask_TSS = fu.AllocatePromoters(experiment, IDs)
	mask = np.any([mask_peak,mask_TSS],axis=0)
	Y_masked = fu.BinaryOneHotEncoder(mask)
	Y_full = np.vstack((Y_masked, Y_TSS))
	X_full = np.vstack((X, X_TSS))
	
	return X_full, Y_full 

def load_model(model_label, ratio, motifs, motif_length, stdev, stdev_out, w_decay, 
				w_out_decay, train_step):

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, shape=[None, 50, 4], name="x")
	y_ = tf.placeholder(tf.float32, shape=[None, 2], name="y_")
	keep_prob = tf.placeholder(tf.float32)
	accuracy = tf.placeholder(tf.float32)
	class_weight = tf.constant([[ratio, 1.0 - ratio]])
	with tf.name_scope('Model'):
		softmax_linear = mu.SelectModel(x, keep_prob, model_label, motifs, motif_length, 
										stdev, stdev_out, w_decay, w_out_decay)

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


def run_model(x, y_, keep_prob, var_init, summary_op, softmax_op, acc_op, loss_op, step_op, X, Y, model_label, epochs, motif_length, batch_size, test_size):

	X_train, X_test, Y_train, Y_test = fu.CreateBalancedTrainTest(X,Y, test_size)
	A_X, A_Y, B_X, B_Y, R_X, R_Y, M_X, M_Y, D_X, D_Y = fu.LoadValidationData()
	
	
	v_out = [v for v in tf.trainable_variables() if v.name == "out/weights:0"]
	if model_label == "PC":
		par_conv = math.ceil(50/motif_length)
		par_conv_names = []
		for i in range(par_conv):
			par_conv_names.append("conv{}/weights:0".format(i))
		v = [v for v in tf.trainable_variables() if v.name in par_conv_names]
	
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
				_, model_pred, model_loss, model_acc, summary = sess.run([step_op, softmax_op, loss_op, acc_op, summary_op],
					feed_dict={x: X_batch , y_: Y_batch, keep_prob: 1})
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
			losses.append(avg_loss)
			if (avg_spear[-1] > .73) and (np.all(avg_spear[:-1]<avg_spear[-1])):
				saver.save(sess,"../models/model_{:%m:%d:%H:%M}_{}_epoch{}.ckpt".format(dt.datetime.now(), avg_spear[-1], epoch))
			summary_writer.add_summary(summary, epoch)
		saver.save(sess,"../models/model_{:%m:%d:%H:%M}_{:.3f}_epoch{}.ckpt".format(dt.datetime.now(), avg_spear[-1], epoch))
	results = pd.DataFrame({"A_spear": A_spear, "B_spear": B_spear, "D_spear": D_spear, "R_spear": R_spear, "M_spear": M_spear, "AVG_spear": avg_spear, "AUC":AUCs})
	
	return results

def ExecuteFunction(function, model_label, experiment, epochs, repeats, cutoff, motifs, motif_length, stdev,
					stdev_out, w_decay, w_out_decay, batch_size=40, train_step=1e-4, test_size=0.1):

	localarg = locals()
	LOGFILENAME, MAINLOG, RESULTLOG = log.LogInit(function, model_label, localarg)
	#X, Y = load_data(experiment, cutoff)
	X, Y = fu.LoadDataTSS("../data/external/promotor_list_exp_growth.csv", experiment)
	#ratio = sum(Y[:,1]==1)/len(Y)
	ratio = 0.012
	
	for repeat in range(repeats):
		x, y_, keep_prob, var_init, summary_op, softmax_op, acc_op, loss_op, step_op = load_model(model_label, ratio, 
									motifs, motif_length, stdev, stdev_out, w_decay, w_out_decay, train_step)
		results = run_model(x, y_, keep_prob, var_init, summary_op, softmax_op, acc_op, loss_op, step_op, 
							X, Y, model_label, epochs, motif_length, batch_size, test_size)
		log.LogWrap(MAINLOG, RESULTLOG, repeat, results)


def main():
	parser = argparse.ArgumentParser(description='high-end script function for prompred')
	parser.add_argument('function', type=str,choices=('eval','no'), help="function to execute")
	parser.add_argument('-d', '--data', type=str, choices=('RPOD'), help='chooses data experiment')
	parser.add_argument('-e', '--epochs', type=int, help='amount of epochs to train')
	parser.add_argument('-r', '--repeats', type=int, help='amount of repeats of the experiment')
	parser.add_argument('-m', '--model', type=str, choices=('PC'), help=' type of architecture of the model')
	parser.add_argument('-c', '--cutoff', type=float, help="cutoff value to select datasets from ")
	parser.add_argument('-b', '--batch_size', type=int, help="determines batch size")
	parser.add_argument('-M', '--motifs', type=int, help="amount of motifs")
	parser.add_argument('-ML', '--motif_length', type=int, help="motif length")
	parser.add_argument('-S', '--stdev', type=float, help="stdev of the conv. layer")
	parser.add_argument('-SO', '--stdev_out', type=float, help="stdev out layer")
	parser.add_argument('-W', '--weight_dec', type=float, help="weight decay")
	parser.add_argument('-WO', '--weight_dec_out', type=float, help="weight decay out")
	args = parser.parse_args()
	ExecuteFunction(args.function,  args.model, args.data, args.epochs, args.repeats, args.cutoff, args.motifs,
					args.motif_length, args.stdev, args.stdev_out, args.weight_dec, args.weight_dec_out)



if __name__ == "__main__":
    sys.exit(main())

	#filter_list.append(np.reshape(sess.run(v),(par_conv,MOTIFS,MOTIF_LENGTH,4)))
	#filter_out.append(np.array(sess.run(v_out))[0,:,:])
#			accs_train.append(avg_acc)
#			accs_test.append(acc_op.eval(feed_dict={x: X_test , y_: Y_test, keep_prob: 1}))
