
# coding: utf-8

import os
import sys

import pandas as pd
import numpy as np
import codecs
import tensorflow as tf

import argparse
from scipy.stats import norm
from Dataset import DataSet
from autoencoder_kmeans import AutoEncoder
from select_gene import *
from sklearn.preprocessing import MinMaxScaler

__author__ = "future_chi"


flags = tf.app.flags
flags.DEFINE_integer("epoch", 800, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 256, "The size of batch data [64]")
flags.DEFINE_integer("feature_nums", None, "The dimension of data to use")
flags.DEFINE_integer("feature_sel", None, "The dimension of data to use")
flags.DEFINE_integer("early_stopping", 20, "The dimension of data to use")

flags.DEFINE_string("activation", "relu", "auto encoder activation")
flags.DEFINE_string("train_datapath", "data/drop80-0-1.train", "Dataset directory.")
# flags.DEFINE_string("infer_complete_datapath", "data/drop80-0-1.infer", "path of infer complete path")
# flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("model_name", "auto_encoder0", "model name will make dir on checkpoint_dir")
flags.DEFINE_string("method", "None", "method to select genes")
# flags.DEFINE_float("truly_mis_pro", -1.0 , "the prob of truly missing values for value 0 [ truly_mis_pro <=0 mean don't random mask, all is missing]")
# flags.DEFINE_string("random_mask_path", "None", "the path of probs of letting value 0's prob trained[None]")
flags.DEFINE_float("dropout", 2.0, "dropout layer prob > 1 mean no layer")
# flags.DEFINE_float("random_sample_mu",0.0,"random mu")
# flags.DEFINE_float("random_sample_sigma", 1.0, "random sigma")
# flags.DEFINE_float("normal_factor", 1e6, "normal factor")

flags.DEFINE_boolean("plot_complete", False, "plot fig after every test or prediction")
# flags.DEFINE_integer("test_freq_steps", 500, "test freq steps")
flags.DEFINE_integer("save_freq_steps", 400, "save freq steps")
flags.DEFINE_integer("log_freq_steps", 10, "log freq steps")
flags.DEFINE_boolean("gene_scale", True, "if have checkpoint, whether load prev model first")
flags.DEFINE_boolean("load_checkpoint", False, "if have checkpoint, whether load prev model first")
# flags.DEFINE_integer("sample_steps", 500, "every sample_steps, will sample the generate mini batch data")
flags.DEFINE_float("gpu_ratio", 1.0, "per_process_gpu_memory_fraction[1.0]")
flags.DEFINE_float("gamma",10.0, "tuning parameter of sparse_loss")
flags.DEFINE_float("beta", 0.05, "tuning parameter of rank_loss")
flags.DEFINE_integer("num_clusters", 10, "log freq steps")
flags.DEFINE_string('outDir', 'prediction', "output dir")
flags.DEFINE_string('data_type', 'count', "output dir")
flags.DEFINE_boolean('trans', 'False', "output dir")

FLAGS = flags.FLAGS

print("make dataset from {}...".format(FLAGS.train_datapath))
df = pd.read_csv(FLAGS.train_datapath, sep=",", index_col=0)
if FLAGS.trans:
  df = df.transpose()
print("have {} samples, {} features".format(df.shape[0], df.shape[1]))
if FLAGS.data_type == 'count':
  df = row_normal(df)
  # df = sizefactor(df)
elif FLAGS.data_type == 'rpkm':
  df = np.log(df+1)
if FLAGS.gene_scale:
  scaler = MinMaxScaler()
  data = scaler.fit_transform(df)
  df = pd.DataFrame(data=data, columns=df.columns)
columns = list(df.columns)
if FLAGS.method is not None:
  columns = select_gene(columns, df, method=FLAGS.method, feature_sel=FLAGS.feature_sel, save=False)
  df = df[columns]
data = df.values
samples, feature_nums = data.shape
train_size = np.int(np.round(samples*0.8))
val_size = samples-train_size
df_shuffle = df.sample(frac=1).reset_index(drop=True)
df_train = df_shuffle.iloc[0:train_size]
df_val = df_shuffle.iloc[train_size:]

train_dataset = DataSet(df_train, FLAGS.batch_size)
val_dataset = DataSet(df_val, FLAGS.batch_size, shuffle=False)
test_dataset = DataSet(df, FLAGS.batch_size, shuffle=False)

model = AutoEncoder(feature_nums, num_clusters=FLAGS.num_clusters,beta=FLAGS.beta, dropout= FLAGS.dropout,learning_rate=FLAGS.learning_rate,
                    activation=FLAGS.activation, model_name=FLAGS.model_name)
model.train(train_dataset, val_dataset, test_dataset, columns, FLAGS)

