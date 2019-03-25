# encoding=utf-8

import os
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
from Dataset import DataSet
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras import constraints
from keras import backend as K
from tensorflow.contrib import distributions

from layers import *
import shutil
import time
from datetime import datetime

activation_dict = {
    "tanh": tf.tanh,
    "sigmoid": tf.sigmoid,
    "relu": tf.nn.relu
}

__author__ = "future_chi"

class AutoEncoder(object):
    def __init__(self, feature_num, num_clusters, bn=True, beta=0.01, dropout=None,
                 learning_rate=0.001, activation="relu", model_name="auto_encoder", **kwargs):

        self.feature_num = feature_num
        self.beta = beta
        self.activation = activation_dict[activation]
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.create_conf = kwargs
        self.dropout = dropout
        self.dim = [500, 500, 50]
        self.num_clusters = num_clusters
        self._create_model()
        self.bn = bn

    def _create_model(self):

        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.X = tf.placeholder(tf.float32, [None, self.feature_num], name="X")

        if len(self.dim) > 1:
            self.encoder_out = self.creat_net_enc(self.X, self.dim, self.bn)  # through activation
            self.decoder_out = self.creat_net_dec(self.encoder_out, self.dim, self.bn)  # must not through activation
        else:
            self.encoder_out = Dense(self.dim[0], activation="relu")(self.X)
            self.decoder_out = Dense(self.feature_num)(self.encoder_out)

        with tf.variable_scope('cluster'):
            centroid_init_val = np.random.normal(0.0, 1.0, (self.num_clusters, self.dim[-1]))
            self.centroids = tf.get_variable('centroids', shape=(self.num_clusters, self.dim[-1]),
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(centroid_init_val))
            self.cluster_distsqs = tf.reduce_sum(
                tf.square(tf.tile(tf.expand_dims(self.encoder_out, axis=1), multiples=[1, self.num_clusters, 1]) - \
                          tf.tile(tf.expand_dims(self.centroids, axis=0),
                                  multiples=[tf.shape(self.encoder_out)[0], 1, 1])), axis=2)


        tf.summary.histogram("encoder_out", self.encoder_out)
        tf.summary.histogram("decoder_out", self.decoder_out)

        self.mse_loss = tf.reduce_mean(tf.pow(self.X - self.decoder_out, 2)) / 2
        self.cluster_loss = tf.reduce_mean(tf.reduce_min(self.cluster_distsqs, axis=1))
        self.loss = self.mse_loss + self.beta * self.cluster_loss

        tf.summary.scalar("train_loss", self.loss)
        tf.summary.scalar("mse_loss", self.mse_loss)
        tf.summary.scalar("cluster_loss", self.cluster_loss)

        with tf.name_scope('optimizer'):
            # Gradient Descent
            # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.step = tf.Variable(0, trainable=False)
            rate = tf.train.exponential_decay(self.learning_rate, self.step, 1, 0.9999)
            optimizer = tf.train.AdamOptimizer(rate)
            # Op to calculate every variable gradient
            grads = tf.gradients(self.loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            # Op to update all variables according to their gradient
            self.apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

        # Create summaries to visualize weights
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        # Summarize all gradients
        # for grad, var in grads:
        #   tf.summary.histogram(var.name + '/gradient', grad)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=1)

        '''
          `max_to_keep` indicates the maximum number of recent checkpoint files to
          keep.  As new files are created, older files are deleted.  If None or 0,
          all checkpoint files are kept.  Defaults to 5 (that is, the 5 most recent
          checkpoint files are kept.)
        '''

    def creat_net_enc(self, input, dim, bn=False):
        with tf.variable_scope("encoder"):
            # input layer
            out = Dense(dim[0], activation="linear")(input)
            if bn:
                out = batch_norm(out, is_training=self.is_training)
            out = keras.layers.activations.relu(out)
            # hidden layers
            if self.dropout < 1.0:
                out = keras.layers.Dropout(self.dropout)(out)
            for i in range(len(dim) - 2):
                out = Dense(dim[i + 1], activation="linear")(out)
                if bn:
                    out = batch_norm(out, is_training=self.is_training)
                out = keras.layers.activations.relu(out)
                # if self.dropout < 1.0:
                #   out = keras.layers.Dropout(self.dropout)(out)
            # output layer
            out = Dense(dim[-1], activation="relu")(out)
        return out

    def creat_net_dec(self, input, dim, bn=False):
        with tf.variable_scope("decoder"):
            # input layer
            out = Dense(dim[-2], activation="linear")(input)
            if bn:
                out = batch_norm(out, is_training=self.is_training)
            out = keras.layers.activations.relu(out)
            # hidden layers
            if self.dropout < 1.0:
                out = keras.layers.Dropout(self.dropout)(out)
            for i in range(len(dim) - 2, 0, -1):
                out = Dense(dim[i - 1], activation="linear")(out)
                if bn:
                    out = batch_norm(out, is_training=self.is_training)
                out = keras.layers.activations.relu(out)
                # if self.dropout < 1.0:
                #   out = keras.layers.Dropout(self.dropout)(out)
            # output layer
            out = Dense(self.feature_num, activation="relu")(out)
        return out

    def predict_tmp(self, sess, step, dataset, columns, config):
        print("testing for {}th...".format(step))
        decoder_out_list, encoder_out_list = [], []
        mask_data = []
        while (1):
            batch_data = dataset.next()
            if batch_data is None:
                break
            mask = (batch_data == 0.0)
            # mask = np.float32(mask)
            mask_data.append(mask)
            # keep_bools = np.float32( np.zeros_like(batch_data) )

            decoder_out, encoder_out = sess.run([self.decoder_out, self.encoder_out], feed_dict={self.X: batch_data,
                                                                                                 self.is_training: False,
                                                                                                 K.learning_phase(): 0})
            decoder_out_list.append(decoder_out)
            encoder_out_list.append(encoder_out)
        decoder_out = np.reshape(np.concatenate(decoder_out_list, axis=0), (-1, dataset.feature_nums))
        encoder_out = np.reshape(np.concatenate(encoder_out_list, axis=0), (-1, self.dim[-1]))
        mask_data = np.reshape(np.concatenate(mask_data, axis=0), (-1, dataset.feature_nums))

        mask_data = np.float32(mask_data)
        decoder_out_replace = mask_data * decoder_out + dataset.data  ##missing value now is completed, other values remain same

        df = pd.DataFrame(decoder_out, columns=columns)
        df_replace = pd.DataFrame(decoder_out_replace, columns=columns)

        if os.path.exists(config.outDir) == False:
            os.makedirs(config.outDir)
        outDir = os.path.join(config.outDir, self.model_name)
        if os.path.exists(outDir) == False:
            os.makedirs(outDir)
        outPath = os.path.join(outDir, "{}.{}.complete".format(self.model_name, step))
        df.to_csv(outPath, index=None, float_format='%.4f')
        df_replace.to_csv(outPath.replace(".complete", ".complete.sub"), index=None, float_format='%.4f')

        print("save complete data from {} to {}".format(config.train_datapath, outPath))
        # pd.DataFrame(rev_normal_predict_data, columns=dataset.columns).to_csv(outPath.replace(".complete", ".revnormal"),index=None)
        # print("save rev normal data to {}".format(outPath.replace(".complete", ".revnormal")))

        pd.DataFrame(encoder_out).to_csv(outPath.replace(".complete", ".encoder.out"), float_format='%.4f')

    def train(self, train_dataset, val_dataset, test_dataset, columns, config):

        t0 = time.time()
        # gpu_conf = tf.ConfigProto()
        # gpu_conf.gpu_options.per_process_gpu_memory_fraction = config.gpu_ratio
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as session:

            log_dirs = os.path.join("./logs", self.model_name)
            if os.path.exists(log_dirs) == False:
                os.makedirs(log_dirs)

            load_model_dir = os.path.join('./backup', self.model_name)
            if config.load_checkpoint and os.path.exists(load_model_dir):
                self.load(session, load_model_dir)
            elif os.path.exists(load_model_dir):
                shutil.rmtree(load_model_dir)

            if config.load_checkpoint is False and os.path.exists(log_dirs):
                shutil.rmtree(log_dirs)
                os.makedirs(log_dirs)

            self.writer = tf.summary.FileWriter(log_dirs, session.graph)

            tf.global_variables_initializer().run()
            step_to_save = config.save_freq_steps * train_dataset.steps
            cost_val = []
            cost_train = []
            step = 0
            for epoch in range(config.epoch):
                t = time.time()
                tot_train_loss = 0.
                tot_mse_loss = 0.
                tot_cluster_loss = 0.
                while True:
                    batch_data = train_dataset.next()
                    if batch_data is None:
                        train_dataset.reset()
                        break
                    mask = (batch_data == 0.0)

                    # print(step, batch_data.shape)
                    if step % config.log_freq_steps != 0:
                        _, loss, mse_loss, cluster_loss = session.run(
                            [self.apply_grads, self.loss, self.mse_loss, self.cluster_loss],
                            feed_dict={self.X: batch_data,
                                       self.is_training: True, K.learning_phase(): 1})
                    else:
                        _, summary_str, loss, mse_loss, cluster_loss = session.run(
                            [self.apply_grads, self.merged_summary_op,
                             self.loss, self.mse_loss, self.cluster_loss],
                            feed_dict={self.X: batch_data, self.is_training: True,
                                       K.learning_phase(): 1})
                        self.writer.add_summary(summary_str, tf.train.global_step(session, self.step))
                    tot_train_loss += loss * len(batch_data)
                    tot_mse_loss += mse_loss * len(batch_data)
                    tot_cluster_loss += cluster_loss * len(batch_data)
                    step += 1
                avg_train_loss = tot_train_loss / train_dataset.samples
                avg_mse_loss = tot_mse_loss / train_dataset.samples
                avg_cluster_loss = tot_cluster_loss / train_dataset.samples
                print(
                    "epoch {}th, train_loss: {:.6f}, mse_loss: {:.6f}, cluster_loss: {:.6f}".format(
                        epoch + 1, avg_train_loss,
                        avg_mse_loss, avg_cluster_loss))
                cost_train.append(avg_train_loss)

                tot_val_loss = 0.
                tot_mse_loss = 0.
                tot_cluster_loss = 0.
                while True:
                    batch_data = val_dataset.next()
                    if batch_data is None:
                        val_dataset.reset()
                        break
                    mask = (batch_data == 0.0)
                    # mask = np.float32(mask)
                    _, val_loss, mse_loss, cluster_loss = session.run(
                        [self.apply_grads, self.loss, self.mse_loss, self.cluster_loss],
                        feed_dict={self.X: batch_data,
                                   self.is_training: False, K.learning_phase(): 0})
                    # val step
                    tot_val_loss += val_loss * len(batch_data)
                    tot_mse_loss += mse_loss * len(batch_data)
                    tot_cluster_loss += cluster_loss * len(batch_data)
                avg_val_loss = tot_val_loss / val_dataset.samples
                avg_mse_loss = tot_mse_loss / val_dataset.samples
                avg_cluster_loss = tot_cluster_loss / val_dataset.samples
                print(
                    "epoch {}th, val_loss: {:.6f}, mse_loss: {:.6f}, cluster_loss: {:.6f}, cost time: {:.6f}".format(
                        epoch + 1,
                        avg_val_loss, avg_mse_loss, avg_cluster_loss, time.time() - t))

                cost_val.append(avg_val_loss)
                if epoch * train_dataset.steps > 300 and np.mean(cost_val[-(config.early_stopping + 1):-1]) > \
                        np.mean(cost_val[-(config.early_stopping * 2 + 1):-(config.early_stopping + 1)]) \
                        and np.mean(cost_train[-(config.early_stopping + 1):-1]) < \
                                np.mean(cost_train[-(config.early_stopping * 2 + 1):-(config.early_stopping + 1)]):
                    print("Early stopping...")
                    break
            t1 = time.time()
            print("Optimization Finished!")
            print("total cost time {}".format(t1 - t0))

            self.predict_tmp(session, epoch, test_dataset, columns, config)

    def save(self, sess, save_dir, step):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.saver.save(sess,
                        os.path.join(save_dir, self.model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

