#encoding=utf-8
import os
import sys
import codecs
import numpy as np
import pandas as pd
from select_gene import *
from sklearn.preprocessing import MinMaxScaler

__author__ = "turingli"

class DataSet:

  def __init__(self, df, batch_size=128, shuffle=True):
    data = np.float32(df.values)
    self.data = data
    self.samples, self.feature_nums = data.shape
    self.index = list(range(self.samples))

    self.cur_pos = 0
    self.batch_counter = 0

    self.batch_size = batch_size
    self.shuffle = shuffle

    if shuffle:
      self.shuffle_index()

    print("batch_size is {}, have {} samples, {} features, step nums is {}".format(batch_size, self.samples,
                                                                                   self.feature_nums, self.steps))

  def next(self):

    batch_size = self.batch_size

    if self.cur_pos >= self.samples:  # for infer mode
      return None

    be, en = self.cur_pos, min(self.samples, self.cur_pos + batch_size)
    batch_index = self.index[be:en]
    batch_data = self.data[batch_index]
    # print(batch_data.shape)
    self.cur_pos = en
    self.batch_counter += 1
    # print("getting {}th batch end".format(self.batch_counter))
    return batch_data

  def sample_batch(self):
    index = np.random.choice(list(range(self.samples)), self.batch_size)
    return self.data[index]

  def shuffle_index(self):
    self.index = np.random.permutation(list(range(self.samples)))

  def reset(self):
    self.cur_pos = 0
    self.batch_counter = 0
    self.index = list(range(self.samples))
    if self.shuffle:
      self.shuffle_index()

  @property
  def steps(self):
    return self.samples // self.batch_size

  @property
  def mode(self):
    return self.samples % self.batch_size


if __name__ == "__main__":
  # dataset = DataSet()
  print("xx")