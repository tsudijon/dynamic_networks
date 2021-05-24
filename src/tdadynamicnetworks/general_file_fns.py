''' August 1, 2017
For opening and closing files'''
from __future__ import division
import pickle


def load_file(filename):
    fr = open(filename, 'rb')
    data = pickle.load(fr, encoding='latin1')
    fr.close()
    return data


def save_file(data, filename):
    fw = open(filename, 'wb')
    pickle.dump(data, fw)
    fw.close()
    return 1
