'''
Build a tweet sentiment analyzer
'''

# from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import json
setence_list_fname='assignment_training_data_word_segment.json'
setence_list=json.load(open(setence_list_fname,'r'))

voc_fname='voc-d.pkl'
voc_d=pickle.load(open(voc_fname,'rb'))
idx2word=voc_d['idx2word']
word2idx=voc_d['word2idx']
