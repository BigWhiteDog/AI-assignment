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
sentence_list_fname='assignment_training_data_word_segment.json'
sentence_list=json.load(open(sentence_list_fname,'r'))

voc_fname='voc-d.pkl'
voc_d=pickle.load(open(voc_fname,'rb'))
idx2word=voc_d['idx2word']
word2idx=voc_d['word2idx']




# Set the random number generators' seeds for consistency



SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
	return numpy.asarray(data, dtype=config.floatX)




def build_tuples(sentence_index,tuples):
	pp = sentence_list[sentence_index]
	kk=0
	for tt in pp["times"]:
		for aa in pp["attributes"]:
			for vv in pp["values"]:
				tuples[kk] = tt,aa,vv,0
				for rr in pp["results"]:
					if rr == [tt,aa,vv]:
						tuples[kk] = tt,aa,vv,1
				kk+=1

pt={}
build_tuples(2,pt)

longest=0
for zz in range(0,len(sentence_list)):
	kk=0
	for tt in sentence_list[zz]["times"]:
			for aa in sentence_list[zz]["attributes"]:
				for vv in sentence_list[zz]["values"]:
					kk+=1
	if(kk>longest):
		longest=kk



print longest

#our params: longest is 800

pa=tensor.alloc(numpy_floatX(0.), 3, 5)