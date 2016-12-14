from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import sys
import time
import numpy
import json
sentence_list_fname='assignment_training_data_word_segment.json'

sentence_list=json.load(open(sentence_list_fname,'r'))
res_list=json.load(open('my_res.json','r'))

def numpy_floatX(data):
    return numpy.asarray(data, dtype=float)

my_1=0
truth_1=0
both_1=0
for i in range(0,len(res_list)):
	truth_1+=len(sentence_list[i]['results'])
	my_1+=len(res_list[i]['results'])
	for j in range(0,len(sentence_list[i]['results'])):
		for k in range(0,len(res_list[i]['results'])):
			if(sentence_list[i]['results'][j]==res_list[i]['results'][k]):
				both_1+=1

p = numpy_floatX(both_1)/numpy_floatX(my_1)
r = numpy_floatX(both_1)/numpy_floatX(truth_1)
f1 = 2*p*r/(p+r)
print(p)
print(r)
print(f1)