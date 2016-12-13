import json
#from __future__ import print_function
#from six.moves import xrange


import numpy
import theano


def prepare_data(seqs, labels, t, a, v, maxlen=None):
    """Create the matrices from the datasets.   

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    # ATTENTION s is a 1-d list
    lengths = len(seqs)
    new_seqs = []
    zuhe = len(labels)
    x = numpy.zeros((lengths, zuhe)).astype('int64')
    x_mask = numpy.zeros((3,lengths,zuhe)).astype(theano.config.floatX)
    for ix in range(0,zuhe):
        for jx in range(0,lengths):
            x[jx][ix] = seqs[jx]
    cnt = 0
    for tt in t:
        for aa in a:
            for vv in v:
                x_mask[0][vv][cnt] = 1.
                x_mask[1][aa][cnt] = 1.
                x_mask[2][tt][cnt] = 1.
                cnt += 1
    return x, x_mask, labels

def load_data(path="assignment_training_data_word_segment.json", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
#sentence_list_fname = path
    sentence_list_fname = "assignment_training_data_word_segment.json"
    sentence_list = json.load(open(sentence_list_fname,'r'))
    sentences = []
    results = []
    times = []
    attributes = []
    values = []
    for dic in sentence_list[:]:
        sentences.append(dic['indexes'])
        times.append(dic['times'])
        attributes.append(dic['attributes'])
        values.append(dic['values'])
        flag = 0
        t_list = []
        for t in dic['times']:
            for a in dic['attributes']:
                for v in dic ['values']:
                    for list in dic['results']:
                        if [t,a,v]==list:
                            flag = 1
                    t_list.append(flag)
                    flag = 0
        results.append(t_list)
    train_set = (sentences,results,times,attributes,values)

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        new_train_set_t = []
        new_train_set_a = []
        new_train_set_v = []
        for x, y, t, a, v in zip(train_set[0], train_set[1], train_set[2], train_set[3], train_set[4]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                new_train_set_t.append(t)
                new_train_set_a.append(a)
                new_train_set_v.append(v)
        train_set = (new_train_set_x, new_train_set_y, new_train_set_t, new_train_set_a, new_train_set_v)
        del new_train_set_x, new_train_set_y, new_train_set_t, new_train_set_a, new_train_set_v

    # split training set into validation set
    train_set_x, train_set_y, train_set_t, train_set_a,train_set_v = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_t = [train_set_t[s] for s in sidx[n_train:]]
    valid_set_a = [train_set_a[s] for s in sidx[n_train:]]
    valid_set_v = [train_set_v[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_t = [train_set_t[s] for s in sidx[:n_train]]
    train_set_a = [train_set_a[s] for s in sidx[:n_train]]
    train_set_v = [train_set_v[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y, train_set_t, train_set_a, train_set_v)
    valid_set = (valid_set_x, valid_set_y, valid_set_t, valid_set_a, valid_set_v)


    '''def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]


    valid_set_x, valid_set_y, valid_set_t, valid_set_a, valid_set_v = valid_set
    train_set_x, train_set_y, train_set_t, train_set_a, train_set_v = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)'''


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))


    if sort_by_len:
    #if True
        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        valid_set_t = [valid_set_t[i] for i in sorted_index]
        valid_set_a = [valid_set_a[i] for i in sorted_index]
        valid_set_v = [valid_set_v[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        train_set_t = [train_set_t[i] for i in sorted_index]
        train_set_a = [train_set_a[i] for i in sorted_index]
        train_set_v = [train_set_v[i] for i in sorted_index]

    train = (train_set_x, train_set_y, train_set_t, train_set_a, train_set_v)
    valid = (valid_set_x, valid_set_y, valid_set_t, valid_set_a, valid_set_v)

    return train, valid