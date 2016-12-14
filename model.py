# coding=utf8

'''
You should compelte this code.
I suggest you to write your email and mobile phone number here
#############  Contact information   ##############
email: legend.z@qq.com
phone: 18810999168
############  Contact information end  ############

'''

###############   import start    #################
from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time
import json
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

##############     import end     #################

def my_prepare_data(seqs, labels, t, a, v, maxlen=None):
    """Create the matrices from the datasets.   

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    lengths = len(seqs)
    new_seqs = []
    combination = len(labels)
    x = numpy.zeros((2,lengths, combination)).astype('int64')
    x_mask = numpy.zeros((6,lengths,combination)).astype(theano.config.floatX)
    for ix in range(0,combination):
        for jx in range(0,lengths):
            x[0][jx][ix] = seqs[jx]
            x[1][lengths-1-jx][ix] = seqs[jx]
    cnt = 0
    for tt in t:
        for aa in a:
            for vv in v:
                x_mask[0][vv][cnt] = 1.
                x_mask[1][aa][cnt] = 1.
                x_mask[2][tt][cnt] = 1.
                x_mask[3][lengths-1-vv][cnt]=1.
                x_mask[4][lengths-1-aa][cnt]=1.
                x_mask[5][lengths-1-tt][cnt]=1.
                cnt += 1
    return x, x_mask, labels

def my_load_data(path="assignment_training_data_word_segment.json", maxlen=None):
    sentence_list_fname = path
    sentence_list = json.load(open(sentence_list_fname,'r'))
    sentences = []
    times = []
    attributes = []
    values = []
    for dic in sentence_list[:]:
        sentences.append(dic['indexes'])
        times.append(dic['times'])
        attributes.append(dic['attributes'])
        values.append(dic['values'])
        t_list = []
        for t in dic['times']:
            for a in dic['attributes']:
                for v in dic ['values']:
                    t_list.append([t,a,v])
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
    return train_set

def get_result()
	return result

datasets = {'train_tuple': (my_load_data, my_prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 666
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_dataset(name):
    return datasets[name][0], datasets[name][1]

def _p(pp, name):
    return '%s_%s' % (pp, name)
def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='word')
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='tav')
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix='wordr')
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def get_layer(name):
    fns = layers[name]
    return fns

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm'):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step( x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))
        c = f * c_ + i * c
        h = o * tensor.tanh(c)
        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}

def build_model(tparams, options):

    x = tensor.tensor3('x', dtype='int64')
    mask = tensor.tensor3('mask', dtype=config.floatX)
    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    emb = tparams['Wemb'][x[0].flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    embr = tparams['Wemb'][x[1].flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='word')
    projr= get_layer(options['encoder'])[1](tparams, embr, options,
                                            prefix='wordr')

    if options['encoder'] == 'lstm':
        projt = (proj * mask[0][:, :, None]).sum(axis=0)
        proja = (proj * mask[1][:, :, None]).sum(axis=0)
        projv = (proj * mask[2][:, :, None]).sum(axis=0)
        
        projtr = (projr * mask[3][:, :, None]).sum(axis=0)
        projar = (projr * mask[4][:, :, None]).sum(axis=0)
        projvr = (projr * mask[5][:, :, None]).sum(axis=0)
        
        projt = (projt+projtr)/2.
        proja = (proja+projar)/2.
        projv = (projv+projvr)/2.
        
        tav = tensor.stack(projt,proja,projv)
        tav_proj = get_layer(options['encoder'])[1](tparams,tav,options,prefix='tav')

        proj=tav_proj[2]

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    return x, mask, f_pred

def prepared_lstm(
    dim_proj=32,  # word embeding dimension and LSTM number of hidden units.
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    maxlen=None,  # Sequence longer then this get ignored
    dataset='train_tuple',
    reload_model=True,  # Path to a saved model we want to start from.
    input_file_path="assignment_training_data_word_segment.json",
):

    # Model options
    model_options = locals().copy()
    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    the_test=load_data(path=input_file_path)
    ydim = 2
    model_options['ydim'] = ydim
    print('Building model')
    params = init_params(model_options)
    if reload_model:
        print('Reloading')
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    (x,mask,f_pred) = build_model(tparams, model_options)
    my_result_list=[]
    for test_index in range(0,len(the_test)):
        y = the_test[1][test_index]
        x = the_test[0][test_index]
        my_t = the_test[2][test_index]
        my_a = the_test[3][test_index]
        my_v = the_test[4][test_index]
        # Get the data in numpy.ndarray format
        # This swap the axis!
        # Return something of shape (minibatch maxlen, n samples)
        x, mask = prepare_data(x, y, my_t, my_a, my_v)
        my_result=f_pred(x,mask)
        my_result_list.append(my_result)
        if saveto and numpy.mod(uidx, saveFreq) == 0:
            print('Saving...')
            numpy.savez(saveto, history_errs=history_errs, **params)
            pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
            print('Done')

    return train_err, valid_err

def predict(input_file_path, output_file_path):
    '''
    This function should load the test file, 
    use the model to predict all correct triples in each sentence, 
    and write the result to the output file.

    input_file_path is the path to the test data json file, input file format:
    [
        {
            "sentenceId": "eadf4c4d7eaa6cb767fa6d8c02555f5-eb85e9fb6ec57b2dd9ba53a8cc4b1625b18",
            "indexes": [0, 6, 0, 6, 0, 7, 13, 104, 146, 33, 1, 11, 8, 2, 6, 2, 9, 2, 7, 11, 14, 17, 18, 1, 12, 2, 6, 2, 9, 2, 10 ],  
            "times": [0, 2, 4 ],  
            "attributes": [23, 10 ],
            "values": [13, 15, 17, 25, 27, 29 ],
        }, 
        { ... }, 
        ...
    ]

    output_file_path is the path to the output json file, output file format:
    [
        {
            "sentenceId": "eadf4c4d7eaa6cb767fa6d8c02555f5-eb85e9fb6ec57b2dd9ba53a8cc4b1625b18",
            "results": [
                [0, 10, 13 ],
                [4, 10, 17 ], 
                [2, 10, 15 ]],
        },
        { ... }, 
        ...
    ]
    '''

    ############ Complete the code below ############

    json.dump(results, open(output_file_path, 'w'))
