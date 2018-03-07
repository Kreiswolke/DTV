#!/usr/bin/python

import numpy as np
import tensorflow as tf
import itertools,time
import sys, os, glob
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import argparse
from random_topics import STV
from utils import get_data, set_up_folder
import gensim

    
'''------------Word Embeddings---------------'''
word_model = gensim.models.Word2Vec.load_word2vec_format('/home/oliver/GloVe/glove2word2vec300d.txt')


'''------------Methods---------------'''
def create_minibatch(data, batch_size):
    rng = np.random.RandomState(10)
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]


def write_topic_files(res, dst, vocabulary, maxtops=10,remove_mean = False):
    T = res['topics']
    n_topics = res['n_topics']
    assert np.isclose(np.mean(np.linalg.norm(T,axis=1)), 1.)
    
    if remove_mean == True:
    	T,T_mean = remove_mean(T)
    
    index = gensim.similarities.SparseMatrixSimilarity(T, num_features = 300) # N_topics x N_docs/num_best x (topic_id,weight)
    sims = index[T]

    with open(dst, 'wb') as f:
        
        for i,t in enumerate(T):
            print(i)
            W = []
            out = word_model.similar_by_vector(t, topn=5)
                
            for word in word_model.similar_by_vector(t, topn=100000):
                if word[0] in vocabulary and len(W) < maxtops:
                #if len(W) < maxtops:
                    W.append(word[0])
                else:
                    pass

            line = ' '.join(W) + '\n'
            print(line)
            f.write(line.encode('utf-8'))

def get_weights(dir_):
    W0 = pickle.load(open(dir_,'r'))['topics']
    return W0


def train(args, corpus_train, corpus_test, vocab, W0):
    res =  {}

    N = len(corpus_train)
    g = tf.Graph()
    lr = args.lr
    lr_decay = args.gamma
    batchsize = args.bs
    max_iter = args.maxiter
    min_iter = args.miniter
    n_topics = args.t
    criterion = []

    print('Training corpus: {}, Test corpus: {}, Vocabulary: {} '.format(np.shape(corpus_train), np.shape(corpus_test), len(vocab)))
    minibatches = create_minibatch(corpus_train.astype('float32'), batchsize)
    with g.as_default():

        # Create STV Architecture
        model = STV(n_topics, 'l2','train')

	# Check if corpus is normalized to unit norm
        assert np.isclose(np.mean(np.linalg.norm(corpus_test,axis=1)), 1.)
        assert np.isclose(np.mean(np.linalg.norm(corpus_train,axis=1)), 1.)

        L_train, L_test, L_train_batches, L_train_sep = [], [], [], []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for t in range(max_iter):

		total_batch = int(np.ceil(N / batchsize))
		# Loop over all batches
                in_batch_loss = []
                in_batch_sep_loss = []
		for i in range(total_batch):
		    batch_xs = minibatches.next()

                
		    # Training
		    feed = {model.X: batch_xs, model.lr: lr}
		    train_loss, train_loss_sep, T = sess.run([model.train_op, model.loss_sep, model.W], feed)

                    in_batch_loss.append(train_loss)
		    L_train_batches.append(train_loss)
		    in_batch_sep_loss.append(train_loss_sep)

		T = T.squeeze()
                index = gensim.similarities.SparseMatrixSimilarity(T, num_features = 300) 
	        sims = index[T]

	        L_train.append(np.mean(in_batch_loss))
                L_train_sep.append((np.mean(zip(*in_batch_sep_loss)[0]), np.mean(zip(*in_batch_sep_loss)[1])))
	        assert np.isclose(np.mean(np.linalg.norm(T,axis=1)), 1.)
                
                
                # Evaluation
                feed_test = {model.X: corpus_test, model.W: T}
                
                test_loss = sess.run([model.loss], feed_test)[0]
                L_test.append(test_loss)

                print(t,train_loss, test_loss, lr)
                
                # Check for stopping criterion
                if len(L_test) >= min_iter:
                    last_3 = np.mean(L_test[-3::])
                    
                    if (last_3 >= L_test[-5]) and (last_3 >= L_test[-7]):
                        print('Stop criterion fulfilled after {} steps'.format(t))
                        criterion = [L_test[-7], L_test[-5], last_3]
                        print('crit', criterion)
                        break
                
                lr = lr_decay*lr
                
    # Collect results            
    res['loss_train'] = L_train
    res['loss_test'] = L_test
    res['loss_train_sep']=L_train_sep
    res['criterion'] = criterion
    res['W0'] = W0
    res['topics'] = T
    res['sims'] = sims
    res['n_topics'] = n_topics
    res['lr'] = lr
    res['lr_decay'] = lr_decay
    res['max_iter'] = max_iter
    res['min_iter'] = min_iter
    res['L_train_batches'] = L_train_batches
    res['batchsize'] = batchsize

    return res


'''------------Main---------------'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str,
                       help='directory to store results')
    parser.add_argument('--maxiter', type=int, default=10000,
                       help='Maximal number of training epochs')
    parser.add_argument('--miniter', type=int, default=100,
                       help='Minimal number of training epochs')
    parser.add_argument('--t', type=int,
                        help='Number of topics')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='learning rate decay')
    parser.add_argument('--case', type=str, default='glove_RCV1',
	               help='specifies which data to use')
    parser.add_argument('--expcase', type=str, default='',
	               help='specifies where to save data')
    parser.add_argument('--bs', type=int, help='batchsize')
    parser.add_argument('--weight', type=str, default=None, help='Dir to dict which contains weights')

    args = parser.parse_args()


    case = args.case
    exp_case = args.expcase
    batchsize = args.bs
    year_str = args.case.split('_')[1]

    # Get data
    corpus_train, doc_dict_train, corpus_test, doc_dict_test, vocab = get_data(case)
    corpus_inds = list(range(np.shape(corpus_train)[0]))

    # Initialize network
    if args.weight:
        print('Initialize W0 from dict')
        W0 = get_weights(args.weight)
    else:
        W0 =  np.random.normal(0,1,(args.t,300))


    # Run optimization
    res_dict = train(args,corpus_train, corpus_test, vocab, W0)

    # Save results dictionary
    res_dict['case'] = case
    res_dir = os.path.join(str(args.savedir),'t2v')
    res_str =  'res_n_{}.p'.format(res_dict['n_topics'])
    set_up_folder(res_dir)
    res_dict_file = os.path.join(res_dir, res_str)
    pickle.dump(res_dict, open(res_dict_file, "wb" ))

    # Write topic words
    topic_file = os.path.join(res_dir, 'topics_n_{}.txt'.format(res_dict['n_topics']))
    write_topic_files(res_dict, topic_file, vocab, remove_mean = False)


if __name__ == "__main__":
    main()
