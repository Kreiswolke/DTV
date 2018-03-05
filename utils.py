#!/usr/bin/python

import numpy as np 
import os
import cPickle as pickle

def set_up_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def get_data(case, src_dir = '/data/'):
    if case == 'glove_20NG':
	src_dir = os.path.join(src_dir, '20NewsGroups_glove')
        corp_train = pickle.load(open(os.path.join(src_dir, 'glove_train_corpus_N_11293.p'),'r'))
        doc_dict_train = pickle.load(open(os.path.join(src_dir, 'glove_train_doc_dict_N_11293.p'),'r'))
        corpus_train = corp_train['corpus']

        corp_test = pickle.load(open(os.path.join(src_dir, 'glove_test_corpus_N_7528.p'),'r'))
        doc_dict_test = pickle.load(open(os.path.join(src_dir, 'glove_test_doc_dict_N_7528.p'),'r'))
        corpus_test = corp_test['corpus']
        vocab = pickle.load(open(os.path.join(src_dir, 'vocab.pkl'),'r'))

    elif case == 'glove_RCV1':
	src_dir = os.path.join(src_dir, 'RCV1_glove')
        corp_train = pickle.load(open(os.path.join(src_dir, 'glove_rcv1_train_corpus_N_781265.p'),'r'))
        doc_dict_train = None
        corpus_train = corp_train['corpus']

        corp_test = pickle.load(open(os.path.join(src_dir, 'glove_rcv1_test_corpus_N_23149.p'),'r'))
        doc_dict_test = None
        corpus_test = corp_test['corpus']
        vocab =  pickle.load(open(os.path.join(src_dir, 'vocab.pkl'),'r'))


    elif case == 'glove_iclr17_20NG':
	src_dir = os.path.join(src_dir, '20NewsGroups_ICLR17')
        corp_train = pickle.load(open(os.path.join(src_dir, 'glove_iclr17_train_corpus_N_11258.p'),'r'))
        doc_dict_train = None
        corpus_train = corp_train['corpus']
        
        corp_test = pickle.load(open(os.path.join(src_dir, 'glove_iclr17_test_corpus_N_7487.p'),'r'))
        doc_dict_test = None
        corpus_test = corp_test['corpus']
	vocab =  pickle.load(open(os.path.join(src_dir, 'vocab.pkl'),'r'))


    elif case.split('_')[0] == 'FN':
	src_dir = os.path.join(src_dir, 'FinancialNews')
        year = case.split('_')[1]
        data_dir = os.path.join(src_dir, year)

        corp_train = pickle.load(open(glob.glob(os.path.join(data_dir, 'train_FinNews_{}_corpus_*'.format(year)))[0],'r'))
        corpus_train = corp_train['corpus']
        doc_dict_train = None

        corp_test = pickle.load(open(glob.glob(os.path.join(data_dir, 'test_FinNews_{}_corpus_*'.format(year)))[0],'r'))
        corpus_test = corp_test['corpus']
        doc_dict_test = None

        vocab = pickle.load(open(os.path.join(src_dir, 'vocab.pkl'),'r'))

    else:
        raise

    return corpus_train, doc_dict_train, corpus_test, doc_dict_test, vocab



