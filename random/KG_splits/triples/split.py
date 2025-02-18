#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import gzip
import bz2

from tqdm import tqdm

import argparse
import logging
import sys


def iopen(file, *args, **kwargs):
    _open = open
    if file.endswith('.gz'):
        _open = gzip.open
    elif file.endswith('.bz2'):
        _open = bz2.open
    return _open(file, *args, **kwargs)


def read_triples_incremental(path):
    logging.debug('Importing the Knowledge Graph ..')

    triples, symbols = [], set()
    with iopen(path, mode='rt') as f:
        for line in tqdm(f):
            s, p, o = line.strip().split('\t')
            triples += [(s, p, o)]
            symbols |= {s, p, o}

    sym2idx = {sym: idx for idx, sym in enumerate(symbols)}
    idx2sym = {idx: sym for sym, idx in sym2idx.items()}

    idx_triples = []
    for s, p, o in triples:
        idx_triples += [(sym2idx[s], sym2idx[p], sym2idx[o])]

    kb_triples = np.array(idx_triples)
    return kb_triples, idx2sym

def parse_args(args=None):
    argparser = argparse.ArgumentParser(
        description='KGSplitter',
        usage='train.py [<args>] [-h | --help]'
    )
    argparser.add_argument('--triples', type=str, default='triples_ICEWS18.tsv',
                           help='Path of the file containing the KG triples')

    argparser.add_argument('--train', type=argparse.FileType('w'), default='train.txt',
                           help='Path of the training set')

    argparser.add_argument('--valid',type=argparse.FileType('w'), default='valid.txt',
                           help='Path of the validation set')
    argparser.add_argument('--valid-size', action='store', type=int, default=26664,
                           help='Size of the validation set')

    argparser.add_argument('--test',  type=argparse.FileType('w'), default='test.txt',
                           help='Path of the test set')
    argparser.add_argument('--test-size',  action='store', type=int, default=26663,
                           help='Size of the test set')

    argparser.add_argument('--seed', action='store', type=int, default=0, help='Seed for the PRNG')

    return argparser.parse_args(args)
def main(args):
    triples_path = args.triples
    train_fd = args.train

    valid_fd = args.valid
    valid_size = args.valid_size
    assert valid_size > 0

    test_fd = args.test
    test_size = args.test_size
    assert test_size > 0

    seed = args.seed

    kb_triples, idx2sym = read_triples_incremental(triples_path)

    NT = kb_triples.shape[0]

    logging.debug('Number of triples in the Knowledge Graph: %s' % NT)

    train_size = NT - (valid_size + test_size)
    assert train_size > 0

    logging.debug('Generating a random permutation of RDF triples ..')

    random_state = np.random.RandomState(seed=seed)
    permutation = random_state.permutation(NT)

    shuffled_triples = kb_triples[permutation]

    logging.debug('Building the training, validation and test sets ..')

    train_triples = shuffled_triples[:train_size]
    valid_triples = shuffled_triples[train_size:][:valid_size]
    test_triples = shuffled_triples[train_size:][valid_size:][:test_size]

    logging.debug('Saving ..')

    train_fd.writelines(['\t'.join(idx2sym[idx] for idx in triple) + '\n' for triple in train_triples])
    valid_fd.writelines(['\t'.join(idx2sym[idx] for idx in triple) + '\n' for triple in valid_triples])
    test_fd.writelines(['\t'.join(idx2sym[idx] for idx in triple) + '\n' for triple in test_triples])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    main(args)
