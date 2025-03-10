#!/usr/bin/env python3

import os
import csv

import argparse

from typing import Tuple, List, Optional, Dict, Union

Symbol = Union[int, str]
Quad = Tuple[Symbol, Symbol, Symbol, int]
Triple = Tuple[Symbol, Symbol, Symbol]


def parse_mapping(path: str) -> Dict[int, str]:
    res: Dict[int, str] = {}
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            symbol = row[0]
            idx = int(row[1])
            res[idx] = symbol
    return res
def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)
def split_by_time(quad_lst: List[Quad]) -> Tuple[List[Triple], List[Triple], List[Triple]]:
    # Sort quads by timestep 't'
    sorted_quads = sorted(quad_lst, key=lambda x: x[3])
    #sorted_triples = tuple2list(sorted_quads)

    sorted_triples = [(t[0], t[1], t[2]) for t in sorted_quads]
    total_quads = len(sorted_triples)
    train_end = int(0.8 * total_quads)
    dev_end = train_end + int(0.1 * total_quads)
    
    train_set = sorted_triples[:train_end]
    dev_set = sorted_triples[train_end:dev_end]
    test_set = sorted_triples[dev_end:]
    
    return train_set, dev_set, test_set

def filter_minimum_timestep(quad_lst: List[Quad]) -> List[Quad]:
    min_timestep_dict = {}
    for quad in quad_lst:
        spo = (quad[0], quad[1], quad[2])
        if spo not in min_timestep_dict or quad[3] < min_timestep_dict[spo][3]:
            min_timestep_dict[spo] = quad
    return list(min_timestep_dict.values())


def parse_tsv(path: str,
              entity2idx_path: Optional[str] = None,
              relation2idx_path: Optional[str] = None) -> List[Quad]:
    
    entity2idx = relation2idx = None
    if entity2idx_path is not None and relation2idx_path is not None:
        entity2idx = parse_mapping(entity2idx_path)
        relation2idx = parse_mapping(relation2idx_path)

    res: List[Quad] = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            subject_ = row[0]
            predicate_ = row[1]
            object_ = row[2]
            timestep_ =row[3]
            res += [(subject_, predicate_, object_, timestep_)]
    return res

def save_to_tsv(tuple_lst: List[Quad], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for tuple_ in tuple_lst:
            writer.writerow(tuple_)

def save_triple_to_tsv(tuple_lst: List[Triple], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for tuple_ in tuple_lst:
            writer.writerow(tuple_)

def main():
    parser = argparse.ArgumentParser(description="Dataset creator")
    #parser.add_argument("prefix")
    #args = parser.parse_args()

    prefix = "/Users/cosimogregucci/Desktop/ICRL-rebutt/is-cqa-complex/new_benchmarks/ICEWS18_random+H/KG_splits"
    entity2idx_path = os.path.join(prefix, 'entity2id.txt')
    relation2idx_path = os.path.join(prefix, 'relation2id.txt')

    full_path = "/Users/cosimogregucci/Desktop/ICRL-rebutt/is-cqa-complex/new_benchmarks/ICEWS18+H/KG_splits/quadruples/ICEWS18.tsv"

    quads = parse_tsv(full_path)
    triples = [(t[0], t[1], t[2]) for t in quads]

    save_triple_to_tsv(tuple_lst=triples, path=os.path.join(prefix, 'triples_ICEWS18.tsv'))

if __name__ == "__main__":
    main()