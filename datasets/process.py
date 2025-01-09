"""Knowledge Graph dataset pre-processing functions."""

import collections
import os
import pickle

import numpy as np


def get_idx(path):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}
    return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")


def get_filters(examples, n_relations):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final


def create_ood_and_easy_splits(train, valid, test):
    """
    Given original (train, valid, test) splits, create:
      1) ood_test, easy_test
      2) ood_valid, easy_valid
    
    Returns:
      ood_test, easy_test, ood_valid, easy_valid 
      (all as NumPy arrays)
    """

    # Build sets for train + valid (to filter test)
    train_valid_hr = set()
    train_valid_rt = set()

    # Concatenate train & valid for test filtering
    tv_combined = np.concatenate([train, valid], axis=0)
    for (h, r, t) in tv_combined:
        train_valid_hr.add((h, r))
        train_valid_rt.add((r, t))

    # Build sets for train only (to filter valid)
    train_hr = set()
    train_rt = set()

    for (h, r, t) in train:
        train_hr.add((h, r))
        train_rt.add((r, t))

    # Create OOD test & easy test
    ood_test = []
    easy_test = []

    for (h, r, t) in test:
        # OOD test if:
        #   1) (h, r) not in train+valid
        #   2) (r, t) not in train+valid
        if (h, r) not in train_valid_hr and (r, t) not in train_valid_rt:
            ood_test.append((h, r, t))
        else:
            easy_test.append((h, r, t))

    # Create OOD valid & easy valid
    ood_valid = []
    easy_valid = []

    for (h, r, t) in valid:
        # OOD valid if:
        #   1) (h, r) not in train
        #   2) (r, t) not in train
        if (h, r) not in train_hr and (r, t) not in train_rt:
            ood_valid.append((h, r, t))
        else:
            easy_valid.append((h, r, t))

    # Convert Python lists to NumPy arrays
    ood_test = np.array(ood_test, dtype=np.int64)
    easy_test = np.array(easy_test, dtype=np.int64)
    ood_valid = np.array(ood_valid, dtype=np.int64)
    easy_valid = np.array(easy_valid, dtype=np.int64)

    return ood_test, easy_test, ood_valid, easy_valid

        

def process_dataset(path):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    ent2idx, rel2idx = get_idx(dataset_path)
    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


if __name__ == "__main__":
    data_path = os.environ["DATA_PATH"]
    for dataset_name in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset_name)
        dataset_examples, dataset_filters = process_dataset(dataset_path)
        ood_test, easy_test, ood_valid, easy_valid = create_ood_and_easy_splits(dataset_examples["train"], dataset_examples["valid"], dataset_examples["test"])
        print(f"Dataset: {dataset_name}, OOD Test: {ood_test.shape}, Easy Test: {easy_test.shape}, "
        f"OOD Valid: {ood_valid.shape}, Easy Valid: {easy_valid.shape}")
        temp_dict = {"ood_test": ood_test, "easy_test": easy_test, "ood_valid": ood_valid, "easy_valid": easy_valid}
        #for dataset_split in ["train", "valid", "test"]:
        save_path = os.path.join(dataset_path, "newdata.pickle")
        with open(save_path, "wb") as save_file:
             pickle.dump(temp_dict, save_file)
        #with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
            #pickle.dump(dataset_filters, save_file)
