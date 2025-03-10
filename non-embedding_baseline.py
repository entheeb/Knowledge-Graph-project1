"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os
import numpy as np
from collections import defaultdict
import pickle

import torch
import torch.optim
from torch import nn

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WN18RR", choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10", "ICEWS18R", "ICEWS18T"],
    help="Knowledge Graph dataset"
)


class NEWMODEL(nn.Module):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        gamma: torch.nn.Parameter for margin in ranking-based loss
        data_type: torch.dtype for machine precision (single or double)
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: torch.nn.Embedding with entity embeddings
        rel: torch.nn.Embedding with relation embeddings
        bh: torch.nn.Embedding with head entity bias embeddings
        bt: torch.nn.Embedding with tail entity bias embeddings
    """

    def __init__(self, sizes, train_dataset, valid_dataset):
        super(NEWMODEL, self).__init__()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.concatenated_dataset = torch.cat((train_dataset, valid_dataset), dim=0)
        self.sizes = sizes
        

    
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
            
        """
        hr_pair = torch.tensor(queries[:, :2], dtype=torch.long)
        return hr_pair



    
    def get_rhs(self, queries):
        """Get embeddings and biases of target entities.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
             rhs_e: torch.Tensor with targets' embeddings
                    if eval_mode=False returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: torch.Tensor with targets' biases
                         if eval_mode=False returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        
        tail = torch.tensor(queries[:, 2], dtype=torch.long)
        return tail

    
    def score(self, hr_pair, rhs, entity_relations, dict_hr, dict_tr, prediction_direction):

     """
        Computes the score tensor and target tensor for entity prediction.

        Args:
            hr_pair (torch.Tensor): Tensor of shape (num_queries, 2) containing (head, relation) pairs.
            rhs (torch.Tensor): Tensor of shape (num_queries, 1) containing the true tail (or head).
            entity_relations (dict): Dictionary mapping each entity to its set of relations.
            dict_hr (dict): Dictionary mapping (h, r) to a relation occurrence dictionary (for tail prediction).
            dict_tr (dict): Dictionary mapping (t, r) to a relation occurrence dictionary (for head prediction).
            prediction_direction (str): Either "rhs" (predict tail) or "lhs" (predict head).
            n_entities (int): Total number of entities.

        Returns:
        scores (torch.Tensor): Tensor of shape (num_queries, n_entities) with computed scores.
        target (torch.Tensor): Tensor of shape (num_queries, 1) with true entity and its score.

     """
     num_queries = hr_pair.shape[0]
     n_entities = self.sizes[0]

     # Initialize scores tensor with zeros
     total_scores = torch.zeros((num_queries, n_entities), dtype=torch.float32)
     #relation_count_scores = torch.zeros((num_queries, n_entities), dtype=torch.float32)

     # Determine which dictionary to use for lookup
     lookup_dict = dict_hr if prediction_direction == "rhs" else dict_tr

     # Compute scores
     for i, (h, r) in enumerate(hr_pair):  # Iterate over queries
        h = int(h)
        r = int(r)

        # Get the dictionary key
        key = (h, r)

        # Get the relation occurrence dictionary for the key, or default to empty
        relation_occurrences = lookup_dict.get(key, {})

        # Loop over all entities
        for entity, relations in entity_relations.items():
            total_score = 0  # Sum of occurrences for this entity

            # Sum up the occurrences of all relations of this entity
            for rel in relations:
                total_score += relation_occurrences.get(rel, 0)

            # Assign total score for the entity
            total_scores[i, entity] = total_score

            # Compute relation count score: Count relations between 'h' and each entity
            #relation_count_scores[i, entity] = len(entity_relations[entity] & entity_relations.get(h, set()))


     normalized_total_scores = self.z_score_norm(total_scores)
     #normalized_relation_count_scores = self.min_max_norm(relation_count_scores)

     # Final score
     scores =  normalized_total_scores

     # Compute target tensor
     target = torch.zeros((num_queries, 1), dtype=torch.float32)
     for i, true_entity in enumerate(rhs):
        target[i, 0] = scores[i, int(true_entity)]  # Store the score for the true entity

     return scores, target
    

    
    def min_max_norm(self, scores):
        min_val, max_val = scores.min(), scores.max()
        return (scores - min_val) / (max_val - min_val + 1e-6) if max_val > min_val else scores
    
    def z_score_norm(self, scores):
     """
       Performs Z-score normalization on scores.

       Args:
        scores (torch.Tensor): The tensor to normalize.

       Returns:
        torch.Tensor: Normalized scores.
      """
     mean = scores.mean()
     std = scores.std()

     # Avoid division by zero if std == 0
     return (scores - mean) / (std + 1e-6) if std > 0 else scores



    def forward(self):
         
     """
        Creates:
        1. dict_hr: (h, r) → related relations for t
        2. dict_tr: (t, r) → related relations for h
        3. entity_relations: entity → set of all relations connected to the entity

        Args:
            n_relations: int for the number of relations in the dataset

    
        Returns:
         tuple: (dict_hr, dict_tr, entity_relations)
             - dict_hr[(h, r)][r'] = count of r' relations related to tails of (h, r).
             - dict_tr[(t, r)][r'] = count of r' relations related to heads of (t, r).
             - entity_relations[entity] = set of all relations connected to this entity.

     """
     dict_hr = defaultdict(lambda: defaultdict(int))  # (h, r) → {r' : count}
     dict_tr = defaultdict(lambda: defaultdict(int))  # (t, r) → {r' : count}
     entity_relations = defaultdict(set)  # entity → set of connected relations
    
     num_triples = self.concatenated_dataset.shape[0]
    
     for i, (h, r, t) in enumerate(self.concatenated_dataset):
            # Create a mask that excludes the current triple (efficient slicing)
            exclude_mask = torch.ones(num_triples, dtype=torch.bool)
            exclude_mask[i] = False  # Exclude current (h, r, t)
            filtered_triples = self.concatenated_dataset[exclude_mask]

            # Compute relations for dict_hr (h, r) → related relations for t
            tail_mask = filtered_triples[:, 2] == t  # Find all triples where t appears as tail
            head_mask = filtered_triples[:, 0] == t  # Find all triples where t appears as head
            related_relations_hr = torch.unique(torch.cat((filtered_triples[tail_mask, 1], filtered_triples[head_mask, 1]), dim=0))
            
            for r_prime in related_relations_hr:
                dict_hr[(int(h), int(r))][int(r_prime)] += 1

            # Compute relations for dict_tr (t, r) → related relations for h
            tail_mask = filtered_triples[:, 2] == h  # Find all triples where h appears as tail
            head_mask = filtered_triples[:, 0] == h  # Find all triples where h appears as head
            related_relations_tr = torch.unique(torch.cat((filtered_triples[tail_mask, 1], filtered_triples[head_mask, 1]), dim=0))

            for r_prime in related_relations_tr:
                dict_tr[(int(t), int(r + (self.sizes[1] // 2)))][int(r_prime)] += 1

            # Update entity_relations dictionary (O(1) operation)
            entity_relations[int(h)].add(int(r))
            entity_relations[int(t)].add(int(r))

     return dict_hr, dict_tr, entity_relations

    def get_ranking(self, queries, filters, prediction_direction, entity_relations, dict_hr, dict_tr, batch_size=1000):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = entity_relations
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries)

                scores, targets  = self.score(q, rhs, candidates, dict_hr, dict_tr, prediction_direction)

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks

    def compute_metrics(self, examples, filters, entity_relations, dict_hr, dict_tr, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], m , entity_relations, dict_hr, dict_tr, batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at



def train(args):
    save_dir = get_savedir("unique_relation_normalized_z", args.dataset)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, debug=False, easy_test=True)
    args.sizes = dataset.get_shape()
    logging.info(f"Number of entities: {args.sizes[0]}, number of total relation x2: {args.sizes[1]}")

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()
    all_dataset = torch.cat((train_examples, valid_examples, test_examples), dim=0)
    n_relations = int(len(set(all_dataset[:, 1].tolist())))
    logging.info(f"Number of relations: {n_relations}")

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = NEWMODEL(args.sizes, train_examples, valid_examples)
    #total = count_params(model)
    #logging.info("Total number of parameters {}".format(total))
    #device = "cuda"
    #model.to(device)

    # get optimizer
    #regularizer = getattr(regularizers, args.regularizer)(args.reg)
    #optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    #optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
     #                       bool(args.double_neg))
    #counter = 0
    #best_mrr = None
    #best_epoch = None
    #logging.info("\t Start training")

    dict_hr, dict_tr, entity_relations = model.forward()
    logging.info(f"numbers of entity_relations: {len(entity_relations)}")

    '''with open("logs/03_09/FB237/unique_relation_normalized_15_42_39/dict_hr_tr_entity.pkl", "rb") as f:
        pickle_dict = pickle.load(f)
    
    dict_hr = pickle_dict["dict_hr"]
    dict_tr = pickle_dict["dict_tr"]
    entity_relations = pickle_dict["entity_relations"]
    logging.info(f"numbers of entity_relations: {len(entity_relations)}")'''
    with open(os.path.join(save_dir, "dict_hr_tr_entity.pkl"), "wb") as f:
        pickle.dump({
            "entity_relations": dict(entity_relations),
            "dict_hr": {k: dict(v) for k, v in dict_hr.items()},
            "dict_tr": {k: dict(v) for k, v in dict_tr.items()}
        }, f)

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, entity_relations, dict_hr, dict_tr))
    logging.info(format_metrics(test_metrics, split="test"))
   
if __name__ == "__main__":
    train(parser.parse_args())
