"""Evaluation script."""

import argparse
import json
import os

import torch

import models
from datasets.kg_dataset import KGDataset
from utils.train import avg_both, format_metrics

parser = argparse.ArgumentParser(description="Test")
parser.add_argument(
    '--model_dir',
    help="Model path"
)


def test(model_dir):
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, False)
    dataset2 = KGDataset(dataset_path, False, easy_test=True)
    test_examples = dataset.get_examples("test")
    test_examples2 = dataset2.get_examples("test")
    filters = dataset.get_filters()

    # load pretrained model weights
    model = getattr(models, args.model)(args)
    device = 'cuda'
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))

    # eval
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    test_metrics2 = avg_both(*model.compute_metrics(test_examples2, filters))
    return test_metrics, test_metrics2


if __name__ == "__main__":
    args = parser.parse_args()
    test_metrics, test_metrics2 = test(args.model_dir)
    print(format_metrics(test_metrics, split='test'))
    print("Finshied with ood and now starting Easy test")
    print(format_metrics(test_metrics2, split='test'))