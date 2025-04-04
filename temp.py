import numpy as np
import os
import pickle

train = np.loadtxt("data/WN18RR/train_ood_cleaned_WN18RR.txt", dtype=int, delimiter="\t")
test = np.loadtxt("data/WN18RR/test_ood_cleaned_WN18RR.txt", dtype=int, delimiter="\t")
valid = np.loadtxt("data/WN18RR/valid_ood_cleaned_WN18RR.txt", dtype=int, delimiter="\t")

print(train.shape)
print(test.shape)
print(valid.shape)

# Save each matrix to a pickle file
with open('train_ood_only_cleaned_NELL-995-h50.pickle', 'wb') as f:
    pickle.dump(train, f)

with open('valid_ood_only_cleaned_NELL-995-h50.pickle', 'wb') as f:
    pickle.dump(valid, f)

with open('test_ood_only_cleaned_NELL-995-h50.pickle', 'wb') as f:
    pickle.dump(test, f)

'''with open('data/ICEWS18R/train_50_easy_ood_ICEWS18R.pickle', 'rb') as f:
    train = pickle.load(f)

with open('data/ICEWS18R/test_50_easy_ood_ICEWS18R.pickle', 'rb') as f:
    test = pickle.load(f)

with open('data/ICEWS18R/valid_50_easy_ood_ICEWS18R.pickle', 'rb') as f:
    valid = pickle.load(f)

print(train.shape)
print(test.shape)
print(valid.shape)

ood_test = temp_dict["ood_test"]
easy_test = temp_dict["easy_test"]
ood_valid = temp_dict["ood_valid"]
easy_valid = temp_dict["easy_valid"]

print(f"ood_test: {ood_test.shape}")
print(f"easy_test: {easy_test.shape}")
print(f"ood_valid: {ood_valid.shape}")
print(f"easy_valid: {easy_valid.shape}")'''
