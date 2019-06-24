import pycrfsuite
import os
import pickle

from crfsuite_data import prepare_data

with open(os.path.join("data/out", "train.pkl"), "rb") as f:
    train = pickle.load(f)

trainer = pycrfsuite.Trainer(verbose=True)

trainer.set_params({'c1': 0.1,   # coefficient for L1 penalty
                    'c2': 0.1,  # coefficient for L2 penalty
                    'max_iterations': 250,  # stop earlier
                    'feature.possible_transitions': False})

for i, data in enumerate(train):
    for features, ylabel in prepare_data(data):
        trainer.append(features, ylabel)
    trainer.train("exp_{}".format(i))
    print("Model {} Trained".format(i))
