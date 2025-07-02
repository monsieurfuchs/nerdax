from NERDA.datasets import get_conll_data, download_conll_data
from NERDA.models import NERDA
from NERDA.gridsearch import NerdaEstimator
import warnings
import json


warnings.filterwarnings("ignore")

# load training data
with open('data/data_train.json', 'r') as file:
     training = json.load(file)
     
# load validation data
with open('data/data_validate.json', 'r') as file:
     validation = json.load(file)
     
# load final test data
with open('data/data_test.json', 'r') as file:    
     test = json.load(file) 


# define the tags to be trained on
tag_scheme = [
        'B-PER',
        'I-PER',
        'B-ORG',
        'I-ORG',
        'B-LOC',
        'I-LOC',
        'B-MISC',
        'I-MISC'
]


# define the model you want to use
model_name = 'bert-base-multilingual-uncased'
#model_name = 'microsoft/deberta-v3-base'


# for grid search define a parameter grid
param_grid = {
   'epochs': [1, 2],
   'learning_rate': [0.0001],
   'dropout': [0.15],
   'train_batch_size': [15]
}

# define the tag for tokens that are none of the above etities
tag_outside = 'O'

# use the Nerda Estimator to perform a hyperparameter search
estimator = NerdaEstimator(
                  param_grid,
                  model_name,
                  training,
                  validation,
                  tag_scheme,
                  tag_outside)


result=estimator.search()

print("Best result:\n-----------------")
print(result)



