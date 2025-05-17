from NERDA.datasets import get_conll_data, download_conll_data
from NERDA.models import NERDA
from NERDA.gridsearch import NerdaEstimator
import warnings
warnings.filterwarnings("ignore")

download_conll_data()
training = get_conll_data('train')
validation = get_conll_data('valid')
test = get_conll_data('test')

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


model_name = 'bert-base-multilingual-uncased'
#model_name = 'microsoft/deberta-v3-base'

param_grid = {
   'epochs': [3,4,5],
   'learning_rate': [0.0001, 0.000001],
   'dropout': [0.15, 0.2, 0.3],
   'train_batch_size': [15]
}

tag_outside = 'O'

estimator = NerdaEstimator(
                  param_grid,
                  model_name,
                  training,
                  validation,
                  tag_scheme,
                  tag_outside)


result=estimator.search()
print(result)
"""
training_hyperparameters = {
        'epochs' : 1,
        'warmup_steps' : 500,
        'train_batch_size': 13,
        'learning_rate': 0.0001
}

dropout = 0.15
tag_outside = 'O'
model = NERDA(
        dataset_training = training, 
        dataset_validation = validation, 
        tag_scheme = tag_scheme, 
        tag_outside = tag_outside, 
        transformer = model_name, 
        dropout = dropout, 
        hyperparameters = training_hyperparameters
)

model.train()
print(model.evaluate_performance(test))
print(model.evaluate_validation())
"""

