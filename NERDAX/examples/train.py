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
# model_name = 'FacebookAI/xlm-roberta-large'

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
        max_len = 300, 
        hyperparameters = training_hyperparameters
)

def run_train():
    model.train()
    print(model.evaluate_performance(test))

    print("\nSample prediction:")
    text = "Der ehrenwerte Heinz Müller lag völlig falsch, als er die Existenz der Stadt Bielefeld anzweifelte."
    print(text)
    return model.predict_text(text, return_tensors=True, return_transformer_outputs=True)
