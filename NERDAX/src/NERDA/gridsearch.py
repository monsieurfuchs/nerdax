from NERDA.models import NERDA
from NERDA.models import NERDA
from itertools import product

class NerdaEstimator():

    def __init__(self,
                 param_grid,
                 transformer,
                 dataset_training,
                 dataset_validation,
                 tag_scheme,
                 tag_outside
                 ):
        self.transformer = transformer
        self.dataset_training = dataset_training
        self.dataset_validation = dataset_validation
        self.tag_scheme = tag_scheme
        self.tag_outside = tag_outside
        self.param_grid = param_grid
        """
        self.param_grid = {
           'epochs': [],
           'learning_rate': [],
           'dropout': [], 
           'train_batch_size': []
        }"""

    def search(self):
        keys, values = zip(*self.param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        results = []

        print("Starting grid search\n----------------------")
        count = 0 
        for params in param_combinations:
            
            count += 1
            print(f"Round {count} / {len(param_combinations)}")
            print(f"Params: {str(params)}")
            model = NERDA(
                      transformer='bert-base-multilingual-uncased',
                      dataset_training=self.dataset_training,
                      dataset_validation=self.dataset_validation,
                      hyperparameters={
                         'epochs': params['epochs'],
                         'warmup_steps': 500,
                         'train_batch_size': params['train_batch_size'],
                         'learning_rate': params['learning_rate']
                      },
                      dropout=params['dropout']
                   )

            model.train()
            eval_result = model.evaluate_performance(self.dataset_validation)
            f1_score = eval_result.loc[eval_result['Level'] == 'AVG_MICRO', 'F1-Score'].values[0]
            results.append({'params': params, 'f1_score': f1_score})

        best_run = max(results, key=lambda x: x['f1_score'])
        return best_run

