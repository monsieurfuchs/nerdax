from NERDA.models import NERDA
from sklearn.base import BaseEstimator

class NerdaEstimator(BaseEstimator):

    def __init__(self, transformer='bert-base-uncased', max_epochs=5, learning_rate=0.000001, dropout=0.1):
        self.transformer = transformer
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.dropout = dropout


    def fit(self, x, y=None):

        self.model = NERDA(
            transformer=self.transformer,
            max_epochs=self.max_epochs,
            hyperparameters={'learning_rate': self.learning_rate, 'dropout': self.dropout}
        )
        self.model.train()
        return self

    def score(self, x, y):
        # Use validation accuracy or F1 score as the metric
        evaluation = self.model.evaluate_validation
        return self.model.evaluate_validation()['f1']
