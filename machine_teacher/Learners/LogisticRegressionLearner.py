from ..GenericLearner import Learner
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionLearner(Learner):
	name = "LogisticRegressionLearner"

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs

	def start(self):
		self.model = LogisticRegression(*self.args, **self.kwargs)
		super().start()

	def fit(self, X, y):
		return self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def get_params(self):
		return self.model.get_params()