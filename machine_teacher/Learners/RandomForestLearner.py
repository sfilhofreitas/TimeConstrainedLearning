from ..GenericLearner import Learner
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestLearner(Learner):
	name = "RandomForestLearner"

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs

	def start(self):
		self.model = RandomForestClassifier(*self.args, **self.kwargs)

	def fit(self, X, y):
		return self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def get_params(self):
		return self.model.get_params()