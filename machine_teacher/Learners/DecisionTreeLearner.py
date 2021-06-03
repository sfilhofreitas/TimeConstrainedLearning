from ..GenericLearner import Learner
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class DecisionTreeLearner(Learner):
	name = "DecisionTreeLearner"

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs

	def start(self):
		self.model = DecisionTreeClassifier(*self.args, **self.kwargs)

	def fit(self, X, y):
		return self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def get_params(self):
		return self.model.get_params()	