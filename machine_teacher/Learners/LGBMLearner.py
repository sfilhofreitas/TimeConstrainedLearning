from ..GenericLearner import Learner
import lightgbm as LGBM
import numpy as np


class LGBMLearner(Learner):
	name = "LGBMLearner"

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs

	def start(self):
		self.model = LGBM.LGBMClassifier(*self.args, **self.kwargs)
		
	def fit(self, X, y):
		return self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def get_params(self):
		return self.model.get_params()