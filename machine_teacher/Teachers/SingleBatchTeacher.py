from ..GenericTeacher import Teacher
import numpy as np
import warnings
from sklearn import preprocessing

class SingleBatchTeacher(Teacher):
	name = "SingleBatchTeacher"

	def __init__(self, seed=0, frac_dataset = 1.0):
		self.seed = seed
		self.frac_dataset = frac_dataset
		

	def start(self, X, y, time_left: float):
		self.m = len(y)
		self.shuffled_ids = self._get_shuffled_ids()
		return self._start(X, y, time_left)

	def get_first_examples(self, time_left: float) -> np.ndarray:
		size = np.round(self.frac_dataset*len(self.y))
		return self.shuffled_ids[:int(size)]

	def get_new_examples(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		return np.array([])

	def get_new_test_ids(self, test_ids, test_labels,
		time_left: float) -> np.ndarray:
		return []

	def _get_shuffled_ids(self):
		ids = np.arange(self.m, dtype=int)
		f_shuffle = np.random.RandomState(self.seed).shuffle
		f_shuffle(ids)
		return ids

	def get_params(self) -> dict:
		return {
			"seed": self.seed,
			"frac_dataset": self.frac_dataset
			}
