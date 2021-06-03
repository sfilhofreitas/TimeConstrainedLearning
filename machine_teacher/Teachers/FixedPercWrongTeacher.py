
import numpy as np
from ..GenericTeacher import Teacher
from ..Utils.Sampler import get_first_examples
from sklearn import preprocessing
import warnings

class FixedPercWrongTeacher(Teacher):
	"""docstring for FixedPercWrongTeacher"""
	name = "FixedPercWrongTeacher"
	_SEED = 0
	_FRAC_START = 0.01
	_STRATEGY_DOUBLE_INCREMENT = 0
	_STRATEGY_DOUBLE_SIZE = 1
	_FRAC_WRONG_INCREMENT = 0.1
	_STATE_DONE = False

	_STATE_SEND_NEW_IDS = 0
	_STATE_CHOOSE_BATCH_SIZE_NEW_IDS = 1
	_STATE_SEND_EMPTY_NEW_IDS = 2
	_SAMPLE_SIZE = 300 #sample size sent to the student to analyze the accuracy

	def __init__(self, seed: int = _SEED,
		frac_start: float = _FRAC_START,
		frac_wrong_increment = _FRAC_WRONG_INCREMENT,
		sample_size = _SAMPLE_SIZE,
		strategy: int = _STRATEGY_DOUBLE_INCREMENT):
		self.seed = seed
		self.frac_start = frac_start
		self.strategy = strategy
		self.frac_wrong_increment = frac_wrong_increment
		self.sample_size = sample_size
		

	def start(self, X, y, time_left: float):
		self._start(X, y, time_left)
		self.num_iters = 0
		self.m = y.size
		self.S_current_size = 0
		self.batch_size = self.sample_size
		self.state_new_ids = self._STATE_SEND_NEW_IDS
		self._STATE_DONE = False
		self.f_shuffle = np.random.RandomState(self.seed).shuffle
		self.shuffled_ids = self._get_shuffled_ids()
		self.last_accuracy = 0.0

		assert len(self.shuffled_ids) == len(self.ids)

	def _get_shuffled_ids(self):
		ids = np.arange(self.m, dtype=int)
		# f_shuffle = np.random.RandomState(self.seed).shuffle
		self.f_shuffle(ids)
		return ids

	def _keep_going(self):
		return self.S_current_size < self.m

	def get_first_examples(self, time_left: float):
		classes = np.unique(self.y) # isso devia sair. devia ser computado for get_first_examples
		f_shuffle = np.random.RandomState(self.seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m, classes, self.y, f_shuffle)
		new_ids = np.array(new_ids)
		
		# update shuffled_ids. Aqui usamos ponteiros de ponteiros.
		# debugue as próximas duas linhas de cabeça vazia
		_new_ids = set(new_ids)
		self.shuffled_ids = np.append(new_ids,
							          [i for i in self.shuffled_ids if i not in _new_ids])
		
		# update batch size, from 1 to len(new_ids), based on strategy
		if self.strategy == self._STRATEGY_DOUBLE_SIZE:
			self.batch_size = len(new_ids)			

		elif self.strategy == self._STRATEGY_DOUBLE_INCREMENT:
			self.batch_size = self.sample_size
		else:
			raise ValueError("Estrategia desconhecida: " + str(self.strategy))

		return self._send_new_ids(new_ids)

	def get_new_examples(self, test_ids, test_labels, time_left: float):
		if not self._keep_going() or self._STATE_DONE:
			return np.array([])

		new_ids = self.shuffled_ids[self.S_current_size:]
		correct_test_labels = self.y[test_ids]
		wrong_ids = test_ids[test_labels != correct_test_labels]
		correct_ids = test_ids[test_labels == correct_test_labels]
		new_ids = np.append(wrong_ids, correct_ids[:self.batch_size-wrong_ids.size])

		unselected_ids = np.copy(correct_ids[self.batch_size-wrong_ids.size:]) 		
		remaining_ids = np.copy(self.shuffled_ids[self.S_current_size+test_ids.size:])
		self.shuffled_ids[self.S_current_size:] = np.concatenate((new_ids, remaining_ids, unselected_ids))


		error = (len(wrong_ids)/len(test_ids))
		self.last_accuracy = 1.0 - error
		

		self.batch_size *= 2
		self.state_new_ids = self._STATE_SEND_NEW_IDS

		return self._send_new_ids(new_ids)


	def get_new_test_ids(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		if not self._keep_going():
			return np.array([])

		if self.state_new_ids == self._STATE_SEND_NEW_IDS:
			self.sample_size = self.S_current_size
			_start = self.S_current_size 
			_end = min(_start + self.sample_size, self.m)
			new_ids = self.shuffled_ids[_start:_end]
			self.state_new_ids = self._STATE_CHOOSE_BATCH_SIZE_NEW_IDS
			assert type(new_ids) == type(np.array([]))
			return new_ids
		
		if self.state_new_ids == self._STATE_CHOOSE_BATCH_SIZE_NEW_IDS:
			correct_test_labels = self.y[test_ids]
			wrong_labels = test_ids[test_labels != correct_test_labels]
			error = (len(wrong_labels)/len(test_ids))
			
			assert (error > 0 or len(wrong_labels) == 0)
			assert (error < 1 or len(wrong_labels) == len(test_ids))

			if np.isclose(error, 0): 
				self._STATE_DONE = True
				return np.array([])


			#testing a dynamic version
			# self.frac_wrong_increment = error

			increment = int((self.frac_wrong_increment*self.batch_size)/error)
			increment -= int(self.frac_wrong_increment*self.batch_size)
			increment = max(increment, 0)
			_start = (self.S_current_size + 
					  self.sample_size)
			_end = (self.S_current_size +
					self.batch_size +
					increment)

			_end = min(_end , self.m)
			new_ids = self.shuffled_ids[_start:_end]
			self.state_new_ids = self._STATE_SEND_EMPTY_NEW_IDS
			return new_ids

		return np.array([])

	def get_log_header(self):
		return ["iter_number", "training_set_size", "accuracy"]

	def get_log_line(self, h):
		accuracy = 1 - self._get_wrong_labels_id(h).size/self.y.size
		log_line = [self.num_iters, self.S_current_size, accuracy]
		return log_line

	def _send_new_ids(self, new_ids):
		self.num_iters += 1
		self.S_current_size += len(new_ids)
		return new_ids


	def get_params(self) -> dict:
		return {
			"seed": self.seed,
			"frac_start": self.frac_start,
			"strategy": self.strategy,
			"frac_wrong_increment": self.frac_wrong_increment,
			}

	def _get_accuracy(self, h=None):		

		return self.last_accuracy

		
