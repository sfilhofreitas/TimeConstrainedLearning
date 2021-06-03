#Versao Original artigo

from ..GenericTeacher import Teacher
from ..Utils.Sampler import get_first_examples
from ..Utils.Sampler import choose_ids
import numpy as np
import warnings

# numpy convetions
_ROW_AXIS = 0
_FRAC_START = 0.01
_FRAC_STOP = 1.0
_SEED = 0
_FIRST_EXAMPLE_SEED = 0

class WTFTeacher(Teacher):
	name = "WTFTeacher"
	
	def __init__(self, seed: int = _SEED,
		frac_start: float = _FRAC_START,
		frac_stop: float = _FRAC_STOP,
		first_example_seed: int = _FIRST_EXAMPLE_SEED):
		self.seed = seed
		self.frac_start = frac_start
		self.frac_stop = frac_stop
		self.first_examples_seed = first_example_seed
		
		assert 0.0 <= frac_start <= 1.0, "frac start most be in [0, 1]"
		assert frac_start <= frac_stop <= 1.0, "frac start most be in [frac_start, 1]"

	def start(self, X, y, time_left: float) -> None:
		super()._start(X, y, time_left)

		m = X.shape[_ROW_AXIS] # number of rows
		self.m = m
		self.S_max_size = int(m * self.frac_stop)
		#self.first_batch_size = int(m * self.frac_start)
		self.num_iters = 0
		self.selected = np.full(m, False)
		self._random = np.random.RandomState(self.seed)
		self.w = np.full(m, 1/(2.0*m))	
		self.samples = []
		self.n = 1
		#self.n = int(m * self.frac_start)
		self.classes = np.unique(self.y)
		self.S_current_size = 0

	def _keep_going(self, test_labels) -> bool:
		assert len(test_labels) == self.m

		if len(self._get_delta_h(test_labels)) == 0:
			return False
		elif self.S_current_size >= self.S_max_size:
			return False
		else:
			return True

	def get_first_examples(self, time_left: float):
		f_shuffle = np.random.RandomState(self.first_examples_seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m,
			self.classes, self.y, f_shuffle)
		new_ids = np.array(new_ids)
		return self._send_new_ids(new_ids)

	def get_new_examples(self, test_ids, test_labels, time_left: float):
		if not self._keep_going(test_labels):
			return np.array([])

		wrong_labels = self._get_delta_h(test_labels)
		
		new_ids = []
		while new_ids == []:
			new_w, delta_w = self._get_new_weights_and_delta_w(wrong_labels)
			self.w = new_w
			new_ids = self._select_examples(wrong_labels, delta_w) #cabe melhoria
			if new_ids == []:
				self.n *= 2
				new_w.fill(1/(2*self.m))

		new_ids = np.array(new_ids)
		if self.S_current_size + len(new_ids) > self.S_max_size:
			size = self.S_max_size - self.S_current_size
			new_ids = new_ids[:size]

		return self._send_new_ids(new_ids)

	def _send_new_ids(self, new_ids):
		# updates
		self.num_iters += 1
		self.S_current_size += len(new_ids)
		self.selected[new_ids] = True
		
		return new_ids

	def _get_delta_h(self, test_labels):
		delta_h = self._get_wrong_labels_id(test_labels)
		self.last_accuracy = (self.m - len(delta_h))/(self.m)
		delta_h = [i for i in delta_h if not self.selected[i]] #analisar se cabe melhoria com setdiff1d
		delta_h = np.array(delta_h)
		return delta_h

	def _get_new_weights_and_delta_w(self, wrong_labels):
		new_w = np.copy(self.w)
		v = np.sum(new_w[wrong_labels])

		if v >= 1.0: #The algorithm failed
			self.n *= 2
			new_w.fill(1/(2*self.m))
			v = (1/(2*self.m)) * wrong_labels.size
		
		k = 1
		while v*k < 1.0:
			k = k*2

		old_w = np.copy(new_w[wrong_labels])
		new_w[wrong_labels] *= k
		delta_w = (new_w[wrong_labels] - old_w)/2
		return (new_w, delta_w)

	def _select_examples(self, wrong_labels, delta_w):
		random_numbers = self._random.rand(self.n)
		random_numbers = np.sort(random_numbers)

		j = 0
		i = 0
		aux = 0
		flag = True
		N = wrong_labels.size
		S = []
		while (j < self.n) and (i < N):
			if flag:
				aux += delta_w[i]
			flag = False
			if random_numbers[j] <= aux:				
				if random_numbers[j] > (aux-delta_w[i]):
					S.append(wrong_labels[i])
					flag = True
					i+=1
				j+=1      
			else:
				i+=1
				flag = True
		
		return S
		
	def get_log_header(self):
		return ["iter_number", "n", "training_set_size", "accuracy"]

	def get_log_line(self, test_labels):
		assert len(test_labels) == self.m

		accuracy = 1 - self._get_wrong_labels_id(test_labels).size/self.y.size
		log_line = [self.num_iters, self.n, self.S_current_size, accuracy]
		return log_line

	def get_params(self):
		return {
			"seed": self.seed, 
			"frac_start": self.frac_start,
			"frac_stop": self.frac_stop,
		}

	def _get_accuracy(self, h=None):		

		return self.last_accuracy

