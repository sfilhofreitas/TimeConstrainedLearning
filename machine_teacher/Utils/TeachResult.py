"""
This modules implements the class TeachResult,
the result of a set of interactions between a teacher T
and a Learner L over a Dataset (X, y)

The result contains the teaching set ids (ids of the
examples) provided by T to L, so that L tries to fit
these examples, the hypothesis h (how L classifies
each example in the entire dataset) and some statistics
of the intereactions

"""

from datetime import datetime
from copy import deepcopy
from copy import copy
from math import isclose

from ..GenericTeacher import Teacher
from ..GenericLearner import Learner
from ..Definitions import Labels
from .Timer import Timer

class TeachResult:
	_DT_FORMAT = "%Y-%m-%d %H:%M"
	_DATASET_STD_NAME = "???"

	def __init__(self, T: Teacher, L: Learner,
		S_ids, h: Labels, timer: Timer,
		qtd_iters: int,
		qtd_attributes: int,
		log, # vetor de linhas do log, cada linha é o estado de uma iteração
		time_limit: float,
		qtd_classes: int,
		dist_classes, # vetor com o % de cada classe,
		validation_set_accuracy: float,
		dataset_name: str = _DATASET_STD_NAME):

		# output
		self.h = h
		self.S_ids = S_ids

		# stats
		self.main_infos = _MainInfos(
			T.name, # teacher_name
			L.name, # learner_name
			dataset_name, # dataset_name
			len(h), #"dataset_qtd_examples"
			qtd_attributes, # qtd_attributes
			qtd_classes, # qtd_classes
			dist_classes, # dist_classes
			time_limit, # time_limit
			timer.total_time, # total_time
			qtd_iters, # qtd_iters
			len(S_ids), # teaching_set_size
			T._get_accuracy(h), # accuracy,
			timer["get_examples"], # get_examples_time
			timer["training"], # training time
			timer["classification"], # classification time,
			validation_set_accuracy
			)
		
		self.timer = timer

		# teacher info
		self.log = log
		self.teacher_params = copy(T.get_params())

		# learner info
		self.learner_params = copy(L.get_params())

		# other stuff
		self.date = datetime.today().strftime(self._DT_FORMAT)

	def __str__(self):
		s1 = "-- main infos"
		s2 = "date: {}".format(self.date)
		s3 = str(self.main_infos)

		s4 = "\n-- times (in seconds)"
		s5 = str(self.timer)
		
		s6 = "\n-- teacher parameters"
		s7 = "\n".join("{}: {}".format(a,b) for (a,b) in self.teacher_params.items())

		s8 = "\n-- learner parameters"
		s9 = "\n".join("{}: {}".format(a,b) for (a,b) in self.learner_params.items())

		return '\n'.join((s1,s2,s3,s4,s5,s6,s7,s8,s9))

	def __add__(self, other):
 		new = deepcopy(self)

 		# Nones, things that does not make sense anymore
 		new.log = None
 		new.teacher_params = dict()
 		new.learner_params = dict()
 		new.date = None

 		# stats
 		new.main_infos += other.main_infos
 		new.timer += other.timer

 		return new

	def __mul__(self, alpha):
		new = deepcopy(self)
		new.main_infos *= alpha
		new.timer *= alpha
		return new

	def __truediv__(self, alpha):
		return self.__mul__(1/alpha)


class _MainInfos:
	def __init__(self, teacher_name: str, learner_name: str,
		dataset_name: str, dataset_qtd_examples: int,
		qtd_attributes: int, qtd_classes: int, dist_classes,
		time_limit: float, total_time: float,
		qtd_iters: int, 
		teaching_set_size: int, accuracy: float,
		get_examples_time: float, training_time: float, classification_time: float,
		validation_set_accuracy: float):
		self.teacher_name = teacher_name
		self.learner_name = learner_name
		self.dataset_name = dataset_name
		self.dataset_qtd_examples = dataset_qtd_examples
		self.qtd_attributes = qtd_attributes
		self.qtd_classes = qtd_classes
		self.dist_classes = dist_classes
		self.time_limit = time_limit

		self.total_time = total_time
		self.teaching_set_size = teaching_set_size
		self.accuracy = accuracy
		self.qtd_iters = qtd_iters
		self.get_examples_time = get_examples_time
		self.training_time = training_time
		self.classification_time = classification_time
		self.validation_set_accuracy = validation_set_accuracy

	@staticmethod
	def get_header():
		return ["teacher_name", "learner_name", "dataset_name",
			"dataset_qtd_examples", "dataset_qtd_attributes",
			"dataset_qtd_classes", "dataset_dist_classes",
			"time_limit", "total_time", "qtd_iters",
			"teaching_set_size", "dataset_accuracy",
			"get_examples_time", "training_time", "classification_time",
			"validation_set_accuracy"]

	def get_infos_list(self):
		return [self.teacher_name, self.learner_name,
			self.dataset_name, self.dataset_qtd_examples,
			self.qtd_attributes, self.qtd_classes, self.dist_classes,
			self.time_limit, self.total_time, 
			self.qtd_iters, self.teaching_set_size, self.accuracy,
			self.get_examples_time, self.training_time, self.classification_time,
			self.validation_set_accuracy]

	def __add__(self, other):
		assert self.teacher_name == other.teacher_name
		assert self.learner_name == other.learner_name
		assert self.dataset_name == other.dataset_name
		assert self.dataset_qtd_examples == other.dataset_qtd_examples
		assert self.qtd_attributes == other.qtd_attributes
		assert self.qtd_classes == other.qtd_classes
		assert self.dist_classes == other.dist_classes
		assert isclose(self.time_limit, other.time_limit)

		new = deepcopy(self)

		new.total_time += other.total_time
		new.teaching_set_size += other.teaching_set_size
		new.accuracy += other.accuracy
		new.qtd_iters += other.qtd_iters
		new.get_examples_time += other.get_examples_time
		new.training_time += other.training_time
		new.classification_time += other.classification_time
		new.validation_set_accuracy += other.validation_set_accuracy
		
		return new

	def __mul__(self, alpha):
		new = deepcopy(self)

		new.total_time *= alpha
		new.teaching_set_size *= alpha
		new.accuracy *= alpha
		new.qtd_iters *= alpha
		new.get_examples_time *= alpha
		new.training_time *= alpha
		new.classification_time *= alpha
		new.validation_set_accuracy *= alpha
		
		return new

	def __truediv__(self, alpha):
		return self.__mul__(1/alpha)

	def __str__(self):
		_v = []
		_v.append("teacher: {}".format(self.teacher_name))
		_v.append("learner: {}".format(self.learner_name))
		_v.append("dataset: {}".format(self.dataset_name))
		_v.append("dataset qtd examples: {}".format(self.dataset_qtd_examples))
		_v.append("qtd attributes: {}".format(self.qtd_attributes))
		_v.append("qtd classes: {}".format(self.qtd_classes))
		_v.append("dataset dist classes: {}".format(self.dist_classes))
		_v.append("time limit {:.3f}".format(self.time_limit))
		_v.append("total time: {:.3f}".format(self.total_time))
		_v.append("qtd iters: {}".format(self.qtd_iters))
		_v.append("teaching set size: {}".format(self.teaching_set_size))
		_v.append("accuracy: {:.3f}".format(self.accuracy))
		_v.append("get examples time: {:.3f}".format(self.get_examples_time))
		_v.append("training time: {:.3f}".format(self.training_time))
		_v.append("classification time: {:.3f}".format(self.classification_time))
		_v.append("validation set accuracy: {:.3f}".format(self.validation_set_accuracy))

		return "\n".join(_v)

