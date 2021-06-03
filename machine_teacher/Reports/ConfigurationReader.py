import configparser
import json
from os.path import basename
from ..Utils.CustomIterator import CustomIterator

_SECTIONS = ('teacher', 'learner', 'dataset', 'destination')
_DATASET_SUPERSET_SECTION = {'path', 'path_teste', 'scale',
							 'is_numeric', 'shuffle_dataset'}
_PROTOCOL_SUPERSET_SECTION = {'time_limit', 'join_sets','save_best_learner'}

class _TestConfiguration:
	def __init__(self, teacher_name: str, learner_name: str,
		teacher_kwargs: dict, learner_kwargs: dict, dest_folder: str,
		dataset_kwargs: dict, protocol_kwargs: dict):
		self.teacher_name = teacher_name
		self.learner_name = learner_name
		self.dataset_kwargs = dataset_kwargs
		self.dataset_name = basename(dataset_kwargs["path"])
		self.teacher_kwargs = teacher_kwargs
		self.learner_kwargs = learner_kwargs
		self.dest_folder = dest_folder
		self.protocol_kwargs = protocol_kwargs

	def __str__(self):
		return "\n".join((
			self.teacher_name,
			self.learner_name,
			str(self.dataset_kwargs),
			str(self.teacher_kwargs),
			str(self.learner_kwargs),
			self.dest_folder,
			str(self.protocol_kwargs)
			))

class _TestConfigurations:
	def __init__(self, teacher_name: str, learner_name: str,
		teacher_kwargs: dict, learner_kwargs: dict, dest_folder: str,
		dataset_kwargs: dict, protocol_kwargs: dict):
		self.teacher_name = teacher_name
		self.learner_name = learner_name
		self.dataset_kwargs = dataset_kwargs
		self.dataset_name = basename(dataset_kwargs["path"])
		self.teacher_kwargs = teacher_kwargs
		self.learner_kwargs = learner_kwargs
		self.dest_folder = dest_folder
		self.protocol_kwargs = protocol_kwargs

		self._teacher_params = list(self.teacher_kwargs.keys())
		self._learner_params = list(self.learner_kwargs.keys())

	def __iter__(self):
		self._teacher_args_iter = self._get_teacher_args_iter()
		self._learner_args_iter = self._get_learner_args_iter()
		self._v_teacher = next(self._teacher_args_iter)

		return self

	def __next__(self):
		if self._learner_args_iter.qtd_left() == 0:
			if self._teacher_args_iter.qtd_left() == 0:
				raise StopIteration
			else:
				self._learner_args_iter = self._get_learner_args_iter()
				self._v_teacher = next(self._teacher_args_iter)

		self._v_learner = next(self._learner_args_iter)

		_d_teacher = self._get_d_teacher(self._v_teacher)
		_d_learner = self._get_d_learner(self._v_learner)

		return _TestConfiguration(self.teacher_name, self.learner_name,
			_d_teacher, _d_learner, self.dest_folder, 
			self.dataset_kwargs, self.protocol_kwargs)

	def _get_teacher_args_iter(self):
		limits = self._get_v_teacher_limits()
		return iter(CustomIterator(self._get_v_teacher_limits()))

	def _get_learner_args_iter(self):
		limits = self._get_v_learner_limits()
		return iter(CustomIterator(self._get_v_learner_limits()))

	def _get_v_teacher_limits(self):
		v = [0] * len(self._teacher_params)
		for (i, key) in enumerate(self._teacher_params):
			upper_bound_i = len(self.teacher_kwargs[key]) - 1
			v[i] = upper_bound_i

		return v

	def _get_v_learner_limits(self):
		v = [0] * len(self._learner_params)
		for (i, key) in enumerate(self._learner_params):
			upper_bound_i = len(self.learner_kwargs[key]) - 1
			v[i] = upper_bound_i

		return v

	def _get_d_teacher(self, v):
		assert len(v) == len(self._teacher_params)
		d = dict()
		for (i, key) in enumerate(self._teacher_params):
			j = v[i]
			value = self.teacher_kwargs[key][j]
			d[key] = value

		return d

	def _get_d_learner(self, v):
		assert len(v) == len(self._learner_params)
		d = dict()
		for (i, key) in enumerate(self._learner_params):
			j = v[i]
			value = self.learner_kwargs[key][j]
			d[key] = value

		return d

def read_configuration_file(path: str):
	config = configparser.ConfigParser()
	config.read(path)

	# some asserts - sections most exist
	for section_name in _SECTIONS:
		assert config.has_section(section_name), "no section {}".format(section_name)

	teacher_name, teacher_kwargs = _parse_teacher_section(config['teacher'])
	learner_name, learner_kwargs = _parse_learner_section(config['learner'])
	dataset_kwargs = _parse_dataset_section(config['dataset'])
	dest_folder = _parse_string(config['destination']['path'])

	# protocol optional args
	if config.has_section("protocol"):
		protocol_kwargs = _parse_protocol_section(config['protocol'])
	else:
		protocol_kwargs = dict()

	return _TestConfigurations(
		teacher_name, learner_name,
		teacher_kwargs, learner_kwargs, dest_folder,
		dataset_kwargs, protocol_kwargs
		)

def _sections_to_lowercase(config):
	lst_sections = list(config.sections())
	for section_name in lst_sections:
		config[section_name.lower()] = config[section_name]

def _parse_teacher_section(section):
	assert 'name' in section
	name = _parse_string(section['name'])

	kwargs = {key: _parse_value(x) for (key,x) in dict(section).items() if key != "name"}

	return (name, kwargs)

def _parse_learner_section(section):
	learner_name, learner_kwargs = _parse_teacher_section(section)
	return (learner_name, learner_kwargs)

def _parse_dataset_section(section):
	assert 'path' in section

	assert set(section) <= _DATASET_SUPERSET_SECTION

	kwargs = dict()
	
	for (key, val) in dict(section).items():
		val = _parse_value(val)

		assert len(val) == 1, "protocol parameters does not support lists"

		val = val[0]
		kwargs[key] = val

	return kwargs

def _parse_protocol_section(section):
	kwargs = dict()
	assert set(section) <= _PROTOCOL_SUPERSET_SECTION
	
	for (key, val) in dict(section).items():
		val = _parse_value(val)

		assert len(val) == 1, "protocol parameters does not support lists"

		val = val[0]
		kwargs[key] = val

	return kwargs

def _parse_value(x):
	x = json.loads(x)

	# claro que eu não vou lembrar o motivo dessas linhas...
	bool1 = isinstance(x, int)
	bool2 = isinstance(x, float)
	bool3 = isinstance(x, str)
	bool4 = isinstance(x, list) and all(isinstance(xi, int) for xi in x)
	bool5 = isinstance(x, list) and all(isinstance(xi, float) for xi in x)
	bool6 = isinstance(x, list) and all(isinstance(xi, str) for xi in x)

	assert (bool1 or bool2 or bool3 or bool4 or bool5 or bool6)

	# get rid of extra quotes
	if bool3:
		x = _parse_string(x)

	if not isinstance(x, list):
		x = [x]
	
	return x

def _parse_string(s):
	return s.strip().replace("'", "").replace('"', '')

def _parse_boolean(s):
	s = _parse_string(s).lower()
	if s in ("1", "yes", "on", "true"):
		return True
	elif s in ("0", "no", "off", "false"):
		return False
	else:
		raise ValueError