"""
This modules implements a collection of methods
to perform experiments from configuration files
or configuration folders (a set of configuration
files)

Each configuration file must especify 4 sets of
parameters, concerning the teacher, the learner,
the dataset and protocol

The result of an experiment is a vector of
objetcs of type TeachResult
"""

import csv
import os
from datetime import datetime
from time import sleep
from copy import copy
from ..Definitions import InputSpace
from ..Definitions import Labels
from ..GenericTeacher import Teacher
from ..GenericLearner import Learner
from ..Utils import TeachResult
from .ConfigurationReader import read_configuration_file
from ..Protocol import teach
from ..Utils.TeacherLearnerLoader import get_teacher
from ..Utils.TeacherLearnerLoader import get_learner
from ..Utils.DatasetLoader import load_dataset_from_path
from ..Utils.DatasetLoader import load_dataset_train_test_from_path

_FAMILY_SUFIX_FORMAT = "%Y_%m_%d_%H_%M_%S"
_SET_SUFIX_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
_RUN_SUFIX_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"

def create_reports_from_configuration_folder(folder_path,
	dest_folder_path, verbose = False):
	assert os.path.isdir(dest_folder_path)

	_sufix = datetime.today().strftime(_FAMILY_SUFIX_FORMAT)

	# create subfolder
	new_folder_name = "family_{}".format(_sufix)
	new_folder_path = os.path.join(dest_folder_path, new_folder_name)
	os.mkdir(new_folder_path)

	TRs = []
	all_TRs = []
	for file_name in os.listdir(folder_path):
		# ignore files that are not configuration files
		if not _is_valid_configuration_file(file_name):
			continue

		if verbose:
			print("\n" + "*"*20)
			print(file_name)

		file_path = os.path.join(folder_path, file_name)
		
		TRs_i = create_reports_from_configuration_file(file_path,
			new_folder_path, verbose)
		all_TRs = all_TRs + TRs_i
		
		# get average result
		avg_TR = TRs_i[0]
		for TR in TRs_i[1:]:
			avg_TR += TR
		avg_TR *= 1/len(TRs_i)
		
		TRs.append(avg_TR)

	return (new_folder_path, all_TRs)

def create_reports_from_configuration_file(src_path: str,
	dest_folder_path: str = None, verbose = False):
	configs = read_configuration_file(src_path)

	dataset_name = configs.dataset_name
	protocol_kwargs = configs.protocol_kwargs
	dataset_kwargs = configs.dataset_kwargs

	if dest_folder_path is None:
		dest_folder_path = configs.dest_folder

	if "path_teste" in dataset_kwargs:
		X, y, X_test, y_test = load_dataset_train_test_from_path(**dataset_kwargs)
	else:
		X, y = load_dataset_from_path(**dataset_kwargs)
		X_test = None
		y_test = None

	dataset_name = configs.dataset_name
	protocol_kwargs = configs.protocol_kwargs
	
	TRs = []
	for conf in configs:
		T = get_teacher(conf.teacher_name, conf.teacher_kwargs)
		L = get_learner(conf.learner_name, conf.learner_kwargs)
		TR_i = teach(T, L, copy(X), copy(y), copy(X_test), copy(y_test),
			dataset_name=dataset_name,
			**protocol_kwargs)
		TRs.append(TR_i)

		if verbose:
			print("-"*20)
			print(TR_i.main_infos)

	if len(TRs) == 1:
		create_report(TRs[0], dest_folder_path)
	else:
		create_reports(TRs, dest_folder_path)

	return TRs

def create_reports(v, dest_folder_path: str):
	assert os.path.isdir(dest_folder_path)

	_sufix = datetime.today().strftime(_SET_SUFIX_FORMAT)

	# create subfolder
	new_folder_name = "set_{}".format(_sufix)
	new_folder_path = os.path.join(dest_folder_path, new_folder_name)
	os.mkdir(new_folder_path)

	for teach_result in v:
		create_report(teach_result, new_folder_path)
		sleep(0.001) # avoid name colision

	return new_folder_path


def create_report(TR: TeachResult, dest_folder_path: str) -> None:
	assert os.path.isdir(dest_folder_path)

	# seconds since epoch, unique id
	_sufix = datetime.today().strftime(_RUN_SUFIX_FORMAT)

	# create subfolder
	new_folder_name = "run_{}".format(_sufix)
	new_folder_path = os.path.join(dest_folder_path, new_folder_name)
	os.mkdir(new_folder_path)

	# create summary file
	summary_file_name = "summary_{}.txt".format(_sufix)
	summary_file_path = os.path.join(new_folder_path,
		summary_file_name)
	_convert_teach_result_to_txt(TR, summary_file_path)

	# create csv file
	log_file_name = "log_{}.csv".format(_sufix)
	log_file_path = os.path.join(new_folder_path,
		log_file_name)
	_convert_log_to_csv(TR.log, log_file_path)

	return (summary_file_path, log_file_path)

def _convert_log_to_csv(log, path: str):
	assert os.path.isdir(os.path.dirname(path))

	with open(path, "w", newline='') as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerows(log)

def _convert_teach_result_to_txt(TR: TeachResult, path: str):
	assert os.path.isdir(os.path.dirname(path))

	with open(path, "w") as fp:
		fp.write(str(TR))

def _is_valid_configuration_file(file_name):
	return file_name.endswith("conf")