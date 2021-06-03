"""
Esse script verifica se existe muita diferenca entre
classificar (1) n exemplos 1 por 1 e (2) classificar n exemplos de uma vez
para o learner SVM

O tamanho de cada dataset foi limitado em 100.000 exemplos
"""

import os
import sys
import pandas as pd
import numpy as np
import csv

_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(_PATH, os.path.pardir)))

import machine_teacher
from machine_teacher.Utils.Timer import Timer
from machine_teacher.Utils.DatasetLoader import load_dataset_from_path

def do_it(L, X, y, batch_size, timer):
	L.start()

	# treina
	timer.tick("{}__treina".format(batch_size))
	L.fit(X, y)
	timer.tock()
	
	# classifica
	labels = np.array([], dtype=int)
	qtd_batchs = len(y) // batch_size
	if (len(y) % batch_size) != 0:
		qtd_batchs += 1

	timer.tick("{}__classifica".format(batch_size))
	for i in range(qtd_batchs):
		ini = i*batch_size
		fim = min(len(y), ini+batch_size)
		labels = np.append(labels, L.predict(X[ini:fim, :]))
	timer.tock()

	return labels

_DATASET_NAMES = (
	"agaricus-lepiota.csv",
	"avila_tr.csv",
	"bank_marketing.csv",
	"car.csv",
	"ClaveVectors_Firm_Teacher_Model.csv",
	"covtype.csv",
	"crowdsourced.csv",
	"default_of_credit_card_clients.csv",
	"Electrical_grid_stability_simulated_data.csv",
	"HTRU.csv",
	"mnist_train.csv",
	"nursery.csv",
	"poker_hand_train.csv",
	"Sensorless_drive_diagnosis.csv",
	"shuttle.csv",
	"Skin_NonSkin.csv",
	)

def main():
	output_name = "classification_times_test.xlsx"
	learner_args = {"random_state": 0, "dual": False}
	_MAX_ROWS = 100_000
	_QTD_RUNS = 5

	#print(os.path.basename(dataset_path))

	_HEADER = ["dataset", "qtd_exemplos", "qtd_atribtos",
			   "single batch - treina", "single batch - classifica",
			   "tam=1 - treina", "tam=1 - classifica",
			   "tam=100 - treina", "tam=100 - classifica",
			   "tam=0.5% - treina", "tam=0.5% - classifica"
			  ]

	body = [ [0]*len(_HEADER) for __ in range(len(_DATASET_NAMES)) ]

	for __ in range(_QTD_RUNS):
		for (i, dataset_name) in enumerate(_DATASET_NAMES):
			print("\n\nInciando experimento com o dataset " + dataset_name)
			timer = Timer()
			timer.start()
			dataset_path = os.path.join(os.pardir, "garagem", "datasets", dataset_name)

			# single batch
			X, y = load_dataset_from_path(dataset_path)
			X = X[0:min(len(y), _MAX_ROWS), :]
			y = y[0:min(len(y), _MAX_ROWS)]
			batch_size = len(y)
			L = machine_teacher.Learners.SVMLinearLearner(**learner_args)
			labels_1 = do_it(L, X, y, batch_size, timer)

			# batch size: one
			X, y = load_dataset_from_path(dataset_path)
			X = X[0:min(len(y), _MAX_ROWS), :]
			y = y[0:min(len(y), _MAX_ROWS)]
			batch_size = 1
			L = machine_teacher.Learners.SVMLinearLearner(**learner_args)
			labels_2 = do_it(L, X, y, batch_size, timer)

			# batch size: hundred
			X, y = load_dataset_from_path(dataset_path)
			X = X[0:min(len(y), _MAX_ROWS), :]
			y = y[0:min(len(y), _MAX_ROWS)]
			batch_size = 100
			L = machine_teacher.Learners.SVMLinearLearner(**learner_args)
			labels_3 = do_it(L, X, y, batch_size, timer)

			# batch size: 1%
			X, y = load_dataset_from_path(dataset_path)
			X = X[0:min(len(y), _MAX_ROWS), :]
			y = y[0:min(len(y), _MAX_ROWS)]
			batch_size = int(0.005 * len(y))
			L = machine_teacher.Learners.SVMLinearLearner(**learner_args)
			labels_4 = do_it(L, X, y, batch_size, timer)

			# comparacao
			timer.finish()
			assert labels_1.shape == labels_2.shape
			assert labels_1.shape == labels_3.shape
			assert labels_1.shape == labels_4.shape
			assert all(labels_1 == labels_2)
			assert all(labels_1 == labels_3)
			assert all(labels_1 == labels_4)

			for (j, value_j) in enumerate(timer._d.values()):
				body[i][j+3] += value_j
			
			body[i][0] = dataset_name
			body[i][1] = len(y)
			body[i][2] = X.shape[1]

	for i in range(len(_DATASET_NAMES)):
		for j in range(3, len(_HEADER)):
			body[i][j] /= _QTD_RUNS

	# escreve resultados num arquivo
	df = pd.DataFrame(body, columns = _HEADER)
	df.to_excel(output_name, index = False)

if __name__ == "__main__":
	main()

