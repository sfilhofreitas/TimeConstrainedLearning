import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
import os

_SEP = ','
_SHUFFLE_RANDOM_STATE = 0

def load_dataset_from_path(path, is_numeric = None,
	*, scale=True, shuffle_dataset =False,
	shuffle_random_state = _SHUFFLE_RANDOM_STATE):
	if is_numeric is None:
		dataset_name = os.path.basename(path)
		is_numeric = _get_is_numeric(dataset_name)

	X, y = _tmp_load_dataset(path, is_numeric, scale)

	if shuffle_dataset:
		shuffle(X, y, random_state = shuffle_random_state)

	return (X, y)

def load_dataset_train_test_from_path(path,
	path_teste, is_numeric = None, *, scale=True,
	shuffle_dataset = False, shuffle_random_state = _SHUFFLE_RANDOM_STATE):
	# carrega treino, aplica transformação em X e em Y
	# carrega teste, aplica as mesmas transformações em X e em Y
	if is_numeric is None:
		dataset_name = os.path.basename(path)
		is_numeric = _get_is_numeric(dataset_name)
	
	X_train, y_train, X_test, y_test = _tmp_load_dataset_train_test(path,
		path_teste, is_numeric, scale)

	if shuffle_dataset:
		shuffle(X_train, y_train, random_state = shuffle_random_state)
		shuffle(X_test, y_test, random_state = shuffle_random_state)

	return (X_train, y_train, X_test, y_test)

def _get_is_numeric(dataset_name):
	_d = {
		"agaricus-lepiota.csv": False,
		"agaricus-lepiota_train.csv": False,
		"agaricus-lepiota_test.csv": False,

		"avila_tr.csv": True,
		"avila_train.csv": True,
		"avila_test.csv": True,

		"bank_marketing.csv": True,
		"bank_marketing_dataset_train.csv": True,
		"bank_marketing_dataset_test.csv": True,

		"car.csv": False,
		
		"ClaveVectors_Firm_Teacher_Model.csv": True,
		"ClaveVectors_Firm_Teacher_Model_train.csv": True,
		"ClaveVectors_Firm_Teacher_Model_test.csv": True,

		"covtype.csv": True,
		"covtype_train.csv": True,
		"covtype_test.csv": True,
		
		"crowdsourced.csv": True,
		
		"default_of_credit_card_clients.csv": True,
		
		"Electrical_grid_stability_simulated_data.csv": True,
		
		"HTRU.csv": True,
		
		"mnist_train.csv": True,
		"mnist_test.csv": True,

		"nursery.csv": False,
		"nursery_train.csv": False,
		"nursery_test.csv": False,

		"poker_hand_train.csv": False,
		"poker_hand_test.csv": False,

		"Sensorless_drive_diagnosis.csv": True,
		"Sensorless_drive_diagnosis_train.csv": True,
		"Sensorless_drive_diagnosis_test.csv": True,
		
		"shuttle.csv": True,
		"shuttle_train.csv": True,
		"shuttle_test.csv": True,
		
		"Skin_NonSkin.csv": True,
		"Skin_NonSkin_train.csv": True,
		"Skin_NonSkin_test.csv": True,

		"codrna_test.csv": True,
		"codrna_train.csv": True,

		"BNG_satimage_test.csv": True,
		"BNG_satimage_train.csv": True,

		"BNG_spectf_test_train.csv": True,
		"BNG_spectf_test_test.csv": True,

		"BNG_wine_train.csv": True,
		"BNG_wine_test.csv": True,

		"BNG_eucalyptus_train.csv": True,
		"BNG_eucalyptus_test.csv": True,

		"BNG_letter_5000_1_train.csv": True,
		"BNG_letter_5000_1_test.csv": True,

		"aloi_train.csv": True,
		"aloi_test.csv": True,

		"BayesianNetworkGenerator_spambase_train.csv": False,
		"BayesianNetworkGenerator_spambase_test.csv": False,

		"BNG_mfeat_fourier_train.csv": True,
		"BNG_mfeat_fourier_test.csv": True,

		"cifar_10_train.csv": True,
		"cifar_10_test.csv": True,

		"Diabetes130US_train.csv": True,
		"Diabetes130US_test.csv": True,

		"GTSRB-HueHist_train.csv": True,
		"GTSRB-HueHist_test.csv": True,

		"jannis_train.csv": True,
		"jannis_test.csv": True,

		"MiniBooNE_train.csv": True,
		"MiniBooNE_test.csv": True,

		"nomao_train.csv": True,
		"nomao_test.csv": True,

		"SantanderCustomerSatisfaction_train.csv": True,
		"SantanderCustomerSatisfaction_test.csv": True,

		"vehicle_sensIT_train.csv": True,
		"vehicle_sensIT_test.csv": True,

		"volkert_train.csv": True,
		"volkert_test.csv": True
	}

	assert dataset_name in _d, "dataset {} " + str(dataset_name) + " nao cadastrado"

	return _d[dataset_name]

def _tmp_load_dataset(path, is_numeric, scale):
	data  = pd.read_csv(path, header = None, sep = _SEP)
	y = data[0].values

	# transforma os rótulos em inteiros a partir de zero
	le = preprocessing.LabelEncoder()
	y = le.fit_transform(y)

	# tira colunas (atributos) categóricos
	if is_numeric:
		X = data.drop(columns = [0]).values
	else:
		data = data.drop(columns = [0])
		X = pd.get_dummies(data, columns = data.columns).values

	if scale:
		preprocessing.scale(X, copy = False)
			
	return (X,y)


def _tmp_load_dataset_train_test(path_train, path_test, is_numeric, scale):
	data_train  = pd.read_csv(path_train, header=None, sep=',')
	data_test  = pd.read_csv(path_test, header=None, sep=',')
	y_train = data_train[0].values
	y_test = data_test[0].values

	if is_numeric:			
		X_train = data_train.drop(columns=[0]).values
		X_test = data_test.drop(columns=[0]).values
	else:
		data_train = data_train.drop(columns=[0])
		data_test = data_test.drop(columns=[0])
		X_train = pd.get_dummies(data_train, columns=data_train.columns).values
		X_test = pd.get_dummies(data_test, columns=data_test.columns).values

	# aplica mesmas transformações (de soma e divisão) ao
	# dataset de treino e ao dataset de teste
	if scale:
		scaler = preprocessing.StandardScaler()
		scaler.fit(X_train)
		scaler.transform(X_train, copy = False)
		scaler.transform(X_test, copy = False)


	# transforma rótulos (labels) em inteiros sequenciais
	# a partir de 0. Aplica as mesmas transformações nos
	# rótulos de treino e de teste
	le = preprocessing.LabelEncoder()
	le.fit(y_train)

	y_train = le.transform(y_train)
	y_test = le.transform(y_test)

	return (X_train, y_train, X_test, y_test)
