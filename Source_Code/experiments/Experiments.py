from copy import deepcopy
import numpy as np
import random
import operator
from itertools import combinations
import csv
import time
import os
import gc
import sys
import inspect

from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from .RNN import generate_model
from .useful import *

random.seed(12)
np.random.seed(12)

class Experiment():
	"""docstring for Experiment"""
	def __init__(self, parameters, search_algorithm="grid", 
		x_test=None, y_test=None,
		x_train=None, y_train=None,
		x_val=None,y_val=None,
		data=None, folds=10, 
		folder_name=str(time.time()),
		thresholding=False, threshold=0.5, model_type="rnn"):

		self.headers = ['memory', 'tx_packets',  'rx_bytes','swap','rx_packets', 'cpu_sys','total_pro','cpu_user', 'max_pid','tx_bytes']

		#hyperparameters
		self.original_h_params = parameters
		self.h_params = parameters		
		
		self.x_val = x_val
		self.y_val = y_val

		# set up parameter search space depending on algorithm
		self.search_algorithm = search_algorithm
		self.current_params = {}

		if self.search_algorithm == "grid":
			self.h_params = dict([(key, list(self.h_params[key])) for key in self.h_params])
			self.original_h_params = deepcopy(self.h_params)
			self.current_params = dict([(key, self.h_params[key][0]) for key in self.h_params])
		else :
			self.__list_to_dict_params() 
			self.__map_to_0_1() 

		self.folder_name = check_filename(folder_name)
		self.experiment_id = 0
		self.metrics_headers = None
		self.metrics_writer = None 

		
		self.model_type = model_type
		
		self.thresholding = thresholding
		if self.thresholding:
			self.min_threshold = threshold + K.epsilon()
			self.temp_min_threshold = threshold + K.epsilon()

		# test-train experiment
		if (data == None) or (folds == None):
			self.folds = None
			self.X_TRAIN = x_train
			self.Y_TRAIN = y_train
			self.X_TEST = x_test
			self.Y_TEST = y_test
			print("Test-train experiment")
		# k-fold cross-validation experiment
		else:
			assert folds != None, "Supply number of folds for k-fold cross validation or supplt x_train, y_train, x_test, y_test"
			self.folds = folds
			self.x = data[0]
			self.y = data[1]
			print(self.folds, "- fold cross validation experiment")



	def __list_to_dict_params(self):
			for key in self.h_params:
				if type(self.h_params[key]) is list:
					self.h_params[key] = dict([(x, 1/(len(self.h_params[key]) + K.epsilon() )) for x in self.h_params[key]])

	def __map_to_0_1(self):
		for key in self.h_params:
			running_total = 0
			scalar = 1/(sum(self.h_params[key].values()))
			for possible_value in self.h_params[key]:
				if self.h_params[key][possible_value] < 0:
					raise ValueError("Negative hyperparameter probabilities are not allowed ({} for {})").format(self.h_params[key][possible_value], possible_value)
				new_value = self.h_params[key][possible_value] * scalar
				self.h_params[key][possible_value] = new_value + running_total
				running_total += new_value

	def __random_config(self):
		for key in self.h_params:
			choice = random.random()
			sorted_options = sorted(self.h_params[key].items(), key=operator.itemgetter(1))
			for option in sorted_options:
				if choice < option[1]:
					self.current_params[key] = option[0]
					break
		if self.current_params["optimiser"] == "adam":
			self.current_params["learning_rate"] = 0.001
		print()

	def run_one_experiment(self):
		print("run one experiment - orig")
		#Get new configuration if random search
		if self.search_algorithm == "random":
			self.__random_config()	

		self.experiment_id += 1
		print("running expt", self.experiment_id, "of", self.num_experiments)
		print(self.current_params)

		self.metrics = {}
		self.current_fold = 1
		self.accuracy_scores = []
		print(self.folds)
		# k-fold cross-validation
		if self.folds != None:
			y = deepcopy(self.y)
			x = deepcopy(self.x)
			print('===================================')
			x, y, identifiers = remove_short_idx(x, y, list(range(len(y))), self.current_params["sequence_length"])
			labels = {}
			temp_y = deepcopy(y).flatten() # get labels as flat array to find stratified folds

			for i, class_label in zip(identifiers, temp_y):
				if class_label in labels:
					labels[class_label].append(i)
				else:
					labels[class_label] = [i]

		
			fold_indicies = [[] for x in range(self.folds)]

			for key in labels:
				labels[key] = to_chunks(labels[key], self.folds)
				for i, fold_ids in enumerate(labels[key]):
					fold_indicies[i] += fold_ids

			fold_indicies = np.array([[int(x) for x in index_set] for index_set in fold_indicies],dtype=object)
			#take new copies of the original to use indicies from the original sets
			x, y = deepcopy(self.x), deepcopy(self.y)

			for i in list(range(self.folds)):
				test = np.array([(i) % self.folds]) # one fold is test set
				train = np.array([i for i in range(self.folds) if i not in [test]]) # remaining folds are training set ( not in [test, val])

				test_idxs = np.concatenate(tuple([fold_indicies[i] for i in test]))
				train_idxs = np.concatenate(tuple([fold_indicies[i] for i in train]))

				self.x_train, self.y_train = truncate_and_tensor(x[train_idxs], y[train_idxs], self.current_params["sequence_length"])
				self.x_test, self.y_test = truncate_and_tensor(x[test_idxs], y[test_idxs], self.current_params["sequence_length"])

				self.test_idxs = test_idxs
				
				stop = self.set_up_model()
				if stop:
					return

				self.current_fold += 1

		# test-train
		else:
			self.x_train = deepcopy(self.X_TRAIN)
			self.y_train = deepcopy(self.Y_TRAIN)
			self.x_test = deepcopy(self.X_TEST)
			self.y_test = deepcopy(self.Y_TEST)

			self.test_idxs = np.array(range(1, len(self.y_test) + 1)) / 10 #divide by 10 to distinguish from training/10-fold data

			# remove short sequences - store indicies for test data
			self.x_train, self.y_train = remove_short(self.x_train, self.y_train, self.current_params["sequence_length"])
			self.x_test, self.y_test, self.test_idxs = remove_short_idx(self.x_test, self.y_test, self.test_idxs, self.current_params["sequence_length"])

			self.set_up_model()
		
		

	def set_up_model(self):
		# Leave out feature if specified in dictionary
		if "leave_out_feature" in self.current_params:
			print("Omitting feature:", self.headers[self.current_params["leave_out_feature"]])
			self.x_train = np.delete(self.x_train, self.current_params["leave_out_feature"], 2)
			self.x_test = np.delete(self.x_test, self.current_params["leave_out_feature"], 2)

		#Shuffle data 
		self.x_train, self.y_train = unison_shuffled_copies([self.x_train, self.y_train])
		self.x_test, self.y_test, self.test_idxs = unison_shuffled_copies([self.x_test, self.y_test, self.test_idxs])
		
		#scale data by test data mean and variance
		means, stdvs = get_mean_and_stdv(self.x_train)
		self.x_train = scale_array(self.x_train, means, stdvs)
		self.x_test = scale_array(self.x_test, means, stdvs)

		#Output size - in future delete any cols for categorical which are all zero 
		print("train, test set size (x):", self.x_train.shape, self.x_test.shape)
		model = generate_model(self.x_train, self.y_train, self.current_params, model_type=self.model_type)

		#if self.current_fold == 1:
		#	print(model.summary())

		return self.train_model(model) #Returns TRUE if accuracy below threshold 
			

	def train_model(self, model):
		"""run one fold and write up results"""
		print("		fold ", self.current_fold, "of", self.folds)
		metrics = self.metrics
		reset_states = ResetStatesCallback()
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=8, verbose=0, mode='auto')

		h = model.fit(
			self.x_train, self.y_train, 
			batch_size=self.current_params["batch_size"], 
			epochs=self.current_params["epochs"],
			shuffle=True,	
			verbose=0,
			callbacks=[reset_states])

		

		metrics["train_acc"] = h.history["acc"]

		pred_Y = model.predict(self.x_test, batch_size=self.current_params["batch_size"])
		p = [np.round(x[0]) for x in pred_Y]
		k = self.y_test.flatten().tolist()
		metrics["fscore"] = f1_score(k, p)
		metrics["accuracy"] = accuracy_score(k, p)		
		metrics["experiment_id"] = self.experiment_id
		metrics["fold_id"] = self.current_fold

		tn, fp, fn, tp =  confusion_matrix(k, p).ravel()
		self.metrics["tp"] = tp/k.count(1)
		self.metrics["tn"] = tn/k.count(0)
		self.metrics["fp"] = fp/k.count(0)
		self.metrics["fn"] = fn/k.count(1)
		
		if not self.metrics_headers:
			#Create files and Write file headers
			os.mkdir(self.folder_name)
			self.metrics_headers = list(metrics.keys()) + list(self.current_params.keys())
			self.metrics_file = open("{}/results.csv".format(self.folder_name), "w")
			self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=self.metrics_headers)
			self.metrics_writer.writeheader()

		#Write up metric results
		
		self.metrics_writer.writerow(merge_two_dicts(self.current_params, self.metrics))

		#Search type changes
		print("acc:", metrics["accuracy"], "fscore:", metrics["fscore"])
		for x in ['tn', 'fp', 'fn', 'tp']:
			print("{}: {}".format(x, self.metrics[x]), end=" ")
		print()
		
		#make space in memory
		del model
		gc.collect()

		self.accuracy_scores.append(metrics["accuracy"])
		if self.current_fold == self.folds:
			average_acc = sum(self.accuracy_scores) / len(self.accuracy_scores)
			print("average acc:", average_acc)

		if self.thresholding:
			if metrics["accuracy"] < self.temp_min_threshold:
				return True
			#On last fold check if average accuracy > current threshold, update temporary minimum to smallest from folds accuracy
			elif (self.current_fold == self.folds) and (average_acc > self.min_threshold):
				self.temp_min_threshold = min(self.accuracy_scores)
				self.min_threshold = average_acc
				print("* * * NEW RECORD avg acc:", average_acc, "min acc:", self.temp_min_threshold)

		return False # Only return true to stop models running


	def run_experiments(self, num_experiments=100):		
		# GRID SEARCH 
		#Find total possible configurations from options
		self.total = 1 
		for key in self.original_h_params:
			self.total *= len(self.original_h_params[key])

		if self.search_algorithm == "grid":
			header_list = list(self.h_params.keys()) #Fixed keys list to loop in order
			countdown = len(self.h_params) - 1
			self.num_experiments = self.total
			print("grid search of ", self.total, "configurations...")
			self.loop_values(header_list, countdown)

		# RANDOM SEARCH 
		elif self.search_algorithm == "random":
			self.num_experiments = num_experiments
			print("random search of ", self.num_experiments, "configurations of a possible", self.total, "configurations")
			while(self.experiment_id <= self.num_experiments):
				self.run_one_experiment()


		# Experiments run - close data files
		print(self.experiment_id, " models run.")

		self.metrics_file.close()


	def loop_values(self, header_list, countdown):
		# loop through all possible configurations in original parameter dictionary
		# http://stackoverflow.com/questions/7186518/function-with-varying-number-of-for-loops-python
		if (countdown > 0):
			for i in self.original_h_params[header_list[countdown]]:
				self.current_params[header_list[countdown]] = i
				self.loop_values(header_list, countdown - 1)
		else:	
			for i in self.original_h_params[header_list[countdown]]:
				self.current_params[header_list[countdown]] = i
				self.run_one_experiment()




