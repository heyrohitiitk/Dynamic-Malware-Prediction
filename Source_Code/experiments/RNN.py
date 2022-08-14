from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Bidirectional
from keras.regularizers import l1, l2, l1_l2
from .useful import *
from .Configs import get_all


class RecurrentModel():
	def __init__(self, x_train, y_train,
		parameters):
		self.current_params = parameters
		self.x_train = x_train
		self.output_dim = len(y_train[0])
		for key in get_all():
			if key not in parameters:
				raise KeyError(key, "parameter missing")


	def get_model(self):
		dims = tuple(list(self.x_train.shape)[1:]) # tuple to list, omit dataset size
		model = Sequential()
		#input
		return_sequences = True
		if self.current_params["depth"] == 1:
			return_sequences = False

		model.add(self.input_layer(dims, return_sequences)) 
		#hidden
		if self.current_params["depth"] > 2:
			for i in range(self.current_params["depth"] - 2):
				model.add(self.hidden_layer(True))
		if self.current_params["depth"] >= 2:
			model.add(self.hidden_layer(False))
		#output
		model.add(self.output_layer(self.output_dim)) 

		model.compile(
			optimizer=self.current_params["optimiser"],
			  loss=self.current_params["loss"],
			  metrics=['acc']
		)		
		return model

	def generate_regulariser(self, l1_value, l2_value):
		if l1_value and l2_value:
			return l1_l2(l1=l1_value, l2=l2_value)
		elif l1_value and not l2_value:
			return l1(l1_value)
		elif l2_value:
			return l2(l2_value)
		else:
			return None


	def input_layer(self, dims, return_sequences):
		if self.current_params["bidirectional"] == True:
			return Bidirectional(self.middle_hidden_layer(return_sequences), input_shape=dims)

		else:	
			if self.current_params["layer_type"]  == "GRU":
				return GRU(self.current_params["hidden_neurons"], 
					input_shape=dims,
					return_sequences=return_sequences, 
					kernel_initializer=self.current_params["kernel_initializer"], 
					recurrent_initializer=self.current_params["recurrent_initializer"], 
					recurrent_regularizer=self.generate_regulariser(self.current_params["r_l1_reg"], self.current_params["r_l2_reg"]), 
					bias_regularizer=self.generate_regulariser(self.current_params["b_l1_reg"], self.current_params["b_l2_reg"]),
					dropout=self.current_params["dropout"], 
					recurrent_dropout=self.current_params["recurrent_dropout"]
				)

			return LSTM(self.current_params["hidden_neurons"], 
				input_shape=dims,
				return_sequences=return_sequences, 
				kernel_initializer=self.current_params["kernel_initializer"], 
				recurrent_initializer=self.current_params["recurrent_initializer"], 
				recurrent_regularizer=self.generate_regulariser(self.current_params["r_l1_reg"], self.current_params["r_l2_reg"]), 
				bias_regularizer=self.generate_regulariser(self.current_params["b_l1_reg"], self.current_params["b_l2_reg"]),
				dropout=self.current_params["dropout"], 
				recurrent_dropout=self.current_params["recurrent_dropout"] 
			)

	def hidden_layer(self, return_sequences):
		layer = self.middle_hidden_layer(return_sequences)
		if self.current_params["bidirectional"] == True:
			return Bidirectional(layer)
		return layer

	def middle_hidden_layer(self, return_sequences):
		if self.current_params["layer_type"]  == "GRU":
			return GRU(self.current_params["hidden_neurons"], 
				return_sequences=return_sequences, 
				kernel_initializer=self.current_params["kernel_initializer"], 
				recurrent_initializer=self.current_params["recurrent_initializer"], 
				recurrent_regularizer=self.generate_regulariser(self.current_params["r_l1_reg"], self.current_params["r_l2_reg"]), 
				bias_regularizer=self.generate_regulariser(self.current_params["b_l1_reg"], self.current_params["b_l2_reg"]),
				dropout=self.current_params["dropout"], 
				recurrent_dropout=self.current_params["recurrent_dropout"]
			)
		
		return LSTM(self.current_params["hidden_neurons"], 
			return_sequences=return_sequences, 
			kernel_initializer=self.current_params["kernel_initializer"], 
			recurrent_initializer=self.current_params["recurrent_initializer"], 
			recurrent_regularizer=self.generate_regulariser(self.current_params["r_l1_reg"], self.current_params["r_l2_reg"]), 
			bias_regularizer=self.generate_regulariser(self.current_params["b_l1_reg"], self.current_params["b_l2_reg"]),
			dropout=self.current_params["dropout"], 
			recurrent_dropout=self.current_params["recurrent_dropout"]
			) 

	def output_layer(self, possible_classes):
		return Dense(
			possible_classes,
			activation=self.current_params["activation"],
		)


def generate_model(x,y,params,model_type="rnn"):
	model_gen = RecurrentModel(x,y,params)
	a = model_gen.get_model()
	#plot_model(a,to_file='rnn.png',show_shapes=True,show_layer_names=True,show_layer_activations=True) 
	#visualizer(a,format='png',view=True)
	#visualkeras.layered_view(a).show()
	return a


	
	
