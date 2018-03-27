from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json 
from keras.callbacks import EarlyStopping

import numpy as np

class MLPClassifier:

	def __init__(self,init_data=None):
		if init_data:
			self.num_labels=init_data["num_labels"]
			self.num_features=init_data["num_features"]
			self.create_model(init_data)
		self.epochs_trained=0


	def create_model(self,init_data):
		inp_shape=init_data["input_shape"]
		layers=init_data["layers"]

		assert(layers[-1]==self.num_labels)

		model=Sequential()
		model.add(Dense(layers[0],input_shape=inp_shape,activation='sigmoid'))
		for i in range(1,len(layers)):
			model.add(Dense(layers[i],activation='sigmoid'))

		self.model=model

	def train(self,trainX,trainY,epochs,use_validation=True,validation_split=0.1):
		self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy','binary_accuracy'])
		if use_validation and validation_split >0:
			#first shuffle array, since keras takes last fraction as validation set
			rand_perm = np.random.permutation(trainX.shape[0])
			trainX=trainX[rand_perm,:]
			trainY=trainY[rand_perm,:]
			early_stopping = EarlyStopping(monitor='val_loss', patience=5)
			self.model.fit(trainX,trainY, epochs=epochs, batch_size=10,validation_split=validation_split,
							callbacks=[early_stopping]) # epochs=150
		else:
			self.model.fit(trainX,trainY, epochs=epochs, batch_size=10) # epochs=150
		self.epochs_trained+=epochs
		print("Finished total "+str(self.epochs_trained)+" epochs")

		
	def set_model(self,model):
		self.model=model

	def save(self,foldername,i):
		filename=foldername+"/class_"+str(i)		
		model_json = self.model.to_json()
		with open(filename+"_mod.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights(filename+'_wts.h5')

	def load(self,foldername,i,num_features,num_labels):
		filename=filename=foldername+"/class_"+str(i)
		import os
		if not os.path.exists(filename+"_mod.json"):
			return None
		
		with open(filename+'_mod.json', 'r') as json_file:
			loaded_model_json = json_file.read()
		
		loaded_model = model_from_json(loaded_model_json) # need python-h5py (or python3-h5py) installed for this
		# load weights into new model
		loaded_model.load_weights(filename+'_wts.h5')
		self.model=loaded_model
		self.num_features=num_features
		self.num_labels=num_labels

		return self
		
	def predict(self,sample):
		return self.model.predict(sample.reshape((1,self.num_features))).reshape((self.num_labels))

