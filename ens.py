from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten
from keras.models import model_from_json 

import numpy as np
import random
import pickle

from weighted_loss import weighted_categorical_crossentropy

def save_model_to_file(model,id,folder_name="."):
	filename = folder_name+'/'+str(id)
	model_json = model.to_json()
	with open(filename+'.json', "w") as json_file:
		json_file.write(model_json)
	model.save_weights(filename+'.h5')

def load_model_from_file(id,folder_name="."):
	filename=folder_name+'/'+str(id)
	json_file = open(filename+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json) # need python-h5py (or python3-h5py) installed for this
	# load weights into new model
	loaded_model.load_weights(filename+'.h5')
	return loaded_model


def get_samples(dict,this_classes,TOTAL_CLASSES,CLASSES_PER_CLASSIFER,SAMPLES_PER_NEG_CLASS):
	
	full_set= set(range(0,TOTAL_CLASSES))
	NUM_NEG_SAMPLES=(TOTAL_CLASSES-CLASSES_PER_CLASSIFER)*SAMPLES_PER_NEG_CLASS

	data = dict["data"]
	labels=dict["labels"]
	divisions=dict["div"]

	this_classes=sorted(this_classes)
	other_classes = full_set - set(this_classes)

	assert(len(other_classes)==(TOTAL_CLASSES-CLASSES_PER_CLASSIFER))

	#aggregate positive samples and labels
	for j in range(0,CLASSES_PER_CLASSIFER):
		class_label=this_classes[j]
		start=divisions[class_label]
		end = divisions[class_label+1]
		
		thisX=data[start:end]
		thisY=np.zeros(((end-start),CLASSES_PER_CLASSIFER+1))
		thisY[:,j]=np.ones((end-start))

		if j==0:
			posX=thisX
			posY=thisY
		else:
			posX = np.concatenate((posX,thisX),axis=0)
			posY = np.concatenate((posY,thisY))
	
	# print("Pos samples")
	# print(posY.shape)
	# print(posX.shape)
	# print(len(other_classes))

	#aggregate negative samples
	first=True
	for c in other_classes:
		start=divisions[c]
		end=divisions[c+1]
		r_indices=np.random.randint(low=start,high=end,size=SAMPLES_PER_NEG_CLASS)
		thisX=data[r_indices]
		if first:
			negX=thisX
			first=False
		else:
			negX=np.concatenate((negX,thisX))
	negY= np.zeros((NUM_NEG_SAMPLES,CLASSES_PER_CLASSIFER+1))
	negY[:,-1]=np.ones(NUM_NEG_SAMPLES)
	
	# print("Neg samples")
	# print(negX.shape)
	# print(negY.shape)

	allX=np.concatenate((posX,negX))
	allY=np.concatenate((posY,negY))

	allX=allX/255.0

	# print("Total samples")
	# print(allX.shape)
	# print(allY.shape)

	return allX,allY

def get_model(CLASSES_PER_CLASSIFER):
	model = Sequential()
	model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,padding="same",
				data_format="channels_first",activation="relu",input_shape=(3,32,32)))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
	model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,padding="same",
				data_format="channels_first",activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
	model.add(Flatten())
	model.add(Dense(512,activation="sigmoid"))
	model.add(Dense(CLASSES_PER_CLASSIFER+1,activation="sigmoid"))
	return model



random.seed(2)
np.random.seed(2)

NUM_CLASSIFIERS=10
CLASSES_PER_CLASSIFER=15
TOTAL_CLASSES=100
NUM_TRAINING_SAMPLES=50*1000

import sys
save=True
load=int(sys.argv[1])==1
print(load)
epochs=5

trainDict={}
with open("./cifarsorted/train","rb") as fo:
	trainDict = pickle.load(fo,encoding="bytes")

testDict={}
with open("./cifarsorted/test","rb") as fo:
	testDict = pickle.load(fo,encoding="bytes")

classmapping=[]
if load:
	with open("./models/classmap","rb") as fo:
		classmapping = pickle.load(fo,encoding="bytes")


for i in range(0,NUM_CLASSIFIERS):
	print("Training classifier "+str(i))
	this_classes = []
	if load:
		this_classes=classmapping[i]
	else:
		#random set of classes
		while(len(this_classes)<CLASSES_PER_CLASSIFER):
			x=np.random.randint(low=0,high=TOTAL_CLASSES,size=1)
			if x not in this_classes:
				this_classes.append(np.asscalar(x))

		classmapping.append(sorted(this_classes))

	trainX,trainY=get_samples(trainDict,this_classes,
							TOTAL_CLASSES,CLASSES_PER_CLASSIFER,
							SAMPLES_PER_NEG_CLASS=20)



	if load:
		model=load_model_from_file(i,"models")
	else:
		model= get_model(CLASSES_PER_CLASSIFER)



	
	SAMPLES_PER_NEG_CLASS=20
	NUM_NEG_SAMPLES=(TOTAL_CLASSES-CLASSES_PER_CLASSIFER)*SAMPLES_PER_NEG_CLASS

	# weights = np.zeros(CLASSES_PER_CLASSIFER+1) + 1.0
	# weights[-1]=500.0/NUM_NEG_SAMPLES
	# print(weights[-1])
	# loss=weighted_categorical_crossentropy(weights)
	# model.compile(loss=loss, optimizer="adam", metrics=['accuracy'])

	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
	
	model.fit(trainX,trainY, epochs=epochs, batch_size=10) # epochs=150
	

	if(save):
		save_model_to_file(model,i,"models")
		print("Saved model to disk")


	SAMPLES_PER_NEG_CLASS=20
	NUM_NEG_SAMPLES=(TOTAL_CLASSES-CLASSES_PER_CLASSIFER)*SAMPLES_PER_NEG_CLASS

	testX,testY=get_samples(testDict,this_classes,
							TOTAL_CLASSES,CLASSES_PER_CLASSIFER,
							SAMPLES_PER_NEG_CLASS)

	posX=testX[:-(NUM_NEG_SAMPLES)]
	posY=testY[:-(NUM_NEG_SAMPLES)]
	print("Accuracy on positive test examples:")
	print(model.evaluate(x=posX,y=posY))

	negX=testX[-(NUM_NEG_SAMPLES):]
	negY=testY[-(NUM_NEG_SAMPLES):]
	print("Accuracy on negative test examples:")
	print(model.evaluate(x=negX,y=negY))

	# break
if not load:
	with open("./models/classmap","wb")  as fi:
		pickle.dump(classmapping,fi)

	

