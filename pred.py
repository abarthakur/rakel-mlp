from keras.models import model_from_json 
import numpy as np
import pickle


def load_model_from_file(id,folder_name="."):
	filename=folder_name+'/'+str(id)
	json_file = open(filename+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json) # need python-h5py (or python3-h5py) installed for this
	# load weights into new model
	loaded_model.load_weights(filename+'.h5')
	return loaded_model


testDict={}
with open("./cifarsorted/test","rb") as fo:
	testDict = pickle.load(fo,encoding="bytes")


classmapping=[]
with open("./models/classmap","rb") as fo:
	classmapping = pickle.load(fo,encoding="bytes")


NUM_CLASSIFIERS=10
CLASSES_PER_CLASSIFIER=15
TOTAL_CLASSES=100
OTHER=TOTAL_CLASSES-CLASSES_PER_CLASSIFIER
full_set=set(range(0,TOTAL_CLASSES))


ensemble=[]
for i in range(0,NUM_CLASSIFIERS):
	m=load_model_from_file(i,"models")
	ensemble.append(m)

testX=testDict["data"]
testY=testDict["labels"]

predictions=np.zeros(testX.shape[0])
correct_count=0
for i in range(0,testX.shape[0]):
	y_pred=np.zeros(TOTAL_CLASSES)
	for j in range(0,NUM_CLASSIFIERS):
		this_classes = classmapping[j]
		other_classes = full_set - set(this_classes)
		other_classes=list(other_classes)
		sample=testX[i].reshape((1,3,32,32))
		y_part=ensemble[j].predict(sample).reshape((CLASSES_PER_CLASSIFIER+1))
		# print(y_part.shape)
		y_pred[this_classes]+=y_part[:-1]
		y_pred[other_classes]+= (y_part[-1])/OTHER
	pred_label=np.argmax(y_pred)
	correct_count+= int(pred_label==testY[i]) 

accuracy = float(correct_count)/testY.shape[0]
print("Accuracy of the ensemble is:")
print(accuracy)





