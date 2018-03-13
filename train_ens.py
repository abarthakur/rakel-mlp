import numpy as np
import random
import pickle

class Ensemble :

	def __init__(self,num_classifiers,num_labels=-1):
		self.num_classifiers=num_classifiers
		self.num_labels=num_labels
		self.classifier_info=[]
		for i in range(0,num_classifiers):
			self.classifier_info.append({})
			self.classifier_info[i]["label_list"]=[]
			self.classifier_info[i]["pos_tset"]=None
			self.classifier_info[i]["neg_tset"]=None
		self.classifier_list=[]
		for i in range(0,num_classifiers):
			self.classifier_list.append(None)
	

	def save(self,foldername):
		print("Saving "+foldername)
		dict={}
		dict["num_classifiers"]=self.num_classifiers
		dict["num_labels"]=self.num_labels
		dict["classifier_info"]=self.classifier_info
		import os
		if not os.path.exists(foldername):
			os.makedirs(foldername)

		for i in range(0,self.num_classifiers):
			if (self.classifier_list[i]):
				self.classifier_list[i].save(foldername,i)
				dict["classifier_info"][i]["epochs_trained"]=self.classifier_list[i].epochs_trained

		filename=foldername+"/ens.py"
		with open(filename,"wb") as fo:
			pickle.dump(dict,fo)

	def load(self,foldername,classifier):
		print("Loading "+foldername)
		filename=foldername+"/ens.py"
		dict=None
		with open(filename,"rb") as fi:
			dict= pickle.load(fi)
		if not dict:
			print("Failed to load!")
			return
		self.num_classifiers=dict["num_classifiers"]
		self.num_labels=dict["num_labels"]
		self.classifier_info=dict["classifier_info"]

		for i in range(0,self.num_classifiers):
			self.classifier_list[i]=classifier()
			self.classifier_list[i]=self.classifier_list[i].load(foldername,i)
			if self.classifier_list[i]:
				self.classifier_list[i].epochs_trained=dict["classifier_info"][i]["epochs_trained"]

	def predict(self,sample):
		predictions=np.zeros(self.num_labels)
		for i in range(0,self.num_classifiers):
			this_classifier=self.classifier_list[i]
			labels=self.classifier_info[i]["label_list"]
			p=this_classifier.predict(sample)
			for i in range(0,len(labels)):
				predictions[labels[i]]+=p[i]
		predictions=predictions/np.sum(predictions)
		return predictions


def select_pos_training_set(MAX_LABELS,num_labels,all_labels,reverse_dict):
	#random set of classes
	labels=[]
	pos_tset=set()
	while(len(labels)<MAX_LABELS):
		new_lab=np.random.randint(low=0,high=num_labels,size=1)
		new_lab=np.asscalar(new_lab)
		labels.append(new_lab)
		pos_tset= pos_tset | set(reverse_dict[new_lab])

	pos_tset=sorted(list(pos_tset))
	return labels,pos_tset

def select_neg_training_set(trainX,NUM_POINTS_NEG,pos_tset):
	num_points=trainX.shape[0]
	full_tset = set(range(0,num_points))
	full_neg_tset=full_tset-set(pos_tset)

	full_neg_tset=list(full_neg_tset)

	full_neg_X=trainX[full_neg_tset]

	from sklearn.cluster import KMeans
	import time
	
	a=time.time()
	print("Starting K means")
	kmeans = KMeans(n_clusters=NUM_POINTS_NEG).fit(full_neg_X)
	print(time.time()-a)
	
	centers = kmeans.cluster_centers_
	partition= kmeans.labels_
	min_dists = np.zeros((NUM_POINTS_NEG,1))+np.inf
	min_ids=[-1]*NUM_POINTS_NEG
	
	# a=time.time()
	for i in range(0,len(full_neg_tset)):
		tid=full_neg_tset[i]
		cluster= partition[i]
		dist = np.linalg.norm(full_neg_X[i]-centers[cluster])
		if dist < min_dists[cluster]:
			min_dists[cluster]=dist
			min_ids[cluster]=tid
	
	# print(time.time()-a)
	return min_ids


	
def build_ensemble(ensemble,trainX,trainY,all_labels,reverse_dict,MAX_LABELS,NUM_POINTS_NEG):
	num_classifiers= ensemble.num_classifiers
	num_labels=ensemble.num_labels
	num_points=len(trainX)

	for i in range(0,num_classifiers):
		print("Sampling for classifier "+str(i))
		this_info = ensemble.classifier_info[i]
		if this_info["label_list"]==[]:
			label_list,pos_tset=select_pos_training_set(MAX_LABELS,num_labels,all_labels,reverse_dict)
			neg_tset = select_neg_training_set(trainX,NUM_POINTS_NEG,pos_tset)
			this_info["label_list"]=label_list
			this_info["pos_tset"]=pos_tset
			this_info["neg_tset"]=neg_tset

#always call build ensemble before
def train_ensemble(trainX,trainY,ensemble,epochs,indices_to_train,classifier,init_data,MAX_LABELS):
	num_classifiers=ensemble.num_classifiers

	for i in indices_to_train:
		this_info=ensemble.classifier_info[i]
		this_classifier=ensemble.classifier_list[i]

		if not this_classifier:
			print("Creating classifier "+str(i))
			ensemble.classifier_list[i]=classifier(init_data)
			this_classifier=ensemble.classifier_list[i]

		posX=trainX[this_info["pos_tset"]]
		negX=trainX[this_info["neg_tset"]]
		this_trainX=np.concatenate((posX,negX),axis=0)
		
		posY=trainY[this_info["pos_tset"]]
		posY=posY[:,this_info["label_list"]]
		negY=np.zeros((len(this_info["neg_tset"]),MAX_LABELS))
		this_trainY=np.concatenate((posY,negY),axis=0)
		print("Training classifier "+str(i))
		this_classifier.train(this_trainX,this_trainY,epochs)

