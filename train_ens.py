import numpy as np
import random
import pickle

class Ensemble :

	def __init__(self,num_classifiers,num_labels=-1,num_features=-1,labels_per_classifier=-1):
		self.num_classifiers=num_classifiers
		self.num_labels=num_labels
		self.num_features=num_features
		self.labels_per_classifier=-1
		self.classifier_info=[]
		for i in range(0,num_classifiers):
			self.classifier_info.append({})
			self.classifier_info[i]["label_list"]=[]
			self.classifier_info[i]["pos_tset"]=None
			self.classifier_info[i]["neg_tset"]=None
		self.classifier_list=[]
		for i in range(0,num_classifiers):
			self.classifier_list.append(None)

	def expand_ensemble(self,num_classifiers):
		n1=self.num_classifiers
		self.num_classifiers=num_classifiers
		for i in range(n1,num_classifiers):
			self.classifier_info.append({})
			self.classifier_info[i]["label_list"]=[]
			self.classifier_info[i]["pos_tset"]=None
			self.classifier_info[i]["neg_tset"]=None
		for i in range(n1,num_classifiers):
			self.classifier_list.append(None)
	

	def save(self,foldername):
		print("Saving "+foldername)
		dict={}
		dict["num_classifiers"]=self.num_classifiers
		dict["num_labels"]=self.num_labels
		dict["num_features"]=self.num_features
		dict["labels_per_classifier"]=self.labels_per_classifier
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
		# self.num_classifiers=dict["num_classifiers"]
		self.num_labels=dict["num_labels"]
		self.num_features=dict["num_features"]
		self.labels_per_classifier=dict["labels_per_classifier"]
		self.classifier_info=dict["classifier_info"]

		for i in range(0,self.num_classifiers):
			self.classifier_list[i]=classifier()
			self.classifier_list[i]=self.classifier_list[i].load(foldername,i,self.num_features,
															self.labels_per_classifier)
			if self.classifier_list[i]:
				try :
					self.classifier_list[i].epochs_trained=dict["classifier_info"][i]["epochs_trained"]
				except KeyError:
					self.classifier_list[i].epochs_trained=0 #should have saved it...

	def predict(self,sample):
		predictions=np.zeros(self.num_labels)
		for i in range(0,self.num_classifiers):
			this_classifier=self.classifier_list[i]
			labels=self.classifier_info[i]["label_list"]
			p=this_classifier.predict(sample)
			# p = np.around(p)
			# print("For each classifier")
			# print(labels)
			# print(np.max(p))
			for i in range(0,len(labels)):
				predictions[labels[i]]+=p[i]
			# break
		# predictions=predictions/np.sum(predictions)
		# print(predictions)
		return predictions

	def predict_batch(self,sample_batch):
		num_samples=sample_batch.shape[0]
		predictions=np.zeros((num_samples,self.num_labels))
		for i in range(0,sample_batch.shape[0]):
			prediction= self.predict(sample_batch[i,:])
			predictions[i,:]=prediction
		return predictions

	def predict_batch_with_metrics(self,sample_batch,sample_batch_labels,k_values,printResults=True):
		num_samples=sample_batch.shape[0]
		predictions=self.predict_batch(sample_batch)
		#precision at k = # of correctly predicted labels in top k / k
		metrics={}
		precision_at_k_values=[]
		#propensity scored precision at k = # of co
		# psp_at_k=np.zeros(len(k_values))
		for k in k_values:
			p_at_k=0
			# psp_at_k=0
			for i in range(0,num_samples):
				prediction = predictions[i,:]
				N=prediction.shape[0]
				top_k_preds=np.argpartition(prediction,N-k)[(N-k):].tolist()
				# print(top_k_preds)
				#note : labels are 0-centered
				intersect = set(top_k_preds) & set(sample_batch_labels[i])
				val = float(len(intersect))/float(k)
				p_at_k+=val
			p_at_k= p_at_k/float(num_samples)
			precision_at_k_values.append(p_at_k)

		metrics["p_at_k"]=precision_at_k_values

		for j in range(0,len(k_values)):
			k=k_values[j]
			print("Precision @ "+str(k)+" : "+str(metrics["p_at_k"][j]))

		return predictions,metrics

	def calculate_overlap(self):
		max_overlap=0
		avg_overlap=0
		count=0
		for i in range(0,self.num_classifiers):
			for j in range(i+1,self.num_classifiers):
				x1=set(self.classifier_info[i]["label_list"])
				x2=set(self.classifier_info[j]["label_list"])
				overlap=len(x1 & x2)
				avg_overlap+=overlap
				max_overlap=max(max_overlap,overlap)
				count+=1
		avg_overlap=avg_overlap/float(count)
		dict = {"max":max_overlap,"avg":avg_overlap}
		return dict

	def training_data_splits(self):
		splits = []
		for i in range(0,self.num_classifiers):
			np=len(self.classifier_info[i]["pos_tset"])
			nn=len(self.classifier_info[i]["neg_tset"])
			split=float(np)/float(nn+np)
			splits.append(split)
		return splits





def select_pos_training_set(MAX_LABELS,num_labels,all_labels,reverse_dict):
	#random set of classes
	labels=set()
	pos_tset=set()
	while(len(labels)<MAX_LABELS):
		new_lab=np.random.randint(low=0,high=num_labels,size=1)
		new_lab=np.asscalar(new_lab)
		if new_lab in labels:
			continue
		labels.add(new_lab)
		pos_tset= pos_tset | set(reverse_dict[new_lab])

	labels=sorted(list(labels))
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

