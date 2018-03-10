import numpy as np


def read_datafile_xcv(data_folder):
	
	data_file_name=data_folder+"data.txt"
	with open(data_file_name,"r") as data_f:
		info=data_f.readline().split()
		assert(len(info)==3)
		[num_points ,num_features,num_labels] = map(int,info)

		allX=np.zeros((num_points,num_features))
		all_labels=[]
		i=0
		anomolous_ids=[]
		anomolous_lines=[]
		for line in data_f:
			l1 = line.split()
			if(":" not in l1[0]):# label set is not empty
				labels=l1[0].split(",")
				labels = sorted(map(int,labels))#0 indexed?
				l1 = l1[1:]
			else:
				labels=[]
				anomolous_ids.append(i)
				anomolous_lines.append(line)

			for feature in l1:
				fsplit=feature.split(":")
				feat_id=int(fsplit[0])
				feat_val=float(fsplit[1])
				allX[i,feat_id]=feat_val
			i+=1
			all_labels.append(labels)

		assert(i==num_points)
	dataset={}
	dataset["points"]=allX
	dataset["metadata"]={"num_points":num_points,"num_features":num_features,"num_labels":num_labels}
	dataset["sparse_labels"]=all_labels
	dataset["anomolous_ids"]=anomolous_ids
	return dataset


def read_splitfile_xcv(file_name):
	splits=None
	num_cols=None
	with open(file_name,"r") as fi:
		first=True
		for line in fi:
			l = line.split()
			if first:				
				num_cols =  len(l)
				splits = [[] for i in range(0,num_cols)]
				first=False

			for i in range(0,num_cols):
				splits[i].append(int(l[i]))

	return splits


# XCV Readme note
# The data files for all the datasets are in the following sparse representation format:
# Header Line: Total_Points Num_Features Num_Labels
# 1 line per datapoint : label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
# Reads an xcv dataset, data.txt,trSplit.txt,tstSplit.txt
# Returns a dict with following fields
# "points": (num_points,num_features) np array
# "sparse_labels":list of lists
# "train_splits":list of lists
# "test_splits": list of lists 

def read_dataset_xcv(data_folder):
	dataset=read_datafile_xcv(data_folder)

	tr_split_file_name=data_folder+"trSplit.txt"
	tst_split_file_name=data_folder+"tstSplit.txt"

	tr_splits= read_splitfile_xcv(tr_split_file_name)
	tst_splits= read_splitfile_xcv(tst_split_file_name)

	#check
	num_points=dataset["metadata"]["num_points"]
	assert(len(tr_splits)==len(tst_splits))
	for i in range(0,len(tr_splits)):
		assert (len(tr_splits[i])+len(tst_splits[i])==num_points)

	dataset["metadata"]["num_splits"]=len(tr_splits)
	dataset["train_splits"]=tr_splits
	dataset["test_splits"]=tst_splits

	return dataset

def make_label_vectors(all_labels,metadata):
	num_points=metadata["num_points"]
	num_labels=metadata["num_labels"]
	allY=np.zeros((num_points,num_labels))
	reverse_dict={}
	for i in range(0,num_points):
		labels=all_labels[i]
		for lab in labels:
			allY[i][lab]=1.0
			if lab not in reverse_dict:
				reverse_dict[lab]=[]
			reverse_dict[lab].append(i)
	return allY,reverse_dict


def read_dataset(datasetname):

	if datasetname=="cifar":
		#do something
		print("Oops, not ready yet")
	#a xcv dataset
	foldernames={"mediamill":"./data/Mediamill/","delicious":"./data/Delicious/"}

	import os.path
	import pickle

	filename="./data/"+str(datasetname)+".p"
	if os.path.isfile(filename):
		with open(filename,"rb") as fi:
			return pickle.load(fi)


	data_folder=foldernames[datasetname]
	dataset=read_dataset_xcv(data_folder)
	allY,reverse_dict=make_label_vectors(dataset["sparse_labels"],dataset["metadata"])
	dataset["vector_labels"]=allY
	dataset["reverse_dict"]=reverse_dict
	with open(filename,"wb") as fo:
		pickle.dump(dataset,fo)

	return dataset
	

# x=read_dataset("mediamill")









			





