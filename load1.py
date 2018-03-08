import pickle
dict={}
with open("./cifar-100-python/test","rb") as fo:
	dict = pickle.load(fo,encoding="bytes")
# for key in dict :
# 	print(key)

data = dict[b'data']
labels = dict[b'fine_labels']
# print(data.shape)
# print(labels.shape)

TOTAL_CLASSES=100

#sort data a/t labels and save again
import numpy as np

labels=np.asarray(labels)
indices=np.argsort(labels)
sortedlabels= labels[indices]
sorteddata= data[indices,:]
sdict={}

divisions=[0]
cur=0
#assume contiguous labels
for i in range(0,sortedlabels.shape[0]):
	if cur != sortedlabels[i]:
		divisions.append(i)
		cur=cur+1
		assert(cur==sortedlabels[i])
divisions.append(i+1)
assert (len(divisions)==TOTAL_CLASSES+1)


#preprocessing

data=sorteddata

#data too large preprocess later
# data=data/255.0

data2 = np.reshape(data,(data.shape[0],3,32,32),order='C')

assert(data2[32,2,22,22]==data[32,1024*2+22*32+22*1])
data=data2



sdict["data"]=data
sdict["labels"]=sortedlabels
sdict["div"]=divisions
with open("./cifarsorted/test","wb")  as fi:
	pickle.dump(sdict,fi)


