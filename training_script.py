from train_ens import *

#set random seeds
random.seed(2)
np.random.seed(2)

NUM_CLASSIFIERS=10
MAX_LABELS=20
# TOTAL_CLASSES=100
# NUM_TRAINING_SAMPLES=50*1000

from load_data import read_dataset,create_reverse_dict,data_statistics

dataset= read_dataset("mediamill")
metadata=dataset["metadata"]

num_points= metadata["num_points"]
num_features=metadata["num_features"]
num_labels=metadata["num_labels"]

#create training set
allX=dataset["points"]
allY=dataset["vector_labels"]
tr_split=dataset["train_splits"][0]

trainX=allX[tr_split]
trainY=allY[tr_split]
all_labels=[ dataset["sparse_labels"][i] for i in tr_split]

reverse_dict= create_reverse_dict(all_labels)
statistics= data_statistics(all_labels,num_labels)
NUM_POINTS_NEG=int(statistics["avg_points_per_label"])
# NUM_POINTS_NEG=2
print(NUM_POINTS_NEG)
ensemble=Ensemble(NUM_CLASSIFIERS,num_labels)

# build_ensemble(ensemble,trainX,trainY,all_labels,reverse_dict,MAX_LABELS,NUM_POINTS_NEG)
# ensemble.save("./models/mediamill_split1")

from classifier import MLPClassifier
ensemble=Ensemble(NUM_CLASSIFIERS)
ensemble.load("./models/mediamill_split1",MLPClassifier)

epochs=1
indices_to_train=range(0,NUM_CLASSIFIERS)

init_data={}
init_data["num_labels"]=MAX_LABELS
init_data["num_features"]=num_features
init_data["input_shape"]=(num_features,)
init_data["layers"]=[240,120,60,MAX_LABELS]
train_ensemble(trainX,trainY,ensemble,epochs,indices_to_train,MLPClassifier,init_data)
ensemble.save("./models/mediamill_split1")
# print(ensemble.classifier_list[0].epochs_trained)
# tst_split=dataset["train_splits"][0]
# testX=allX[tst_split]
# testY=allY[tst_split]

# print(ensemble.predict(testX[0,:].reshape(1,num_features)))