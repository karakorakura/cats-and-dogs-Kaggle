# USAGE
# python simple_neural_network.py --dataset kaggle_dogs_vs_cats

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
#from keras.layers import Activation
from keras.optimizers import SGD
#from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# Import libraries
#import os,cv2
#import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split

from keras import backend as K
# K.set_image_dim_ordering('th')
K.set_image_dim_ordering('tf')

#from keras.utils import np_utils
#from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
#####################################################################################################################
#%%
num_classes = 2
num_epoch=20

def image_to_feature_vector(image, size=(64,64),flatten=True):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    if flatten == False : return cv2.resize(image,size);
    return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset")
#args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
#imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images('train'))

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	imageName=imagePath.split(os.path.sep)[-1]
	label = imageName.split(".")[0]

	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	features = image_to_feature_vector(image,size=(50,50),flatten=False)
	data.append(features)
	labels.append(label)

	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
print (labels)
labels = le.fit_transform(labels)
print (labels)
# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`
data = np.array(data) / 255.0
# labels = np_utils.to_categorical(labels, 2)
labels = np_utils.to_categorical(labels, num_classes)



#Shuffle the dataset
data,labels = shuffle(data,labels, random_state=2)
# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.25, random_state=42)

# define the architecture of the network


#%%
# Defining the model
input_shape=data[0].shape
num_classes = 2
num_of_samples = data.shape[0]
print (num_of_samples)

model = Sequential()

model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

#%%
# Training
hist = model.fit(trainData, trainLabels, batch_size=50, nb_epoch=num_epoch, verbose=1, validation_data=(testData, testLabels))


# model = Sequential()
# model.add(Dense(768, input_dim=3072, init="uniform",
# 	activation="relu"))
# model.add(Dense(384, init="uniform", activation="relu"))
# model.add(Dense(2))
# model.add(Activation("softmax"))
#
# # train the model using SGD
# print("[INFO] compiling model...")
# sgd = SGD(lr=0.01)
# model.compile(loss="binary_crossentropy", optimizer=sgd,
# 	metrics=["accuracy"])
# model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128,
# 	verbose=1)
#
# # show the accuracy on the testing set
# print("[INFO] evaluating on testing set...")
# (loss, accuracy) = model.evaluate(testData, testLabels,
# 	batch_size=128, verbose=1)
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
# 	accuracy * 100))
