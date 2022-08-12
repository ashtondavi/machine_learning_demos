"""
### Description

This script uses keras to find the centre point of a variable number of black 
objects on a white background. It starts by creating a dataset of randomly 
generated images and then trains a keras model to detect the centre points.

### Important notes

For multiple object detection the output needs to be of a defined length. So it 
fails if there are variable numbers of objects. To fix this the output can be 
padded out with zeros. The zeros are replaced with xy coordinates as they are 
detected.

If a random set of xy coordinates are generated as the output for the training 
set then the coordinates need to be sorted. Otherwise when multiple coordinates 
are present in a single image it trains the NN to detect a position halfway 
between the two points.
"""

#load libraries
import json
import numpy
import matplotlib.pyplot as plt
import random

#load parameters from json file
with open('parameters.json', 'r') as f:
    parameters = json.load(f)

#generate the images and coordinates
bboxes = numpy.zeros((parameters['number_images'], parameters['number_objects'], 2))
imgs = numpy.zeros((parameters['number_images'], parameters['image_size'], parameters['image_size']))

for i_img in range(parameters['number_images']):
    #init an empty box list
    box_list = []
    #generate these in a specific order
    for i_object in range(parameters['number_objects']):
        if random.uniform(0,1) < parameters['missing_rate']:
            box_list += [[0,0]]
        else:
            w, h = numpy.random.randint(parameters['min_object_size'], parameters['max_object_size'], size=2)
            x = numpy.random.randint(0, parameters['image_size'] - w)
            y = numpy.random.randint(0, parameters['image_size'] - h)
            imgs[i_img, x:x+w, y:y+h] = 1.
            box_list += [[x+(w/float(2)), y+(h/float(2))]]
    #sort by x and then y
    box_list.sort()
    for x in range(len(box_list)):
        bboxes[i_img, x] = box_list[x]
    
imgs.shape, bboxes.shape

i = 0
plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', 
           extent=[0, parameters['image_size'], 0, parameters['image_size']])
for bbox in bboxes[i]:
    plt.scatter(bbox[0], bbox[1])
   
# Reshape and normalize the image data to mean 0 and std 1. 
X = (imgs.reshape(parameters['number_images'], -1) - numpy.mean(imgs)) / numpy.std(imgs)
X.shape, numpy.mean(X), numpy.std(X)

print(X.shape)

# Normalize x, y, w, h by img_size, so that all values are between 0 and 1.
# Important: Do not shift to negative values (e.g. by setting to mean 0), because the IOU calculation needs positive w and h.
y = bboxes.reshape(parameters['number_images'], -1) / parameters['image_size']
y.shape, numpy.mean(y), numpy.std(y)

# Split training and test.
i = int(0.8 * parameters['number_images'])
train_X = X[:i]
#print(train_X[0])
test_X = X[i:]
train_y = y[:i]
#print(train_y[0])
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]

# Build the model.
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

#initiate a sequential model
model = Sequential()
#add dense layer
model.add(Dense(200, input_dim=X.shape[-1]))
#add dense layer
model.add(Dense(100))
#add activation layer
model.add(Activation('relu'))
#add dropout layer
model.add(Dropout(0.2))
#add dense layer
model.add(Dense(y.shape[-1]))
#compile the model
model.compile('adadelta', 'mse')

#fit model
model.fit(train_X, train_y, nb_epoch=parameters['epochs'], validation_data=(test_X, test_y), verbose=2)  

# Predict bounding boxes on the test images.
pred_y = model.predict(test_X)
pred_bboxes = pred_y * parameters['image_size']
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), parameters['number_objects'], -1)
pred_bboxes.shape

# Show a few images and predicted bounding boxes from the test dataset. 
plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = numpy.random.randint(len(test_imgs))
    plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', 
               extent=[0, parameters['image_size'], 0, parameters['image_size']])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.scatter(pred_bbox[0], pred_bbox[1])
        