

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
import tensorflow as tf
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json



column_names = ['center', 'left', 'right', 'steering']
dataframe = pd.read_csv('./data/driving_log.csv')



# Import center images, resize and crop

center_images = []
center_steering = []
center = dataframe.center.tolist()
steering = dataframe.steering.tolist()
for c, s in zip(center, steering):
    if s >= 0.04 or s <= - 0.04:
        image = mpimg.imread('./data/' + c)
        image = image[80:140, 20:300]
        center_images.append(image)
        center_steering.append(s)


# Flip center images

center_images_flipped = []
center_steering_flipped = []
for c, s in zip(center, steering):
    if s >= 0.04 or s <= - 0.04:
        image = mpimg.imread('./data/' + c)
        image = image[80:140, 20:300]
        image = cv2.flip(image, 1)
        center_images_flipped.append(image)
        center_steering_flipped.append(-s)



# import left images and resize and crop and flip vertically

left_images = []
left_images_flipped = []
left_steering = []
left_steering_flipped = []
left = dataframe.left.tolist()

lefty = [i.strip() for i in left]

for c, s in zip(lefty, steering):
    if s >= 0.05 or s <= -0.05:
        image = mpimg.imread('./data/' + c)
        image = image[80:140, 20:300]
        left_images.append(image)
        left_steering.append(s + 0.15)

for c, s in zip(lefty, steering):
    if s >= 0.05 or s <= -0.05:
        image = mpimg.imread('./data/' + c)
        image = image[80:140, 20:300]
        image = cv2.flip(image, 1)
        left_images_flipped.append(image)
        left_steering_flipped.append(-s - 0.15)


# import right images, resize and crop

right_images = []
right_images_flipped = []
right_steering = []
right_steering_flipped = []
right = dataframe.right.tolist()

righty = [i.strip() for i in right]

for c, s in zip(righty, steering):
    if s >= 0.05 or s <= -0.05:

        image = mpimg.imread('./data/' + c)
        image = image[80:140, 20:300]
        right_images.append(image)
        right_steering.append(s - 0.15)

for c, s in zip(righty, steering):
    if s >= 0.05 or s <= -0.05:

        image = mpimg.imread('./data/' + c)
        image = image[80:140, 20:300]
        image = cv2.flip(image, 1)
        right_images_flipped.append(image)
        right_steering_flipped.append(-s + 0.15)




center_img = (center_images + center_images_flipped)
cent_steer = (center_steering + center_steering_flipped)
left_img = (left_images + left_images_flipped)
l_steer = (left_steering + left_steering_flipped)

right_img = (right_images + right_images_flipped)
r_steer = (right_steering + right_steering_flipped)


all_images = center_img + left_img + right_img
all_steering = cent_steer + l_steer + r_steer

# Split into left, right, and straight driving

new_straight_images = []
new_straight_angles = []
new_left_images= []
new_left_angles = []
new_right_images = []
new_right_angles = []

for image, angle in zip(all_images, all_steering):
    if angle < -0.15: #left turn
        new_left_images.append(image)
        new_left_angles.append(angle)
    elif angle > 0.15: #right turn
        new_right_images.append(image)
        new_right_angles.append(angle)
    else: #straight
        new_straight_images.append(image)
        new_straight_angles.append(angle)



new_straight_images = new_straight_images[:5300]
new_straight_angles = new_straight_angles[:5300]


final_images = (new_straight_images + new_left_images + new_right_images)
final_angles = (new_straight_angles + new_left_angles + new_right_angles)


# Add in recovery data

second_column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
second_dataframe = pd.read_csv('./driving_log.csv', header=0)

second_dataframe.columns = [second_column_names]


## Extra bridge and turn data

bridge_center_images = []
bridge_left_images = []
bridge_right_images = []
bridge_steering_c = []
bridge_steering_l = []
bridge_steering_r = []
centerb = second_dataframe.center.tolist()
leftb = second_dataframe.left.tolist()
rightb = second_dataframe.right.tolist()
bridge_ang = second_dataframe.steering.tolist()
rightyb = [i.strip() for i in rightb]
leftyb = [i.strip() for i in leftb]

for c, s in zip(centerb, bridge_ang):
        image = mpimg.imread(c)
        image = image[80:140, 20:300]
        bridge_center_images.append(image)
        bridge_steering_c.append(s)
        

for c, s in zip(leftyb, bridge_ang):
        image = mpimg.imread(c)
        image = image[80:140, 20:300]
        bridge_left_images.append(image)
        bridge_steering_l.append(s + 0.15)
    

for c, s in zip(rightyb, bridge_ang):
        image = mpimg.imread(c)
        image = image[80:140, 20:300]
        bridge_right_images.append(image)
        bridge_steering_r.append(s - 0.15)

bridge_images = (bridge_center_images + bridge_left_images + bridge_right_images)
bridge_steering_angles = (bridge_steering_c + bridge_steering_l + bridge_steering_r)

# Create final image variables

final_images = final_images + bridge_images
final_angles = final_angles + bridge_steering_angles


# Generator for training

def train_gen(batch_size=128):
    
    while 1:
        image_batch = np.zeros((batch_size, 60, 280, 3))
        angle_batch = np.zeros(batch_size)
        
        for i in range(batch_size):
            image, angle = X_train[i], y_train[i]
            
            image_batch[i] = image
            angle_batch[i] = angle
            
            
        yield image_batch, angle_batch



def validation_gen(images, angles, batch_size=50):
    
    while 1:
        image_batch = np.zeros((batch_size, 60, 280, 3))
        angle_batch = np.zeros(batch_size)
        
        
        for i in range(batch_size):
            image, angle = images[i], angles[i]
            
            image_batch[i] = image
            angle_batch[i] = angle
            
                
        yield image_batch, angle_batch
               
            
           

# split up data and shuffle

X_train, x_validation, y_train, y_validation = train_test_split(final_images, final_angles, test_size = 0.1, random_state=534)

X_train, y_train = shuffle(X_train, y_train)


# Nvidia Architecture

epochs = 12

kernal = [5, 5]
kernal2 = [2, 2]

model = Sequential()

model.add(Lambda(lambda x: x/ 127.5 - 1.0, input_shape=(60, 280, 3)))

model.add(Convolution2D(24, kernal[0], kernal[1], subsample=(2,2), input_shape=(60, 280, 3), dim_ordering='tf'))
model.add(Activation('elu'))

model.add(Convolution2D(36, kernal[0], kernal[1], subsample=(2,2), dim_ordering='tf'))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Convolution2D(48, kernal[0], kernal[1], subsample=(2,2), dim_ordering='tf'))
model.add(Activation('elu'))


model.add(Convolution2D(64, kernal2[0], kernal2[1], subsample=(1,1), dim_ordering='tf'))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Convolution2D(64, kernal2[0], kernal2[1], subsample=(1,1)))
model.add(Activation('elu'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('elu'))

model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('elu'))

model.add(Dense(1))

#optimizer, loss, accuracy
from keras.optimizers import Adam
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

#train the model
history = model.fit_generator(train_gen(), samples_per_epoch=12800, nb_epoch=epochs, verbose=2, validation_data=validation_gen(x_validation, y_validation), nb_val_samples=1000)

#evaluate the accuracy of the model
model.summary()


# Save

with open('./model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

model.save_weights('model.h5')

