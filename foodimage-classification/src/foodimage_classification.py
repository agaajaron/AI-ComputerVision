# %% [markdown]
# # **Context**
# Image classification has become less complicated with deep learning and the availability of larger datasets and computational assets. The Convolution neural network is the most popular and extensively used image classification technique in the latest days.
# 
# Clicks is a stock photography company and is an online source of images available for people and companies to download. Photographers from all over the world upload food-related images to the stock photography agency every day. Since the volume of the images that get uploaded daily will be high, it will be difficult for anyone to label the images manually.
# 
# Clicks have decided to use only three categories of food (**`Bread`**, **`Soup`**, and  **`Vegetables-Fruits`**) for now, and you as a data scientist at Clicks, need to build a classification model using a dataset consisting of images that would help to label the images into different categories.
# 
# 
# ## **Dataset**
# 
# The dataset folder contains different food images. The images are already split into Training and Testing folders.
# Each folder has 3 more subfolders named **`Bread`**, **`Soup`**, and  **`Vegetables-Fruits`**. These folders have images of the respective classes.
# 
# **Instructions** to access the data through Google Colab:
# 
# Follow the below steps:
# 
# 1) Download the zip file from Olympus.
# 
# 2) Upload the file into your drive and unzip the folder using the code provided in notebook.
# 
# 3) Mount your Google Drive using the code below.
# 
# NOTE: **Change the run time to GPU**
# 
# 
# ```
# from google.colab import drive
# drive.mount('/content/drive')
# ```
# 
# 
# 4) Now, you can read the dataset.
# 
# 
# 
# 
# 
# 

# %% [markdown]
# ###Mount the Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ### **Importing the libraries**

# %%
#Reading the training images from the path and labelling them into the given categories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 # this is an important module to get imported which may even cause issues while reading the data if not used
import seaborn as sns # for data visualization
import tensorflow as tf
import keras
import os
from tensorflow.keras.models import Sequential #sequential api for sequential model
from tensorflow.keras.layers import Dense, Dropout, Flatten #importing different layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Input, LeakyReLU,Activation
from tensorflow.keras import backend
from tensorflow.keras.utils import to_categorical #to perform one-hot encoding
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam,SGD #optimiers for optimizing the model
from tensorflow.keras.callbacks import EarlyStopping  #regularization method to prevent the overfitting
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import losses, optimizers
# Importing all the required sub-modules from Keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

# %%
# !unzip "/content/drive/MyDrive/Food_Case_Study/Food_Data.zip"

# %% [markdown]
# ### **Reading the Training Data**

# %%
# Storing the training path in a variable named DATADIR, and storing the unique categories/labels in a list

DATADIR = "/content/Food_Data/Training"                                        # Path of training data after unzipping
CATEGORIES = ["Bread","Soup","Vegetable-Fruit"]                                # Storing all the categories in categories variable
IMG_SIZE=150                                                                   # Defining the size of the image to 150

# %%
# Here we will be using a user defined function create_training_data() to extract the images from the directory
training_data = []                                                             # Storing all the training images
def create_training_data():
    for category in CATEGORIES:                                                # Looping over each category from the CATEGORIES list
        path = os.path.join(DATADIR,category)                                  # Joining images with labels
        class_num = category
        for img in os.listdir(path):
          img_array = cv2.imread(os.path.join(path,img))                       # Reading the data
          new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))                # Resizing the images
          training_data.append([new_array,class_num])                          # Appending both the images and labels
create_training_data()

# %% [markdown]
# ### **Reading the Testing Dataset**
# 

# %%
DATADIR_test = "/content/Food_Data/Testing"                                    # Path of training data after unzipping
CATEGORIES =  ["Bread","Soup","Vegetable-Fruit"]                               # Storing all the categories in categories variable
IMG_SIZE=150                                                                   # Defining the size of the image to 150

# %%
# Here we will be using a user defined function create_testing_data() to extract the images from the directory
testing_data = []

def create_testing_data():                                                     # Storing all the testing images
    for category in CATEGORIES:                                                # Looping over each category from the CATEGORIES list
        path = os.path.join(DATADIR_test,category)                             # Joining images with labels
        class_num = category
        for img in os.listdir(path):
          img_array = cv2.imread(os.path.join(path,img))                       # Reading the data
          new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))                # Resizing the images
          testing_data.append([new_array,class_num])                           # Appending both the images and labels

create_testing_data()

# %% [markdown]
# **Let's visualize images randomly from each of the four classes.**
# 
# 
# 

# %%
# Creating 3 different lists to store the image names for each category by reading them from their respective directories.
bread_imgs = [fn for fn in os.listdir(f'{DATADIR}/{CATEGORIES[0]}') ]          # Looping over the path of each image from the bread directory
soup_imgs = [fn for fn in os.listdir(f'{DATADIR}/{CATEGORIES[1]}')]            # Looping over the path of each image from the soup directory
veg_fruit_imgs = [fn for fn in os.listdir(f'{DATADIR}/{CATEGORIES[2]}') ]      # Looping over the path of each image from the vegetables-fruit directory



# Ranodmly selecting 3 images from each category
select_bread = np.random.choice(bread_imgs, 3, replace = False)
select_soup = np.random.choice(soup_imgs, 3, replace = False)
select_veg_fruit = np.random.choice(veg_fruit_imgs, 3, replace = False)

# %%

# plotting 4 x 3 image matrix
fig = plt.figure(figsize = (10,10))

# Plotting three images from each of the four categories by looping through their path
for i in range(9):
    if i < 3:
        fp = f'{DATADIR}/{CATEGORIES[0]}/{select_bread[i]}'                    # Here datadir is a path to the training data and categories[0] indicate the first label bread and here we are looping over to take the three random images that we have stored in select_galo variable
        label = 'Bread'
    if i>=3 and i<6:
        fp = f'{DATADIR}/{CATEGORIES[1]}/{select_soup[i-3]}'                   # Here datadir is a path to the training data and categories[1] indicate the second label soup and here we are looping over to take the three random images that we have stored in select_menin variable
        label = 'Soup'
    if i>=6 and i<9:
        fp = f'{DATADIR}/{CATEGORIES[2]}/{select_veg_fruit[i-6]}'              # Here datadir is a path to the training data and categories[2] indicate the third label vegetables-fruit and here we are looping over to take the three random images that we have stored in select_no_t variable
        label = 'Vegetable_Fruit'
    ax = fig.add_subplot(4, 3, i+1)

    # Plotting each image using load_img function
    fn = image.load_img(fp, target_size = (150,150))
    plt.imshow(fn, cmap='Greys_r')
    plt.title(label)
    plt.axis('off')
plt.show()

# %% [markdown]
# ### **Data Preprocessing**

# %%
# Creating two different lists to store the Numpy arrays and the corresponding labels
X_train = []
y_train = []
np.random.shuffle(training_data)                                               # Shuffling data to reduce variance and making sure that model remains general and overfit less
for features,label in training_data:                                           # Iterating over the training data which is generated from the create_training_data() function
    X_train.append(features)                                                   # Appending images into X_train
    y_train.append(label)                                                      # Appending labels into y_train

# %%
# Creating two different lists to store the Numpy arrays and the corresponding labels
X_test = []
y_test = []

np.random.shuffle(testing_data)                                                # Shuffling data to reduce variance and making sure that model remains general and overfit less
for features,label in testing_data:                                            # Iterating over the training data which is generated from the create_testing_data() function
    X_test.append(features)                                                    # Appending images into X_train
    y_test.append(label)                                                       # Appending labels into y_train

# %%
# Converting the list into DataFrame
y_train = pd.DataFrame(y_train, columns=["Label"],dtype=object)
y_test = pd.DataFrame(y_test, columns=["Label"],dtype=object)

# %% [markdown]
# ### **Exploratory Data Analysis**

# %%
# Storing the value counts of target variable
count=y_train.Label.value_counts()
print(count)
print('*'*10)
count=y_train.Label.value_counts(normalize=True)
print(count)

# %%
# Converting the pixel values into Numpy array
X_train= np.array(X_train)
X_test= np.array(X_test)

# %%
X_train.shape

# %% [markdown]
# Since the given data is stored in X_train, X_test,y_train, and y_test variables, there is no need to split the data further.

# %% [markdown]
# **Normalizing the data**

# %% [markdown]
# 
# 1. **Normalization makes the training faster and reduces the chances of getting stuck at local optima.**
# 2. In deep neural networks, **normalization helps to avoid exploding gradient problems.** Gradient exploding problem occurs when large error gradients accumulate and result in very large updates to neural network model weights during training. This makes a model unstable and unable to learn from the training data.
# 
# As we know image pixel **values range from 0-255**, here we are simply **dividing all the pixel values by 255 to standardize all the images to have values between 0-1.**

# %%
## Normalizing the image data
X_train= X_train/255.0

# %% [markdown]
# ### **Encoding Target Variable**

# %% [markdown]
# LabelBinarizer is another technique used to encode the target variables which reduces the sparsity as compared to one hot encoder.

# %% [markdown]
# For Example:
# If we have 4 classes as "Good","Better","Okay","Bad".
# After applying LabelBinarizer on these 4 classes, the output result will be in the form of array.
# * [1, 0, 0, 0] --------- Good
# * [0, 1, 0, 0] --------- Better
# * [0, 0, 1, 0] --------- Okay
# * [0, 0, 0, 1] --------- Bad
# 
# Each class will be represented in the form of array
# 
# 
# 

# %%
from sklearn.preprocessing import LabelBinarizer
# Storing the LabelBinarizer function in lb variable
lb = LabelBinarizer()
# Applying fit_transform on train target variable
y_train_e = lb.fit_transform(y_train)
# Applying only transform on test target variable
y_test_e = lb.transform(y_test)

# %% [markdown]
# ### **Model Building**
# 
# - **CNN** (Convolutional Neural Network)

# %% [markdown]
# #### **Convolutional Neural Network (CNN)**
# 
# **Model 1:**

# %%
from tensorflow.keras import backend
backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %% [markdown]
# * **Filters**: 256- Number of filters in the first hidden layer.
# * **Kernel_Size**: The kernel size here refers to the widthxheight of the filter mask. The kernel_size must be an odd integer as well. Typical values for kernel_size include: (1, 1) , (3, 3) , (5, 5) , (7, 7)
# * **Padding**: The padding type is called SAME because the output size is the same as the input size(when stride=1). Using 'SAME' ensures that the filter is applied to all the elements of the input. Normally, padding is set to "SAME" while training the model. Output size is mathematically convenient for further computation.
# * **MaxPool2D**: Max Pooling is a pooling operation that calculates the maximum value for patches of a feature map, and uses it to create a downsampled (pooled) feature map.
# * **Flatten**: Flattening is converting the data into a 1-dimensional array for giving them as input to the next layer.

# %%
# Intializing a sequential model
model = Sequential()

# Adding first conv layer with 64 filters and kernel size 3x3 , padding 'same' provides the output size same as the input size
# Input_shape denotes input image dimension of MNIST images
model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(150,150,3)))

# Adding max pooling to reduce the size of output of first conv layer
model.add(MaxPooling2D((2, 2), padding = 'same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2), padding = 'same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2), padding = 'same'))

# flattening the output of the conv layer after max pooling to make it ready for creating dense connections
model.add(Flatten())

# Adding a fully connected dense layer with 100 neurons
model.add(Dense(100, activation='relu'))

# Adding the output layer with 10 neurons and activation functions as softmax since this is a multi-class classification problem
model.add(Dense(3, activation='softmax'))

# Using SGD Optimizer
opt = SGD(learning_rate=0.01, momentum=0.9)

# Compile model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Generating the summary of the model
model.summary()

# %% [markdown]
# As we can see from the above summary, this CNN model will train and learn **1,185,107 parameters (weights and biases).**

# %% [markdown]
# Let's now compile and train the model using the train data. Here, we are using the loss function - categorical_crossentropy as this is a multi-class classification problem. We will try to minimize this loss at every iteration using the optimizer of our choice. Also, we are choosing accuracy as the metric to measure the performance of the model.

# %%
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# Fitting the model with 30 epochs and validation_split as 10%
history=model.fit(X_train,
          y_train_e,
          epochs=30,
          batch_size=32,validation_split=0.10,callbacks=[es, mc])

# %% [markdown]
# **Plotting Accuracy vs Epoch Curve**

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% [markdown]
# From the above plot, it seems that the model is overfitting, lets try to use Dropout in the next model.

# %%
model.evaluate(X_test,(y_test_e))

# %%
# Test Prediction
y_test_pred_ln = model.predict(X_test)
y_test_pred_classes_ln = np.argmax(y_test_pred_ln, axis=1)
normal_y_test = np.argmax(y_test_e, axis=1)

# %% [markdown]
# Since we have converted the target variable into a NumPy array using labelbinarizer, now we are converting the target variable into its original form by using the numpy. argmax() function which returns indices of the max element of the array in a particular axis and this original form of target will be used in calculating the accuracy, and plotting the confusion matrix.

# %%
# Test Accuracy
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score((normal_y_test), y_test_pred_classes_ln)

# %%
cf_matrix = confusion_matrix(normal_y_test, y_test_pred_classes_ln)

# Confusion matrix normalized per category true value
cf_matrix_n1 = cf_matrix/np.sum(cf_matrix, axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(cf_matrix_n1, xticklabels=CATEGORIES, yticklabels=CATEGORIES, annot=True)

# %% [markdown]
# * The model is giving about 73% accuracy on the test data

# %% [markdown]
# ### Model 2:
# 
# Lets try to build another CNN model with more layers added to the model.

# %%
from tensorflow.keras import backend
backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %%
# initialized a sequential model
model_3 = Sequential()
# adding first conv layer with 256 filters and kernel size 5x5 , with ReLU activation and padding 'same' provides the output size same as the input size
#input_shape denotes input image dimension of images
model_3.add(Conv2D(filters = 256, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (150,150,3)))
# adding max pooling to reduce the size of output of first conv layer
model_3.add(MaxPool2D(pool_size=(2,2)))
#  adding dropout to randomly switch off 25% neurons to reduce overfitting

# adding second conv layer with 256 filters and with kernel size 3x3 and ReLu activation function
model_3.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
# adding max pooling to reduce the size of output of first conv layer
model_3.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#  adding dropout to randomly switch off 25% neurons to reduce overfitting

# adding third conv layer with 256 filters and with kernel size 3x3 and ReLu activation function
model_3.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
# adding max pooling to reduce the size of output of first conv layer
model_3.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#  adding dropout to randomly switch off 30% neurons to reduce overfitting
model_3.add(Dropout(0.3))

# adding forth conv layer with 256 filters and with kernel size 3x3 and ReLu activation function
model_3.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
# adding max pooling to reduce the size of output of first conv layer
model_3.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#  adding dropout to randomly switch off 30% neurons to reduce overfitting
model_3.add(Dropout(0.3))


# flattening the 3-d output of the conv layer after max pooling to make it ready for creating dense connections
model_3.add(Flatten())
# adding first fully connected dense layer with 1024 neurons
model_3.add(Dense(64, activation = "relu"))
#  adding dropout to randomly switch off 50% neurons to reduce overfitting
model_3.add(Dropout(0.5))
# adding second fully connected dense layer with 512 neurons
model_3.add(Dense(32, activation = "relu"))
#  adding dropout to randomly switch off 50% neurons to reduce overfitting
model_3.add(Dropout(0.5))

# adding the output layer with 3 neurons and activation functions as softmax since this is a multi-class classification problem with 3 classes.
model_3.add(Dense(3, activation = "softmax"))

# %%
model_3.summary()

# %% [markdown]
# As we can see from the above summary, this CNN model will train and learn **1,099,171 parameters (weights and biases).**

# %% [markdown]
# Let's now compile and train the model using the train data. Here, we are using the loss function - categorical_crossentropy as this is a multi-class classification problem. We will try to minimize this loss at every iteration using the optimizer of our choice. Also, we are choosing accuracy as the metric to measure the performance of the model.

# %%
optimizer = Adam(lr=0.001)
model_3.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history=model_3.fit(X_train,
          y_train_e,
          epochs=30,
          batch_size=32,validation_split=0.10,callbacks=[es, mc],use_multiprocessing=True)

# %% [markdown]
# **Plotting Accuracy vs Epoch Curve**

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()

# %% [markdown]
# Train and Validation accuracy seems fine and let's calculate the accuracy on the test data

# %%
model_3.evaluate(X_test,y_test_e)

# %% [markdown]
# By comparing the train and test accuracy, it seems the model is not overfitting, so adding more layers to the model worked, and we can say that this CNN model is good. We can also try building different models by increasing the hidden layers and see if we get good accuracy.

# %%
y_test_pred_ln3 = model_3.predict(X_test)
y_test_pred_classes_ln3 = np.argmax(y_test_pred_ln3, axis=1)

# %%
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(normal_y_test, y_test_pred_classes_ln3)

# %%
cf_matrix = confusion_matrix(normal_y_test, y_test_pred_classes_ln3)

# Confusion matrix normalized per category true value
cf_matrix_n1 = cf_matrix/np.sum(cf_matrix, axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(cf_matrix_n1, xticklabels=CATEGORIES, yticklabels=CATEGORIES, annot=True)

# %% [markdown]
# ### Model: 3
# #### Transfer Learning (VGG16) with Data Agumentation

# %% [markdown]
# #### **Data Augmentation**
# 
# In most real-life case studies, it is generally difficult to collect lots of images and then train CNNs. In that case, one idea we can take advantage of is Data Augmentation. CNNs have the property of **translational invariance**, i.e., they can recognize an object as an object, even when its appearance varies translationally in some way. Taking this property into consideration, we can augment the images using the following techniques:  **Data Augmentation**
# 
# In most real-life case studies, it is generally difficult to collect lots of images and then train CNNs. In that case, one idea we can take advantage of is Data Augmentation. CNNs have the property of **translational invariance**, i.e., they can recognize an object as an object, even when its appearance varies translationally in some way. Taking this property into consideration, we can augment the images using the following techniques:

# %% [markdown]
# **1. Horizontal Flip** (should be set to True/False) <br>
# **2. Vertical Flip** (should be set to True/False) <br>
# **3. Height Shift** (should be between 0 and 1) <br>
# **4. Width Shift** (should be between 0 and 1) <br>
# **5. Rotation** (should be between 0 and 180) <br>
# **6. Shear** (should be between 0 and 1) <br>
# **7. Zoom** (should be between 0 and 1) etc. <br>**1. Horizontal Flip** (should be set to True/False) <br>
# **2. Vertical Flip** (should be set to True/False) <br>
# **3. Height Shift** (should be between 0 and 1) <br>
# **4. Width Shift** (should be between 0 and 1) <br>
# **5. Rotation** (should be between 0 and 180) <br>
# **6. Shear** (should be between 0 and 1) <br>
# **7. Zoom** (should be between 0 and 1) etc. <br>

# %%
# sTORING TRAIN AND TEST DATA PATH
train_dir = "/content/Food_Data/Training"
test_dir  = "/content/Food_Data/Testing"

# All images to be rescaled by 1/255.
train_datagen = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True,
                              vertical_flip = False,
                              height_shift_range= 0.1,
                              width_shift_range=0.1,
                              rotation_range=20,
                              shear_range = 0.1,
                              zoom_range=0.1)
test_datagen  = ImageDataGenerator(rescale = 1.0/255.)

# Flowing training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))

# Flowing validation images in batches of 20 using test_datagen generator
test_generator =  test_datagen.flow_from_directory(test_dir,
                                                         batch_size=20,
                                                         class_mode  = 'categorical',
                                                         target_size = (150, 150))

# %% [markdown]
# -  **Transfer Learning** with Data Agumentation.  **VGG16**, which was trained on the ImageNet dataset and finished runner-up in the ImageNet competition in 2014. Below is a schematic of the VGG16 model.
# 
# 
# - For training VGG16, we will directly use the convolutional and pooling layers and freeze their weights i.e. no training will be done on them. We will remove the already-present fully-connected layers and add our own fully-connected layers for this binary classification task.

# %%
# Loading VGG16 model
model = VGG16(weights='imagenet')
# Summary of the whole model
model.summary()

# %%
# Getting only the conv layers for transfer learning.
transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs=model.input, outputs=transfer_layer.output)

# %%
vgg_model.summary()

# %%
vgg_model = VGG16(weights='imagenet', include_top = False, input_shape = (150,150,3))
vgg_model.summary()

# %%
# Making all the layers of the VGG model non-trainable. i.e. freezing them
for layer in vgg_model.layers:
    layer.trainable = False

# %%
for layer in vgg_model.layers:
    print(layer.name, layer.trainable)

# %%
backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)
import random
random.seed(42)
tf.random.set_seed(42)

# %%
# Initializing the model
new_model = Sequential()

# Adding the convolutional part of the VGG16 model from above
new_model.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
new_model.add(Flatten())

# Adding a dense input layer
new_model.add(Dense(32, activation='relu'))
# Adding dropout
new_model.add(Dropout(0.2))
# Adding second input layer
new_model.add(Dense(32, activation='relu'))
# Adding output layer
new_model.add(Dense(3, activation='softmax'))

# %%
# Compiling the model
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Summary of the model
new_model.summary()

# %%
## Pulling a single large batch of random validation data for testing after each epoch
testX, testY = test_generator.next()

# %%
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

## Fitting the VGG model
new_model_history = new_model.fit(train_generator,
                                  validation_data = (testX, testY),
                                  epochs=30,callbacks=[es, mc],use_multiprocessing=True)

# %%
plt.plot(new_model_history.history['accuracy'])
plt.plot(new_model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()

# %%
# Evaluating on the Test set
new_model.evaluate(test_generator)

# %%
y_test_pred_ln4 = new_model.predict(X_test)
y_test_pred_classes_ln4 = np.argmax(y_test_pred_ln4, axis=1)

# %%
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(normal_y_test, y_test_pred_classes_ln4)

# %% [markdown]
# We were able to get good accuracy as compared to previous model.

# %%
cf_matrix = confusion_matrix(normal_y_test, y_test_pred_classes_ln4)

# Confusion matrix normalized per category true value
cf_matrix_n1 = cf_matrix/np.sum(cf_matrix, axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(cf_matrix_n1, xticklabels=CATEGORIES, yticklabels=CATEGORIES, annot=True)

# %% [markdown]
# **Classification Report for each class**
# 
# - **Precision:** precision is the fraction of relevant instances among the retrieved instances.
# 
# - **Recall:** recall is the fraction of relevant instances that were retrieved.
# 
# - **F1 score:** The F1 score is the harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.

# %% [markdown]
# **CNN Model 1**

# %%
from sklearn.metrics import classification_report
print(classification_report((normal_y_test), y_test_pred_classes_ln))

# %% [markdown]
# **CNN Model 2**

# %%
from sklearn.metrics import classification_report
print(classification_report(normal_y_test, y_test_pred_classes_ln3))

# %%
from sklearn.metrics import classification_report
print(classification_report((normal_y_test), y_test_pred_classes_ln4))

# %% [markdown]
# ### **Conclusion**

# %% [markdown]
# 1. As we have seen, the CNN model - 3 was able to predict the test image correctly with a test accuracy of 87%.
# 
# 2. **There might still be scope for improvement in the accuracy of the CNN model** chosen here. Try adding a more dense layer to the VGG16 model and see if you can get more accuracy than the best model.
# 
# 3. Once the desired performance is achieved from the model, the company can use it to classify different images being uploaded to the website.
# 

