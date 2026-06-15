# %% [markdown]
# <a href="https://colab.research.google.com/github/aj4di/AI-ComputerVision/blob/main/MRI_braintumor_Transfer%2BLearning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # **Brain Tumor Image Classifier**

# %% [markdown]
# ## **Context**
# 
# In this notebook, we will build an image classifier that can distinguish Pituitary Tumor from "No Tumor" MRI Scan images.
# 
# The dataset used in this notebook is available for download from [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).
# 
# Although this dataset actually has a total of 3,264 images belonging to 4 classes - Glioma Tumor, Meningioma Tumor, Pituitary Tumor and No Tumor, for this project we have only taken two classes, and **we are building a binary classification model to classify between the Pituitary Tumor category vs No Tumor.**
# 
# For this project, we will only use 1000 of these images (830 training images and 170 Testing images). For the training dataset, we will take 395 MRI scans of No Tumor and 435 MRI scans of Pituitary Tumor. In our problem, we will also be using Data Augmentation to prevent overfitting, and to make our model model more generalised and robust.
# 
# We will use this to build an image classification model for this problem statement, and then show how we can improve our performance by simply "importing" a popular pre-trained model architecture and leveraging the idea of **Transfer Learning**.
# 
# ## **Objectives**
# The objectives of this project are to:
# 1. Load and understand the dataset
# 2. Automatically label the images
# 3. Perform Data Augmentation
# 4. Build a classification model for this problem using CNNs
# 5. Improve the model's performance through Transfer Learning
# 
# 

# %% [markdown]
# ## **Importing Libraries**

# %%
# Library for creating data paths
import os

# Library for randomly selecting data points
import random

# Library for performing numerical computations
import numpy as np

# Library for creating and showing plots
import matplotlib.pyplot as plt

# Library for reading and showing images
import matplotlib.image as mpimg

# Importing all the required sub-modules from Keras
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# %% [markdown]
# Mounting the drive to load the dataset

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# We have stored the images in a structured folder, and below we create the data paths to load images from those folders. This is required so that we can extract images in an auto-labelled fashion using Keras **flow_from_directory**.

# %%
# Parent directory where images are stored in drive
parent_dir = 'brain_tumor'

# Path to the training and testing datasets within the parent directory
train_dir = os.path.join(parent_dir, 'Training')
validation_dir = os.path.join(parent_dir, 'Testing')

# Directory with our training pictures
train_pituitary_dir = os.path.join(train_dir, 'pituitary_tumor')
train_no_dir = os.path.join(train_dir, 'no_tumor')

# Directory with our testing pictures
validation_pituitary_dir = os.path.join(validation_dir, 'pituitary_tumor')
validation_no_dir = os.path.join(validation_dir, 'no_tumor')

# %% [markdown]
# ## **Visualizing a few images**

# %% [markdown]
# Before we move ahead and perform data augmentation, let's randomly check out some of the images and see what they look like:

# %%
train_pituitary_file_names = os.listdir(train_pituitary_dir)
train_no_file_names = os.listdir(train_no_dir)

fig = plt.figure(figsize=(16, 8))
fig.set_size_inches(16, 16)

pituitary_img_paths = [os.path.join(train_pituitary_dir, file_name) for file_name in train_pituitary_file_names[:8]]
no_img_paths = [os.path.join(train_no_dir, file_name) for file_name in train_no_file_names[:8]]

for i, img_path in enumerate(pituitary_img_paths + no_img_paths):
    ax = plt.subplot(4, 4, i + 1)
    ax.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# %% [markdown]
# As we can see, the images are quite different in size from each other.
# 
# This represents a problem, as most CNN architectures, including the pre-built model architectures that we will use for Transfer Learning, **expect all the images to have the same size.**
# 
# So we need to crop these images from the center to make sure they all have the same size. We can do this automatically while performing Data Augmentation, as shown below.

# %% [markdown]
# ## **Data Augmentation**
# 
# In most real-life case studies, it is generally difficult to collect lots of images and then train CNNs. In that case, one idea we can take advantage of is Data Augmentation. CNNs have the property of **translational invariance**, i.e., they can recognize an object as an object, even when its appearance varies translationally in some way. Taking this property into consideration, we can augment the images using the following techniques:

# %% [markdown]
# **1. Horizontal Flip** (should be set to True/False) <br>
# **2. Vertical Flip** (should be set to True/False) <br>
# **3. Height Shift** (should be between 0 and 1) <br>
# **4. Width Shift** (should be between 0 and 1) <br>
# **5. Rotation** (should be between 0 and 180) <br>
# **6. Shear** (should be between 0 and 1) <br>
# **7. Zoom** (should be between 0 and 1) etc. <br>
# 
# Remember ***not to use data augmentation in the validation/test data set***.

# %% [markdown]
# Also, as mentioned above, we need to have images of the same size. So below,we resize the images by using the parameter **target_size**. Here we are resizing it to **224 x 224**, as we will be using the **VGG16** model for Transfer Learning, which takes image inputs as **224 x 224**.
# 
# As this is a binary classification problem, we will need class labels. This is directly handled by the **flow_from_directory** function. It will take the images from the folder inside our specified directory, and the images from one folder will belong to same class.
# 
# As the train directory has 2 folders pituitary_tumor and no_tumor, it will read the directory and each folder will be considered a separate class. We specify **class_model = 'binary'** as this is a binary classification problem.
# 
# As the folders inside the directory will be read in an alphabetical order, the no_tumor folder will be given a label 0, and pituitary_tumor will be label 1.

# %%
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
                                                    class_mode='binary',
                                                    target_size=(224, 224))

# Flowing testing images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (224, 224))

# %% [markdown]
# Let's look at some examples of our augmented training data.
# 
# This is helpful for understanding the extent to which data is being manipulated prior to training, and can be compared with how the raw data looks prior to data augmentation.

# %%
images, labels = next(train_generator)
fig, axes = plt.subplots(4, 4, figsize = (16, 8))
fig.set_size_inches(16, 16)
for (image, label, ax) in zip(images, labels, axes.flatten()):
    ax.imshow(image)
    if label == 1:
        ax.set_title('pituitary tumor')
    else:
        ax.set_title('no tumor')
    ax.axis('off')

# %% [markdown]
# ## **CNN Model Building**

# %% [markdown]
# Once the data is augmented and cropped to have the same size, we are now ready to build a first baseline CNN model to classify no_tumor vs pituitary_tumor.
# 
# When building our custom model, we have used Batch Normalization and Dropout layers as regularization techniques to prevent overfitting.

# %%
cnn_model = Sequential()
cnn_model.add(Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3), padding = 'same'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(32, (3,3), activation='relu', padding = 'same'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(32, (3,3), activation='relu', padding = 'same'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Conv2D(16, (3,3), activation='relu', padding = 'same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid')

# %%
cnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ['accuracy'])
cnn_model.summary()

# %%
# Pulling a single large batch of random testing data for testing after each epoch
testX, testY = validation_generator.next()

# %%
model_history = cnn_model.fit(train_generator,
                              validation_data=(testX, testY),
                              epochs=10)

# %%
# Evaluating on the Test dataset
cnn_model.evaluate(validation_generator)

# %% [markdown]
# ### **Findings**
# 
# - Our model had 840,369 trainable parameters.
# - After running 10 epochs, we were able to achieve a training accuracy of ~95% and a testing accuracy of ~72%.
# - Even after using Data Augmentation, Batch Normalization and the Dropout Layers, the model seems to have highly overfit on the training dataset and is performing somewhat poorly.

# %% [markdown]
# ## **Model Building using Transfer Learning: VGG 16**

# %% [markdown]
# - Now, let's try again, but this time, using the idea of **Transfer Learning**. We will be loading a pre-built architecture - **VGG16**, which was trained on the ImageNet dataset and finished runner-up in the ImageNet competition in 2014. Below is a schematic of the VGG16 model.
# 
# - For training VGG16, we will directly use the convolutional and pooling layers and freeze their weights i.e. no training will be done on them. We will remove the already-present fully-connected layers and add our own fully-connected layers for this binary classification task.
# 

# %% [markdown]
# ![vgg16](https://upload.wikimedia.org/wikipedia/commons/2/24/VGG16.png)

# %%
# Summary of the whole model
model = VGG16(weights='imagenet')
model.summary()

# %%
# Getting only the conv layers for transfer learning.
transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs=model.input, outputs=transfer_layer.output)

# %%
vgg_model.summary()

# %% [markdown]
# - To remove the fully-connected layers of the imported pre-trained model, while calling it from Keras we can also specify an additonal keyword argument that is **include_top**.
# 
# - **If we specify include_top = False, then the model will be imported without the fully-connected layers.** Here we won't have to do the above steps of getting the last convolutional layer and creating a separate model.
# 
# - If we are specifying include_top = False, we will also have to specify our input image shape.
# 
# - Keras has this keyword argument as generally while importing a pre-trained CNN model, we don't require the fully-connected layers and we train our own fully-connected layers for our task.

# %%
vgg_model = VGG16(weights='imagenet', include_top = False, input_shape = (224,224,3))
vgg_model.summary()

# %%
# Making all the layers of the VGG model non-trainable. i.e. freezing them
for layer in vgg_model.layers:
    layer.trainable = False

# %%
for layer in vgg_model.layers:
    print(layer.name, layer.trainable)

# %%
new_model = Sequential()

# Adding the convolutional part of the VGG16 model from above
new_model.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
new_model.add(Flatten())

# Adding a dense output layer
new_model.add(Dense(32, activation='relu'))
new_model.add(Dense(32, activation='relu'))
new_model.add(Dense(1, activation='sigmoid'))

# %%
new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
new_model.summary()

# %%
## Fitting the VGG model
new_model_history = new_model.fit(train_generator,
                                  validation_data=(testX, testY),
                                  epochs=5)

# %%
# Evaluating on the Test set
new_model.evaluate(validation_generator)

# %%
# Function to plot loss, val_loss,
def plot_history(history):
    N = len(history.history["accuracy"])
    plt.figure()
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training accuracy Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("accuracy")
    plt.legend(loc="upper right")

# %%
# Plotting the loss vs epoch curve for the basic CNN model without Transfer Learning
plot_history(model_history)

# %%
# Plotting the loss vs epoch curve for the Transfer Learning model
plot_history(new_model_history)

# %% [markdown]
# ### Findings
# 
# - Our model has 803,937 Trainable parameters.
# - After running 5 epochs we were able to achieve a training accuracy of ~98% and a validation accuracy of ~ 91%.

# %% [markdown]
# ## **Conclusions**
# 
# - The difference in both models is evident. Both models had nearly the same number of trainable parameters. However even after training the custom CNN model for 10 epochs, it could not attain accuracies as high as we achieved with Transfer Learning.
# 
# - The Transfer Learning model has converged faster than the custom CNN model in only 5 epochs.
# 
# - That's a good level of improvement just by directly using a pre-trained architecture such as VGG16.
# 
# - This model can, in fact, further be tuned to achieve the accuracies required for practical applicability in the medical domain.

