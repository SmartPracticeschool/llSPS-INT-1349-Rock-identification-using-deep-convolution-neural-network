from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import load_model
import numpy as np
import cv2
#from skimage.transform import resize 
import PIL

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

x_train=train_datagen.flow_from_directory(r"C:\Users\tirumala guddanti\Desktop\rocks\trainset",target_size=(64,64),batch_size=32,class_mode="categorical")

x_test=test_datagen.flow_from_directory(r"C:\Users\tirumala guddanti\Desktop\rocks\testset",target_size=(64,64),batch_size=32,class_mode="categorical")

model = Sequential()

model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(output_dim=128,activation="relu",init="random_uniform"))

model.add(Dense(output_dim=5,activation="softmax",init="random_uniform"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit_generator(x_train,samples_per_epoch=8000,epochs=10,validation_data=x_test,nb_val_samples=2000)

model.save("mymodel.h5")

##model = load_model("mymodel.h5")
##
##model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
##
##def detect(frame):
##    try:
##        img = resize(frame,(64,64))
##        img = np.expand_dims(img,axis=0)
##        if(np.max(img)>1):
##            img = img / 255.0
##        prediction = model.predict(img)
##        print(prediction)
##        prediction_class = model.predict_classes(img)
##        print(prediction_class)
##    except AttributeError:
##        print("shape not found")
##
##frame = cv2.imread(r"C:\Users\tirumala guddanti\Desktop\rocks\trainset\sandstone\images(10).jpg")
##data = detect(frame)
##
##frame = cv2.imread("G:\data\sedimentary_data\10.jpg")
##data = detect(frame)
##
##
##frame = cv2.imread("G:\data\sedimentary_data\11.jpg")
##data = detect(frame)
