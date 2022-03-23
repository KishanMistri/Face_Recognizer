# Convolution_Neural_Network_Method

#Part 1 : Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

#Initialzing CNN
classifier = Sequential()

#Step-1: Convolution
classifier.add(Convolution2D(filters = 32,kernel_size = 3,input_shape = (64,64,3),activation = 'relu'))

#Step-2: Max-Pooling
classifier.add(MaxPooling2D(2,2))


#Adding 2nd convolution layer to increase efficiency 
classifier.add(Convolution2D(filters = 32,kernel_size = 3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3: Flattening
classifier.add(Flatten()) #Input Layer is ready
 
#Step-4: Full connetion
classifier.add(Dense(activation= 'relu',units=128))   #Hidden Layer is ready
classifier.add(Dense(activation= 'sigmoid',units=1))  #Output Layer Is ready

#Compiling ANN -> MEans Stochastic Gradient Decent
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics=['accuracy'] )

# Path-2: Fitting CNN to the images
#Image augementation to prevant overfitting to data
#https://keras.io/preprocessing/image/
'''
#Features we can use:
    keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=K.image_data_format())
'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set data path',
        target_size=( 64 , 64 ),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test set path',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)