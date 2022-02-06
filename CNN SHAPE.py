from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()
model.add(Convolution2D(16, 3, 3, input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(56, activation='relu', kernel_initializer='uniform'))
model.add(Dense(3, activation='softmax', kernel_initializer='uniform'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('S:\\Project\\shapes\\train', target_size=(28, 28),color_mode="grayscale", batch_size=1, class_mode='categorical')
test_set = test_datagen.flow_from_directory('S:\\Project\\shapes\\test', target_size=(28, 28),color_mode="grayscale",batch_size=1, class_mode='categorical')

steps_per_epoch = len(training_set.filenames) 
validation_steps = len(test_set.filenames) 

model_info = model.fit_generator(training_set, steps_per_epoch=steps_per_epoch, epochs=25, validation_data=test_set,
                                      validation_steps=validation_steps)

model.save("S:\\Project\\trained model\\model.h5")

