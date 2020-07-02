from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D 
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import backend as K

class SmallCNN:
    @staticmethod
    def build(width, height, depth,):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension 
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', padding="same" ))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2))) 

        model.add(Flatten())

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))             
        model.add(Dense(1 , activation='sigmoid'))
        
        return model
