from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
# from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.optimizers import adam



# MLP
def create_mlp(test=False):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=43))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.1))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.1))


    if test:
        model.add(Dense(4, activation='softmax'))
        optimizer = adam(learning_rate=0.0005, decay=1e-6)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return model


# CNN
def create_cnn(width, height, rgb=True, filtersNb=(4, 8, 16, 32), test=False):
    if rgb:
        inShape = (width, height, 3)
    else:
        inShape = (width, height)

    inLayer = Input(shape=inShape)

    # for i, filtersCount in enumerate(filtersNb):
    #     if i == 0:
    #         prev = inLayer
    #
    #     prev = Conv2D(filtersCount, kernel_size=(3, 3), padding="same")(prev)
    #     prev = MaxPooling2D(pool_size=(2, 2))(prev)
    #     prev = BatchNormalization()(prev)
    #     prev = Dropout(0.4)(prev)

    prev = Conv2D(16, (3, 3), padding='same', activation='relu')(inLayer)
    prev = MaxPooling2D(pool_size=(2, 2))(prev)
    prev = Dropout(0.4)(prev)

    prev = Conv2D(32, (3, 3), padding='same', activation='relu')(inLayer)
    prev = MaxPooling2D(pool_size=(3, 3))(prev)
    prev = Dropout(0.4)(prev)

    prev = Conv2D(64, (3, 3), padding='same', activation='relu')(inLayer)
    prev = MaxPooling2D(pool_size=(3, 3))(prev)
    prev = Dropout(0.4)(prev)

    prev = Flatten()(prev)
    prev = (Dense(256, activation='relu'))(prev)
    prev = (Dropout(0.5))(prev)

    # prev = Dense(128, activation='relu')(prev)
    # prev = Dense(64, activation='relu')(prev)

    # prev = Dense(128, activation='relu')(prev)
    # prev = Dense(16, activation='relu')(prev)

    if test:
        prev = Dense(33, activation="softmax")(prev)

    model = Model(inLayer, prev)
    if test:
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
    return model


def create_final_model(summary=False):
    modelCNN = create_cnn(64, 64, filtersNb=(4, 8, 16, 32))
    modelMLP = create_mlp()
    combinedInput = concatenate([modelMLP.output, modelCNN.output])
    x = Dense(128, activation="relu")(combinedInput)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(4, activation="softmax")(x)

    model = Model(inputs=[modelMLP.input, modelCNN.input], outputs=x)
    if summary:
        model.summary()
    optimizer = adam(learning_rate=0.0005, decay=1e-6)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    create_cnn(64, 64, test=False)
