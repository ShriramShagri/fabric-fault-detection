from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

from .data_preprocessing import preprocessing

import matplotlib.pyplot as plt

def generateModel(lPath1, lPath2, overflow = False):
    """Generate model and run training

    Args:
        lPath1 (str): Path of first dataset
        lPath2 (str): Path of second dataset
        overflow (bool, optional): If dataset has image count > 2000 set to True. Defaults to False.
    """    

    # Split the dataset into train and validation set
    X_train, X_val, y_train, y_val = preprocessing(lPath1, lPath2, overflow = overflow)

    # Get length of training and validation data
    ntrain = len(X_train)
    nval = len(X_val)

    # We will use the batch size for this network as 32. Note this has to be a power of 2
    batch_size = 32

    # Generate Network Architecture called VGGNet
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Print the model architecture
    model.summary()

    # Loss ['binary_crossentropy']: We specify a loss function that our optimizer will minimize. 
    # Using optimizer as rmsprop. This is the prt of hyperparameter tuning
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    # Create ImageDataGenerator object
    train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                  )
    val_datagen = ImageDataGenerator(rescale=1./255)

    # call flow on the data generator and call fit to 
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    history = model.fit_generator(train_generator, 
                                steps_per_epoch = ntrain//batch_size,
                                epochs = 64,
                                validation_data = val_generator,
                                validation_steps = nval//batch_size
                                )
    
    # try:
    #     # Save models to a path
    #     model.save_weights('./src/saved_models/model_weights.h5')
    #     model.save('./src/saved_models/model_keras.h5')
    # except:
    #     model.save_weights('model_weights.h5')
    #     model.save('model_keras.h5')

    # Display graphs of accuracy and loss
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'r', label="Validation Accuracy")
    plt.title("Training and validation accuracy for classification")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label="Training loss")
    plt.plot(epochs, val_loss, 'r', label="Validation loss")
    plt.title("Training and validation loss for classification")
    plt.legend()

    plt.show()