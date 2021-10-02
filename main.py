import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

for dirpath, dirnames, filenames in os.walk("data_klasifikasi_batik"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}")

def view_random_image(target_dir, target_class):
    target_folder = target_dir + target_class

    # Take 1 sample
    random_image = random.sample(os.listdir(target_folder), 1)

    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Picture size: {img.shape}")
    return img
  
def show_batik():
    for cls in ["Kawung", "Ceplok", "Parang", "Nitik", "Mix motif", "Lereng"]:
        view_random_image("data_klasifikasi_batik/train/", cls)
        plt.show()
    
tf.random.set_seed(3244)

train_dir = "data_klasifikasi_batik/train/"
test_dir = "data_klasifikasi_batik/test/"

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label = "Training loss")
    plt.plot(epochs, val_loss, label = "Validation loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, accuracy, label = "Training accuracy")
    plt.plot(epochs, val_accuracy, label = "Validation accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

def test_model_1():
    train_datagen = ImageDataGenerator(rescale = 1./255)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_data = train_datagen.flow_from_directory(train_dir,
                                                   batch_size = 16,
                                                   target_size = (500, 500),
                                                   class_mode = "categorical",
                                                   seed = 3244)

    valid_data = train_datagen.flow_from_directory(test_dir,
                                                   batch_size = 16,
                                                   target_size = (500, 500),
                                                   class_mode = "categorical",
                                                   seed = 3244)
    
    model_1 = Sequential([
                          Conv2D(10, 3, activation = 'relu'), # 10 filters, 3x3 kernel size
                          MaxPool2D(2),
                          Conv2D(10, 3, activation = 'relu'), # 10 filters, 3x3 kernel size
                          MaxPool2D(2),
                          Flatten(),
                          Dense(6, activation = 'softmax') # 6 classes
    ])

    model_1.compile(loss = 'categorical_crossentropy',
                    optimizer = Adam(),
                    metrics = ['accuracy'])

    history_1 = model_1.fit(train_data,
                            epochs = 10, # probably try 30 to check overfit also
                            steps_per_epoch = len(train_data),
                            validation_data = valid_data,
                            validation_steps = len(valid_data))
    
    plot_loss_curves(history_1)

def test_model_2():
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       horizontal_flip = True)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_data_augmented = train_datagen.flow_from_directory(train_dir,
                                                             batch_size = 16,
                                                             target_size = (500, 500),
                                                             class_mode = "categorical",
                                                             shuffle = True)

    model_2 = Sequential([
                          Conv2D(10, 3, activation = 'relu'), # 10 filters, 3x3 kernel size
                          MaxPool2D(2),
                          Conv2D(10, 3, activation = 'relu'), # 10 filters, 3x3 kernel size
                          MaxPool2D(2),
                          Flatten(),
                          Dense(6, activation = 'softmax') # 6 classes
    ])

    model_2.compile(loss = 'categorical_crossentropy',
                    optimizer = Adam(),
                    metrics = ['accuracy'])

    history_2 = model_2.fit(train_data_augmented,
                            epochs = 10, # probably try 30 to check overfit also
                            steps_per_epoch = len(train_data),
                            validation_data = valid_data,
                            validation_steps = len(valid_data))

    plot_loss_curves(history_2)
    
def test_model_3():
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       horizontal_flip = True)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_data_augmented = train_datagen.flow_from_directory(train_dir,
                                                             batch_size = 32,
                                                             target_size = (500, 500),
                                                             class_mode = "categorical",
                                                             shuffle = True)
    
    model_3 = Sequential([
                          Conv2D(50, 3, activation = 'relu'),
                          Conv2D(50, 3, activation = 'relu'),
                          MaxPool2D(2),
                          Conv2D(50, 3, activation = 'relu'),
                          Conv2D(50, 3, activation = 'relu'),
                          MaxPool2D(2),
                          Flatten(),
                          Dense(6, activation = 'softmax') # 6 classes
    ])

    model_3.compile(loss = 'categorical_crossentropy',
                    optimizer = Adam(),
                    metrics = ['accuracy'])

    # Remove steps because of data shortage, let the fitter decide
    history_3 = model_3.fit(train_data_augmented,
                            epochs = 10,
                            validation_data = valid_data)
    
    plot_loss_curves(history_3)
    
def test_model_4():
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       horizontal_flip = True)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_data_augmented = train_datagen.flow_from_directory(train_dir,
                                                             batch_size = 32,
                                                             target_size = (500, 500),
                                                             class_mode = "categorical",
                                                             shuffle = True)
    
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (450, 450, 3),
                                                                include_top = False,
                                                                weights = "imagenet")
    
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = Dense(6)
    soft = Activation('softmax')

    inputs = tf.keras.Input(shape = (450, 450, 3))
    x = base_model(inputs, training = False)
    x = global_average_layer(x)
    x = Dropout(0.2)(x)
    outputs = prediction_layer(x)
    outputs = soft(outputs)

    model_4 = tf.keras.Model(inputs, outputs)
    model_4.compile(loss = 'categorical_crossentropy',
                    optimizer = Adam(),
                    metrics = ['accuracy'])

    history_4 = model_4.fit(train_data_augmented,
                            epochs = 10,
                            validation_data = valid_data)

    plot_loss_curves(history_4)
    
models = [test_model_1, test_model_2, test_model_3, test_model_4]
def test_model(x):
    if x == 0:
        return
    models[x - 1]()
    
test_model(0)
