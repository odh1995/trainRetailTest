import tensorflow as tf
from tensorflow.keras import *
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import wrap_frozen_graph

x = pickle.load(open("x.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
x = np.array(x/255)
y = np.array(y)


def gpuSetting():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def myModel(x, y):

    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (5, 5), input_shape=x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, (5, 5), input_shape=x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    return model


gpuSetting()

model = myModel(x, y)

history = model.fit(x, y, batch_size=64, epochs=20, validation_split=0.3)

# plt.figure(1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.title('Loss')
# plt.xlabel('epoch')
# plt.figure(2)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training', 'validation'])
# plt.title('Accuracy')
# plt.xlabel('epoch')
# plt.show()

score = model.evaluate(x, y, verbose=0)

print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

# Save model to SavedModel format
tf.saved_model.save(model, "./models/simple_model")
model.save("./models/model.h5")

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="simple_frozen_graph.pb",
                  as_text=False)

# Save frozen graph from frozen ConcreteFunction to hard drive text readable
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="simple_frozen_graph.pbtxt",
                  as_text=True)






























