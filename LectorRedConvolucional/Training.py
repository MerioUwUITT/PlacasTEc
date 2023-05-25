import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import to_categorical

# Cargar los datos de entrenamiento y prueba de MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar los datos de imagen y convertir las etiquetas en vectores one-hot
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Crear un modelo secuencial con dos capas convolucionales y dos capas totalmente conectadas
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar el modelo con una función de pérdida, un optimizador y una métrica
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5, batch_size=64)

# Evaluar el modelo con los datos de prueba
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print('Precisión del modelo en los datos de prueba:', test_acc)

# Guardar el modelo en un archivo HDF5
model.save('modelo.h5')
