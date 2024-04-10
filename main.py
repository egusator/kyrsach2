import warnings

import numpy as np

from DPCNN import create_dpcnn_model

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pandas

import EEGModels
from EEGModels import EEGNet_SSVEP


HCvMCI = pandas.read_csv('C:\\Users\\egusa\\Alzheimer-s-Classification-EEG\\data\\MCIvsHCFourier.csv')
MCIvAD = pandas.read_csv('C:\\Users\\egusa\\Alzheimer-s-Classification-EEG\\data\\MCIvsADFourier.csv')
ADvHC = pandas.read_csv('C:\\Users\\egusa\\Alzheimer-s-Classification-EEG\\data\\ADvsHCFourier.csv')

y = ADvHC["class"]
x = ADvHC.iloc[:, 1:305]
x_train_reshaped = np.expand_dims(x, axis=-1) # Adds an extra dimension to make it (None, 304, 1)
x_train_reshaped = np.expand_dims(x_train_reshaped, axis=1) # Ad

from sklearn.preprocessing import LabelEncoder

# Assuming y_train is your target labels with string values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y)

# Now, you can use to_categorical with the encoded labels
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train_encoded, num_classes=2)

x_train, x_test, y_train, y_test = train_test_split(x_train_reshaped, y_train_one_hot, test_size=0.2, random_state=42)

kernels, chans, samples = 1, 1, 304

model = create_dpcnn_model((None, 304), 2)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Assuming you have your data and labels ready
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Example evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Example prediction
predictions = model.predict(y_test)