from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout, Embedding

def create_dpcnn_model(input_shape, num_classes):
    model = Sequential()

    # First convolutional block
    model.add(Conv1D(304, 3, activation='relu'))
    model.add(Conv1D(304, 3, activation='relu'))
    model.add(Conv1D(304, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())

    # Second convolutional block
    model.add(Conv1D(304, 4, activation='relu'))
    model.add(Conv1D(304, 4, activation='relu'))
    model.add(Conv1D(304, 4, activation='relu'))
    model.add(GlobalMaxPooling1D())

    # Third convolutional block
    model.add(Conv1D(304, 5, activation='relu'))
    model.add(Conv1D(304, 5, activation='relu'))
    model.add(Conv1D(304, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(304, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model