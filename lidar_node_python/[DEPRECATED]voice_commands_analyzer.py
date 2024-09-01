# THIS FILE IS ONLY FOR ARCHIVE AS A TRAIN FIELD
import os
from pydub import AudioSegment
from pydub.playback import play

import tensorflow as tf
from keras import layers, models, optimizers

import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class Voice_analyzer:
    def __init__(self) -> None:
        self.input_data = []
        self.input_labels = []
        self.commands = ['forward', 'backward', 'left', 'right', 'stop']
        self.samples_path = '../samples/'
        self.input_data_train = None
        self.input_data_test = None
        self.input_labels_train = None
        self.input_labels_test = None
        self.seed = 42
        self.sound_frequency = 16000
        self.model = None
        self.training_history = None

        # START ANALYZE AND LEARNING PROCESS
        self.input_data, self.input_labels = self.load_data()
       
        self.normalize_data()

        self.build_model()

        self.train_model()

        self.rate_model()
        

    def _resize_array(self, spectogram_db, max_shape):
        # This method helps to fit np.array to the same shape
        # sub-sample for each audio has unregular count of values
        # np.array demend from us to set array in regular shape for instance 128x128
        result = np.zeros(max_shape)
        shape = spectogram_db.shape
        slices = tuple(slice(0, min(s, ts)) for s, ts in zip(shape, max_shape))
        result[slices] = spectogram_db[slices]
        return result


    def load_data(self):
        # loads and parses sound samples from wav format into data
        # for model to machine learning
        print('[DEBUG] START loading data')
        x = []
        y = []
        max_shape = (128, 96) # 128 is const, number of samples per one audio, 96 is max of taking values inside of sub-sample
        for index, command in enumerate(self.commands):
            print(f'[DEBUG] {index} - {command}')
            files = os.listdir(os.path.join(self.samples_path, command))
            print(f'Available files in {command}\n{files}')
            for file in files:
                file_path = os.path.join(self.samples_path, command, file)
                print(f'[DEBUG] Analyzing file {file}')
                audio, _ = librosa.load(file_path, sr=self.sound_frequency)
                mel_spectogram = librosa.feature.melspectrogram(y=audio, sr=self.sound_frequency)

                mel_spectogram_db = librosa.power_to_db(mel_spectogram, ref=np.max)

                # DEBUG to check how many values per one sub-sample
                for i in mel_spectogram_db:
                    print(f'[DEBUG] {file_path} - {len(i)}')

                mel_spectrogram_db_resized = self._resize_array(mel_spectogram_db, max_shape)

                x.append(mel_spectrogram_db_resized)
                y.append(index)
        
        print(f'[DEBUG] Data parsed successfully')

        return (np.array(x), np.array(y))
    

    def normalize_data(self):
        print(f'[DEBUG] Start to normalize data')
        self.input_data = self.input_data / np.max(self.input_data)
        print(f'[DEBUG] Data normalization finished')


    def build_model(self):
        # Division into training and test collection
        self.input_data_train, self.input_data_test, self.input_labels_train, self.input_labels_test = train_test_split(self.input_data, self.input_labels, test_size=0.2, random_state=self.seed)
        
        # Adaptation of input data dimensions to the requirements of the convolution model (CNN)
        # CNN models expect the input data to have four dimensions:

        # 1 - Number of samples: The number of samples in the dataset (batch size).
        # 2 - Height: The height of the image or feature.
        # 3 - Width: The width of the image or feature.
        # 4 - Number of channels: The number of channels (e.g., 1 for grayscale images, 3 for color images).
        self.input_data_train = self.input_data_train[..., np.newaxis]
        self.input_data_test = self.input_data_test[..., np.newaxis]

        print(f'DUPA {self.input_data_train.shape[1]}')

        print(f'DUPA2 {self.input_data_train.shape[2]}')

        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.input_data_train.shape[1], self.input_data_train.shape[2], 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.commands), activation='softmax')])
        
        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
    def train_model(self):
        # Training the model on prepared data.
        print(f'[DEBUG] Start model training')
        self.training_history = self.model.fit(self.input_data_train, self.input_labels_train, epochs=30, batch_size=32, validation_data=(self.input_data_test, self.input_labels_test))
        self.model.save('AI.h5')
    

    def rate_model(self):
        test_loss, test_acc = self.model.evaluate(self.input_data_test, self.input_labels_test, verbose=2)
        plt.plot(self.training_history.history['loss'])
        plt.plot(self.training_history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

        plt.plot(self.training_history.history['accuracy'])
        plt.plot(self.training_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        print(f'Test accuracy: {test_acc}\nTest loss: {test_loss}')


Voice_analyzer()