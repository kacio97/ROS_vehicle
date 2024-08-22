
import os
from pydub import AudioSegment
from pydub.playback import play

import tensorflow as tf
from keras import layers, models

import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# y, sr = librosa.load("../samples/forward/forward_1.wav")
# mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
# log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

# plt.figure(figsize=(10, 4))
# librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.show()



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

        try:
            self.input_data, self.input_labels = self.load_data()
        except:
            print(f'[ERROR] Failed to load data into data-parser')

        try:
            self.normalize_data()
        except:
            print('[ERROR] Failed to normalize data')

        try:
            self.build_model()
        except:
            print('[ERROR] Failed to build model')

        try:
            self.train_model()
            self.rate_model()
        except:
            print('[ERROR] Failed to train model')
        

    def load_data(self):
        # loads and parses sound samples from wav format into data
        # for model to machine learning
        print('[DEBUG] START loading data')
        for index, command in enumerate(self.commands):
            print(f'[DEBUG] {index} {command}')
            x = []
            y = []
            files = os.listdir(os.path.join(self.samples_path, command))
            print(f'Available files in {command}\n{files}')
            for file in files:
                file_path = os.path.join(self.samples_path, command, file)
                print(f'[DEBUG] Analyzing file {file}')
                audio, _ = librosa.load(file_path, sr=self.sound_frequency)
                mel_spectogram = librosa.feature.melspectrogram(audio, sr=self.sound_frequency)
                mel_spectogram_db = librosa.power_to_db(mel_spectogram, ref=np.max)
                x.append(mel_spectogram_db)
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

        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.input_data_train.shape[1], self.input_data_train.shape[2], 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.commands), activation='softmax')])
        
        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
    def train_model(self):
        # Training the model on prepared data.
        print(f'[DEBUG] Start model training')
        self.model.fit(self.input_data_train, self.input_labels_train, epochs=10, validation_data=(self.input_data_test, self.input_labels_test))
        self.model.save('AI.h5')
    

    def rate_model(self):
        test_loss, test_acc = self.model.evaluate(self.input_data_test, self.input_labels_test, verbose=2)
        print(f'Test accuracy: {test_acc}\nTest loss: {test_loss}')


Voice_analyzer()