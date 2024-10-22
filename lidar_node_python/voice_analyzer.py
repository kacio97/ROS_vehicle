import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from keras import layers, models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
DATASET_PATH = 'E:/ROS_vehicle/samples/'
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 1000

class VoiceAnalyzer:
  def __init__(self) -> None:
    self.data_dir = None
    self.commands = None
    self.filenames = None
    self.num_samples = None
    self.train_files = None
    self.val_files = None
    self.test_files = None
    self.files_data_set = None
    self.model = None
    self.history = None


  def start_analyze(self):
    self.prepare_data_set()

    self.waveform_data_set = self.files_data_set.map(map_func=self._get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    self.show_charts_for_waveform()

    self.show_plot_for_one_waveform_and_spectogram()

    self.spectrogram_data_set = self.waveform_data_set.map(map_func=self._get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    self.train_data_set = self.spectrogram_data_set
    self.val_data_set = self._preprocess_dataset(self.val_files)
    self.test_data_set = self._preprocess_dataset(self.test_files)

    self.train_data_set = self.train_data_set.batch(BATCH_SIZE)
    self.val_data_set = self.val_data_set.batch(BATCH_SIZE)

    self.train_data_set = self.train_data_set.cache().prefetch(AUTOTUNE)
    self.val_data_set = self.val_data_set.cache().prefetch(AUTOTUNE)

    self.prepare_model()
    self.train_model()
    self.show_training_and_validation_loss_curves()
    self.test_model_efficiency()


  def show_charts_for_waveform(self):
    # These variables define the number of rows and columns in the grid layout that will be used to display the charts.
    rows = 3
    cols = 3
    # the number of charts that will be generated (3 * 3)
    number_of_charts = rows * cols # x = 9
    # Creating charts for later visualization of audio data.
    _, axes = plt.subplots(rows, cols, figsize=(10, 12))

    for i, (audio, label) in enumerate(self.waveform_data_set.take(number_of_charts)):
      row = i // cols # calculates the row number in the chart grid for the current subplot
      column = i % cols # calculates the column number in the chart grid for the current subplot
      ax = axes[row][column]
      ax.plot(audio.numpy())
      ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
      label = label.numpy().decode('utf-8')
      ax.set_title(label)

    plt.show()


  def show_charts_for_spectograms(self):
    # These variables define the number of rows and columns in the grid layout that will be used to display the charts.
    rows = 3
    cols = 3
    # the number of charts that will be generated (3 * 3)
    number_of_charts = rows * cols
    _, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (spectrogram, label_id) in enumerate(self.spectrogram_data_set.take(number_of_charts)):
      rows = i // cols
      cols = i % cols
      ax = axes[rows][cols]
      self._plot_spectrogram(spectrogram.numpy(), ax)
      ax.set_title(self.commands[label_id.numpy()])
      ax.axis('off')

    plt.show()


  def _plot_spectrogram(self, spectrogram, ax):
    if len(spectrogram.shape) > 2:
      assert len(spectrogram.shape) == 3
      spectrogram = np.squeeze(spectrogram, axis=-1) # removes the last axis of dimension 1 from the array

    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


  def show_plot_for_one_waveform_and_spectogram(self):
    # Print the tensorized waveforms from one example and the corresponding spectrogram
    for waveform, label in self.waveform_data_set.take(1):
      label = label.numpy().decode('utf-8')
      spectrogram = self._get_spectrogram(waveform)
    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)

    # Plot the waveform over time from the example and the corresponding spectrogram (frequencies over time):
    _, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 64000])
    axes[1].set_title('Spectrogram')

    self._plot_spectrogram(spectrogram.numpy(), axes[1])
    plt.show()


  def prepare_data_set(self):
    self.data_dir =pathlib.Path(DATASET_PATH)
    self.commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
    self.filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*')

    # The files are randomly mixed, which is important so that the model does not train on the structured data
    self.filenames = tf.random.shuffle(self.filenames)

    self.num_samples = len(self.filenames)
    print(f'[DEBUG] Number of total examples: {self.num_samples}')
    print(f'[DEBUG] Number of examples per label: {len(tf.io.gfile.listdir(str(self.data_dir/self.commands[0])))}')
    # print(f'[DEBUG] Example file tensor: {self.filenames[0]}')

    # Split data into 3 parts 80% train files, 10% per validation and testing
    self.train_files = self.filenames[:240]
    self.val_files = self.filenames[240: 240 + 30]
    self.test_files = self.filenames[-30:]

    print(f'[DEBUG] Training set size {len(self.train_files)}')
    print(f'[DEBUG] Validation set size {len(self.val_files)}')
    print(f'[DEBUG] Test set size {len(self.test_files)}')

    self.files_data_set = tf.data.Dataset.from_tensor_slices(self.train_files)


  def prepare_model(self):
    for spectrogram, _ in self.spectrogram_data_set.take(1):
      input_shape = spectrogram.shape
    print(f'[DEBUG] Input shape:{input_shape}')
    num_labels = len(self.commands)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=self.spectrogram_data_set.map(map_func=lambda spec, label: spec))

    self.model = models.Sequential([
      layers.Input(shape=input_shape),
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      norm_layer,
      layers.Conv2D(32, 3, activation='relu'),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(num_labels),
    ])

    self.model.summary()

    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
    )


  def train_model(self):
    self.history = self.model.fit(
      self.train_data_set,
      validation_data=self.val_data_set,
      epochs=EPOCHS,
      callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),)
    self.model.save('AI.h5')
    

  def show_training_and_validation_loss_curves(self):
    metrics = self.history.history
    plt.plot(self.history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()

  def test_model_efficiency(self):
    test_audio = []
    test_labels = []

    for audio, label in self.test_data_set:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(self.model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')


  def _decode_audio(self, audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. 
    # @return - float32 mono_audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels=2)

    # Since all the data is single channel (mono), 
    # drop the `channel` to get a one-dimensional audio signal
    mono_audio = tf.reduce_mean(audio, axis=-1)
    return mono_audio


  def _get_waveform_and_label(self, file_path):
    # The method is mapped to each file to get the waveform data (waveform) and the label (label). 
    # The processing is done in parallel (AUTOTUNE parameter).
    # @return wave data and label
    label = self._get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = self._decode_audio(audio_binary)
    return waveform, label


  def _get_label(self, file_path):
    # Extracts the label from the file name
    parts = tf.strings.split(
    input=file_path,
    sep=os.path.sep)
    return parts[-2]
  

  def _get_spectrogram(self, waveform):
    # Zero-padding for an audio waveform with less than 64,000 samples.
    input_len = 64000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([64000] - tf.shape(waveform), dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
  

  def _get_spectrogram_and_label_id(self, audio, label):
    spectrogram = self._get_spectrogram(audio)
    label_id = tf.argmax(label == self.commands)
    return spectrogram, label_id


  def _preprocess_dataset(self, files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(map_func=self._get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(map_func=self._get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


VoiceAnalyzer().start_analyze() # RUN VOICE ANALYZER
