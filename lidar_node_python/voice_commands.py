import io
import os
import math
import pathlib
import struct
from time import sleep, time
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pyaudio
import numpy as np
import wave
import speech_recognition

# audio settings
RATE = 48000  # (48kHz)
CHUNK = 1024  # Buffer size (sample number on each call)
LABELS = ['backward', 'forward', 'left', 'right', 'stop']
THRESHOLD = 25  # Próg głośności (RMS), ~-40DB
SWIDTH = 2
SHORT_NORMALIZE = (1.0/32768.0)
DELAY_AFTER_THRESHOLD = 0.4
FORMAT = pyaudio.paInt16
CHANNELS = 1
AUTOTUNE = tf.data.AUTOTUNE


class Voice_movement:
    def __init__(self) -> None:
        # Load model
        self.model = tf.keras.models.load_model('AI.h5')
        self.audio = pyaudio.PyAudio()
        self.stream =  self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                input_device_index=0, frames_per_buffer=CHUNK)
        self.current_direction = None
        

    def listen_and_predict_command(self):
        while True:
            print("[DEBUG] Waiting for the sound threshold of 25 dB to be exceeded.")
            recording = False
            frames = []
            silence_start_time = None

            while True:
                input = self.stream.read(CHUNK, exception_on_overflow=False)
                rms_value = self._rms(input)

                if rms_value > THRESHOLD:
                    if not recording:
                        print("[DEBUG] Recording started")
                        recording = True
                    frames.append(input)
                    silence_start_time = None
                elif recording:
                    if silence_start_time is None:
                        silence_start_time = time()  # Rozpocznij mierzenie czasu ciszy
                    elif (time() - silence_start_time) >= DELAY_AFTER_THRESHOLD :
                        print("[DEBUG] Recording finished")
                        break
                    frames.append(input)

            
            if frames:    
                audio_data = b''.join(frames)
                self._save_audio(audio_data)
                arr = ['recording.wav']
                
                file = tf.data.Dataset.from_tensor_slices(arr)
                output = file.map(map_func=self._get_waveform, num_parallel_calls=AUTOTUNE)
                waveform = list(output.as_numpy_iterator())[0]
                num_samples_to_silence = int(0.05 * RATE)
                waveform = tf.concat([tf.zeros(num_samples_to_silence), waveform[num_samples_to_silence:]], 0)
                spectrogram = output.map(map_func=self._get_spectrogram, num_parallel_calls=AUTOTUNE)
                
                audio = []

                for x in spectrogram:
                    audio.append(x.numpy())
                audio = np.array(audio)
                prediction = np.argmax(self.model.predict(audio), axis=1)
                self.current_direction = f'{LABELS[prediction[0]]}'
                print(f"[INFO] Predicted command: {LABELS[prediction[0]]}")
            os.remove('recording.wav')
        

    def show_plot_for_spectrogram(self, waveform, spectrogram):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Wykres fali dźwiękowej (Waveform)
        axs[0].plot(waveform)
        axs[0].set_title("Waveform")
        axs[0].set_xlabel("Czas")
        axs[0].set_ylabel("Amplituda")

        # Wykres spektrogramu
        spectrogram = np.squeeze(spectrogram)
        axs[1].imshow(np.log(spectrogram.T), aspect='auto', origin='lower')
        axs[1].set_title("Spectrogram")
        axs[1].set_xlabel("Czas")
        axs[1].set_ylabel("Częstotliwość")

        plt.show()



    def _rms(self, frame):
        count = len(frame)/SWIDTH
        format = "%dh"%(count)
        # short is 16 bit int
        shorts = struct.unpack( format, frame )

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n*n
        # compute the rms 
        rms = math.pow(sum_squares/count,0.5)
        return rms * 1000


    def _decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(contents=audio_binary, desired_channels=2)
        mono_audio = tf.reduce_mean(audio, axis=-1)
        return mono_audio
    

    def _get_waveform(self, audio):
        audio_binary = tf.io.read_file(audio)
        waveform = self._decode_audio(audio_binary)
        return waveform
    
    def _get_spectrogram(self, waveform):
        input_len = 48000
        waveform = waveform[:input_len]
        zero_padding = tf.zeros([48000] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, dtype=tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram
    

    def _save_audio(self, data):
        # Parametry WAV
        wf = wave.open('recording.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()
        print("Plik WAV zapisany jako 'recording.wav'")
    