import io
import os
import math
import pathlib
import struct
from time import sleep, time
import tensorflow as tf
from tensorflow import keras
import pyaudio
import numpy as np
import wave
import speech_recognition

# audio settings
RATE = 44100  # (44kHz)
CHUNK = 1024  # Buffer size (sample number on each call)
LABELS = ['forward', 'backward', 'left', 'right', 'stop']
THRESHOLD = 10  # Próg głośności (RMS), ~-40DB
SWIDTH = 2
SHORT_NORMALIZE = (1.0/32768.0)
DELAY_AFTER_THRESHOLD = 1
FORMAT = pyaudio.paInt16
CHANNELS = 2
AUTOTUNE = tf.data.AUTOTUNE


class Voice_movement:
    def __init__(self) -> None:
        # Load model
        self.model = tf.keras.models.load_model('AI.h5')
        self.audio = pyaudio.PyAudio()
        self.stream =  self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK)
        

    def listen_and_predict_command(self):
        print("Czekam na przekroczenie progu...")
        recording = False
        frames = []
        silence_start_time = None

        while True:
            input = self.stream.read(CHUNK, exception_on_overflow=False)
            rms_value = self._rms(input)

            if rms_value > THRESHOLD:
                if not recording:
                    print("Nagrywanie rozpoczęte...")
                    recording = True
                frames.append(input)
                silence_start_time = None
            elif recording:
                if silence_start_time is None:
                    silence_start_time = time()  # Rozpocznij mierzenie czasu ciszy
                elif (time() - silence_start_time) >= DELAY_AFTER_THRESHOLD :
                    print("Nagrywanie zakończone.")
                    break
                frames.append(input)

        
        if frames:    
            audio_data = b''.join(frames)
            self._save_audio(audio_data)
            arr = ['recording.wav']
            file = tf.data.Dataset.from_tensor_slices(arr)
        
            output = file.map(map_func=self._get_waveform, num_parallel_calls=AUTOTUNE)
            output = output.map(map_func=self._get_spectrogram, num_parallel_calls=AUTOTUNE)
            
            audio = []
            for x in output:
                audio.append(x.numpy())
            audio = np.array(audio)
            print(audio)
            prediction = np.argmax(self.model.predict(audio), axis=1)
            print(f"Przewidywana komenda: {LABELS[prediction[0]]}")
        os.remove('recording.wav')
    
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
        input_len = 64000
        waveform = waveform[:input_len]
        zero_padding = tf.zeros([64000] - tf.shape(waveform), dtype=tf.float32)
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
    
while True:
    Voice_movement().listen_and_predict_command()