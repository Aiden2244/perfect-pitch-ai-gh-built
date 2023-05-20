import sounddevice as sd
import keyboard
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
import os
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import argparse
from chroma_utils import *
from model_utils import *

# Pitches mapping
PITCHES = {
    0: "A",
    1: "Ab",
    2: "B",
    3: "Bb",
    4: "C",
    5: "D",
    6: "Db",
    7: "E",
    8: "Eb",
    9: "F",
    10: "G",
    11: "Gb",
}

SAMPLE_RATE = 22050
DURATION = 5  # duration of the recording in seconds
HOP_LENGTH = 512  # hop length for the spectrogram
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
MAX_FRAMES = 512
RECORDING_FILENAME = 'recording.wav'
MODEL_FILENAME = './models/my_model_weights.keras'

def record_audio(duration, sample_rate):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # wait until recording is finished
    print("Recording complete.")
    return recording


def predict_pitch(model, chroma):
    # Add an extra dimension to match model's input shape
    chroma = chroma[np.newaxis, ..., np.newaxis]
    prediction = model.predict(chroma)
    return np.argmax(prediction, axis=1)


def load_model(model_weights_filename):
    print(f"Loading model weights from {model_weights_filename}...")

    # Rebuild the model
    model = create_model()

    # Load the model weights
    model.load_weights(model_weights_filename)

    # Compile the model
    model.compile(loss=SparseCategoricalCrossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    print("Model loaded.")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chromagram", action="store_true",
                        help="display chromagram of the recording")
    args = parser.parse_args()
    
    model = load_model(MODEL_FILENAME)
    print("Welcome to PerfectPitch.ai!")
    print("Press the spacebar to record for 5 seconds or the escape key to quit...")
    while True:
        if keyboard.is_pressed('space'):
            recording = record_audio(DURATION, SAMPLE_RATE)
            write(RECORDING_FILENAME, SAMPLE_RATE, recording)
            chroma = create_chromagram(RECORDING_FILENAME, sr=SAMPLE_RATE)
            pitch = predict_pitch(model, chroma)
            print(f"Predicted pitch: {PITCHES[pitch[0]]}")
            if args.chromagram:
                plot_chromagram(chroma)
            os.remove(RECORDING_FILENAME)
            print()
            print("Press the spacebar to record again or the escape key to quit...")
        elif keyboard.is_pressed('esc'):  # Checks for the escape key press
            print()
            print("Exiting the program. Goodbye!")
            break

if __name__ == "__main__":
    main()
