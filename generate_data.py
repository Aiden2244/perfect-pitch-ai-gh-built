import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
from tqdm import tqdm
import warnings

AUDIO_DIR = './data/audio'
SAMPLE_RATE = 22050
DURATION = 5  # duration of your audio files in seconds
HOP_LENGTH = 512  # hop length for the spectrogram
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
MAX_FRAMES = 600

notes = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]
le = LabelEncoder()
le.fit(notes)

def get_chromagram(file_path, sr=None, hop_length=512, n_fft=2048):
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0]<2:
        wav = np.pad(wav, int(np.ceil((2-wav.shape[0])/2)), mode='reflect')
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr, hop_length=hop_length, n_fft=n_fft)
    chroma = np.pad(chroma, ((0, 0), (0, MAX_FRAMES - chroma.shape[1])), mode='constant')
    return chroma

X = []
y = []

print("Processing audio files...")

total_files = sum([len(files) for r, d, files in os.walk(AUDIO_DIR)])
pbar = tqdm(total=total_files)

logfile = './logs/data_log.txt'
log = open(logfile, 'w')

for root, dirs, files in os.walk(AUDIO_DIR):
    for name in files:
        if name.endswith('.wav'):
            filepath = os.path.join(root, name)
            # inside your loop
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")  # Catch all warnings
                    chroma = get_chromagram(filepath, sr=SAMPLE_RATE)
                    X.append(chroma)
                    y.append(le.transform([name.split('-')[0]]))
                    for warning in w:
                        log.write(f'Warning when processing file {filepath}: {str(warning.message)}\n')
            except Exception as e:
                print(f"Could not process file {filepath}: {str(e)}")
            pbar.update()

log.close()
            
pbar.close()

X = np.array(X)
y = np.array(y)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

X = X / np.amax(np.abs(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Save to disk
np.savez('./data/data_arrays', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Numpy arrays saved successfully!")

