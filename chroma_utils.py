import numpy as np
import librosa
import matplotlib.pyplot as plt

MAX_FRAMES = 600

def create_chromagram(file_path, sr=None, hop_length=512, n_fft=2048):
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0]<2:
        wav = np.pad(wav, int(np.ceil((2-wav.shape[0])/2)), mode='reflect')
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr, hop_length=hop_length, n_fft=n_fft)
    chroma = np.pad(chroma, ((0, 0), (0, MAX_FRAMES - chroma.shape[1])), mode='constant')
    return chroma

def plot_chromagram(chroma):
    plt.figure(figsize=(10, 4))
    plt.imshow(chroma, aspect='auto', origin='lower', cmap='jet')
    plt.title('Chromagram')
    plt.colorbar(format='%+2.0f dB')
    
    # Change y-axis labels
    plt.yticks(np.arange(12), ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])

    plt.tight_layout()
    plt.show()

# need to commits