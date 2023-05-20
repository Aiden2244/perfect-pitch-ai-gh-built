import sys
from sklearn.preprocessing import MinMaxScaler
from chroma_utils import *

def main(argv):
    if len(argv) < 2:
        print("Please provide the path to a .wav file.")
        return
    
    wav_file_path = argv[1]

    # Generate chromagram
    chroma = create_chromagram(wav_file_path)

    # Normalize chromagram
    scaler = MinMaxScaler()
    chroma = scaler.fit_transform(chroma.T).T

    # Plotting the chromagram
    plot_chromagram(chroma)

if __name__ == "__main__":
    main(sys.argv)
