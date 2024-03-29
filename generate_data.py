import tkinter as tk
import threading
import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display
import os
from test_model import *

class PerfectPitchAIApp:
    def __init__(self, root):
        self.root = root
        root.title("Perfect Pitch.AI")
        root.geometry("800x600") 

        self.model = load_model(MODEL_FILENAME)

        # Welcome Label
        self.welcome_label = tk.Label(root, text="Welcome to Perfect Pitch.AI")
        self.welcome_label.pack(pady=10)

        # Record Button
        self.record_button = tk.Button(root, text="Record", command=self.start_recording)
        self.record_button.pack(pady=10)

        # Recording Status Label
        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        # Predicted Pitch Label
        self.pitch_label = tk.Label(root, text="")
        self.pitch_label.pack(pady=10)

        # Chromagram Plot
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Playback Button (simplified without slider)
        self.playback_button = tk.Button(root, text="Play Audio", command=self.toggle_playback, state='disabled')
        self.playback_button.pack(pady=5)
        self.playback_thread = None
        self.playback_wave_obj = None
        self.is_playing = False

    def start_recording(self):
        self.status_label.config(text="Recording...")
        self.record_button.config(state='disabled')
        threading.Thread(target=self.record_and_predict).start()

    def record_and_predict(self):
        recording = record_audio(DURATION, SAMPLE_RATE)
        write(RECORDING_FILENAME, SAMPLE_RATE, recording)
        chroma = create_chromagram(RECORDING_FILENAME, sr=SAMPLE_RATE)
        pitch = predict_pitch(self.model, chroma)
        pitch_text = f"Predicted pitch: {PITCHES[pitch[0]]}"
        self.pitch_label.config(text=pitch_text)

        self.ax.clear()
        librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm', ax=self.ax)
        self.ax.set_title('Chromagram')
        self.ax.set_yticks(np.arange(12))
        # Change labels to flat notes
        self.ax.set_yticklabels(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])
        self.canvas.draw()

        self.status_label.config(text="")
        self.record_button.config(state='normal')
        self.playback_button.config(state='normal')

    def toggle_playback(self):
            self.playback_thread = threading.Thread(target=self.playback_recording)
            self.playback_thread.start()

    def playback_recording(self):
        try:
            self.playback_wave_obj = sa.WaveObject.from_wave_file(RECORDING_FILENAME)
            play_obj = self.playback_wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            print(f"Error during playback: {e}")

    def on_closing(self):
        if os.path.exists(RECORDING_FILENAME):
            os.remove(RECORDING_FILENAME)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PerfectPitchAIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

