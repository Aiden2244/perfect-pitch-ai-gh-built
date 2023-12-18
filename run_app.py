import tkinter as tk
import threading
import simpleaudio as sa  # Import simpleaudio for playback
from test_model import *

class PerfectPitchAIApp:
    def __init__(self, root):
        self.root = root
        root.title("Perfect Pitch.AI")
        root.geometry("600x400") 

        self.model = load_model(MODEL_FILENAME)

        # Welcome Label
        self.welcome_label = tk.Label(root, text="Welcome to Perfect Pitch.AI")
        self.welcome_label.pack(pady=10)

        # Record Button
        self.record_button = tk.Button(root, text="Record", command=self.start_recording)
        self.record_button.pack(pady=10)

        # Playback Button
        self.playback_button = tk.Button(root, text="Play Recording", command=self.playback_recording, state='disabled')
        self.playback_button.pack(pady=10)

        # Recording Status Label
        self.status_label = tk.Label(root, text="")
        self.status_label.pack()

        # Predicted Pitch Label
        self.pitch_label = tk.Label(root, text="")
        self.pitch_label.pack(pady=10)

        # Chromagram Checkbox
        self.chromagram_var = tk.BooleanVar()
        self.chromagram_check = tk.Checkbutton(root, text="Display Chromagram", variable=self.chromagram_var)
        self.chromagram_check.pack()

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
        if self.chromagram_var.get():
            plot_chromagram(chroma)
        self.status_label.config(text="")
        self.record_button.config(state='normal')
        self.playback_button.config(state='normal')  # Enable playback button

    def playback_recording(self):
        try:
            wave_obj = sa.WaveObject.from_wave_file(RECORDING_FILENAME)
            wave_obj.play().wait_done()
        except Exception as e:
            print(f"Error during playback: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PerfectPitchAIApp(root)
    root.mainloop()
