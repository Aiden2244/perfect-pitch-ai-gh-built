import os
import shutil
from midi2audio import FluidSynth

# Specify paths
MIDI_DIR = './data/midifiles'
AUDIO_DIR = './data/audio'
SOUND_FONT = './soundfont/soundfont.sf2'

# Delete and recreate main directory
if os.path.exists(AUDIO_DIR):
    shutil.rmtree(AUDIO_DIR)
os.mkdir(AUDIO_DIR)

# Instantiate FluidSynth
fs = FluidSynth(SOUND_FONT)

# Prepare label encoder for notes
notes = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]

# Process MIDI files
for dirpath, dirnames, filenames in os.walk(MIDI_DIR):
    for filename in filenames:
        if filename.endswith('.mid'):
            # Convert MIDI to audio
            midi_path = os.path.join(dirpath, filename)
            audio_path = os.path.join(AUDIO_DIR, filename.replace('.mid', '.wav'))
            fs.midi_to_audio(midi_path, audio_path)
