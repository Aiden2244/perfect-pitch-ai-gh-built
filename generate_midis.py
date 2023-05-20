import os
import shutil
import pretty_midi
from tqdm import tqdm

BASE_NOTE_NUMBER = 60  # This is MIDI number for Middle C.
VELOCITY = 100
DURATION = 4.0
START_TIME = 0.0

NOTES = ["c", "db", "d", "eb", "e", "f", "gb", "g", "ab", "a", "bb", "b"]
OCTAVES = ["2", "3", "4", "5"]

count = 0
instrument_number = 0

file_prefix = "./data/midifiles/"

# Delete and recreate main directory
if os.path.exists(file_prefix):
    shutil.rmtree(file_prefix)
os.mkdir(file_prefix)

log_file_path = "./logs/midis_log.txt"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
log_file = open(log_file_path, 'w')

total_instruments = len(NOTES) * len(OCTAVES) * 128  # Total iterations
progress_bar = tqdm(total=total_instruments)

print("Generating MIDI files...")

for octave_string in OCTAVES:
    octave_offset = int(octave_string) - 4

    for note_string in NOTES:
        note_offset = NOTES.index(note_string)
        note_number = BASE_NOTE_NUMBER + (octave_offset * 12) + note_offset

        full_path = file_prefix + note_string + octave_string + "/"

        # Create subdirectory
        os.mkdir(full_path)

        log_file.write("file prefix: " + file_prefix + "\n")

        for instrument_number in range(128):  # There are 128 General MIDI instruments.
            log_file.write("Generating midi for instrument " + str(instrument_number) + "\n")

            midi_file = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=instrument_number, is_drum=False)

            note = pretty_midi.Note(velocity=VELOCITY, pitch=note_number, start=START_TIME, end=START_TIME + DURATION)
            instrument.notes.append(note)

            midi_file.instruments.append(instrument)

            filename = full_path + note_string + "-" + octave_string + "_" + str(instrument_number) + ".mid"
            log_file.write("Writing midi to file " + filename + "\n")
            midi_file.write(filename)
            log_file.write("\n")
            count += 1
            progress_bar.update(1)  # Update progress bar
            
log_file.close()
progress_bar.close()

print("Generated " + str(count) + " MIDI files.")
