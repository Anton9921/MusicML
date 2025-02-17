# necessary libraries for MIDI processing and machine learning
import pretty_midi
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# STEP 1: Processing the MIDI files into sequences for training
def process_midi(folder_path, seq_length=100, fs=100, max_files=100):
    """
    seq_length (int): Length of each sequence for the LSTM model
    fs (int): Sampling frequency for the piano roll
    max_files (int): The Maximum number of MIDI files to process to prevent memory overload
    """
    sequences = []  # used to store all piano roll sequences
    file_count = 0  # used to count the processed files

    # searching through the folder to find the MIDI files
    print(f"Listing all files in the directory: {folder_path}")
    for root, _, files in os.walk(folder_path):
        print(f"Found directory: {root}")
        for file in files:
            print(f"Found file: {file}")  # display to confirm the file being checked
            if file.endswith('.midi') or file.endswith('.mid'):
                try:
                    midi_path = os.path.join(root, file)  # full path to the file
                    midi_data = pretty_midi.PrettyMIDI(midi_path)  # loading the MIDI file
                    # converting MIDI to a piano roll with time-major format
                    'time-major format: data is organzied by time first for LSTM models'
                    piano_roll = midi_data.get_piano_roll(fs=fs).T
                    # splitting piano roll into sequences of length seq_length
                    for i in range(0, len(piano_roll) - seq_length, seq_length):
                        sequences.append(piano_roll[i:i + seq_length])
                    file_count += 1
                    # error handling to stop processing once max_files limit is reached
                    if file_count >= max_files:
                        break
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    if len(sequences) == 0:
        print("No valid sequences were processed.")
    # converting the list of piano roll sequences to a numpy array for training
    return np.array(sequences)

# path to the folder containing MIDI files
folder_path = '/Users/antonpervukhin/Documents/Music_ML/maestro-v3.0.0'
# processing the MIDI files to extract sequences
sequences = process_midi(folder_path)
print(f"Loaded sequences: {sequences.shape}")

# error handling if the sequences are empty or malformed
if sequences.shape[0] == 0:
    print("No sequences loaded. Please check the folder path and the MIDI files.")
else:
    # preparing training data: X as inputs and y as targets
    X = sequences[:, :-1, :]  # Input sequences (all but the last timestep)
    y = sequences[:, 1:, :]   # Target sequences (all but the first timestep)

    print(f"Input shape: {X.shape}, Target shape: {y.shape}")

    # STEP 2: Build the LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),  # dropout to reduce overfitting
        LSTM(128, return_sequences=True),
        Dense(128, activation='softmax') # dense layer for predicting the next note
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    model.summary()
    model.fit(X, y, epochs=1, batch_size=128) # training the model with prepared data

# STEP 3: Generate new music using the trained model
def sample_with_temperature(predictions, temperature=1.0):
    """
    sampling note index from a probability distribution with temperature scaling
    (prevents any errors with generated notes)
    predictions (np.array): Array of probabilities for each note.
    temperature (float): Controls randomness (higher = more random).
    """
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature  # applying temperature to predictions
    exp_preds = np.exp(predictions)
    preds = exp_preds / np.sum(exp_preds)  # normalizing  probabilities
    return np.random.choice(range(len(preds)), p=preds)

def generate_music(model, sequence_length=100, num_notes_to_generate=100, fs=100, min_note=36, max_note=84, note_duration=0.25, temperature=1.0):
    """
    sequence_length (int): Length of input sequences.
    num_notes_to_generate (int): Number of notes to generate.
    fs (int): Sampling frequency for piano roll.
    min_note (int): Minimum MIDI note.
    max_note (int): Maximum MIDI note.
    note_duration (float): Duration of each note in seconds.
    temperature (float): Controls randomness of note selection.
    """
    # starting with a random sequence as seed
    start_idx = np.random.randint(0, X.shape[0])
    input_sequence = X[start_idx]  # the seed sequence
    generated_sequence = []  # storing the generated notes

    # generating notes iteratively
    for i in range(num_notes_to_generate):
        prediction = model.predict(input_sequence[np.newaxis, :, :])  # predicting the next note
        predicted_note = sample_with_temperature(prediction[0, -1], temperature=temperature)
        predicted_note = np.clip(predicted_note, min_note, max_note)  # keeping the note within range

        # creating a new piano roll frame for the predicted note
        next_piano_roll = np.zeros((128, 1))  # 128 MIDI notes, 1 time step
        next_piano_roll[predicted_note, 0] = 1

        generated_sequence.append(next_piano_roll[:, 0])  # appending the new note
        # updating the input sequence for the next prediction
        input_sequence = np.roll(input_sequence, shift=-1, axis=0)
        input_sequence[-1, :] = next_piano_roll[:, 0]

    # finally, we convert the list to a 2D numpy array
    generated_sequence = np.array(generated_sequence)
    return np.reshape(generated_sequence, (generated_sequence.shape[0], 128))

def piano_roll_to_midi(piano_roll, output_path='output.mid', fs=100, note_duration=2):
    """
    Convert a piano roll into a MIDI file.
    piano_roll (np.array): Piano roll to convert.
    output_path (str): File path to save the MIDI file.
    fs (int): Sampling frequency.
    note_duration (float): Duration of each note in seconds.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
      # 0 = regular piano, 5 = electric piano, which ever one works well
    midi.instruments.append(instrument)

    current_time = 0  # Start time of the notes

    # Convert piano roll rows to MIDI notes
    for note_idx in range(piano_roll.shape[0]):
        start_time = current_time
        end_time = start_time + note_duration  # Note duration
        played_notes = np.where(piano_roll[note_idx] > 0)[0]  # Get active notes
        if len(played_notes) > 0:
            note = played_notes[0]
            velocity = 100  # Note velocity
            instrument.notes.append(pretty_midi.Note(velocity, note, start_time, end_time))
        current_time = end_time  # Update time

    midi.write(output_path)  # Save MIDI file
    print(f"MIDI file saved at {output_path}")

# Example of generating and saving music
generated_music = generate_music(model, num_notes_to_generate=200, note_duration=0.25, temperature=1)
piano_roll_to_midi(generated_music, 'AI_music.mid', note_duration=0.25)