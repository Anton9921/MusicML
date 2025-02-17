# MusicML: AI-Generated Music with LSTMs

## Overview
MusicML is a machine learning project designed to generate piano music using an LSTM-based neural network. The model is trained on the **maestro-v3.0.0** dataset and processes MIDI files to generate new compositions.

## Repository Contents
- `MusicML.py`: The main script that processes MIDI data, trains the LSTM model, and generates music.
- `maestro-v3.0.0.csv`: Metadata file containing information about the training dataset.
- `AI_music.mid`: Example output of generated music.
- `README.md`: Instructions for setting up and running the project.

## Setup Instructions
To replicate this project exactly, follow these steps:

### Step 1: Install Python 3.9.13
Ensure you have **Python 3.9.13** installed. You can download it from:
[https://www.python.org/downloads/release/python-3913/](https://www.python.org/downloads/release/python-3913/)

### Step 2: Set Up a Virtual Environment
Create and activate a virtual environment called `tensorflow_env`:
```bash
python3 -m venv tensorflow_env
source tensorflow_env/bin/activate  # On macOS/Linux
tensorflow_env\Scripts\activate    # On Windows
```

### Step 3: Install Dependencies
Install the required Python libraries:
```bash
pip install pretty_midi numpy tensorflow
```

### Step 4: Download and Extract the Maestro Dataset
Download the **maestro-v3.0.0** dataset from [Google's Magenta Project](https://magenta.tensorflow.org/datasets/maestro) and place it in the same directory as `MusicML.py`.

### Step 5: Run the MusicML Script
Execute the script to train the model and generate music:
```bash
python MusicML.py
```

### Step 6: Generated Output
- After training, the script will create a new MIDI file called `AI_music.mid`.
- You can listen to the output using any MIDI player (e.g., GarageBand, LMMS, or MuseScore).

## Notes
- The script may take time to process large MIDI files, depending on your hardware.
- Modify `MusicML.py` to tweak hyperparameters such as sequence length, number of training epochs, and temperature for more diverse results.

## Troubleshooting
If you encounter errors:
1. Ensure your virtual environment is activated.
2. Verify dependencies are installed using `pip list`.
3. Check that the dataset is correctly downloaded and placed in the right directory.

## License
This project is for educational purposes and experiments in AI-generated music. Modify and expand it as needed!

