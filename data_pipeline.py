import pandas as pd
import os
import librosa
import numpy as np
import pretty_midi
import h5py
from tqdm import tqdm
import tensorflow as tf

MAESTRO_PATH = '/home/amine/maestro-dataset/maestro-v3.0.0/'
N_MELS=128
N_FFT = 2048
HOP_LENGTH=512
SR = 16000
CHUNK_SECONDS = 5
CHUNK_SIZE = 160

def get_pairs(df):
    return [(os.path.join(MAESTRO_PATH, row['midi_filename']), os.path.join(MAESTRO_PATH, row['audio_filename'])) for _, row in df.iterrows()]
def process_file_pair(midi_path, audio_path):
    # 1. Spectogram generation
    y, sr = librosa.load(audio_path, sr=SR)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. Piano roll
    midi = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi.get_piano_roll(fs=SR / HOP_LENGTH)
    piano_roll = piano_roll[21:109]
    piano_roll = (piano_roll > 0).astype(np.float32)

    # 3. frames for onset detection
    onset_roll = np.zeros_like(piano_roll)
    for note in midi.instruments[0].notes:
        frame = int(round(note.start * SR / HOP_LENGTH))
        pitch_idx = note.pitch - 21
        if 0 <= pitch_idx < 88 and frame < onset_roll.shape[1]:
            onset_roll[pitch_idx, frame] = 1


    # 4. Trimming
    min_frames = min(mel_spec_db.shape[1], piano_roll.shape[1])
    mel_spec_db = mel_spec_db[:, :min_frames]
    piano_roll = piano_roll[:, :min_frames]
    onset_roll = onset_roll[:, :min_frames]

    # 5. Chunking
    n_chunks = min_frames // CHUNK_SIZE

    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        spec_chunk = mel_spec_db[:, start:end]  # shape (128, 156)
        roll_chunk = piano_roll[:, start:end]  # shape (88, 156)
        onset_chunk = onset_roll[:, start:end]
        chunks.append((spec_chunk, roll_chunk, onset_chunk))

    return chunks

def process_split(sets, split_name, hdf5_file):
    # Saving extracted features in h5 file
    with h5py.File(hdf5_file, 'a') as f:
        if f"{split_name}/spectrograms" not in f:
            f.create_dataset(
                f"{split_name}/spectrograms",
                shape=(0, N_MELS, CHUNK_SIZE),
                maxshape=(None, N_MELS, CHUNK_SIZE),
                dtype=np.float32,
            )
            f.create_dataset(
                f"{split_name}/piano_rolls",
                shape=(0, 88, CHUNK_SIZE),
                maxshape=(None, 88, CHUNK_SIZE),
                dtype=np.float32,
            )
            f.create_dataset(
                f"{split_name}/onset_rolls",
                shape=(0, 88, CHUNK_SIZE),
                maxshape=(None, 88, CHUNK_SIZE),
                dtype=np.float32,
            )

        for (midi_path, audio_path) in tqdm(sets):
            try:
                chunks = process_file_pair(midi_path, audio_path)
                spec_chunks = np.array([chunk[0] for chunk in chunks])
                roll_chunks = np.array([chunk[1] for chunk in chunks])
                onset_chunks = np.array([chunk[2] for chunk in chunks])

                current_size = f[f"{split_name}/spectrograms"].shape[0]

                f[f'{split_name}/spectrograms'].resize(current_size + len(chunks), axis=0)
                f[f'{split_name}/spectrograms'][current_size:] = spec_chunks
                f[f'{split_name}/piano_rolls'].resize(current_size + len(chunks), axis=0)
                f[f'{split_name}/piano_rolls'][current_size:] = roll_chunks
                f[f'{split_name}/onset_rolls'].resize(current_size + len(chunks), axis=0)
                f[f'{split_name}/onset_rolls'][current_size:] = onset_chunks
            except Exception as e:
                print(f"Error processing file pair: {e}, Skipping...")
                continue

class MAESTROGenerator(tf.keras.utils.Sequence):
    def __init__(self, hdf5_path, split, batch_size):
        self.hdf5_path = hdf5_path
        self.split = split
        self.batch_size = batch_size

        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_chunks = f[f"{self.split}/spectrograms"].shape[0]

        # Shuffling indices to avoid model bias
        self.indices = np.arange(self.total_chunks)
        np.random.shuffle(self.indices)

        print(f"{split} generator: {self.total_chunks} chunks, {len(self)} batches per epoch")

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return self.total_chunks // self.batch_size

    def __getitem__(self, idx):
        # Get the shuffled indices for this batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Sort them — HDF5 requires indices in ascending order for slicing
        batch_indices = np.sort(batch_indices)

        with h5py.File(self.hdf5_path, 'r') as f:
            X = f[f"{self.split}/spectrograms"][batch_indices] # shape (batch_size, 128, 156)
            y_frame = f[f"{self.split}/piano_rolls"][batch_indices] # shape (batch_size, 88, 156)
            y_onset = f[f"{self.split}/onset_rolls"][batch_indices] # shape (batch_size, 88, 156) (i think> (this is amine from the future yes it is))

        # Add new channel to spectograms for CNN support
        X = X[..., np.newaxis]

        # Transpose to match model output shape
        y_frame = y_frame.transpose(0, 2, 1)
        y_onset = y_onset.transpose(0, 2, 1)

        return X.astype(np.float32), {
            'frame_output': y_frame.astype(np.float32),
            'onset_output': y_onset.astype(np.float32),
        }



def main():
    maestro_df = pd.read_csv(os.path.join(MAESTRO_PATH, 'maestro-v3.0.0.csv'))
    train_df = maestro_df[maestro_df['split'] == 'train']
    validation_df = maestro_df[maestro_df['split'] == 'validation']
    test_df = maestro_df[maestro_df['split'] == 'test']

    train_pairs = get_pairs(train_df)
    validation_pairs = get_pairs(validation_df)
    test_pairs = get_pairs(test_df)

    chunks = process_file_pair(*train_pairs[0])
    spec_chunk, roll_chunk, onset_chunk = chunks[0]

    process_split(train_pairs, 'train', 'maestro_chunks.h5')
    process_split(validation_pairs, 'validation', 'maestro_chunks.h5')
    process_split(test_pairs, 'test', 'maestro_chunks.h5')

if __name__ == "__main__":
    main()

