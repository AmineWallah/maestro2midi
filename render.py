import librosa
import numpy as np
from data_pipeline import N_MELS, N_FFT, HOP_LENGTH, SR, CHUNK_SECONDS, CHUNK_SIZE
import tensorflow as tf
import pretty_midi

REFRACTORY = 2
MIN_NOTE_FRAMES = 2

def generate_chunks(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Split into chunks
    frames = mel_spec_db.shape[1]
    n_chunks = frames // CHUNK_SIZE
    remainder = frames % CHUNK_SIZE
    chunks = []

    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = mel_spec_db[:, start:end]
        chunks.append(chunk)

    # Appending last chunk after padding it
    if (frames % CHUNK_SIZE) != 0:
        last_chunk = mel_spec_db[:, n_chunks * CHUNK_SIZE:]
        padded_last_chunk = np.pad(last_chunk, ((0, 0), (0, CHUNK_SIZE - remainder)), mode='constant')
        chunks.append(padded_last_chunk)


    last_chunk_real_length = remainder if remainder != 0 else CHUNK_SIZE

    return chunks, last_chunk_real_length

def predict_chunks(chunks, last_chunk_real_length):
    model = tf.keras.models.load_model('best_model.keras')

    batch = np.stack(chunks)
    batch = batch[..., np.newaxis]
    frame_pred, onset_pred = model.predict(batch)

    # Stitch frames
    full = frame_pred[:-1].reshape(-1,88)
    last = frame_pred[-1][:min(last_chunk_real_length, CHUNK_SIZE), :]
    frame_roll = np.concatenate([full, last], axis=0)

    # Stitch onsets
    full = onset_pred[:-1].reshape(-1, 88)
    last = onset_pred[-1][:min(last_chunk_real_length, CHUNK_SIZE), :]
    onset_roll = np.concatenate([full, last], axis=0)

    return frame_roll, onset_roll

def roll_to_notes(frame_roll, onset_roll, hop_length=HOP_LENGTH, sr=SR, velocity=100, refractory=REFRACTORY):
    # binary_roll shape: (frames, 88), values 0 or 1
    notes = []
    n_frames = frame_roll.shape[0] # no need for n_onsets, it's gonna be identical
    for pitch_idx in range(88):
        frame_column = frame_roll[:, pitch_idx]
        onset_column = onset_roll[:, pitch_idx]
        in_run = False
        start_frame = 0
        last_onset_frame = -999
        for frame in range(n_frames):
            onset_now = onset_column[frame] == 1
            is_new_onset = onset_now and (frame - last_onset_frame) > refractory

            if is_new_onset and in_run and (frame - start_frame) > MIN_NOTE_FRAMES:
                notes.append(_make_note(pitch_idx, start_frame, frame, hop_length, sr, velocity))
                start_frame = frame
                last_onset_frame = frame
            elif is_new_onset and in_run:
                last_onset_frame = frame
            elif is_new_onset and not in_run:
                in_run = True
                start_frame = frame
                last_onset_frame = frame
            elif frame_column[frame] == 1 and not in_run:
                in_run = True
                start_frame = frame
            elif frame_column[frame] == 0 and in_run:
                in_run = False
                notes.append(_make_note(pitch_idx, start_frame, frame, hop_length, sr, velocity))
        if in_run:
            notes.append(_make_note(pitch_idx, start_frame, n_frames, hop_length, sr, velocity))

    return notes

def _make_note(pitch_idx, start_frame, end_frame, hop_length, sr, velocity):
    return pretty_midi.Note(
        velocity=velocity,
        pitch=pitch_idx + 21,
        start=start_frame * hop_length / sr,
        end=end_frame * hop_length / sr,
    )

def notes_to_midi(notes, out_path):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # acoustic grand
    piano.notes.extend(notes)
    pm.instruments.append(piano)
    pm.write(out_path)

def main():
    chunks, last_len = generate_chunks('queen.wav')
    frame_roll, onset_roll = predict_chunks(chunks, last_len)
    frame_binary = (frame_roll > 0.8).astype(int)
    onset_binary = (onset_roll > 0.5).astype(int)
    notes = roll_to_notes(frame_binary, onset_roll=onset_binary)
    notes_to_midi(notes, 'output/output.mid')


if __name__ == "__main__":
    main()
