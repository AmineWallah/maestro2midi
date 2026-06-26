# !! Still a WIP !!

# Maestro2MIDI

Polyphonic piano transcription: converts piano audio (`.wav`) into MIDI using a
U-Net with dual onset and frame detection heads, trained on MAESTRO v3.0.0, spiritual successor of [NSynth2MIDI](https://github.com/AmineWallah/nsynth2midi)

Built using mainly TensorFlow 2.0 and Keras. (full list of dependencies in
`pyproject.toml`)

## Architecture

- **Input:** mel spectrogram (128 mel bins, 16 kHz audio, 512 hop length), chunked
  into 160-frame (~5 second) windows
- **Backbone:** U-Net (3 encoder/decoder blocks, 2.6M+ parameters)
- **Two output heads** branching from shared decoder features:
  - **Frame head** — predicts whether each of 88 piano keys is active per frame
  - **Onset head** — predicts where each note *begins*, enabling re-articulation
    detection that frame-only models miss
- **Output:** per-frame `(160, 88)` predictions → thresholded → note events → MIDI

## Pipeline

1. Load audio at 16 kHz, compute mel spectrogram
2. Split into 160-frame chunks (final chunk is zero padded to compensate for it being shorter)
3. Predict frame + onset rolls per chunk, stitch back into full piano rolls
4. Threshold both rolls
5. Convert to note events: a note starts at each onset spike (or frame-run start),
   and sustains until the key releases or is re-struck
6. Export to MIDI via `pretty_midi`

## Usage

```bash
# Install dependencies (uses uv)
uv sync

# Generate the dataset (requires MAESTRO v3.0.0 downloaded locally)
python data_pipeline.py

# Train
python train.py

# Transcribe an audio file
python render.py   # edit the input path in main()
```

## To-Do list
- Remove all the comments that r fucking stupid and replace them with appropriate ones
- Stop using bum ass swear words in the README.md AND code comments
- Publish evaluation metrics results
- More detailed overview of the architecture
- Limitations of the model/challenges faced
- CLI version
- Potential GUI frontend fork(?)

## Acknowledgments

This project uses the **MAESTRO v3.0.0 dataset** from Google Magenta. If you use
MAESTRO in your own work, please cite the paper that introduced it:

> Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
> Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling Factorized
> Piano Music Modeling and Generation with the MAESTRO Dataset." In *International
> Conference on Learning Representations (ICLR)*, 2019.

The dual-head architecture is inspired by **Onsets and Frames**:

> Curtis Hawthorne, Erich Elsen, Jialin Song, Adam Roberts, Ian Simon, Colin
> Raffel, Jesse Engel, Sageev Oore, and Douglas Eck. "Onsets and Frames:
> Dual-Objective Piano Transcription." In *ISMIR*, 2018.
> [arXiv:1710.11153](https://arxiv.org/abs/1710.11153)

## License

MIT — see [LICENSE](LICENSE).
