from data_pipeline import MAESTROGenerator
import tensorflow as tf
import numpy as np
import mir_eval
from collections import Counter

from render import roll_to_notes


def main():
    evaluations = []
    model = tf.keras.models.load_model('best_model.keras')
    test_gen = MAESTROGenerator('maestro_chunks.h5', 'test', batch_size=32)

    for i in range(len(test_gen)):
        X, labels = test_gen[i]
        frame_labels = labels['frame_output']
        onset_labels = labels['onset_output']

        frame_pred, onset_pred = model.predict(X, verbose=0)

        for b in range(X.shape[0]):
            pred_notes = roll_to_notes(
                (frame_pred[b] > 0.8).astype(int),
                (onset_pred[b] > 0.5).astype(int),
            )
            true_notes = roll_to_notes(
                frame_labels[b].astype(int),
                onset_labels[b].astype(int),
            )

            if len(pred_notes) == 0 or len(true_notes) == 0:
                continue

            pred_intervals = np.array([[n.start, n.end] for n in pred_notes])
            pred_pitches = np.array([mir_eval.util.midi_to_hz(n.pitch) for n in pred_notes])
            true_intervals = np.array([[n.start, n.end] for n in true_notes])
            true_pitches = np.array([mir_eval.util.midi_to_hz(n.pitch) for n in true_notes])

            # onset-only
            p_on, r_on, f_on, o_on = mir_eval.transcription.precision_recall_f1_overlap(
                true_intervals, true_pitches, pred_intervals, pred_pitches,
                offset_ratio=None,
            )
            # with offset
            p_off, r_off, f_off, o_off = mir_eval.transcription.precision_recall_f1_overlap(
                true_intervals, true_pitches, pred_intervals, pred_pitches,
                offset_ratio=0.2,
            )

            evaluations.append({
                'precision_onset': p_on, 'recall_onset': r_on,
                'f_measure_onset': f_on, 'overlap_onset': o_on,
                'precision_offset': p_off, 'recall_offset': r_off,
                'f_measure_offset': f_off, 'overlap_offset': o_off,
            })

    evaluations_total = Counter()
    for e in evaluations:
        evaluations_total.update(e)

    n = len(evaluations)
    print(f"Scored on {n} chunks\n")
    print("Note F1 (onset-only):")
    print(f"  Precision: {evaluations_total['precision_onset'] / n:.4f}")
    print(f"  Recall:    {evaluations_total['recall_onset'] / n:.4f}")
    print(f"  F1:        {evaluations_total['f_measure_onset'] / n:.4f}\n")
    print("Note F1 (with offset):")
    print(f"  Precision: {evaluations_total['precision_offset'] / n:.4f}")
    print(f"  Recall:    {evaluations_total['recall_offset'] / n:.4f}")
    print(f"  F1:        {evaluations_total['f_measure_offset'] / n:.4f}")



if __name__ == "__main__":
    main()