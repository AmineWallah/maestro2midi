import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help='Path to the audio file to render')
    parser.add_argument('-o', '--output_path', type=str, default='./', help='Path to the output directory (default: current directory)')
    parser.add_argument('-ft', '--frame_threshold', type=float, default=0.8, help='Threshold for frame prediction (default: 0.8)')
    parser.add_argument('-ot', '--onset_threshold', type=float, default=0.5, help='Threshold for onset prediction (default: 0.5)')
    parser.add_argument('-mp', '--model_path', type=str, default='best_model.keras', help='Path to the model checkpoint (default: best_model.keras)')
    parser.add_argument('-mnf', '--min_note_frames', type=int, default=2, help='EXPERIMENTAL: Adjust this to reduce hallucinated double notes firing (default: 2)')
    parser.add_argument('-r', '--refractory', type=int, default=2, help='EXPERIMENTAL: Pretty much same as min_notes_frames (default: 2)')
    args = parser.parse_args()

    if args.frame_threshold <= 0 or args.frame_threshold > 1:
        parser.error("Frame threshold must be between 0 and 1")
    if args.onset_threshold <= 0 or args.onset_threshold > 1:
        parser.error("Onset threshold must be between 0 and 1")
    if args.min_note_frames <= 0:
        parser.error("min_note_frames must be positive")
    if args.refractory <= 0:
        parser.error("refractory must be positive")
    if not os.path.isfile(args.audio_path):
        parser.error(f"Audio file not found: {args.audio_path}")

    import render # import here to avoid delay from initializing tensorflow

    render.render(args.audio_path, output_path=args.output_path, frame_threshold=args.frame_threshold, onset_threshold=args.onset_threshold, refractory=args.refractory, min_note_frames=args.min_note_frames, model_path=args.model_path)

if __name__ == "__main__":
    main()