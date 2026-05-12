"""Extract frames from video for Gaussian Splatting training."""

import argparse
import os
import sys

import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, target_fps=2, max_frames=None):
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save extracted frames.
        target_fps: Frames per second to extract (default: 2).
        max_frames: Maximum number of frames to extract.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    frame_interval = max(1, int(video_fps / target_fps))
    expected_frames = total_frames // frame_interval
    if max_frames:
        expected_frames = min(expected_frames, max_frames)

    print(f"Video: {video_path}")
    print(f"  Duration: {duration:.1f}s | FPS: {video_fps:.1f} | Total frames: {total_frames}")
    print(f"  Extracting every {frame_interval} frames (~{target_fps} fps)")
    print(f"  Expected output: ~{expected_frames} frames")

    frame_idx = 0
    saved_count = 0

    with tqdm(total=expected_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                filename = f"frame_{saved_count:05d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                pbar.update(1)

                if max_frames and saved_count >= max_frames:
                    break

            frame_idx += 1

    cap.release()
    print(f"\nExtracted {saved_count} frames to: {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for 3DGS training")
    parser.add_argument("--input", "-i", required=True, help="Path to input video")
    parser.add_argument("--output", "-o", required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2, help="Target FPS for extraction (default: 2)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to extract")
    args = parser.parse_args()

    frames_dir = os.path.join(args.output, "frames")
    extract_frames(args.input, frames_dir, args.fps, args.max_frames)

    print(f"\nNext step:")
    print(f"  ns-process-data images --data {frames_dir} --output-dir {args.output}/processed")


if __name__ == "__main__":
    main()
