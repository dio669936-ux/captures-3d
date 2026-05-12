"""Export trained Nerfstudio model to .splat format for web viewer."""

import argparse
import glob
import os
import subprocess
import sys


def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in the model directory."""
    config_files = glob.glob(os.path.join(model_dir, "**", "config.yml"), recursive=True)
    if not config_files:
        return None
    config_files.sort(key=os.path.getmtime, reverse=True)
    return config_files[0]


def export_splat(model_dir, output_path, num_points=None):
    """Export model to .splat format.

    Args:
        model_dir: Path to trained model directory.
        output_path: Output .splat file path.
        num_points: Number of Gaussian points to export.
    """
    config_path = find_latest_checkpoint(model_dir)
    if not config_path:
        print(f"Error: No config.yml found in {model_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ns-export", "gaussian-splat",
        "--load-config", config_path,
        "--output-dir", os.path.dirname(output_path) or ".",
    ]

    if num_points:
        cmd.extend(["--num-points", str(num_points)])

    print(f"Exporting to: {output_path}")
    print(f"  Config: {config_path}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print(f"\nExport complete: {output_path}")
            print(f"\nTo view in the web viewer:")
            print(f"  1. Open the Captures 3D web app")
            print(f"  2. Click Upload > Local File")
            print(f"  3. Drop the .splat file")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nExport failed with exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: ns-export not found. Install nerfstudio: pip install nerfstudio")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export model to .splat format")
    parser.add_argument("--model", "-m", required=True, help="Path to trained model directory")
    parser.add_argument("--output", "-o", default="scene.splat", help="Output .splat file path")
    parser.add_argument("--num-points", type=int, default=None, help="Number of Gaussian points")
    args = parser.parse_args()

    export_splat(args.model, args.output, args.num_points)


if __name__ == "__main__":
    main()
