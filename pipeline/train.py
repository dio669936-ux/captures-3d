"""Train a Gaussian Splatting model using Nerfstudio."""

import argparse
import os
import subprocess
import sys


def train(data_dir, method="splatfacto", output_dir=None, max_iterations=15000):
    """Run Nerfstudio training.

    Args:
        data_dir: Path to processed data directory (from ns-process-data).
        method: Training method (default: splatfacto).
        output_dir: Output directory for trained model.
        max_iterations: Maximum training iterations.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    transforms_path = os.path.join(data_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        print(f"Error: transforms.json not found in {data_dir}")
        print("Run ns-process-data first to generate camera poses.")
        sys.exit(1)

    cmd = [
        "ns-train", method,
        "--data", data_dir,
        "--max-num-iterations", str(max_iterations),
        "--steps-per-eval-image", "250",
        "--pipeline.model.num-downscales", "2",
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    print(f"Training with {method}...")
    print(f"  Data: {data_dir}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Output: {output_dir or 'outputs/'}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining complete!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: ns-train not found. Install nerfstudio: pip install nerfstudio")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting model")
    parser.add_argument("--data", "-d", required=True, help="Path to processed data directory")
    parser.add_argument("--method", "-m", default="splatfacto", choices=["splatfacto", "nerfacto", "instant-ngp"],
                        help="Training method (default: splatfacto)")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--max-iterations", type=int, default=15000, help="Max training iterations (default: 15000)")
    parser.add_argument("--quick", action="store_true", help="Quick demo mode (7000 iterations)")
    args = parser.parse_args()

    iterations = 7000 if args.quick else args.max_iterations
    success = train(args.data, args.method, args.output, iterations)

    if success:
        print(f"\nNext step: Export the model")
        print(f"  python export.py --model {args.output or 'outputs/'} --output scene.splat")


if __name__ == "__main__":
    main()
