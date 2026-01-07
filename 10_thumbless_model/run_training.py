import os
import sys
import argparse
import subprocess
import time
import webbrowser


def run_training_with_tensorboard(config_path, port=6006):
    """
    Run training and TensorBoard simultaneously

    Args:
        config_path: Path to config file
        port: Port for TensorBoard
    """

    # Extract person from config path
    if 'sj' in config_path:
        person = 'sj'
    elif 'tk' in config_path:
        person = 'tk'
    else:
        person = 'unknown'

    log_dir = f'./logs/{person}'
    log_dir_abs = os.path.abspath(log_dir)

    # Ensure log directory exists
    os.makedirs(log_dir_abs, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Starting Training with Real-time TensorBoard Monitoring")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Person: {person}")
    print(f"Log directory: {log_dir_abs}")
    print(f"TensorBoard port: {port}")
    print(f"{'='*70}\n")

    # Start TensorBoard in background
    print("Starting TensorBoard...")
    tensorboard_cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", log_dir_abs,
        "--port", str(port),
        "--reload_interval", "5"
    ]

    tensorboard_process = subprocess.Popen(
        tensorboard_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for TensorBoard to start
    print("Waiting for TensorBoard to start...")
    time.sleep(3)

    # Open browser
    url = f"http://localhost:{port}"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)

    print(f"\n{'='*70}")
    print(f"  TensorBoard is running at {url}")
    print(f"{'='*70}\n")

    time.sleep(2)

    # Start training
    print("Starting training...\n")
    train_cmd = [sys.executable, "train.py", "--config", config_path]

    try:
        # Run training (this will block until training completes)
        training_process = subprocess.Popen(train_cmd)
        training_process.wait()

        print("\n" + "="*70)
        if training_process.returncode == 0:
            print("  Training completed successfully!")
        else:
            print(f"  Training ended with return code: {training_process.returncode}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        training_process.terminate()
        training_process.wait()

    finally:
        # Ask if user wants to keep TensorBoard running
        print("\n" + "="*70)
        print(f"TensorBoard is still running at {url}")
        print("="*70)

        keep_running = input("\nKeep TensorBoard running? (y/n): ").strip().lower()

        if keep_running != 'y':
            print("Stopping TensorBoard...")
            tensorboard_process.terminate()
            tensorboard_process.wait()
            print("TensorBoard stopped.")
        else:
            print("\nTensorBoard will continue running.")
            print(f"View at: {url}")
            print("To stop later, press Ctrl+C\n")

            try:
                tensorboard_process.wait()
            except KeyboardInterrupt:
                print("\nStopping TensorBoard...")
                tensorboard_process.terminate()
                tensorboard_process.wait()
                print("TensorBoard stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Run training with automatic TensorBoard monitoring'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/config_sj.yaml)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='Port for TensorBoard (default: 6006)'
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Run training with TensorBoard
    run_training_with_tensorboard(args.config, args.port)


if __name__ == '__main__':
    main()
