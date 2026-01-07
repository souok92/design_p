import os
import sys
import argparse
import subprocess
import webbrowser
import time


def start_tensorboard(log_dir, port=6006, auto_open=True):
    """
    Start TensorBoard server and optionally open browser

    Args:
        log_dir: Directory containing TensorBoard logs
        port: Port number for TensorBoard server
        auto_open: Whether to automatically open browser
    """

    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"Warning: Log directory does not exist: {log_dir}")
        print("Creating directory...")
        os.makedirs(log_dir, exist_ok=True)

    # Start TensorBoard
    print(f"Starting TensorBoard on port {port}...")
    print(f"Monitoring logs from: {log_dir}")
    print("\nTensorBoard will update in real-time as training progresses.")
    print("Press Ctrl+C to stop TensorBoard.\n")

    # Build command
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", log_dir,
        "--port", str(port),
        "--reload_interval", "5"  # Refresh every 5 seconds
    ]

    try:
        # Start TensorBoard process
        process = subprocess.Popen(cmd)

        # Wait a bit for server to start
        time.sleep(3)

        # Open browser
        if auto_open:
            url = f"http://localhost:{port}"
            print(f"Opening browser at {url}...")
            webbrowser.open(url)

        print(f"\n{'='*60}")
        print(f"TensorBoard is running at http://localhost:{port}")
        print(f"{'='*60}\n")

        # Keep process running
        process.wait()

    except KeyboardInterrupt:
        print("\n\nStopping TensorBoard...")
        process.terminate()
        process.wait()
        print("TensorBoard stopped.")

    except Exception as e:
        print(f"Error starting TensorBoard: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Start TensorBoard for real-time training monitoring'
    )
    parser.add_argument(
        '--person',
        type=str,
        choices=['sj', 'tk', 'all'],
        default='all',
        help='Which person to monitor (sj, tk, or all)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='Port number for TensorBoard (default: 6006)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )

    args = parser.parse_args()

    # Determine log directory
    if args.person == 'all':
        log_dir = './logs'
    else:
        log_dir = f'./logs/{args.person}'

    # Convert to absolute path
    log_dir = os.path.abspath(log_dir)

    # Start TensorBoard
    start_tensorboard(
        log_dir=log_dir,
        port=args.port,
        auto_open=not args.no_browser
    )


if __name__ == '__main__':
    main()
