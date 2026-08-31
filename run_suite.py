import os
import sys
import subprocess
import time
import webbrowser
import signal

# Try to import psutil for cleaner process group killing
try:
    import psutil
except ImportError:
    psutil = None

processes = []

def cleanup(sig=None, frame=None):
    print("\nShutting down all dashboard services...")
    for p in processes:
        try:
            if psutil:
                # Kill process and all its children
                parent = psutil.Process(p.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
            else:
                p.terminate()
        except Exception:
            pass
    print("Cleanup complete. Goodbye!")
    sys.exit(0)

# Register cleanup handler for Ctrl+C
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("          VERILUMEN SEMICONDUCTOR DASHBOARD SUITE")
    print("=" * 60)
    print("Starting all local microservices...")

    # 1. Start Shmoo ML Flask backend (Port 5000)
    shmoo_dir = os.path.join(root, "tools", "shmoo_ml")
    print("\n[1/3] Starting Shmoo ML flask app on http://127.0.0.1:5000 ...")
    try:
        shmoo_process = subprocess.Popen(
            [sys.executable, "run.py"],
            cwd=shmoo_dir,
            stdout=subprocess.DEVNULL,  # Keep output clean, or redirect to log
            stderr=subprocess.DEVNULL
        )
        processes.append(shmoo_process)
    except Exception as e:
        print(f"Failed to start Shmoo ML: {e}")

    # 2. Start Test Time Optimization Express backend (Port 8787)
    test_time_dir = os.path.join(root, "tools", "test_time_opt")
    print("[2/3] Starting Test Time Opt Node backend on http://127.0.0.1:8787 ...")
    try:
        # Check if node_modules exists
        if not os.path.exists(os.path.join(test_time_dir, "node_modules")):
            print("Installing node_modules for test_time_opt...")
            subprocess.run("npm install", shell=True, cwd=test_time_dir)
        
        test_time_process = subprocess.Popen(
            ["node", "server/index.js"],
            cwd=test_time_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(test_time_process)
    except Exception as e:
        print(f"Failed to start Test Time Opt: {e}")

    # 3. Start ATE Frontend Next.js app (Port 3000)
    ate_frontend_dir = os.path.join(root, "tools", "ate_frontend")
    print("[3/3] Starting ATE Frontend Next.js app on http://127.0.0.1:3000 ...")
    try:
        # Check if node_modules exists
        if not os.path.exists(os.path.join(ate_frontend_dir, "node_modules")):
            print("Installing node_modules for ate_frontend...")
            subprocess.run("npm install", shell=True, cwd=ate_frontend_dir)
        
        # Build environment file pointing to local backends if they exist
        env_path = os.path.join(ate_frontend_dir, ".env.local")
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write("NEXT_PUBLIC_API_BASE_URL=https://wafer-yield-api.onrender.com\n")
                f.write("NEXT_PUBLIC_KPI_M_BIST_SHMOO_URL=http://127.0.0.1:5000\n")
                f.write("NEXT_PUBLIC_KPI_TEST_TIME_URL=http://127.0.0.1:8787\n")

        ate_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=ate_frontend_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(ate_process)
    except Exception as e:
        print(f"Failed to start ATE Frontend Next.js: {e}")

    # Wait for servers to wake up
    print("\nWarming up servers (5 seconds)...")
    time.sleep(5)

    # Open the browser to the Next.js ATE Frontend
    url = "http://127.0.0.1:3000"
    print(f"\nOpening dashboard at {url} in your default browser...")
    webbrowser.open(url)

    print("\nSuite is running. Press Ctrl+C to stop all services.")
    
    # Keep the script running
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            cleanup()

if __name__ == "__main__":
    main()
