import os
import sys
import subprocess
import time
import webbrowser
import signal
import socket

# Try to import psutil for cleaner process group killing
try:
    import psutil
except ImportError:
    psutil = None

processes = []

def cleanup(sig=None, frame=None):
    print("\nShutting down all dashboard suite services...")
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

def wait_for_port(port, host="127.0.0.1", timeout=15):
    start_t = time.time()
    while time.time() - start_t < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.5)
    return False

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
    print("\n[1/5] Starting Shmoo ML flask app on http://127.0.0.1:5000 ...")
    try:
        shmoo_process = subprocess.Popen(
            [sys.executable, "run.py"],
            cwd=shmoo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(shmoo_process)
    except Exception as e:
        print(f"Failed to start Shmoo ML: {e}")

    # 2. Start Test Time Optimization Suite (Express Backend + Vite UI on Port 8787 / 5173)
    test_time_dir = os.path.join(root, "tools", "test_time_opt")
    print("[2/5] Starting Test Time Opt Suite (Node + React)...")
    try:
        if not os.path.exists(os.path.join(test_time_dir, "node_modules")):
            print("Installing node_modules for test_time_opt...")
            subprocess.run("npm install && npm install --prefix client", shell=True, cwd=test_time_dir)
        
        test_time_process = subprocess.Popen(
            ["npm", "run", "dev"],
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
    print("[3/5] Starting ATE Frontend Next.js app on http://127.0.0.1:3000 ...")
    try:
        if not os.path.exists(os.path.join(ate_frontend_dir, "node_modules")):
            print("Installing node_modules for ate_frontend...")
            subprocess.run("npm install", shell=True, cwd=ate_frontend_dir)
        
        # Build environment file pointing to local backends
        env_path = os.path.join(ate_frontend_dir, ".env.local")
        with open(env_path, "w") as f:
            f.write("NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api\n")
            f.write("NEXT_PUBLIC_KPI_M_BIST_SHMOO_URL=http://127.0.0.1:5000\n")
            f.write("NEXT_PUBLIC_KPI_TEST_TIME_URL=http://127.0.0.1:5173\n")
            f.write("NEXT_PUBLIC_KPI_RETEST_URL=http://127.0.0.1:5175\n")

        print("  -> Starting ATE Intelligence Local Backend on http://127.0.0.1:8000 ...")
        ate_backend_proc = subprocess.Popen(
            [sys.executable, "ate_backend.py"],
            cwd=ate_frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(ate_backend_proc)

        ate_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=ate_frontend_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(ate_process)
    except Exception as e:
        print(f"Failed to start ATE Frontend: {e}")

    # 4. Start Dynamic Test Limit (DTL) FastAPI Backend (Port 8001)
    dtl_dir = os.path.join(root, "tools", "dtl")
    print("[4/5] Starting Dynamic Test Limits (DTL) Backend on http://127.0.0.1:8001 ...")
    try:
        # Add src/ folder of DTL to python path so it imports dtl_agent cleanly
        dtl_env = {**os.environ, "PYTHONPATH": os.path.join(dtl_dir, "src")}
        dtl_backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "dtl_agent.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", "8001"],
            cwd=dtl_dir,
            env=dtl_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(dtl_backend_process)
    except Exception as e:
        print(f"Failed to start DTL Backend: {e}")

    # 5. Start DTL React Frontend (Port 5174)
    dtl_frontend_dir = os.path.join(dtl_dir, "frontend")
    print("[5/5] Starting DTL React UI on http://127.0.0.1:5174 ...")
    try:
        if not os.path.exists(os.path.join(dtl_frontend_dir, "node_modules")):
            print("Installing node_modules for DTL frontend...")
            subprocess.run("npm install", shell=True, cwd=dtl_frontend_dir)

        dtl_frontend_process = subprocess.Popen(
            ["npx", "vite", "--host", "127.0.0.1", "--port", "5174"],
            cwd=dtl_frontend_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(dtl_frontend_process)
    except Exception as e:
        print(f"Failed to start DTL Frontend: {e}")

    # 6. Start Retest Benefit Prediction AI Backend (Port 8002)
    retest_dir = os.path.join(root, "tools", "retest_reduction")
    print("[6/6] Starting Retest AI Backend on http://127.0.0.1:8002 ...")
    try:
        retest_env = {**os.environ, "PYTHONPATH": os.path.join(retest_dir)}
        retest_backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "retest_ai.api.main:app", "--host", "127.0.0.1", "--port", "8002"],
            cwd=retest_dir,
            env=retest_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(retest_backend_proc)
    except Exception as e:
        print(f"Failed to start Retest AI Backend: {e}")

    # 7. Start Retest AI React Frontend (Port 5175)
    retest_frontend_dir = os.path.join(retest_dir, "frontend")
    print("  -> Starting Retest AI React UI on http://127.0.0.1:5175 ...")
    try:
        if not os.path.exists(os.path.join(retest_frontend_dir, "node_modules")):
            print("Installing node_modules for Retest AI frontend...")
            subprocess.run("npm install", shell=True, cwd=retest_frontend_dir)

        retest_frontend_proc = subprocess.Popen(
            ["npx", "vite", "--host", "127.0.0.1", "--port", "5175"],
            cwd=retest_frontend_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(retest_frontend_proc)
    except Exception as e:
        print(f"Failed to start Retest AI Frontend: {e}")

    # Wait for servers to wake up
    print("\nWarming up servers and checking ports...")
    wait_for_port(5000, timeout=10)
    wait_for_port(8787, timeout=12)
    wait_for_port(8000, timeout=10)
    wait_for_port(8001, timeout=10)
    wait_for_port(8002, timeout=10)
    wait_for_port(3000, timeout=15)
    wait_for_port(5174, timeout=15)
    wait_for_port(5175, timeout=15)

    # Open the browser to the single unified central dashboard
    print("\nOpening Unified ATE Intelligence Suite in your default browser:")
    print("  -> Central Dashboard (Single Window): http://127.0.0.1:3000")
    
    webbrowser.open("http://127.0.0.1:3000")

    print("\nSuite is running. Press Ctrl+C to stop all services.")
    
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            cleanup()

if __name__ == "__main__":
    main()
