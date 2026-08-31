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
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write("NEXT_PUBLIC_API_BASE_URL=https://wafer-yield-api.onrender.com\n")
                f.write("NEXT_PUBLIC_KPI_M_BIST_SHMOO_URL=http://127.0.0.1:5000\n")
                f.write("NEXT_PUBLIC_KPI_TEST_TIME_URL=http://127.0.0.1:5173\n")

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

    # 4. Start Dynamic Test Limit (DTL) FastAPI Backend (Port 8000)
    dtl_dir = os.path.join(root, "tools", "dtl")
    print("[4/5] Starting Dynamic Test Limits (DTL) Backend on http://127.0.0.1:8000 ...")
    try:
        # Add src/ folder of DTL to python path so it imports dtl_agent cleanly
        dtl_env = {**os.environ, "PYTHONPATH": os.path.join(dtl_dir, "src")}
        dtl_backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "dtl_agent.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", "8000"],
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
            ["npm", "run", "dev", "--", "--port", "5174"],
            cwd=dtl_frontend_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(dtl_frontend_process)
    except Exception as e:
        print(f"Failed to start DTL Frontend: {e}")

    # Wait for servers to wake up
    print("\nWarming up servers (6 seconds)...")
    time.sleep(6)

    # Open the browser to the Central Next.js Dashboard and the DTL Dashboard
    print("\nOpening dashboards in your default browser:")
    print("  -> Central Dashboard: http://127.0.0.1:3000")
    print("  -> Dynamic Test Limits Dashboard: http://127.0.0.1:5174/three-month")
    
    webbrowser.open("http://127.0.0.1:3000")
    webbrowser.open("http://127.0.0.1:5174/three-month")

    print("\nSuite is running. Press Ctrl+C to stop all services.")
    
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            cleanup()

if __name__ == "__main__":
    main()
