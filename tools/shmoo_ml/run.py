import sys, os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("  M-BIST SHMOO ML Optimization System")
    print("  Offline & Local Deployment")
    print("  Running at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, port=5000, host='0.0.0.0')
