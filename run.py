# run.py - Place in PROJECT ROOT (not in backend or frontend)
"""
Unified startup script for RAG Book Bot
Runs both backend and frontend simultaneously
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return the process"""
    return subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def stream_output(process, prefix):
    """Stream output from process with prefix"""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[{prefix}] {line.rstrip()}")

def main():
    print("=" * 70)
    print("🚀 Starting RAG Book Bot")
    print("=" * 70)
    
    # Get project root directory
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"
    frontend_dir = project_root / "frontend"
    
    # Check if directories exist
    if not backend_dir.exists():
        print("❌ Error: backend/ directory not found")
        sys.exit(1)
    
    if not frontend_dir.exists():
        print("❌ Error: frontend/ directory not found")
        sys.exit(1)
    
    print("\n📦 Starting Backend (FastAPI)...")
    print(f"   Location: {backend_dir}")
    
    # Try to use venv, fallback to global Python
    if sys.platform == "win32":
        venv_python = backend_dir / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = backend_dir / "venv" / "bin" / "python"
    
    # Check if venv exists, otherwise use global python
    if venv_python.exists():
        print(f"   ✅ Using virtual environment: {venv_python}")
        python_exec = str(venv_python)
    else:
        print(f"   ⚠️  Virtual environment not found, using global Python")
        python_exec = sys.executable  # Use the same Python running this script
    
    backend_command = f'"{python_exec}" -m uvicorn main:app --reload --port 8000'
    backend_process = run_command(backend_command, cwd=str(backend_dir))
    
    # Wait a bit for backend to start
    print("   Waiting for backend to initialize...")
    time.sleep(3)
    
    print("\n⚛️  Starting Frontend (React + Vite)...")
    print(f"   Location: {frontend_dir}")
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("❌ Error: node_modules not found")
        print("   Run: cd frontend && npm install")
        sys.exit(1)
    
    frontend_command = "npm run dev"
    frontend_process = run_command(frontend_command, cwd=str(frontend_dir))
    
    print("\n" + "=" * 70)
    print("✅ Both servers are starting!")
    print("=" * 70)
    print("\n📡 Server URLs:")
    print("   Backend:  http://localhost:8000")
    print("   Frontend: http://localhost:3000")
    print("   API Docs: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop both servers")
    print("=" * 70 + "\n")
    
    try:
        # Stream output from both processes
        import threading
        
        backend_thread = threading.Thread(
            target=stream_output,
            args=(backend_process, "BACKEND"),
            daemon=True
        )
        frontend_thread = threading.Thread(
            target=stream_output,
            args=(frontend_process, "FRONTEND"),
            daemon=True
        )
        
        backend_thread.start()
        frontend_thread.start()
        
        # Wait for processes
        while backend_process.poll() is None and frontend_process.poll() is None:
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for graceful shutdown
        backend_process.wait(timeout=5)
        frontend_process.wait(timeout=5)
        
        print("✅ Servers stopped successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()