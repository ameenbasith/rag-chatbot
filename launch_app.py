"""
Launch script for the RAG chatbot web interface.
Place this file in the ROOT directory of your project.
"""

import subprocess
import sys
import os
from pathlib import Path


def launch_streamlit_app():
    """Launch the Streamlit app with proper Python path."""
    # Get the directory where this script is located (should be root)
    root_dir = Path(__file__).parent
    app_file = root_dir / "app" / "streamlit_app.py"

    print("🚀 Launching RAG Chatbot Web Interface...")
    print(f"📁 Root directory: {root_dir}")
    print(f"🔍 Looking for app file at: {app_file}")

    # Check if app file exists
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        print("💡 Please create app/streamlit_app.py first!")
        return

    # Set PYTHONPATH to include the root directory
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root_dir)

    print("🌐 Opening web interface in your browser...")

    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file)
        ], env=env, cwd=root_dir)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")
    except Exception as e:
        print(f"❌ Error launching app: {e}")


if __name__ == "__main__":
    launch_streamlit_app()