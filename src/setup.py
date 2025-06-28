#!/usr/bin/env python3
"""
Setup script for RAG Chatbot System
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a system command with error handling"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False

    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama detected: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama not found")
            return False
    except:
        print("❌ Ollama not found")
        return False

def setup_ollama_models():
    """Setup required Ollama models"""
    models = ["llama3.1", "nomic-embed-text"]

    print("\nSetting up Ollama models...")
    print("This may take a while as models need to be downloaded...")

    for model in models:
        print(f"\nPulling {model}...")
        if run_command(f"ollama pull {model}", f"Downloading {model}"):
            print(f"✅ {model} ready")
        else:
            print(f"❌ Failed to download {model}")
            print("You can manually download it later with: ollama pull {model}")

def create_directories():
    """Create necessary directories"""
    directories = ["chroma_db", "uploads", "logs"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🤖 RAG Chatbot System Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    create_directories()

    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)

    # Check Ollama
    if not check_ollama():
        print("\n⚠️  Ollama not found!")
        print("Please install Ollama from: https://ollama.ai/download")
        print("After installation, run this setup script again.")
        return

    # Setup Ollama models
    setup_response = input("\nDo you want to download required Ollama models now? (y/n): ")
    if setup_response.lower() in ['y', 'yes']:
        setup_ollama_models()
    else:
        print("\n⚠️  Remember to download models manually:")
        print("ollama pull llama3.1")
        print("ollama pull nomic-embed-text")

    print("\n🎉 Setup completed!")
    print("\nTo run the application:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Start the Streamlit app: streamlit run app.py")

if __name__ == "__main__":
    main()
