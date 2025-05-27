# Create venv
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv whisper-env

# Activate env whisper-env
source whisper-env/bin/activate

python -m tkinter  # ✅ should work

# Install dependencies
pip install sounddevice pynput numpy scipy openai-whisper
pip install torch

# Install soundfile
pip install soundfile


# Start Ollama
ollama run llama3.2

# Run code
python voice_assistant.py
