import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import font as tkfont
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import requests
import tempfile
import os

SAMPLERATE = 44100

class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Voice Assistant with Whisper + LLaMA 3.2")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        self.root.resizable(False, False)

        self.recording = False
        self.frames = []

        # Set font
        self.custom_font = tkfont.Font(family="Helvetica", size=10)
        self.heading_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

        # Title
        title_label = tk.Label(root, text="Voice Assistant", font=self.heading_font, fg="#333", pady=10)
        title_label.pack()

        # Button frame (horizontal layout)
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.play_btn = tk.Button(
            button_frame,
            text="▶️ Start Recording",
            command=self.start_recording,
            width=20,
            bg="#4CAF50",
            fg="white",
            font=self.custom_font
        )
        self.play_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ Stop Recording",
            command=self.stop_recording,
            state=tk.DISABLED,
            width=20,
            bg="#F44336",
            fg="white",
            font=self.custom_font
        )
        self.stop_btn.grid(row=0, column=1, padx=10)

        # Text area
        self.text_area = ScrolledText(root, width=70, height=20, font=("Courier", 20), wrap=tk.WORD)
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.text_area.insert(tk.END, "[INFO] Ready to record...\n")

        # Load whisper model once
        self.model = whisper.load_model("base")

    def update_text_area(self, message):
        self.text_area.insert(tk.END, message)
        self.text_area.see(tk.END)

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.frames = []

        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_text_area("[INFO] Recording started...\n")

        threading.Thread(target=self.record_audio, daemon=True).start()

    def record_audio(self):
        try:
            device_info = sd.query_devices(kind='input')
            channels = min(device_info['max_input_channels'], 2)

            def callback(indata, frame_count, time_info, status):
                if self.recording:
                    self.frames.append(indata.copy())

            with sd.InputStream(samplerate=SAMPLERATE, channels=channels, callback=callback):
                while self.recording:
                    sd.sleep(100)

        except Exception as e:
            self.root.after(0, self.update_text_area, f"[ERROR] Audio recording failed: {e}\n")
            self.root.after(0, self.stop_recording)  # Reset buttons safely

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False

        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_text_area("[INFO] Recording stopped.\n")

        threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        try:
            audio_np = np.concatenate(self.frames, axis=0)

            # Convert to mono
            if audio_np.ndim == 2 and audio_np.shape[1] > 1:
                audio_np = np.mean(audio_np, axis=1)

            self.root.after(0, self.update_text_area, "[INFO] Saving audio to file...\n")

            # Save to 'recorded_audio.wav' in current directory
            audio_path = os.path.join(os.getcwd(), "recorded_audio.wav")
            sf.write(audio_path, audio_np, SAMPLERATE)

            self.root.after(0, self.update_text_area, "[INFO] Transcribing audio...\n")

            # Use Whisper's built-in transcribe function with the file path
            result = self.model.transcribe(audio_path)
            text = result['text'].strip()

            self.root.after(0, self.update_text_area, f"[USER]: {text}\n")
            self.root.after(0, self.update_text_area, "[INFO] Sending to LLaMA 3.2...\n")

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.2", "prompt": text, "stream": False},
                    timeout=30
                )
                response.raise_for_status()
                llama_response = response.json().get("response", "[No response]")
            except Exception as e:
                llama_response = f"[ERROR] LLaMA request failed: {e}"

            self.root.after(0, self.update_text_area, f"[LLAMA 3.2]: {llama_response}\n\n")

        except Exception as e:
            self.root.after(0, self.update_text_area, f"[ERROR] Audio processing failed: {e}\n")

def main():
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()