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
import json
from tkhtmlview import HTMLScrolledText
import markdown

SAMPLERATE = 44100

class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Voice Assistant with Whisper + LLaMA 3.2")
        # self.root.geometry("600x800")
        self.root.resizable(True, True)

        self.recording = False
        self.frames = []

        # Set font
        self.custom_font = tkfont.Font(family="Helvetica", size=20)
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

        self.listen_speaker_btn = tk.Button(
            button_frame,
            text="🔊 Listen Speaker",
            command=self.start_speaker_recording,
            width=20,
            bg="#FF9800",
            fg="white",
            font=self.custom_font
        )
        self.listen_speaker_btn.grid(row=0, column=2, padx=10)

        # Text area (Markdown/HTML rendering)
        self.text_area = HTMLScrolledText(root, width=70, height=20, font=("Courier", 20), html=True)
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.text_area.config(state="normal")  # Ensure selectable/copyable
        self.html_buffer = "<p><b>[INFO]</b> Ready to record...</p>"
        self.text_area.set_html(self.html_buffer)

        # Query entry and Go button at the bottom
        entry_frame = tk.Frame(root, height=60)  # Double the height (default is ~30)
        entry_frame.pack_propagate(False)  # Prevent frame from shrinking to fit contents
        entry_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.query_var = tk.StringVar()
        self.query_entry = tk.Entry(entry_frame, textvariable=self.query_var, font=self.custom_font, width=60)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=10)  # Double the height with ipady
        self.query_entry.bind('<Return>', self.on_query_enter)
        self.go_btn = tk.Button(entry_frame, text="➡️", command=self.on_query_submit, font=self.custom_font, width=4, bg="#2196F3", fg="white")
        self.go_btn.pack(side=tk.LEFT)

        # Load whisper model once
        self.model = whisper.load_model("small")

    def update_text_area(self, message):
        # Check if user is at the bottom before inserting
        last_visible = self.text_area.yview()[1]
        at_bottom = last_visible >= 0.999  # yview returns (top, bottom) as fractions
        # Render user/info/error messages as markdown for consistency
        html_message = markdown.markdown(message, extensions=['fenced_code', 'codehilite'])
        self.html_buffer += html_message
        self.text_area.set_html(self.html_buffer)
        if at_bottom:
            self.text_area.see("end")

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.frames = []

        self.play_btn.config(state=tk.DISABLED)
        self.listen_speaker_btn.config(state=tk.DISABLED)
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

    def start_speaker_recording(self):
        if self.recording:
            return
        self.recording = True
        self.frames = []
        self.play_btn.config(state=tk.DISABLED)
        self.listen_speaker_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_text_area("[INFO] Speaker recording started...\n")
        threading.Thread(target=self.record_speaker_audio, daemon=True).start()

    def record_speaker_audio(self):
        try:
            # Find BlackHole device index
            devices = sd.query_devices()
            blackhole_index = None
            for idx, dev in enumerate(devices):
                if 'blackhole' in dev['name'].lower() and dev['max_input_channels'] > 0:
                    blackhole_index = idx
                    break
            if blackhole_index is None:
                self.root.after(0, self.update_text_area, "[ERROR] BlackHole input device not found.\n")
                self.root.after(0, self.stop_recording)
                return
            channels = min(devices[blackhole_index]['max_input_channels'], 2)
            def callback(indata, frame_count, time_info, status):
                if self.recording:
                    self.frames.append(indata.copy())
            with sd.InputStream(samplerate=SAMPLERATE, channels=channels, device=blackhole_index, callback=callback):
                while self.recording:
                    sd.sleep(100)
        except Exception as e:
            self.root.after(0, self.update_text_area, f"[ERROR] Speaker recording failed: {e}\n")
            self.root.after(0, self.stop_recording)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False

        self.play_btn.config(state=tk.NORMAL)
        self.listen_speaker_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_text_area("[INFO] Recording stopped.\n")

        threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        try:
            if not self.frames:
                self.root.after(0, self.update_text_area, "[ERROR] No audio data was recorded.\n")
                return
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
                    json={"model": "llama3.2", "prompt": text, "stream": True},
                    stream=True,
                    timeout=60
                )
                response.raise_for_status()
                buffer = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                buffer += chunk
                                # Only update the UI every 10 characters or on newlines to reduce flicker
                                if len(buffer) % 10 == 0 or chunk.endswith("\n"):
                                    self.root.after(0, self.update_llama_response, buffer)
                        except Exception:
                            continue
                self.root.after(0, self.finalize_llama_response, buffer)
            except Exception as e:
                self.root.after(0, self.update_text_area, f"[ERROR] LLaMA request failed: {e}\n")
        except Exception as e:
            self.root.after(0, self.update_text_area, f"[ERROR] Audio processing failed: {e}\n")

    def on_query_enter(self, event=None):
        self.on_query_submit()

    def on_query_submit(self):
        query = self.query_var.get().strip()
        if not query:
            return
        self.query_var.set("")
        self.update_text_area(f"[USER]: {query}\n")
        self.update_text_area("[INFO] Sending to LLaMA 3.2...\n")
        threading.Thread(target=self.send_query_to_ollama, args=(query,), daemon=True).start()

    def send_query_to_ollama(self, query):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": query, "stream": True},
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            buffer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            buffer += chunk
                            # Only update the UI every 10 characters or on newlines to reduce flicker
                            if len(buffer) % 10 == 0 or chunk.endswith("\n"):
                                self.root.after(0, self.update_llama_response, buffer)
                    except Exception:
                        continue
            self.root.after(0, self.finalize_llama_response, buffer)
        except Exception as e:
            self.root.after(0, self.update_text_area, f"[ERROR] LLaMA request failed: {e}\n")

    def update_llama_response(self, buffer):
        import re
        # Remove only the last LLaMA response block at the end, keep previous conversation
        html_buffer = self.html_buffer  # Use the persistent buffer, not get_html()
        html_buffer = re.sub(r"(<div id='llama-response'>.*?</div>)$", "", html_buffer, flags=re.DOTALL)
        html = f"<div id='llama-response'>{markdown.markdown(buffer, extensions=['fenced_code', 'codehilite'])}</div>"
        # Check if user is at the bottom before updating
        last_visible = self.text_area.yview()[1]
        at_bottom = last_visible >= 0.999
        self.text_area.set_html(html_buffer + html)
        if at_bottom:
            self.text_area.see("end")

    def finalize_llama_response(self, buffer):
        import re
        self.html_buffer = re.sub(r"(<div id='llama-response'>.*?</div>)$", "", self.html_buffer, flags=re.DOTALL)
        html = f"<div id='llama-response'>{markdown.markdown(buffer, extensions=['fenced_code', 'codehilite'])}</div>"
        self.html_buffer += html + "<br><br>"
        # Check if user is at the bottom before updating
        last_visible = self.text_area.yview()[1]
        at_bottom = last_visible >= 0.999
        self.text_area.set_html(self.html_buffer)
        if at_bottom:
            self.text_area.see("end")

def main():
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()