import cv2
import subprocess
import threading
import time
import numpy as np
import sounddevice as sd

from datetime import datetime
from pathlib import Path


CAMERA_NAME = "HP HD Camera"
MICROPHONE_NAME = "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)"


# ============================================================
# LIVE MICROPHONE MONITOR
# ============================================================

class AudioMonitor:

    def __init__(self):
        self.level = 0.0
        self.running = False
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):

        if not self.running:
            return

        if status:
            return

        audio = indata[:, 0]

        rms = np.sqrt(np.mean(audio ** 2))

        # Convert RMS into a more visible 0-100 range.
        level = min(rms * 500, 100)

        # Smooth the meter.
        self.level = (self.level * 0.7) + (level * 0.3)

    def start(self):

        self.running = True

        try:

            self.stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                blocksize=1024,
                callback=self.audio_callback
            )

            self.stream.start()

            print("Microphone monitor started.")

        except Exception as e:

            print(f"Microphone monitor could not start: {e}")

            self.stream = None

    def stop(self):

        self.running = False

        if self.stream is not None:

            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

            self.stream = None


# ============================================================
# DRAW AUDIO METER
# ============================================================

def draw_audio_meter(frame, level):

    x = 30
    y = 620

    bar_width = 500
    bar_height = 25

    # Background
    cv2.rectangle(
        frame,
        (x, y),
        (x + bar_width, y + bar_height),
        (60, 60, 60),
        -1
    )

    # Current level
    filled_width = int((level / 100) * bar_width)

    if filled_width > 0:

        cv2.rectangle(
            frame,
            (x, y),
            (x + filled_width, y + bar_height),
            (0, 200, 0),
            -1
        )

    # Border
    cv2.rectangle(
        frame,
        (x, y),
        (x + bar_width, y + bar_height),
        (255, 255, 255),
        2
    )

    # Label
    cv2.putText(
        frame,
        "MICROPHONE LEVEL",
        (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Status
    if level > 8:

        status = "AUDIO DETECTED"

        status_color = (0, 255, 0)

    else:

        status = "LOW / NO AUDIO"

        status_color = (0, 165, 255)

    cv2.putText(
        frame,
        status,
        (x + bar_width + 20, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )


# ============================================================
# RECORD VIDEO + AUDIO
# ============================================================

def record_video():

    recordings_dir = Path("recordings")
    recordings_dir.mkdir(exist_ok=True)

    filename = recordings_dir / (
        f"presentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():

        raise RuntimeError("Could not open camera.")

    width = 1280
    height = 720
    fps = 30

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if actual_fps <= 0:
        actual_fps = 30

    # --------------------------------------------------------
    # START AUDIO MONITOR
    # --------------------------------------------------------

    audio_monitor = AudioMonitor()
    audio_monitor.start()

    # --------------------------------------------------------
    # FFmpeg RECORDING
    # --------------------------------------------------------

    command = [

        "ffmpeg",
        "-y",

        # Microphone
        "-f",
        "dshow",
        "-i",
        f"audio={MICROPHONE_NAME}",

        # Video from OpenCV
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{actual_width}x{actual_height}",
        "-r",
        str(actual_fps),
        "-i",
        "-",

        # Mapping
        "-map",
        "1:v:0",
        "-map",
        "0:a:0",

        # Video encoding
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",

        # Audio encoding
        "-c:a",
        "aac",
        "-b:a",
        "128k",

        "-shortest",

        str(filename)
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    print("\n========================================")
    print("       AI COMMUNICATION COACH")
    print("========================================")

    print("\nLIVE CAMERA + MICROPHONE RECORDING")
    print("Speak in English.")
    print("Watch the microphone level bar.")
    print("Press Q in the camera window to stop.\n")

    # --------------------------------------------------------
    # RECORDING LOOP
    # --------------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("Failed to read camera frame.")

                break

            # Recording indicator
            cv2.putText(
                frame,
                "LIVE RECORDING",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Audio meter
            draw_audio_meter(
                frame,
                audio_monitor.level
            )

            # Instructions
            cv2.putText(
                frame,
                "Press Q to stop",
                (30, 680),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # Show camera
            cv2.imshow(
                "AI Communication Coach",
                frame
            )

            # Send frame to FFmpeg
            try:

                process.stdin.write(
                    frame.tobytes()
                )

            except (BrokenPipeError, OSError):

                print(
                    "FFmpeg process stopped unexpectedly."
                )

                break

            # Stop with Q
            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

    finally:

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        cap.release()

        cv2.destroyAllWindows()

        audio_monitor.stop()

        try:

            process.stdin.close()

        except (BrokenPipeError, OSError):

            pass

        try:

            process.wait(timeout=10)

        except subprocess.TimeoutExpired:

            print(
                "FFmpeg did not finish normally. Terminating it..."
            )

            process.terminate()

            try:

                process.wait(timeout=5)

            except subprocess.TimeoutExpired:

                process.kill()
                process.wait()

    print(
        f"\nRecording saved: {filename}"
    )

    return str(filename)