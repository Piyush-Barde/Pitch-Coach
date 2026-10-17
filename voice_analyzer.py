import librosa
import numpy as np
import subprocess
import tempfile
from pathlib import Path


def extract_audio(video_path):
    temp_audio = Path(tempfile.gettempdir()) / "communication_coach_audio.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(temp_audio)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return temp_audio


def analyze_voice(video_path):

    print("\n========================================")
    print("          VOICE ANALYSIS")
    print("========================================")

    print("\nExtracting audio...")

    audio_path = extract_audio(video_path)

    print("Analyzing audio...")

    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    # Energy
    rms = librosa.feature.rms(y=audio)[0]
    average_energy = float(np.mean(rms))

    # Pitch
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    valid_pitch = f0[~np.isnan(f0)]

    if len(valid_pitch) > 0:
        average_pitch = float(np.mean(valid_pitch))
        pitch_variation = float(np.std(valid_pitch))
    else:
        average_pitch = 0
        pitch_variation = 0

    # Speaking / silence detection
    intervals = librosa.effects.split(
        audio,
        top_db=30
    )

    speaking_duration = sum(
        end - start
        for start, end in intervals
    ) / sample_rate

    pause_duration = max(
        duration - speaking_duration,
        0
    )

    pause_percentage = (
        pause_duration / duration * 100
        if duration > 0
        else 0
    )

    result = {
        "duration_seconds": round(duration, 2),
        "average_energy": round(average_energy, 4),
        "average_pitch_hz": round(average_pitch, 2),
        "pitch_variation_hz": round(pitch_variation, 2),
        "speaking_duration_seconds": round(
            speaking_duration, 2
        ),
        "pause_duration_seconds": round(
            pause_duration, 2
        ),
        "pause_percentage": round(
            pause_percentage, 2
        )
    }

    feedback = []

    if pitch_variation < 15:
        feedback.append(
            "Voice pitch variation is relatively low. "
            "Try adding more vocal variation."
        )
    else:
        feedback.append(
            "Good pitch variation in the speech."
        )

    if pause_percentage > 30:
        feedback.append(
            "There are relatively long pauses. "
            "Try maintaining smoother speech flow."
        )
    else:
        feedback.append(
            "Pausing appears reasonably controlled."
        )

    if average_energy < 0.02:
        feedback.append(
            "Voice energy appears relatively low. "
            "Try speaking with slightly more vocal projection."
        )
    else:
        feedback.append(
            "Voice energy appears reasonably strong."
        )

    result["feedback"] = feedback

    return result


if __name__ == "__main__":

    video = r"recordings\presentation_20261005_230728.mp4"

    result = analyze_voice(video)

    print("\nVOICE METRICS:")

    for key, value in result.items():
        print(f"{key}: {value}")