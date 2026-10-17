import os
import torch
import whisper

from recorder import record_video
from analyzer import analyze_transcript
from voice_analyzer import analyze_voice
from facial_analyzer import analyze_facial_expressions


def main():
    torch.set_num_threads(os.cpu_count())

    # ========================================
    # RECORD
    # ========================================
    video_path = record_video()

    # ========================================
    # WHISPER TRANSCRIPTION
    # ========================================
    print("\n========================================")
    print("          SPEECH TRANSCRIPTION")
    print("========================================")

    print("\nLoading Whisper base.en model...")
    whisper_model = whisper.load_model("base.en")

    print("Transcribing speech...")
    result = whisper_model.transcribe(video_path)

    transcript = result["text"].strip()

    print("\nTRANSCRIPT:")
    print(transcript)

    # ========================================
    # NLP ANALYSIS
    # ========================================
    print("\n========================================")
    print("             NLP ANALYSIS")
    print("========================================")

    nlp_analysis = analyze_transcript(transcript)

    print(f"\nWord Count: {nlp_analysis['word_count']}")
    print(f"Sentence Count: {nlp_analysis['sentence_count']}")
    print(
        f"Average Sentence Length: "
        f"{nlp_analysis['average_sentence_length']}"
    )
    print(f"Filler Words: {nlp_analysis['filler_words']}")
    print(f"Total Fillers: {nlp_analysis['total_fillers']}")
    print(f"Unique Words: {nlp_analysis['unique_words']}")
    print(
        f"Vocabulary Diversity: "
        f"{nlp_analysis['vocabulary_diversity']}"
    )
    print(f"Repeated Words: {nlp_analysis['repeated_words']}")

    # ========================================
    # VOICE ANALYSIS
    # ========================================
    voice_analysis = analyze_voice(video_path)

    print("\nVOICE METRICS:")
    print(
        f"Duration: "
        f"{voice_analysis['duration_seconds']} seconds"
    )
    print(
        f"Average Pitch: "
        f"{voice_analysis['average_pitch_hz']} Hz"
    )
    print(
        f"Pitch Variation: "
        f"{voice_analysis['pitch_variation_hz']} Hz"
    )
    print(
        f"Speaking Duration: "
        f"{voice_analysis['speaking_duration_seconds']} seconds"
    )
    print(
        f"Pause Duration: "
        f"{voice_analysis['pause_duration_seconds']} seconds"
    )
    print(
        f"Pause Percentage: "
        f"{voice_analysis['pause_percentage']}%"
    )
    print(
        f"Average Voice Energy: "
        f"{voice_analysis['average_energy']}"
    )

    # ========================================
    # FACIAL ANALYSIS
    # ========================================
    facial_analysis = analyze_facial_expressions(video_path)

if __name__ == "__main__":
    main()