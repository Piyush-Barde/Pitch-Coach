from ai_coach import generate_ai_feedback


def build_evidence_summary(
    nlp_analysis,
    voice_analysis,
    facial_analysis
):
    evidence = {
        "language": {
            "word_count": nlp_analysis["word_count"],
            "sentence_count": nlp_analysis["sentence_count"],
            "average_sentence_length": nlp_analysis["average_sentence_length"],
            "filler_words": nlp_analysis["filler_words"],
            "total_fillers": nlp_analysis["total_fillers"],
            "vocabulary_diversity": nlp_analysis["vocabulary_diversity"],
            "repeated_words": nlp_analysis["repeated_words"]
        },

        "voice": {
            "duration_seconds": voice_analysis["duration_seconds"],
            "average_pitch_hz": voice_analysis["average_pitch_hz"],
            "pitch_variation_hz": voice_analysis["pitch_variation_hz"],
            "speaking_duration_seconds": voice_analysis["speaking_duration_seconds"],
            "pause_duration_seconds": voice_analysis["pause_duration_seconds"],
            "pause_percentage": voice_analysis["pause_percentage"],
            "average_energy": voice_analysis["average_energy"]
        },

        "facial": {
            "frames_analyzed": facial_analysis["frames_analyzed"],
            "dominant_expression": facial_analysis["dominant_expression"],
            "expression_distribution": facial_analysis[
                "expression_distribution"
            ]
        }
    }

    return evidence


def generate_final_report(
    transcript,
    nlp_analysis,
    voice_analysis,
    facial_analysis
):
    evidence = build_evidence_summary(
        nlp_analysis,
        voice_analysis,
        facial_analysis
    )

    return generate_ai_feedback(
        transcript=transcript,
        nlp_analysis=evidence["language"],
        voice_analysis=evidence["voice"],
        facial_analysis=evidence["facial"]
    )