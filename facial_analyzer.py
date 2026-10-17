import cv2
from deepface import DeepFace
from collections import Counter


def analyze_facial_expressions(video_path, sample_interval=2):
    print("\n========================================")
    print("       FACIAL EXPRESSION ANALYSIS")
    print("========================================")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_interval = int(fps * sample_interval)

    frame_count = 0
    analyzed_frames = 0

    expressions = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv"
                )

                if isinstance(result, list):
                    result = result[0]

                dominant_expression = result["dominant_emotion"]
                confidence = result["face_confidence"]

                if confidence > 0.5:
                    expressions.append(dominant_expression)
                    analyzed_frames += 1

            except Exception as e:
                print(f"Frame analysis skipped: {e}")

        frame_count += 1

    cap.release()

    if not expressions:
        print("\nNo facial expressions could be detected.")
        return {
            "frames_analyzed": 0,
            "dominant_expression": "unknown",
            "expression_distribution": {},
            "feedback": [
                "No reliable facial-expression data was detected."
            ]
        }

    expression_counts = Counter(expressions)

    total = len(expressions)

    expression_distribution = {
        expression: round((count / total) * 100, 2)
        for expression, count in expression_counts.items()
    }

    dominant_expression = expression_counts.most_common(1)[0][0]

    feedback = []

    if dominant_expression == "happy":
        feedback.append(
            "Facial expression appears generally positive."
        )

    elif dominant_expression == "neutral":
        feedback.append(
            "Facial expression appears relatively neutral. "
            "Consider using more expressive facial cues when appropriate."
        )

    elif dominant_expression == "angry":
        feedback.append(
            "The system detected a relatively strong angry-expression "
            "pattern. Maintain a relaxed facial expression during communication."
        )

    elif dominant_expression == "sad":
        feedback.append(
            "A relatively serious or sad facial-expression pattern "
            "was detected. Try maintaining a more engaged expression."
        )

    elif dominant_expression == "fear":
        feedback.append(
            "A relatively fearful expression pattern was detected. "
            "Try to maintain relaxed facial expressions."
        )

    elif dominant_expression == "surprise":
        feedback.append(
            "Some surprised-expression patterns were detected."
        )

    else:
        feedback.append(
            "Facial-expression analysis completed."
        )

    result = {
        "frames_analyzed": analyzed_frames,
        "dominant_expression": dominant_expression,
        "expression_distribution": expression_distribution,
        "feedback": feedback
    }

    print(f"\nFrames Analyzed: {analyzed_frames}")
    print(f"Dominant Expression: {dominant_expression}")

    print("\nExpression Distribution:")

    for expression, percentage in expression_distribution.items():
        print(f"  {expression}: {percentage}%")

    print("\nFACIAL COMMUNICATION FEEDBACK:")

    for item in feedback:
        print(f"- {item}")

    return result


if __name__ == "__main__":

    video = r"recordings\presentation_20261005_232800.mp4"

    analyze_facial_expressions(video)