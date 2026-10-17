from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-base"

print("\nLoading FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("FLAN-T5 loaded successfully.")


def generate_ai_feedback(
    transcript,
    nlp_analysis,
    voice_analysis,
    facial_analysis
):
    prompt = f"""
You are an English communication coach.

Give concise practical feedback based on this communication evidence.

Transcript:
{transcript}

Language evidence:
{nlp_analysis}

Voice evidence:
{voice_analysis}

Facial expression evidence:
{facial_analysis}

Identify:
1. Strengths
2. Areas to improve
3. Practical recommendations

Important:
- Use the evidence provided.
- Do not invent information.
- Do not claim to know the speaker's actual emotions.
- Facial expressions are only observable expression classifications.
- The transcript may contain speech-recognition errors.

Write a short useful coaching report.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )