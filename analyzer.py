import re
from collections import Counter


FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "like",
    "basically", "actually", "literally",
    "you know", "i mean", "sort of", "kind of"
}


def analyze_transcript(text):
    text = text.strip()

    # Words
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())

    word_count = len(words)

    # Sentences
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    sentence_count = len(sentences)

    # Average sentence length
    average_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0
        else 0
    )

    # Filler words
    filler_counts = Counter()

    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        count = len(re.findall(pattern, text.lower()))

        if count > 0:
            filler_counts[filler] = count

    total_fillers = sum(filler_counts.values())

    # Vocabulary diversity
    unique_words = set(words)

    vocabulary_diversity = (
        len(unique_words) / word_count
        if word_count > 0
        else 0
    )

    # Repeated words
    word_frequency = Counter(words)

    repeated_words = {
        word: count
        for word, count in word_frequency.items()
        if count >= 3
    }

    # Basic feedback
    feedback = []

    if total_fillers > 3:
        feedback.append(
            "Try reducing filler words such as um, uh, like, and basically."
        )
    else:
        feedback.append(
            "Filler-word usage is relatively controlled."
        )

    if average_sentence_length > 25:
        feedback.append(
            "Some sentences are long. Try using shorter and clearer sentences."
        )
    elif average_sentence_length < 8:
        feedback.append(
            "Try developing your answers with slightly more complete sentences."
        )
    else:
        feedback.append(
            "Sentence length is reasonably suitable for spoken communication."
        )

    if vocabulary_diversity < 0.45:
        feedback.append(
            "Try using a wider range of vocabulary."
        )
    else:
        feedback.append(
            "Vocabulary variety is reasonably good."
        )

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": round(
            average_sentence_length, 2
        ),
        "filler_words": dict(filler_counts),
        "total_fillers": total_fillers,
        "unique_words": len(unique_words),
        "vocabulary_diversity": round(
            vocabulary_diversity, 2
        ),
        "repeated_words": repeated_words,
        "feedback": feedback
    }


if __name__ == "__main__":
    sample_text = """
    I am currently studying artificial intelligence and data science.
    Basically, I am interested in machine learning and artificial intelligence.
    """

    result = analyze_transcript(sample_text)

    print("\n===== NLP ANALYSIS =====")

    for key, value in result.items():
        print(f"{key}: {value}")