import os
import json
import whisper
import nltk
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Download tokenizer (first time only)
nltk.download('punkt')

# ---------------------------
# 1. Folder setup
# ---------------------------
os.makedirs("output", exist_ok=True)

# ---------------------------
# 2. Load models
# ---------------------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading summarization model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---------------------------
# 3. Audio file path
# ---------------------------
audio_path = "audio/new_podcast.wav"

# ---------------------------
# 4. Transcribe audio
# ---------------------------
print("Transcribing audio...")
result = whisper_model.transcribe(audio_path, verbose=False)
segments = result["segments"]

# ---------------------------
# 5. Topic segmentation 
# ---------------------------
GROUP_SIZE = 10   # ⬅️ Increased to reduce total segments
topic_segments = []

for i in range(0, len(segments), GROUP_SIZE):
    group = segments[i:i + GROUP_SIZE]

    text = " ".join([s["text"] for s in group])
    start_time = group[0]["start"]
    end_time = group[-1]["end"]

    topic_segments.append({
        "text": text,
        "start": start_time,
        "end": end_time
    })

# ---------------------------
# 6. Keyword extraction
# ---------------------------
def extract_keywords(text, n=4):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=n)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# ---------------------------
# 7. Generate JSON output with TITLES
# ---------------------------
output_json = "output/final_topic_segments.json"
final_results = []

for idx, seg in enumerate(topic_segments, 1):

    # Summary
    summary = summarizer(
        seg["text"],
        max_length=70,
        min_length=30,
        do_sample=False
    )[0]["summary_text"]

    # Title (short summary used as title)
    title = summarizer(
        seg["text"],
        max_length=15,
        min_length=5,
        do_sample=False
    )[0]["summary_text"]

    keywords = extract_keywords(seg["text"])

    final_results.append({
        "segment_id": idx,
        "title": title,
        "start_time": f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}",
        "end_time": f"{int(seg['end']//60):02d}:{int(seg['end']%60):02d}",
        "keywords": list(keywords),
        "summary": summary
    })

# Save JSON file
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4, ensure_ascii=False)

print("✅ Topic segmentation with titles completed!")
print(f"📄 JSON output saved to: {output_json}")
