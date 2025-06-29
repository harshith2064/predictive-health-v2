import wikipedia
from textblob import TextBlob

# 🔒 Manual keyword-to-topic mappings
manual_fallbacks = {
    "bp": "Hypertension",
    "what is bp": "Hypertension",
    "blood pressure": "Hypertension",
    "sugar disease": "Diabetes",
    "diabetics": "Diabetes",
    "diabetes": "Diabetes",
    "cancer diagnosis": "Cancer",
    "heart pain": "Heart attack",
    "heart disease": "Cardiovascular disease",
    "liver disease": "Liver disease",
    "liver desease": "Liver disease",
    "stroke": "Stroke (medicine)",
    "insulin": "Insulin",
    "piles": "Hemorrhoid",
    "hemorrhoids": "Hemorrhoid",
    "back pain": "Low back pain"
}

# 🚫 Unsafe keywords to block accidental wrong summaries
bad_keywords = ["slur", "offensive", "racial", "ethnic", "insult"]

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

def is_safe_summary(summary):
    return not any(bad in summary.lower() for bad in bad_keywords)

def get_faq_response(user_input):
    try:
        query = user_input.strip().lower().rstrip("?.!")
        corrected = correct_spelling(query)

        # 🔎 Step 1: Match manually
        for key in manual_fallbacks:
            if key in corrected:
                topic = manual_fallbacks[key]
                try:
                    summary = wikipedia.summary(topic, sentences=2)
                    return summary if is_safe_summary(summary) else None
                except:
                    continue

        # 🔎 Step 2: Wikipedia search fallback
        search_results = wikipedia.search(corrected)
        if not search_results:
            return None

        for title in search_results:
            if any(word in title.lower() for word in corrected.split()):
                summary = wikipedia.summary(title, sentences=2)
                if is_safe_summary(summary):
                    return summary

        summary = wikipedia.summary(search_results[0], sentences=2)
        return summary if is_safe_summary(summary) else None

    except Exception:
        return None


