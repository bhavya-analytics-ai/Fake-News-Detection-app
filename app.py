import os
import streamlit as st
import numpy as np
import joblib
import json
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Optional: URL article extraction
try:
    from newspaper import Article
except Exception:
    Article = None

# Optional: live trend feed
try:
    import feedparser
except Exception:
    feedparser = None


# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================
# Load Model + Tokenizer
# =====================================
@st.cache_resource
def load_all():
    model_path = os.path.join(BASE_DIR, "advanced_cnn_model.h5")
    tok_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    cfg_path = os.path.join(BASE_DIR, "model_config.json")

    model = tf.keras.models.load_model(model_path, compile=False)
    tokenizer = joblib.load(tok_path)

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    return model, tokenizer, cfg["max_len"]


# Actually load model/tokenizer/max_len
model, tokenizer, max_len = load_all()


# =====================================
# Helper functions
# =====================================
def preprocess_text(text: str):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding="post")


def predict_news(text: str):
    padded = preprocess_text(text)
    prob = float(model.predict(padded)[0][0])

    is_fake = prob > 0.5
    label = "FAKE News" if is_fake else "REAL News"
    confidence = prob if is_fake else 1 - prob

    return label, confidence, prob


def extract_article(url: str):
    if Article is None:
        return None, "⚠️ newspaper3k not installed."

    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text.strip():
            return None, "Could not extract article text."

        return article.text, None

    except Exception as e:
        return None, f"❌ Error extracting article: {e}"


def get_trending_items():
    if feedparser is None:
        return [
            {"title": "5G towers cause COVID-19? (False)", "link": "https://www.politifact.com/"},
            {"title": "Cash giveaway hoax", "link": "https://www.snopes.com/"},
            {"title": "Vaccine misinformation", "link": "https://www.factcheck.org/"}
        ]

    try:
        feed = feedparser.parse("https://www.politifact.com/rss/factchecks/")
        items = [{"title": e.title, "link": e.link} for e in feed.entries[:5]]

        return items or [{"title": "Feed unavailable", "link": "https://www.politifact.com/"}]

    except:
        return [{"title": "Feed unavailable", "link": "https://www.politifact.com/"}]


# =====================================
# CSS Theme System
# =====================================
def inject_css(theme: str):
    if theme == "Dark":
        bg_grad = "linear-gradient(145deg, #050816, #111827)"
        card_bg = "#111827"
        text_color = "#e5e7eb"
        subtext_color = "#9ca3af"
        result_bg = "#020617"
        border_color = "#4f46e5"
    else:
        bg_grad = "linear-gradient(145deg, #f6f9ff, #e9efff)"
        card_bg = "#ffffff"
        text_color = "#111827"
        subtext_color = "#4b5563"
        result_bg = "#f1f5f9"
        border_color = "#3b82f6"

    st.markdown(
        f"""
        <style>
            body {{ background: {bg_grad}; }}
            .main-block {{
                background: {card_bg};
                padding: 2rem;
                border-radius: 18px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            }}
            .intro-box {{
                background: rgba(255,255,255,0.12);
                padding: 1.2rem 1.6rem;
                border-radius: 12px;
                text-align: center;
                color: {subtext_color};
                margin-bottom: 1.5rem;
            }}
            .result-box {{
                background: {result_bg};
                padding: 1.4rem 1.7rem;
                border-radius: 14px;
                border-left: 6px solid {border_color};
                margin-top: 1.2rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


# =====================================
# Sidebar
# =====================================
st.sidebar.markdown("## 🧭 Navigation")

theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"])
inject_css(theme_choice)

input_type = st.sidebar.radio("Input Mode", ["📝 Enter Text Manually", "🌐 Paste URL"])

st.sidebar.markdown("### 📰 Trending fact-checks")
for item in get_trending_items():
    st.sidebar.markdown(f"- [{item['title']}]({item['link']})")


# =====================================
# Title Section
# =====================================
try:
    logo = Image.open(os.path.join(BASE_DIR, "logo.png"))
    st.image(logo, width=170)
except:
    pass

st.markdown("<h1 style='text-align:center;'>📡 Fake News Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>AI-powered classifier for headlines, paragraphs, and article URLs</p>",
    unsafe_allow_html=True,
)


# =====================================
# Main block
# =====================================
with st.container():
    st.markdown("<div class='main-block'>", unsafe_allow_html=True)

    # -------- TEXT INPUT MODE --------
    if input_type == "📝 Enter Text Manually":
        st.subheader("✍️ Enter News Text")

        user_text = st.text_area(
            label="News text:",
            placeholder="Type or paste a news headline or short article here...",
            height=200,
            label_visibility="collapsed"
        )

        if st.button("Analyze Text"):
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                label, conf, raw_prob = predict_news(user_text)
                emoji = "🔴" if label.startswith("FAKE") else "🟢"

                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(f"<h3>{emoji} {label}</h3>", unsafe_allow_html=True)
                st.write(f"**Confidence:** {conf:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)

    # -------- URL INPUT MODE --------
    else:
        st.subheader("🌐 Paste Article URL")

        url = st.text_input(
            "URL",
            placeholder="https://www.example.com/news/article",
        )

        if st.button("Fetch & Analyze"):
            if not url.strip():
                st.warning("Please paste a URL.")
            else:
                if Article is None:
                    st.error("newspaper3k is not installed.")
                else:
                    with st.spinner("Extracting article..."):
                        text, err = extract_article(url)

                    if err:
                        st.error(err)
                    else:
                        preview = text[:700] + ("..." if len(text) > 700 else "")
                        st.write("### Extracted Text Preview")
                        st.write(preview)

                        label, conf, raw_prob = predict_news(text)
                        emoji = "🔴" if label.startswith("FAKE") else "🟢"

                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown(f"<h3>{emoji} {label}</h3>", unsafe_allow_html=True)
                        st.write(f"**Confidence:** {conf:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
