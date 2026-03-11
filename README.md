# Fake News Detection
### NLP pipeline with two deployments - lightweight cloud demo and full local CNN with explainability.

![Stack](https://img.shields.io/badge/Stack-TensorFlow%20%7C%20Scikit--learn%20%7C%20Streamlit-3776AB?style=flat&logo=python&logoColor=white)
![Live](https://img.shields.io/badge/Live-Cloud%20Demo-brightgreen?style=flat)
![Explainability](https://img.shields.io/badge/Explainability-LIME-orange?style=flat)

---

## Live Demo

[Launch Cloud App →](https://fake-news-detection-app-adptswkkruuf4keteyadn6.streamlit.app)

---

## Two Versions, One Repo

This repo ships two implementations with a deliberate reason for each.

| | Cloud Version | Local Version |
|-|--------------|---------------|
| Models | Logistic Regression + TF-IDF | CNN + Logistic Regression + LIME |
| Purpose | Public demo, fast and stable | Full deep learning pipeline + explainability |
| Deployment | Streamlit Cloud | Run locally |
| Use case | Sharing, quick testing | Academic evaluation |

The cloud version is lightweight by design - TensorFlow CNN models don't run reliably on cloud platforms. The local version is where the real pipeline lives.

---

## Local Version - Full Pipeline

Three components working together:

- **CNN** - deep learning classifier trained on news-style text
- **Logistic Regression** - baseline comparison model
- **LIME** - explainability layer showing which words drove the prediction

```bash
streamlit run app.py
```

**Requirements:**
```
Python 3.10
TensorFlow 2.10
Streamlit
scikit-learn
LIME
```

---

## Repo Structure

```
Fake-News-Detection-app/
├── local_app/
│   ├── app.py                    # Streamlit app (CNN + LR + LIME)
│   ├── models/
│   │   ├── advanced_cnn_model.h5
│   │   ├── log_reg.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   └── tokenizer.pkl
│   └── notebook/                 # Training and experimentation
└── cloud_app/
    ├── app.py                    # Streamlit app (Logistic Regression only)
    ├── log_reg.pkl
    ├── tfidf_vectorizer.pkl
    └── requirements.txt
```

---

## Note on Input

Models are trained on news-style text. Casual or conversational input may return unexpected results - that's a data distribution issue, not a model failure.

---

**Bhavya Pandya** · [LinkedIn](https://www.linkedin.com/in/bhavya-91p/) · M.S. Data Analytics, LIU Brooklyn
