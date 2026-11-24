# StudyMate AI — Streamlit Edition

## Features
- PDF/TXT upload and text extraction using PyMuPDF.
- Broad topic classification using the trained TF-IDF + classical ML pipeline.
- Abstractive summarization using `sshleifer/distilbart-cnn-12-6`.
- Flashcard and MCQ generation using `google/flan-t5-small`.
- Streamlit interactive web interface.

## Dataset transparency
The classifier uses AG News labels: World, Sports, Business and Sci/Tech. It predicts broad text topics, not university course subjects. The Colab notebook performs model comparison and ROUGE evaluation; ROUGE is not computed live in this app.

## Required repository structure
```text
studymate-ai/
├── app.py
├── requirements.txt
└── artifacts/
    ├── best_classifier.joblib
    ├── tfidf_vectorizer.joblib
    └── model_metadata.json
```

## Deploy on Streamlit Community Cloud
1. Push these files and the `artifacts/` directory to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select Create app, choose your repository, branch and `app.py`.
4. Deploy. The first start downloads the pretrained Transformer models.
5. Add the resulting public `streamlit.app` URL at the top of your Colab notebook.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app caches models and uses FLAN-T5-small to reduce memory usage on free CPU hosting. Transformer inference may still be slow or exceed free-tier memory limits. AI-generated content should be checked against the original notes.
# study-ai
