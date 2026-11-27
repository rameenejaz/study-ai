# StudyMate AI

StudyMate AI is a study assistant built with Machine Learning and Transformer models. A student uploads lecture notes (PDF or TXT, or pastes them in) and gets:

1. **Topic classification.** A TF-IDF + Logistic Regression model trained on AG News.
2. **Smart summary.** DistilBART summarises the *whole* document, at a Short, Medium or Detailed length.
3. **Interactive flashcards.** Question on the front, answer on the back, with navigation and progress.
4. **Practice MCQ quiz.** Four options (A–D), one correct answer, an explanation, a score and a restart button.
5. **Downloads.** Summary (TXT), flashcards (CSV/TXT), MCQs, answer key (TXT) and all MCQs (CSV).

**Live app:** _add your `https://….streamlit.app` link here after deploying_

---

## 1. What was wrong before, and what changed

| Problem in the previous version | Root cause | Fix |
|---|---|---|
| A short note came back as its first paragraph | DistilBART-CNN was trained on news, where the opening sentences carry the story ("lead bias"). The whole note went in as one input, so the model copied the opening | Notes are split at sentence boundaries into ≥2–3 sections (even short ones). Every section is summarised. Unsupported, repeated or cut-off sentences are removed, and a section-wise second pass shortens long results |
| Long notes lost their later parts | Chunks were cut every 550 *words* and only `chunks[:10]` were used, so anything after about 5,500 words was silently dropped. The final pass squeezed everything into one 120-token output | Chunks are sized with the model's real tokenizer, and none are skipped. An input that is too long raises an error instead of being cut off silently |
| The length sliders barely mattered | The same `max_length` was used for every chunk and for the final pass, and `max_length` is only an upper limit | Each length setting controls the target word budget, section size, per-section min/max new tokens and the second pass. The UI explains that a maximum is not a guaranteed length and reports the actual word count |
| "Flashcards" were one paragraph | FLAN-T5-*small* (77M parameters) was asked for "6 flashcards" in one prompt, given only the first 2,500 characters, and its raw text was displayed with no checks | Definitions and key facts are extracted from every section. FLAN-T5-*base* writes one question at a time for a known answer. Each output is parsed, validated and verified, retried with a corrective prompt, and falls back to fill-in-the-blank. Duplicates are removed. Every answer is text from the notes |
| MCQs were plain text with no quiz | Same raw-generation approach, with no options to validate | Correct answers come from the notes. The three distractors are other concepts from the same notes, of the same kind. Each MCQ is checked to have exactly 4 unique options, exactly one correct answer and an explanation. The quiz is interactive |
| Results disappeared after clicking anything | Results were only shown inside `if st.button(...)`, so the next rerun (e.g. clicking Download) hid them | All results live in `st.session_state`. Downloads use `on_click="ignore"`. Card and quiz positions survive reruns |
| Notes could be reset unexpectedly | The text area's value was re-extracted on every run | A file is extracted only when a *new* file is uploaded. Edits are kept, and a "Restore original" button is provided |
| Classifier crashed on string labels | `int(clf.predict(...)[0])` assumed integer class ids | Both integer ids (mapped through `model_metadata.json`, with a range check) and string labels are handled. A confidence is shown only because Logistic Regression has `predict_proba`. A model without it shows no confidence |

### Your trained model was kept

`artifacts/best_classifier.joblib` is your notebook's **Logistic Regression** pipeline (best macro-F1 = 0.9068, accuracy = 0.9070 on the 7,600 AG News test articles). It is loaded as-is and never retrained. Only `model_metadata.json` gained four extra fields (`sklearn_version`, `sklearn_version_note`, `preprocessing`, `label_mapping`); its results are unchanged.

**Compatibility finding:** the pickle was saved with **scikit-learn 1.6.1** and **NumPy 2.x** (it references `numpy._core`). Loading it with other versions gives warnings or errors, so `requirements.txt` pins `scikit-learn==1.6.1` and `numpy>=2`. If you ever need different versions, retrain with `train.py` (same settings as the notebook) instead of loading the old pickle.

---

## 2. Architecture

```
StudyMate_AI_Streamlit/
├── app.py                      Streamlit UI + session state (no ML logic)
├── requirements.txt
├── train.py                    Retraining script (same settings as the notebook)
├── .streamlit/config.toml      Theme + 25 MB upload limit
├── artifacts/                  Your trained model (unchanged) + metadata
├── modules/
│   ├── preprocessing.py        clean_text (identical to training), sentence splitting, chunking, similarity
│   ├── extraction.py           PDF (PyMuPDF, every page, paragraphs kept) and TXT extraction + errors
│   ├── classification.py       Loads the joblib pipeline, label mapping, confidence, per-section topics
│   ├── models.py               Loads DistilBART / FLAN-T5 on CPU; safe generate() that never truncates silently
│   ├── summarization.py        Map → check → reduce summarisation with length presets
│   ├── facts.py                Definition and key-fact extraction, key phrases, section round-robin
│   ├── question_writer.py      FLAN-T5 prompts, output parsing, validation, verification, retry, fallback
│   ├── flashcards.py           Flashcard pipeline, validation, CSV/TXT export
│   └── mcqs.py                 MCQ pipeline, distractors, validation, TXT/CSV export
├── tests/                      114 pytest tests (see §8)
├── scripts/check_real_models.py  Runs the real DistilBART/FLAN-T5 on the samples and checks the acceptance tests
├── sample_data/                OS (2,017 words, TXT + PDF), ML, Business, Football, 118-word note
└── notebooks/StudyMate_AI.ipynb
```

**Flow:** upload → `extraction` → editable notes → each tab calls its module → result stored in `st.session_state` → rendered and downloadable.

---

## 3. Models

| Task | Model | Why |
|---|---|---|
| Topic classification | TF-IDF (1–2-grams, 30,000 features) + Logistic Regression, from your notebook | Best macro-F1 of the 5 algorithms compared (NB, LR, Linear SVM, RF, k-NN), and it gives real probabilities |
| Summarisation | `sshleifer/distilbart-cnn-12-6` (306M params, 1,024-token input) | The same model your notebook evaluated with ROUGE. Good abstractive quality on CPU |
| Question writing | `google/flan-t5-base` (248M params, 512-token input) | Your notebook used `flan-t5-base`. The app previously used `-small` (77M), which was too weak to follow instructions. Base is still CPU-friendly |

**Memory:** Streamlit Community Cloud guarantees about 1 GB and allows up to about 2.7 GB. The app keeps **only one Transformer in memory at a time** (`st.cache_resource(max_entries=1)`): the summariser and the question writer swap when you change tabs (about 10 s to reload from disk). If the app still hits the memory limit, add these in *App settings → Secrets/Environment*:

```
STUDYMATE_SUMMARY_MODEL = "sshleifer/distilbart-cnn-6-6"   # 230M params: faster, slightly lower quality
STUDYMATE_QG_MODEL = "google/flan-t5-small"                # 77M params: weaker questions; more fall back to fill-in-the-blank
```

**How FLAN-T5 is used:** FLAN-T5 does not reliably produce structured JSON or long lists, so it is never asked for "10 flashcards". It is asked for **one** question for a known answer taken from the notes. Its output is parsed (prefixes, quotes, JSON wrappers and missing "?" are repaired), then validated. The question must be a wh-question, must not reveal the answer, must not copy the sentence, and must use the notes' vocabulary. Next, it is verified: the model answers its own question, and the answer must match. If any check fails, the question is retried with a corrective prompt and then replaced by a fill-in-the-blank question. Definition cards ("What is a semaphore?") use templates. The app reports honestly when it cannot make the requested number.

---

## 4. Dataset

* **AG News** (`fancyzhx/ag_news`): 120,000 training and 7,600 test news articles in 4 balanced classes (World, Sports, Business, Sci/Tech). Your model was trained on 32,000 articles (8,000 per class) and evaluated on all 7,600 test articles.
* The classifier predicts **broad news topics, not course names**. Operating Systems or Machine Learning notes are classified as Sci/Tech, football notes as Sports, and management notes as Business.
* **SAMSum** (validation set) is used only in the notebook for the ROUGE evaluation of the summariser.

---

## 5. Installation and local run (macOS)

The pinned scikit-learn 1.6.1 has installers for **Python 3.9–3.13** only. Check your version first:

```bash
python3 --version
```

If it shows 3.14 or newer, install Python 3.12 with `brew install python@3.12` and use `python3.12` in place of `python3` below.

```bash
cd ~/Downloads/StudyMate_AI_Streamlit          # the folder from the ZIP
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The app opens at http://localhost:8501. The first summary or flashcard request downloads the Transformer models (about 1 GB each) into `~/.cache/huggingface`. Later runs load them from disk. Stop the app with `Ctrl + C`.

Each time you come back to the project, run only these:

```bash
cd ~/Downloads/StudyMate_AI_Streamlit
source .venv/bin/activate
python3 -m streamlit run app.py
```

### Model artifact requirements

The app needs these files in `artifacts/`:

| File | Required | Purpose |
|---|---|---|
| `best_classifier.joblib` | **yes** | Full scikit-learn pipeline (TF-IDF + Logistic Regression) |
| `model_metadata.json` | recommended | Label names for class ids 0–3, results table, scikit-learn version check |
| `tfidf_vectorizer.joblib` | only if the classifier file contains a bare classifier | The pipeline already includes the vectorizer |
| `model_comparison.csv` | no | Notebook results (for the report) |

If `best_classifier.joblib` is missing, the Topic tab shows a clear error naming the file, and the other tools still work. To rebuild the artifacts: `python3 -m pip install datasets pandas && python3 train.py`.

---

## 6. Upload to GitHub

The ZIP still contains your repository's `.git` folder (remote `https://github.com/rameenejaz/study-ai.git`). Replace your old project folder with the new one, or copy the new files into it, then:

```bash
cd ~/Downloads/StudyMate_AI_Streamlit
git status                                   # shows the new and changed files
git add .
git commit -m "Rebuild StudyMate AI: full-document summaries, real flashcards and MCQ quiz, tests"
git push origin main
```

If `git push` is rejected with "fetch first", run `git pull --rebase origin main` and then push again.

Finder hides folders whose names start with a dot. If you upload through the GitHub website instead of `git`, make sure `.streamlit/config.toml` is included: press `Cmd + Shift + .` to show hidden files, or create the file on GitHub with **Add file → Create new file** and the name `.streamlit/config.toml`. The app still works without it; only the colour theme changes.

---

## 7. Deploy on Streamlit Community Cloud

1. Push the repository (§6). The `artifacts/` files must be in the repo; they are about 3.3 MB in total.
2. Open https://share.streamlit.io and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Choose the repository `rameenejaz/study-ai`, branch `main`, main file `app.py`.
5. Open **Advanced settings** and choose **Python 3.12**.
6. Click **Deploy**. Installing the packages takes several minutes.
7. Open the app and run each tool once. The first summary and the first flashcard/quiz request each download a model, which takes 1–3 minutes.
8. Paste the URL at the top of this README and in the notebook.

**If the build fails:**
* An error mentioning `download.pytorch.org`: delete the `--extra-index-url` line from `requirements.txt`. torch then installs from PyPI; the install is bigger but works the same on CPU.
* "This app has gone over its resource limits": use the smaller models (§3) and reboot the app from **Manage app**.

---

## 8. Testing

```bash
source .venv/bin/activate
python3 -m pytest -q                       # all automated tests (~40 s)
python3 scripts/check_real_models.py       # real DistilBART + FLAN-T5 on the samples (downloads ~2 GB the first time)
```

### What has actually been run

All results below are from `python3 -m pytest` on Python 3.13, scikit-learn 1.6.1, transformers 4.57, torch 2.15 and Streamlit 1.67: **114 passed, 0 failed.**

| File | Tests | Covers |
|---|---|---|
| `test_extraction.py` | 18 | Multi-page PDF, the 4-page sample PDF extracting word-for-word like the TXT (ligatures such as "ﬁ" fixed), line-end hyphens, paragraph boundaries, blank pages, empty PDF, scanned PDF (OCR message), corrupted and 0-byte PDF, UTF-8/UTF-16/CP1252 TXT, empty/binary TXT, unsupported type, sentence splitting |
| `test_classification.py` | 16 | Your real artifacts on OS, ML, Business and Football notes (Sci/Tech, Sci/Tech, Business, Sports), a World example, per-section topics, missing and corrupted model files, string labels, inconsistent label ids, model without probabilities, bare classifier + vectorizer, version warning |
| `test_summarization.py` | 15 | Short summaries, **118-word note not reduced to its first paragraph**, **every sentence of the 2,017-word note reaches the model**, coverage across sections, token limits, **length setting changes the budgets and output**, invented and repeated sentences removed, model failure, too-short input |
| `test_flashcards.py` | 28 | **10 valid cards from each of the 4 domains** (with and without the AI writer), 5/10/15/20 counts, whole-document coverage, no duplicates, honest shortfall report, garbage/empty/answer-revealing/wrong/JSON model outputs, retry before fallback, parsing, validation, exports |
| `test_mcqs.py` | 24 | **10 valid MCQs from each domain**, 5/10/15/20, exactly 4 unique options, one correct answer, correct-answer validation, explanation, letters spread A–D, invalid model outputs, distractors from the notes, exports |
| `test_models_integration.py` | 5 | The real Hugging Face `generate()` path with tiny randomly initialised T5/BART models built offline. Random output is rejected and never shown. Inputs that are too long are refused. Missing models give a clear download error |
| `test_app_ui.py` | 8 | The real Streamlit app run headless: layout, sample loading, edits surviving reruns, topic tab, missing-artifact error, **length setting in the UI**, **interactive flashcards (show/hide, previous/next, place kept across reruns)**, **full quiz flow (select → submit → feedback → next → score 2/5 = 40% → restart)** |

**Also checked by hand in a real browser (Chrome, headless):** uploading `sample_data/os_lecture_2000_words.pdf` fills the notes with all 2,017 words. After editing the notes and running Predict topic, the edits were still there and were not overwritten. All four tabs were exercised, using the stand-in models for the Transformer tabs.

**Not tested automatically:** the writing quality of the real DistilBART and FLAN-T5. The test machine could not download models from Hugging Face, so the summary, flashcard and MCQ tests use **stand-in models** (`tests/fakes.py`). The stand-in summariser deliberately copies opening sentences to mimic lead bias. These tests prove that the pipeline is correct: chunking, coverage, budgets, parsing, validation and fallbacks. They do not prove the quality of the real models' text. Run `scripts/check_real_models.py` on your Mac to check the real models; it prints PASS/FAIL for the acceptance tests.

**Not done:** the app has not been deployed or checked on Streamlit Community Cloud by me. Follow §7.

---

## 9. Known limitations

* **AG News topics only.** The classifier chooses among World, Sports, Business and Sci/Tech, so most academic notes are Sci/Tech.
* **No OCR.** Scanned PDFs are detected and explained but cannot be read.
* **Speed on free CPU hosting.** A 2,000-word Detailed summary runs the model on about 12 sections plus a second pass, so expect a few minutes on Streamlit Cloud; flashcards and MCQs should take well under a minute. These are estimates — I could not time the real models (see §8).
* **Summary length is approximate.** `max_new_tokens` is an upper limit. The model can stop early, and rejected sentences make the result shorter. The UI shows the target and the actual word count.
* **Short notes give fewer items.** A 118-word note cannot give 20 distinct flashcards. The app says how many valid items it made instead of padding the list.
* **Fill-in-the-blank fallback.** When FLAN-T5's question fails validation, a fill-in-the-blank question is used. It is always correct, but less varied.
* **Distractors can occasionally be arguable.** Distractors are other concepts from your notes, checked not to appear in the fact sentence, but they are not semantically verified. Always check against your notes.
* **English only.** The sentence splitting, key-phrase rules and models assume English text.
