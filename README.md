# Sentiment Analysis using Machine Learning

## 📌 Project Overview
This project implements a Machine Learning pipeline for binary sentiment classification using the IMDb movie reviews dataset.

The objective is to classify reviews as Positive or Negative using text preprocessing, feature engineering, and Logistic Regression.

---

## 📊 Dataset
- Source: IMDb Movie Reviews Dataset (Kaggle)
- 50,000 labeled reviews
- Binary sentiment classification (Positive/Negative)

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 🧠 Methodology

1. Data Loading
2. Data Cleaning & Preprocessing
3. TF-IDF Feature Extraction
4. Train-Test Split (80/20)
5. Model Training (Logistic Regression)
6. Model Evaluation

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Example Output:

Accuracy: ~88-90% (depends on random state)

---

## ❗ Error Analysis

The model misclassified sarcastic or context-heavy reviews where sentiment depended on deeper semantic understanding.

For example:
"I expected this movie to be bad... but it was surprisingly good."

Traditional TF-IDF based models struggle with contextual nuance.

Future improvement:
- Use Word Embeddings (Word2Vec, GloVe)
- Fine-tune BERT Transformer model

---

## 📁 Project Structure
