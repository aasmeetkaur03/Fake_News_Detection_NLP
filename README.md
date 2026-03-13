# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Overview
This project is an **End-to-End Natural Language Processing (NLP) Pipeline** designed to classify News Articles as **Fake** or **True** using Machine Learning.

It demonstrates a practical real-world Text Classification Workflow including :

- Data ingestion and labeling
- Text preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature extraction using **TF-IDF**
- Model training using **Passive Aggressive Classifier**
- Evaluation with classification metrics and confusion matrix
- Real-world prediction on unseen news articles

---

## 🚀 Why This Project Matters
Misinformation spreads rapidly across digital platforms, making Fake News Detection an important Machine Learning Problem.

This project simulates how machine learning can be used to automatically detect potentially misleading news content based on textual patterns.

It demonstrates Practical Skills in :

- **Natural Language Processing (NLP)**
- **Text cleaning and preprocessing**
- **Feature engineering for text data**
- **Supervised machine learning**
- **Model evaluation and interpretation**
- **Reusable inference pipeline for real-world prediction**

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **WordCloud**
- **Jupyter Notebook**

---

## 📂 Dataset
The project uses Two Labeled Datasets :

- **Fake.csv** → Contains fake news articles
- **True.csv** → Contains true / authentic news articles

### Label Encoding
- `0` → Fake News
- `1` → True News

The Datasets are merged into a single labeled corpus for Supervised Learning.

---

## ⚙️ Project Workflow

### 1. Data Loading
- Loaded `Fake.csv` and `True.csv`
- Added labels to each dataset
- Combined both datasets into a single dataframe
- Shuffled data to avoid ordering bias

### 2. Data Cleaning & Preprocessing
Performed preprocessing on textual content to improve model quality :

- Converted text to lowercase
- Removed URLs
- Removed punctuation
- Removed numeric values
- Normalized extra spaces
- Filled missing values in:
  - `title`
  - `text`
  - `subject`

### 3. Feature Engineering
To capture richer semantic context, the model uses a combined text representation :

- `title + text`

This combined text is transformed using **TF-IDF Vectorization**.

### 4. TF-IDF Configuration
The vectorizer is configured for strong performance on text classification tasks :

- `stop_words='english'`
- `max_df=0.8`
- `min_df=3`
- `ngram_range=(1, 2)`
- `sublinear_tf=True`
- `max_features=50000`

### 5. Model Training
The project uses a **Passive Aggressive Classifier**, which is highly effective for sparse, high-dimensional text data.

#### Why Passive Aggressive Classifier?
- Efficient for large text datasets
- Performs well with TF-IDF features
- Fast training and inference
- Suitable for online/stream-style learning tasks

### 6. Model Evaluation
The trained model is evaluated using :

- **Accuracy Score**
- **Precision**
- **Recall**
- **F1-Score**
- **Classification Report**
- **Confusion Matrix**

### 7. Real-World Inference
A reusable function is implemented :

```python
predict_news(news_text)
```

This function allows custom news content to be classified as :

- **Fake News**
- **True News**

---

## 📊 Exploratory Data Analysis (EDA)
The project includes visual analysis to better understand the dataset :

- Class distribution of fake vs true news
- Article length distribution
- Average text length by class
- Word clouds for:
  - Fake news articles
  - True news articles

---

## 📈 Results
The model demonstrates strong performance on the held-out test split for distinguishing Fake and True News Articles.

> **Note :** Since fake news detection is a complex real-world problem, predictions should be interpreted as model-based classifications rather than absolute truth verification.

---

## 💡 Key Features
- End-to-end NLP classification pipeline
- Text cleaning and preprocessing
- TF-IDF based feature extraction
- Efficient machine learning classifier
- EDA with visual insights
- Confusion matrix and classification metrics
- Real-world prediction function for unseen inputs
- Structured and reproducible project workflow

---

## 🔮 Future Improvements
To make this project more Production-Ready and Industry-Grade, Future Enhancements could include :

- Deploying the model using **Streamlit** or **Flask**
- Saving trained artifacts using:
  - `joblib`
  - `pickle`
- Comparing multiple models:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM
  - Random Forest
- Hyperparameter tuning using **GridSearchCV**
- Adding **cross-validation**
- Performing **lemmatization / stemming**
- Improving text normalization
- Trying **deep learning models**:
  - LSTM
  - BiLSTM
  - BERT / DistilBERT
- Building a real-time fake news detector interface

---

## ⚠️ Important Note
This project is intended for **Educational and Portfolio Purposes**.

Although the model can classify news text based on learned patterns, fake news detection in the real world is a nuanced problem involving :

- context
- source credibility
- evolving narratives
- bias in training data

Therefore, predictions should be treated as **model outputs**, not Definitive Factual Judgments.

---

## 👩‍💻 Author
**Aasmeet Kaur**

- Passionate about Machine Learning, NLP and Building Practical End-to-End Projects
- Focused on Writing Clean, Structured and Real-World Applicable Solutions

If you found this Project Useful..Feel Free to ⭐ the Repository!
