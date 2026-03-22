# Assignment 2: Advanced IMDB Sentiment Analysis

## 1. Project Overview
This project aims to improve upon initial sentiment analysis results by exploring advanced neural network architectures and optimization techniques. We evaluated four distinct approaches on the IMDB dataset (50,000 movie reviews) to identify the most effective method for binary sentiment classification.

---

## 2. Key Techniques & Methodology

### Data Preparation
- **Text Cleaning**: Standardized text by lowercasing, removing HTML tags, and normalizing whitespace.
- **Sequence Modeling**: Tokenized text into sequences of 10,000 most frequent words and padded them to a uniform length of 200 tokens.
- **Dataset Split**: Utilized a consistent 70/15/15 split for training, validation, and testing.

### Advanced Strategies
- **Optimization**: Implemented **Adam** and **SGD** optimizers with **Early Stopping** (patience=2) to ensure efficient convergence and prevent overfitting.
- **Hyperparameter Tuning**: Used `GridSearchCV` to exhaustively test MLP configurations (hidden layers, batch sizes, learning rates).
- **Transfer Learning**: Integrated pretrained **Word2Vec** embeddings from Google News (300 dimensions) to leverage external semantic knowledge.

---

## 3. Model Architectures & Performance

### 3.1. Tuned Multi-Layer Perceptron (MLP)
- **Features**: Sentence-level lexicon features (VADER + TextBlob).
- **Best Configuration**: `alpha=0.001`, `batch_size=256`, `hidden_layer_sizes=(128, 64)`, `solver='sgd'`.
- **Test Accuracy**: **77.17%**

### 3.2. Long Short-Term Memory (LSTM)
- **Architecture**: 128D Embedding layer -> 64-unit LSTM (0.2 Dropout) -> 32D Dense -> 1D Sigmoid.
- **Key Strength**: Captures long-term sequential dependencies in text.
- **Test Accuracy**: **87.64%**

### 3.3. 1D Convolutional Neural Network (1D CNN)
- **Architecture**: 128D Embedding layer -> 128-filter Conv1D (kernel=5) -> Global Max Pooling -> 64D Dense -> 1D Sigmoid.
- **Key Strength**: Efficiently extracts local n-gram features and patterns.
- **Test Accuracy**: **90.36%** (Best Performer)

### 3.4. Pretrained Word2Vec + CNN
- **Architecture**: **Non-trainable** 300D Google News Embedding matrix -> 128-filter Conv1D -> Global Max Pooling -> 64D Dense -> 1D Sigmoid.
- **Key Strength**: Leverages massive external pretraining (100 billion words).
- **Test Accuracy**: **~84.15%**

---

## 4. Final Comparison Results

| Model | Validation Acc | Test Acc | Key Optimization |
| :--- | :--- | :--- | :--- |
| Baseline MLP | 77.84% | 77.01% | Lexicon Features |
| **Tuned MLP** | 77.67% | 77.17% | GridSearchCV |
| **LSTM** | 86.48% | 87.64% | RNN Sequence Modeling |
| **1D CNN** | **89.49%** | **90.36%** | Local Feature Extraction |
| **W2V CNN** | 85.12% | 84.15% | Pretrained Embeddings |

---

## 5. Visualizations & Artifacts
The following artifacts are generated in the `results/` directory:
- `model_comparison.png`: Bar chart visualizing the test accuracy across all implemented models.
- `confusion_matrix.png`: Prediction performance for the baseline model.
- `loss_curves.png`: Training history for the MLP classifier.

## 6. Conclusion
The **1D CNN** architecture emerged as the superior model for this dataset, achieving a test accuracy of **90.36%**. 