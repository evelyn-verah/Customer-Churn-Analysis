
# Customer Churn Prediction for Mobile Money Platforms  
### A Deep Learning Approach Using LSTMs & Sequential User Behavior Modeling

**Author:** Everlyn Musembi  
**Exhibit 3.14 – Customer Churn Prediction Project Results**

---

## 1. Project Overview
This project develops a deep learning–based customer churn prediction system designed specifically for **mobile money platforms**.

It leverages:
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) architectures

to analyze sequential user behavior and transactional patterns to predict churn risk early.

This project demonstrates my ability to:
- Design and implement RNNs and LSTMs for sequential financial data  
- Perform time-series feature engineering for transaction logs  
- Model user churn risk to support strategic retention decisions  
- Deliver practical, production-ready AI workflows for FinTech applications  

By using LSTMs, the system captures long-term dependency patterns that traditional ML models often miss.

---

## 2. Real-World Application & Industry Relevance
This churn prediction system can be deployed by:

- Digital banks and neobanks  
- Mobile wallet providers (Venmo, CashApp, PayPal)  
- Telecom-based mobile money platforms  
- FinTech analytics & customer engagement platforms  

### Business Impact
- Early identification of users at high risk of churn  
- Reduced acquisition costs  
- Optimized retention campaigns  
- Improved financial inclusion and user engagement  

---

## 3. Machine Learning Models Used
This project evaluates traditional machine learning approaches alongside advanced deep learning models to understand which techniques are best suited for predicting churn in mobile money environments.

---

### 3.1 Baseline Classical ML Models
Used for benchmarking performance:
- Logistic Regression  
- Random Forest  
- Gradient Boosting (XGBoost)  

**Limitation of classical ML:**  
They treat user data as static snapshots rather than evolving behavioral sequences, which limits their ability to detect gradual churn behavior.

---

### 3.2 Deep Learning Sequence Models

#### Recurrent Neural Network (RNN)
- Captures short-term sequential behavior  

#### Gated Recurrent Unit (GRU)
- Addresses vanishing gradient issues  
- Faster and efficient training  

#### Long Short-Term Memory (LSTM) — Final Production Model
- Captures long-term temporal dependencies  
- Detects gradual engagement decline over weeks/months  
- Provides the highest recall and stability  

This is why **LSTM was selected as the final model**.

---

### 3.3 Model Performance Comparison

| Model | Accuracy | Recall (Churn) | AUC |
|------|---------|----------------|------|
| Logistic Regression | 0.71 | 0.52 | 0.68 |
| Random Forest | 0.78 | 0.61 | 0.74 |
| RNN (GRU) | 0.84 | 0.72 | 0.82 |
| **LSTM (Final Model)** | **0.89** | **0.81** | **0.90** |

### Interpretation
- Traditional ML performed reasonably well  
- Deep learning delivered significantly stronger performance  
- **LSTM achieved the best recall**, which is critical because missing churners is more costly than slightly lower accuracy  

---

## 4. Data & Feature Engineering

Dataset: mobile_money_logs.csv

| Column | Description |
|--------|-------------|
| user_id | Unique customer identifier |
| day | Day index (1–90) |
| txn_count | Number of transactions |
| cashin | Total cash-in amount |
| cashout | Total cash-out amount |
| failed_login | Indicates account access/security issues |
| pin_reset | Security-related events |
| churned | Target label (1 = churned) |

See data_dictionary.md for full definitions.

### Engineered Features
- 7-day and 30-day rolling activity metrics  
- Transaction velocity  
- Behavioral decline curves  
- Event frequency embeddings  
- Sliding sequence windows for LSTM  

---

## 5. Repository Structure

```
customer-churn-lstm/
├── data/
│   ├── mobile_money_logs.csv
│   └── data_dictionary.md
│
├── notebooks/
│   ├── 01_sequence_preprocessing.ipynb
│   ├── 02_lstm_model_training.ipynb
│   ├── 03_rnn_baseline_compare.ipynb
│   └── 04_evaluation_visualization.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── sequence_generator.py
│   ├── lstm_model.py
│   ├── rnn_model.py
│   ├── train.py
│   ├── predict.py
│   └── hyperparam_search.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── churn_risk_heatmap.png
│   ├── architecture_diagram.png
│   └── example_predictions.csv
│
├── requirements.txt
└── README.md
```

---

## 6. Installation & Setup
```
git clone https://github.com/<your-username>/customer-churn-lstm.git
cd customer-churn-lstm
pip install -r requirements.txt
```

---

## 7. Model Training (LSTM)
```
cd src
python train.py
```

Outputs saved to:
```
results/lstm_model.h5
results/metrics.json
```

---

## 8. Hyperparameter Search
```
cd src
python hyperparam_search.py
```

Results saved to:
```
results/hyperparam_results.json
```

---

## 9. Inference & Predictions
```
cd src
python predict.py
```

Creates:
```
results/example_predictions.csv
```

---

## 🔷 Architecture Diagram
<img src="results/architecture_diagram.png" width="700">

**Explanation:**  
Raw mobile money logs are converted into sequential datasets, processed through the RNN/LSTM modeling pipeline, and outputted as churn probability predictions and actionable retention insights.

---

## 11. Evaluation Results

---

### Confusion Matrix
<img src="results/confusion_matrix.png" width="700">

**Summary:**  
The confusion matrix shows that the model accurately distinguishes between churners and non-churners, correctly identifying most loyal users (220) and a substantial portion of churners (65).  
This demonstrates strong predictive performance and reliable classification for real-world retention strategies.

---

### ROC Curve
<img src="results/roc_curve.png" width="700">

Shows strong balance between recall and precision, demonstrating excellent discriminatory power between churn and non-churn users.

---

### Churn Risk Heatmap
<img src="results/churn_risk_heatmap.png" width="700">

**Summary:**  
The churn risk heatmap highlights behavior patterns across user segments over time, revealing periods and groups with increased likelihood of churn. These insights enable targeted retention actions and proactive customer engagement.

---

## 12. Sample Model Interpretation Output
“User 11203 shows declining usage over six weeks, reduced transaction frequency, and multiple failed logins.  
Predicted churn probability: 0.86.  
This pattern aligns with high‑risk churn segments in the training set.”

---

## 13. National Interest Alignment (EB-2 NIW)
This project supports the U.S. national interest because it:

- Strengthens digital financial infrastructure  
- Enables proactive churn mitigation in U.S. FinTech platforms  
- Improves stability in mobile money ecosystems  
- Advances deep learning applications for consumer financial protection  
- Demonstrates expertise in mission-critical AI for the U.S. digital economy  

---

## 14. Future Enhancements
- Transformer models (BERT4Rec, Longformer)  
- Explainability methods (SHAP for LSTMs)  
- Real-time deployment with TensorFlow Serving  
- Integration with customer engagement pipelines  

---

## 15. Contact
**Everlyn Musembi**  
Machine Learning & FinTech Security  
LinkedIn: (Add your link)
