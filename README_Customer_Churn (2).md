
# Customer Churn Prediction for Mobile Money Platforms  
### A Deep Learning Approach Using LSTMs & Sequential User Behavior Modeling  

Author: **Everlyn Musembi**  
Exhibit **3.14 – Customer Churn Prediction Project Results**

---

## 1️⃣ Overview

This project develops a **deep learning–based customer churn prediction system** designed specifically for mobile money platforms.  
It leverages **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** architectures to analyze sequential user behavior and transactional patterns to predict churn risk early.

The model analyzes:

- Transaction sequences  
- Mobile money wallet activity  
- Behavioral time-series trends  
- Frequency & recency of usage  

By using LSTMs, the system captures **long-term dependence patterns** that traditional ML methods often miss.

---

## 2️⃣ Key Technical Contributions

| Model | Accuracy | Recall (Churn) | AUC |
|------|---------|----------------|------|
| Logistic Regression | 0.71 | 0.52 | 0.68 |
| Random Forest | 0.78 | 0.61 | 0.74 |
| RNN (GRU) | 0.84 | 0.72 | 0.82 |
| **LSTM (Final Model)** | **0.89** | **0.81** | **0.90** |

✔️ Designed sequence-modeling pipelines for mobile money behavioral data  
✔️ Implemented RNN and LSTM architectures using TensorFlow/Keras  
✔️ Built sequence windowing for variable-length transaction logs  
✔️ Engineered temporal, behavioral, and financial features  
✔️ Achieved **significant uplift** vs traditional ML approaches  
✔️ Produced user-level churn risk scores and explanations  

---

## 📁 Folder Structure

```
customer-churn-lstm/
│
├── data/
│   ├── user_activity_sequences.csv
│   ├── mobile_money_logs.csv
│
├── notebooks/
│   ├── 01_sequence_preprocessing.ipynb
│   ├── 02_lstm_model_training.ipynb
│   ├── 03_rnn_baseline_compare.ipynb
│   ├── 04_evaluation_visualization.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── sequence_generator.py
│   ├── lstm_model.py
│   ├── rnn_model.py
│   ├── train.py
│   └── predict.py
│
├── results/
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── churn_risk_heatmap.png
│   ├── architecture_diagram.png
│
├── requirements.txt
└── README.md
```

---

## 🧠 Architecture Diagram

### Architecture Diagram
<img src="results/architecture_diagram.png" width="650">

**Explanation:**  
Raw mobile money logs are transformed into sequential user behavior windows, processed through an **LSTM churn prediction model**, and converted into churn-probability outputs with actionable retention recommendations.

---

## 📊 Evaluation Results

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

## 3️⃣ Data & Feature Engineering

This project processes **mobile money behavioral datasets**, including:

- Transaction timestamps  
- Wallet inflow/outflow behavior  
- Failed logins & PIN resets  
- Transaction frequency patterns  
- Gaps between usage sessions  

Key engineered features:

- `txn_count_7d`, `txn_count_30d`
- Days since last transaction  
- Rolling transaction averages  
- Behavioral sequence embeddings  
- Wallet financial flow normalization  

Sequences are processed using **time-windowing and sequence padding** to support neural network training.

---

## 4️⃣ Machine Learning Models Used

### Traditional ML Baselines
Used for benchmarking:
- Logistic Regression  
- Random Forest  
- Gradient Boosting  

### Deep Learning Models

✔ **RNN**
• captures short-term behavior  
• suitable for simpler patterns  

✔ **GRU**
• faster training  
• handles noisy financial logs  

✔ **LSTM (Final Model)**  
• captures long-range dependencies  
• learns complex transactional behavior trends  
• best recall for churn detection  

---

## 5️⃣ Real-World Application & Industry Relevance

Modern mobile money and digital financial platforms face severe financial loss due to churn.  
This system supports:

- Predictive retention strategies  
- Revenue protection  
- Early churn intervention  
- Improved customer engagement  
- Financial inclusion stability  

Benefiting sectors:

- Digital banking / neobanks  
- Mobile payment platforms (CashApp, Venmo, PayPal)  
- Telecom-based wallets  
- FinTech analytics  
- Risk management teams  

---

## 🧾 Sample Model Output

Example churn explanation:

> **User 11203** shows declining transaction frequency over 6 weeks, reduced cash-in amounts, and two failed PIN resets.  
> Sequence patterns match churn trajectories observed in cluster segments 4 and 7.  
> **Predicted Churn Probability: 0.86**

---

## 6️⃣ Business Impact

✔ Reduced churn by **~17%–25%** in simulation tests  
✔ Early retention trigger capability  
✔ Reduced cost of reacquisition  
✔ Strengthens user financial participation  
✔ Supports durable mobile economy ecosystems  

---

## 7️⃣ Deployment & Integration (Future Enhancements)

Planned improvements:

- Attention-based transformers (BERT4Rec / Transformer-XL)
- Real-time scoring APIs
- Kafka streaming + TensorFlow Serving
- Advanced explainability (SHAP for sequences)
- Incorporating demographics & socio-behavioral metadata

---

## 8️⃣ EB-2 NIW Relevance

This project strengthens the U.S. digital financial infrastructure by:

- Enhancing financial system resilience  
- Supporting secure mobile money ecosystems  
- Advancing AI innovation in FinTech  
- Protecting consumers through proactive engagement  

It demonstrates **substantial merit and national importance** through advanced AI application in financial systems.
