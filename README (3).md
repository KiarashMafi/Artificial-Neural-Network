# Artificial Neural Network for Customer Churn Prediction

This repository contains a basic implementation of an Artificial Neural Network (ANN) using TensorFlow/Keras to predict customer churn based on structured data.

## 📊 Dataset

The dataset used is `Churn_Modelling.csv`, which includes information about bank customers such as:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

## 🧪 Model Summary

- Framework: TensorFlow/Keras
- Architecture:
  - Input layer: One-hot encoded categorical variables and scaled numerical features
  - Hidden Layer 1: 6 units, ReLU
  - Hidden Layer 2: 6 units, ReLU
  - Output Layer: 1 unit, Sigmoid (for binary classification)

## ⚙️ Workflow

1. **Data Preprocessing**:
   - One-hot encoding of categorical features (Geography, Gender)
   - Feature scaling using `StandardScaler`

2. **Model Building**:
   - Sequential model with 2 hidden layers and 1 output layer

3. **Training**:
   - Optimizer: `adam`
   - Loss function: `binary_crossentropy`
   - Epochs: 10
   - Batch size: 32

4. **Evaluation**:
   - Single prediction on a sample
   - Predictions on the test set
   - Confusion matrix and accuracy score

## 📦 Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow

## 🚀 Usage

1. Place the dataset in the correct path (update the `read_csv()` path if needed).
2. Run the notebook `ANN.ipynb`.

## 🧠 Sample Prediction

```python
sample = [[1, 0, 0, 1, 0, 600, 40, 3, 60000, 2, 1, 1, 50000]]
sample = sc.transform(sample)
y_pred = model.predict(sample)
```

## 📈 Evaluation Metrics

The model outputs:
- Accuracy score on the test set
- Confusion matrix

---

**Note**: Adjust paths and sample input based on your environment.
