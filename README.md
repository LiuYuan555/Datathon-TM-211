TM-211
Liu Yuan (liuy0317@e.ntu.edu.sg)
Boopathy Kowshik (kowshik001@e.ntu.edu.sg)
Kalepu Sai Sri Akshath (saisriak001@e.ntu.edu.sg)

# 🧠 Datathon TM-211 — Fetal Health Classification

This project trains and evaluates a machine learning model to classify fetal health status (Normal, Suspect, Pathological) based on cardiotocography (CTG) signals.  
It includes scripts for data preparation, model training, and model evaluation.

---

## 🚀 Getting Started

### 🧩 1. Set up your environment

Make sure you have **Python 3.8+** installed.

(Optional but recommended) — Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

---

### 📦 2. Install dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 🧠 3. Train the model

Run the training script to clean data, balance classes using SMOTE, and train a LightGBM model.

```bash
python train.py
```

This will:
- Train the model using the CTG dataset  
- Save the trained model (e.g. `lightgbm_ctg_model.pkl`)  
- Output evaluation metrics such as **Balanced Accuracy** and **Macro F1 Score**

---

### 🧪 4. Test the model

Once training is complete, test the model on the sample dataset:

```bash
python test.py
```

This will:
- Load the trained model  
- Load the test Excel file  
- Display predicted class probabilities, balanced accuracy, and macro F1 score

---

### 📊 5. Test with your own dataset

You can test the trained model on your own CTG Excel file.

edit test.py and replace function call argument with your own CTG excel file

test_model_on_excel("path/to/your_file.xlsx")

results will be saved as test_results_output.xlsx

✅ **Requirements for your Excel file:**
- Header consists of 24 features.

---

## 📁 Project Structure

```
📂 Datathon-TM-211
 ├── train.py                       # Training script
 ├── test.py                        # Testing script
 ├── requirements.txt               # Dependencies
 ├── README.md                      # Project documentation
 ├── CTG.xlsx                       # For training model in train.py
 ├── CTG_first10_combined.xlsx      # For sample usage of test.py
 ├── lightgbm_ctg_model.pkl         # Saved model (after training)
```

---

## ⚙️ Key Features

- ✅ Data preprocessing and cleaning  
- ⚖️ Class balancing using SMOTE  
- 🌲 Model training with LightGBM  
- 📈 Evaluation using Balanced Accuracy and Macro F1  
- 🔍 SHAP & interpretability support (optional)  
- 📊 Easy testing with your own Excel datasets  

---

## 🧮 Metrics Used

- **Balanced Accuracy:** Ensures fair performance across imbalanced classes.  
- **Macro F1 Score:** Measures overall precision and recall balance.  
- **Class Probabilities:** For each sample, predicted likelihood for NSP = 1, 2, or 3.

---