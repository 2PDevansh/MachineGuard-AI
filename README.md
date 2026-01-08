![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![MLOps](https://img.shields.io/badge/MLOps-MLflow-orange)
![RAG](https://img.shields.io/badge/GenAI-RAG-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-lightgrey)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

# **Predictive Maintenance AI with RAG-Powered Engineer Assistant**

An end-to-end **Predictive Maintenance System** that combines classical machine learning with a **Retrieval-Augmented Generation (RAG)** based AI Maintenance Engineer. The system predicts equipment failure probability and provides **human-readable maintenance recommendations** using open-source LLMs.


## **Key Features**

-Failure prediction using XGBoost Classifier

-Domain-driven feature engineering

-Production-style training & inference pipelines

-FAISS-based RAG for maintenance knowledge retrieval

-LLM-generated maintenance assessments

-Streamlit dashboard for interactive inference

-Clean, modular industry-grade project structure

## **System Architecture**
 
Sensor Data

   ↓
   
Preprocessing & Feature Engineering

   ↓
   
XGBoost Failure Prediction

   ↓
   
FAISS Knowledge Retrieval

   ↓
   
LLM Maintenance Reasoning

   ↓
   
Streamlit UI Output

## **Tech Stack**
 
       -Machine Learning

        -Python 3.10+

       -XGBoost

       -Scikit-learn

       -Pandas, NumPy

       -GenAI / RAG

      -Hugging Face Transformers

     -Sentence-Transformers

      -FAISS (CPU)

      -Open-source LLM (Colab-friendly)

      -MLOps & App

     -Streamlit

      -Pickle (model persistence)

      -Modular pipeline design

      -Google Colab (T4 GPU compatible)

## **Project Structure**
```
Predictive_Maintainenance-AI/
│
├── app/
│   └── streamlit_app.py
│
├── artifacts/
│   └── models/
│       └── xgboost_model.pkl
│
├── data/
│   └── knowledge_base/
│       ├── maintenance_guidelines.txt
│       ├── faiss_index.bin
│       └── chunks.pkl
│
├── notebooks/
│   └── ai_engineer_assistant_colab.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_xgboost.py
│   │   └── predict.py
│   │
│   └── llm/
│       ├── build_retriever.py
│       ├── retriever.py
│       └── maintenance_assistant.py
│
├── requirements.txt
└── README.md
```

## **Dataset**

**-** **Source**: UCI AI4I 2020 Predictive Maintenance Dataset

**-** **Type**: Industrial sensor telemetry

**-** **Target**: Machine failure (binary classification)

## **Setup Instructions**
 
 **-** **Install dependencies**
       
  ```
pip install -r requirements.txt

```

  **-Build FAISS Knowledge Index (One-Time)**
      
```

python src/llm/build_retriever.py

```

**This creates**: 
```
   -faiss_index.bin

   -chunks.pkl
```

**-Train ML Model**
```
   python src/models/train_xgboost.py

```

Model is saved to:
```
artifacts/models/xgboost_model.pkl
```

   **Run Streamlit App**
   
  ```
streamlit run app/streamlit_app.py
```

## **Example Output**
<img width="1652" height="1072" alt="image" src="https://github.com/user-attachments/assets/f670e039-b91d-4057-b0e5-59e4be12d7c2" />
