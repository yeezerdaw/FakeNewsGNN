# 📰 Fake News Detection with Graph Neural Networks (GNN)  
🚀 A **Fake News Classifier** using **GraphSAGE + BERT** to analyze news propagation patterns on social networks.

---

## 📌 Features  
✅ **GraphSAGE Model** → Learns from social network structure  
✅ **BERT Embeddings** → Converts news text into numerical representations (static embeddings)  
✅ **Multi-Class Classification** → Classifies news into **6 categories**  
✅ **End-to-End Pipeline** → Includes dataset loading, feature extraction, model training, and evaluation  

---

## 📂 Project Structure  
FakeNewsGNN/ 
    │── dataset.py # Load LIAR dataset & preprocess labels 
    │── feature_extraction.py # Extract BERT embeddings (static) 
    │── graph.py # Build graph structure (edge_index) 
    │── model.py # Define GraphSAGE-based GNN model 
    │── train.py # Training loop & accuracy evaluation 
    │── main.py # Main script to run training & evaluation 
    │── requirements.txt # Dependencies for installation 
    │── README.md # Project documentation


---

## 📊 Dataset  
We use the **LIAR Dataset** from [Hugging Face](https://huggingface.co/datasets/liar), which contains:
- **12.8K manually labeled short news statements**
- **6 truthfulness categories**:  
  🔴 `pants-fire (extremely false)`  
  🔴 `false`  
  🟡 `barely-true`  
  🟡 `half-true`  
  🟢 `mostly-true`  
  🟢 `true`  

---

## 🚀 Installation & Usage  
### 🔹 **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/FakeNewsGNN.git
cd FakeNewsGNN
