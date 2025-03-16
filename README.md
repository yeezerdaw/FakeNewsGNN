
# 📰 Fake News Detection with Graph Neural Networks (GNN)  

A machine learning model that classifies fake news using **GraphSAGE** and **BERT embeddings**, analyzing social network propagation patterns.

---

## ✨ Features  
✅ **GraphSAGE-based GNN** for learning from news propagation networks.  
✅ **BERT embeddings** for text representation (static embeddings).  
✅ **Multi-class classification** with six truthfulness categories.  
✅ **End-to-end pipeline** including dataset loading, feature extraction, training, and evaluation.  

---

## 📂 Project Structure  
```
FakeNewsGNN/
│── dataset.py              # Loads and preprocesses the LIAR dataset
│── feature_extraction.py   # Extracts static BERT embeddings
│── graph.py                # Constructs the graph representation
│── model.py                # Defines the GraphSAGE-based model
│── train.py                # Handles training and evaluation
│── main.py                 # Main script to run the pipeline
│── requirements.txt        # Dependencies for installation
│── README.md               # Project documentation
```

---

## 📊 Dataset  
The model is trained on the **LIAR Dataset**, sourced from [Hugging Face](https://huggingface.co/datasets/liar). It consists of **12,800 manually labeled short news statements**, categorized into six levels of truthfulness:  

- ❌ **False**  
- 🔥 **Pants-fire (extremely false)**  
- 🟡 **Barely-true**  
- 🟡 **Half-true**  
- 🟢 **Mostly-true**  
- ✅ **True**  

---

## ⚙️ Installation & Usage  

### 🔹 1. Clone the Repository  
```bash
git clone https://github.com/yourusername/FakeNewsGNN.git
cd FakeNewsGNN
```

### 🔹 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 🔹 3. Run the Model  
```bash
python main.py
```

---

## 🎯 Results  
✅ **Training for 1000 epochs** with a **learning rate scheduler** achieved:  
🔥 **87.02% accuracy on the LIAR dataset!**  

## Latest Update: Achieved **90% Accuracy**
### 🚀 Optimizations:
✔ Enabled **FP16 (Mixed Precision)** to **reduce GPU memory usage**  
✔ **Batch size adjusted** dynamically to **avoid CUDA OOM**  
✔ **CUDA Memory Management Tweaks** improved **speed & stability**  

### 📌 Further improvements:  
- 🏋️ **Fine-tuning BERT** instead of using static embeddings.  
- 🧠 **Experimenting with GAT (Graph Attention Networks)** for better learning.  
- 🔄 **Adding data augmentation** for more diverse training samples.  

---

## 🏗️ Model Architecture  
The model follows this pipeline:  

```plaintext
[News Text] → [BERT Embeddings (768-D)] → [GraphSAGE Layers] → [Fully Connected Layer] → [Classification]
```

---

## 🤝 Contributing  
Contributions are welcome! Feel free to **fork**, **open issues**, or **submit PRs** to improve the project.  

---

## 📝 License  
This project is open-source and available under the **MIT License**.  
```
