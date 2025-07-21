# 🤝 Federated Learning for Encrypted Traffic Classification

A privacy-preserving machine learning framework to classify encrypted network traffic using federated learning algorithms such as FedAvg and FedProx.

This project leverages multiple preprocessing techniques, deep learning models, and federated architectures to build and evaluate encrypted traffic classification without compromising raw data privacy.

---

## 🧠 Core Features

- **Federated Learning:** Implemented FedAvg and FedProx algorithms to collaboratively train models across decentralized data.
- **Encrypted Traffic Classification:** Detect and classify traffic flow types using features extracted from PCAP files.
- **Data Augmentation & Preprocessing:** CSV and PCAP processing, feature engineering, and augmentation for realistic modeling.
- **Multiple ML Models:** Includes Random Forest, MLP, and custom PyTorch neural networks.
- **Result Analysis:** Automated evaluation of accuracy, classification reports, and exportable metrics for comparison.

---

# 📁 Federated-learning Project Structure

## 📦 Data Preprocessing
- `1process_csv.py` – Convert CSV to usable format  
- `2process_pcap.py` – Extract data from PCAP  
- `3flow_analysis.py` – Analyze network flows  
- `5feature_extraction.py` – Extract flow-level features  
- `6data_augmentation.py` – Augment training data  
- `7data_preparation.py` – Final preprocessing for FL  

## 🤖 Federated Training
- `8fl_implementation.py` – Federated learning orchestration  
- `9model_training.py` – Local model training  
- `10raw_training.py` – Centralized (non-FL) training  
- `11diff_algo.py` – FedAvg vs FedProx comparison  
- `12save_load.py` – Model saving/loading helpers  

## 📊 Models & Evaluation
- `demo.py` – Quick demo on inference  
- `classification_reports.txt` – Evaluation reports  
- `*.pkl / *.pth` – Trained model files  

## 📚 Datasets
- `packets.pcap` – Raw packet data  
- `*.csv` – Preprocessed or augmented data  

## 📈 Results
- `fed_avg_results.csv`  
- `fed_prox_results.csv`  
- `federated_learning_results.csv`  

## ⚙️ Requirements
- `requirements.txt` – Python dependencies
---
## 📈 Sample Output Metrics

- Accuracy across 5 FL rounds with FedAvg: **~92.3%**
- FedProx showed improved stability on non-IID data
- Classification report: Precision > 90%, F1-score > 89%

---

## 🧠 Models Available

- `mlp_model.pkl` – Multi-layer perceptron model
- `rf_model.pkl` – Random forest model
- `simple_nn_model.pth` – PyTorch model for inference

---

## 📦 Datasets Used

- **PCAP Files**: Network packet captures
- **CSV Logs**: Processed and labeled flows
- **Augmented Data**: Synthesized using domain patterns

---

## 🛡️ Use Case

This project is ideal for:

- Network intrusion detection
- Encrypted traffic analytics
- Federated systems in privacy-sensitive environments

