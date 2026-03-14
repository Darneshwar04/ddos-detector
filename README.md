# DDoS Attack Detector 🛡️

A production-ready **Bidirectional LSTM (BiLSTM)** neural network for real-time DDoS attack detection using network flow data.

![Python](https://img.shields.io/badge/Python-3.14+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.94% |
| **F1 Score** | 0.9994 |
| **ROC-AUC** | 0.9935 |
| **Parameters** | 113,346 |

## ✨ Features

- ✅ **BiLSTM Architecture** — Bidirectional LSTM with 2 layers for sequence learning
- ✅ **Real-time Inference** — Process CSV files in seconds
- ✅ **Comprehensive Metrics** — Accuracy, F1, ROC-AUC, confusion matrix, classification report
- ✅ **Beautiful UI** — Dark mode Tailwind CSS frontend with drag-and-drop
- ✅ **Fast API** — Deployable REST API with `/api/status` and `/api/predict` endpoints
- ✅ **Production-Ready** — Configured for Render, Railway, or AWS deployment

## 🚀 Quick Start

### Local Development

**1. Clone the repository**
```bash
git clone https://github.com/Darneshwar04/ddos-detector.git
cd ddos-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the server**
```bash
python app.py
```

**4. Open in browser**
```
http://localhost:8000
```

### Docker

```bash
docker build -t ddos-detector .
docker run -p 8000:8000 ddos-detector
```

## 📁 Project Structure

```
ddos-detector/
├── app.py                    # FastAPI application
├── train_bilstm.ipynb       # Training notebook
├── train_bilstm.py          # Training script
├── utils.py                 # Feature preprocessing & utilities
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Web UI (Tailwind CSS)
├── saved_model/
│   ├── bilstm_ddos.pt      # Trained model weights
│   ├── scaler.pkl          # StandardScaler for feature normalization
│   └── label_encoder.pkl   # Metadata (features, class names, seq_len)
├── Procfile                 # Render/Heroku deployment config
├── runtime.txt              # Python version
└── .gitignore              # Git ignore file
```

## 📊 Data Format

The model expects **CICFlowMeter** format CSV files with **66 network flow features**:

### Supported Datasets
- ✅ **Archive (1)** — CIC-IDS-2018 (10 CSV files)
- ✅ **Archive (2)** — CIC-DDoS-2019 (18 CSV files)
- ✅ use the Test_file for evaluating the model!

### Features (Sample)
```
flow_duration, tot_fwd_pkts, tot_bwd_pkts, flow_byts_s, flow_pkts_s,
syn_flag_cnt, rst_flag_cnt, ack_flag_cnt, psh_flag_cnt, ...
[64 total features + 1 Label column]
```

## 🔧 API Usage

### Get Model Status
```bash
curl http://localhost:8000/api/status
```

**Response:**
```json
{
  "model_loaded": true,
  "torch_version": "2.10.0+cpu",
  "device": "cpu",
  "python_version": "3.14.3",
  "parameters": 113346,
  "seq_len": 10,
  "n_features": 66,
  "class_names": ["Benign", "Attack"]
}
```

### Run Inference
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@data.csv" \
  -F "max_rows=100000"
```

**Response:**
```json
{
  "filename": "data.csv",
  "total_rows_loaded": 1000,
  "sequences_evaluated": 100,
  "predictions": {
    "benign_count": 217,
    "attack_count": 9783,
    "attack_percentage": 97.83
  },
  "metrics": {
    "accuracy": 99.94,
    "f1_score": 0.9994,
    "roc_auc": 0.9935,
    "classification_report": { ... },
    "confusion_matrix": { ... }
  }
}
```

## 🧠 Model Architecture

```
Input (10 × 66 features)
    ↓
BiLSTM Layer 1 (64 units × 2 directions) + Dropout(0.2)
    ↓
BiLSTM Layer 2 (32 units × 2 directions) + Dropout(0.2)
    ↓
Dense Layer (64 units, ReLU) + Dropout(0.3)
    ↓
Output Layer (2 units, Softmax)
    ↓
Binary Classification: [P(Benign), P(Attack)]
```

## 📚 Training

To retrain the model:

```bash
jupyter lab train_bilstm.ipynb
```

Or use the Python script:
```bash
python train_bilstm.py
```

**Training Configuration:**
- **Epochs:** 30
- **Batch Size:** 512
- **Learning Rate:** 0.001
- **Sequence Length:** 10 timesteps
- **Validation Split:** 15%
- **Loss Function:** Cross-Entropy with class weights

## 🌐 Deployment

### Render (Recommended)

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Create new **Web Service**
4. Select your GitHub repo
5. Render auto-detects `Procfile` → Auto-deploy! ✅
6. Get public URL: `https://your-app.onrender.com`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Railway

```bash
railway init
railway up
```

### Heroku/AWS/Azure

Use `Procfile` — compatible with all major platforms.

## 🔐 Security Notes

- Model is frozen (`eval()` mode) — no training on inference
- File uploads limited to:**100MB** by default
- Only CSV files accepted
- Features are validated and missing columns filled with zeros
- No credentials stored in code

## 🛠️ Troubleshooting

### Model not loading
```
[!] Trained model not found.
    Run notebook: jupyter lab train_bilstm.ipynb
```
**Solution:** Train the model first using the notebook

### Permission denied on GitHub push
```
fatal: unable to access 'https://github.com/.../': The requested URL returned error: 403
```
**Solution:** Update git config:
```bash
git config --global user.name "YOUR_USERNAME"
git config --global user.email "your-email@example.com"
```

### Model predicting everything as Benign
**Cause:** Dataset format mismatch (wrong features or labeling)
**Solution:** Use Archive (1) or (2) CSV files which match training data

## 📖 Learning Resources

- [BiLSTM Tutorial](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [CICFlowMeter](https://www.unb.ca/cic/datasets/ids-2017.html)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Darneshwar04** — Network Security & Machine Learning

- GitHub: [@Darneshwar04](https://github.com/Darneshwar04)
- Email: darnesh2104@gmail.com

## 🙏 Acknowledgments

- **CIC-IDS-2018 & CIC-DDoS-2019** datasets
- **PyTorch & FastAPI** communities
- **Scikit-learn** preprocessing tools

---

**⭐ If you found this helpful, please star the repo!**

**Questions?** Open an issue on GitHub or contact me!
