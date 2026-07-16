# 🚀 Get Started with DeepSceneLoc

## For End Users (Demo)

### Option 1: Full Web Interface (Recommended) ⭐

**Professional React UI with maps, metrics, and team info**

```bash
# 1. Install backend dependencies
pip install flask flask-cors

# 2. Start both servers
start_fullstack.bat

# 3. Open browser
http://localhost:3000
```

That's it! Upload an image in the "Interactive Demo" section.

### Option 2: Simple Gradio Interface

**Quick and simple, just upload and predict**

```bash
python demo_app.py
```

Opens at `http://localhost:7860`

---

## For Developers (Training)

### Train Your Own Models

```bash
# On Kaggle T4x2 GPU (recommended)
python scripts/training/run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa
python scripts/training/run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa
python scripts/training/run_training_resnet50.py --batch 128 --epochs 40 --full-finetune
```

See: `docs/guides/KAGGLE_TRAINING_NOTEBOOK.md`

---

## Repository Structure

```
DeepSceneLoc/
├── demo_app.py              # Simple Gradio demo
├── start_fullstack.bat      # Launch web interface
│
├── frontend/                # React web interface ⭐
│   ├── src/
│   └── server.ts
│
├── webapp/
│   └── backend_api.py       # Flask API for frontend
│
├── scripts/training/        # Training scripts
│   ├── run_training_efficientnet_b0.py
│   ├── run_training_vit_b16.py
│   └── run_training_resnet50.py
│
├── src/                     # Core library
│   ├── models/
│   ├── preprocessing/
│   └── evaluation/
│
└── docs/guides/             # Documentation
    ├── QUICK_START.md
    ├── TRAINING_README.md
    └── KAGGLE_TRAINING_NOTEBOOK.md
```

---

## Quick Links

### For Users
- **Web interface setup:** `SETUP_FRONTEND.md`
- **Integration guide:** `FRONTEND_INTEGRATION_GUIDE.md`
- **Integration summary:** `INTEGRATION_SUMMARY.md`

### For Developers
- **Command reference:** `docs/guides/QUICK_START.md`
- **Training guide:** `docs/guides/TRAINING_README.md`
- **Kaggle notebook:** `docs/guides/KAGGLE_TRAINING_NOTEBOOK.md`

### Project Info
- **Repository map:** `DIRECTORY_STRUCTURE.md`
- **File organization:** `FILE_ORGANIZATION.md`
- **Main README:** `README.md`

---

## What Each Interface Offers

### React Frontend (localhost:3000) ⭐

**Best for:** Professional demos, presentations, deployment

Features:
- ✅ Beautiful modern UI
- ✅ Interactive map with location pins
- ✅ Model architecture visualization
- ✅ Performance metrics display
- ✅ Team member profiles
- ✅ Responsive (mobile + desktop)
- ✅ Production-ready

**How to start:**
```bash
start_fullstack.bat
# or
python webapp/backend_api.py    # Terminal 1
cd frontend && npm run dev       # Terminal 2
```

### Gradio Demo (localhost:7860)

**Best for:** Quick testing, simple demos

Features:
- ✅ Simple upload interface
- ✅ Instant predictions
- ✅ Confidence scores
- ✅ One-command launch
- ❌ No maps
- ❌ Basic UI
- ❌ Limited features

**How to start:**
```bash
python demo_app.py
```

---

## System Requirements

### For Demo (Users)
- Python 3.8+
- 4GB RAM minimum
- Optional: GPU for faster predictions

### For Training (Developers)
- Python 3.10+
- 16GB RAM minimum
- GPU with 12GB+ VRAM (T4, P100, V100)
- CUDA 12.6+

---

## Installation

### Basic Setup (Demo Only)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/DeepSceneLoc.git
cd DeepSceneLoc

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Or CPU only
pip install torch torchvision torchaudio

# 5. Install Flask for web interface
pip install flask flask-cors
```

### Frontend Setup (Optional but Recommended)

```bash
cd frontend
npm install
```

---

## Quick Test

### Test Backend API
```bash
# Start backend
python webapp/backend_api.py

# Check health (in another terminal)
curl http://localhost:5000/health
# Should return: {"status": "healthy", ...}
```

### Test Gradio Demo
```bash
python demo_app.py
# Opens browser automatically
```

### Test Full Stack
```bash
start_fullstack.bat
# Opens:
# - Backend: http://localhost:5000
# - Frontend: http://localhost:3000
```

---

## Troubleshooting

### "No module named 'flask'"
```bash
pip install flask flask-cors
```

### "CUDA not available"
Your system doesn't have a GPU or CUDA isn't installed. The models will use CPU (slower but works).

### "Port already in use"
Change ports in:
- Backend: `webapp/backend_api.py` → `port=5001`
- Frontend: `frontend/vite.config.ts` → `port: 3001`

### "Model not found"
The backend will use pretrained models (demo mode). To use trained models:
1. Train on Kaggle
2. Download checkpoints to `models/checkpoints/`
3. Restart backend

---

## Next Steps

### 1. Try the Demo ⭐
```bash
start_fullstack.bat
```
Upload any landscape image!

### 2. Read Documentation
- Start: `SETUP_FRONTEND.md`
- Training: `docs/guides/TRAINING_README.md`
- Commands: `docs/guides/QUICK_START.md`

### 3. Train Your Models
Follow: `docs/guides/KAGGLE_TRAINING_NOTEBOOK.md`

### 4. Deploy
See: `FRONTEND_INTEGRATION_GUIDE.md` (Production section)

---

## Support

- **Issues:** Check troubleshooting sections in docs
- **Questions:** See detailed guides in `docs/guides/`
- **Training:** Follow Kaggle notebook guide

---

## Project Team

- Krishan Yadav
- Aditi Sah
- Anuj Kondawar
- Jensi Paneliya

---

**Choose your interface and start exploring!** 🌍🗺️

**Recommended:** Use the React frontend (`start_fullstack.bat`) for the best experience!
