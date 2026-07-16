# Running DeepSceneLoc Backend on Kaggle

Since your local PyTorch has DLL issues, run the backend on Kaggle where it works.

## Setup on Kaggle

### 1. Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Create new notebook
3. Enable **Internet** in settings
4. Enable **GPU** (T4)

### 2. Upload/Clone Repository

```python
# Option A: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/DeepSceneLoc.git
%cd DeepSceneLoc

# Option B: Upload as Kaggle Dataset
# (if you don't have GitHub)
```

### 3. Install Dependencies

```python
!pip install flask flask-cors
```

### 4. Run Backend

```python
# Run the full backend (with real models)
!python webapp/backend_api.py
```

### 5. Expose with ngrok (for frontend connection)

```python
!pip install pyngrok

from pyngrok import ngrok

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print(f"Backend URL: {public_url}")
```

Then update your local frontend to use this URL.

## Alternative: Run Everything Locally (Fix PyTorch)

### Fix PyTorch DLL Issue

```bash
# Uninstall broken PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Or CPU only
pip install torch torchvision torchaudio
```

Then run:
```bash
python webapp/backend_api.py  # Real backend with models
```

## Quick Test: Use Pretrained Models

Your backend already has fallback to pretrained models. To test quickly:

```bash
# Just run the real backend
python webapp/backend_api.py
```

It will automatically use pretrained models if no trained checkpoints exist.
