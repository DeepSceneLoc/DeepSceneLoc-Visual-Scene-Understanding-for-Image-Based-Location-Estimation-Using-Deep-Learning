# Frontend Integration Guide
## Connecting React Frontend with DeepSceneLoc Model

## Overview

Your frontend integrates with a **FastAPI backend** (`backend.py`) that loads the PyTorch ensemble models (ResNet-50, EfficientNet-B0, ViT-B16) and falls back to **Gemini 3.1 Flash-Lite** for Stage 2 location details.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  React Frontend │  HTTP   │ FastAPI Backend │  PyTorch│  DeepSceneLoc   │
│   (Port 3000)   │────────▶│   (Port 5000)   │────────▶│     Models      │
│                 │◀────────│                 │◀────────│                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
  frontend/                  backend.py                 model_repo/
```

## Setup Steps

### Step 1: Install Python Dependencies

```bash
# Activate your virtual environment
venv\Scripts\activate  # Windows

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Start the Python Backend

```bash
# From project root
python backend.py
```

You should see:
```
============================================================
Loading DeepSceneLoc Ensemble Models on: cuda  (or cpu)
============================================================
[OK] ResNet-50 loaded successfully.
[OK] EfficientNet-B0 loaded successfully.
[OK] ViT-B16 loaded successfully.
[OK] Native Gemini AI initialized successfully
[OK] Stage 2 location analyzer ready. Mode: native
INFO:     Started server process [...]
```

### Step 3: Verify Frontend Proxy Configuration

The React frontend has been configured to read the API keys and endpoints dynamically from the root `.env` file. 

The `frontend/server.ts` Express server automatically proxies `/api/analyze-image` requests directly to the FastAPI backend running on port 5000:

```typescript
// frontend/server.ts
const response = await fetch("http://localhost:5000/api/analyze-image", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(req.body)
});
```

### Step 4: Start Frontend

```bash
cd frontend
npm install  # First time only
npm run dev  # Development mode
```

Or for production:
```bash
npm run build
npm start
```

### Step 5: Test

1. Open browser: `http://localhost:3000`
2. Navigate to "Interactive Demo" section
3. Upload an image
4. Should see predictions from your trained model!

## API Specification

### Endpoint: `POST /api/analyze-image`

**Request:**
```json
{
  "imageBase64": "data:image/jpeg;base64,/9j/4AAQ...",
  "presetId": null  // optional
}
```

**Response:**
```json
{
  "success": true,
  "source": "deepsceneloc_model",
  "data": {
    "sceneCategory": "Mountain",
    "confidence": 96.48,
    "landmarkName": "Rocky Mountain Range",
    "city": "Colorado",
    "country": "United States",
    "latitude": 39.7392,
    "longitude": -105.5130,
    "aiConfidence": 96.48,
    "reasoning": "High-altitude terrain with exposed rock formations...",
    "elevation": "3,200m",
    "bestSeason": "July to September",
    "geologicalAge": "Precambrian Granite Core"
  }
}
```

## Model Loading Priority

The backend tries to load models in this order:

1. **EfficientNet-B0** (best accuracy)
   - `models/checkpoints/efficientnet/EfficientNet-B0_best.pth`

2. **ViT-B/16** (transformer-based)
   - `models/checkpoints/vit/ViT-B_16_best.pth`

3. **ResNet50** (baseline)
   - `models/checkpoints/resnet/best_model.pth`
   - `model_repo/ResNet50/ResNet50_best_model.pth`

4. **Fallback:** Pretrained ResNet-50 (demo mode)

## Location Database

The backend includes a location database that maps scene categories to realistic location details:

- **Coastal** → Pacific/Mediterranean coastlines
- **Forest** → Pacific Northwest/Black Forest
- **Mountain** → Rocky Mountains/Alps
- **Rural** → Midwest farmland/Tuscany
- **Urban** → San Francisco/Vienna

You can expand this in `backend_api.py` → `LOCATION_DATABASE`

## Troubleshooting

### Backend won't start

```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Or use a different port
# Edit backend.py, change uvicorn port argument
```

### CORS errors

Make sure CORS middleware is enabled in `backend.py`. It is set up by default to allow origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Model not found

Check your checkpoint paths:
```bash
# From project root
ls models/checkpoints/efficientnet/
ls models/checkpoints/vit/
ls models/checkpoints/resnet/
```

### Frontend can't connect

Check both servers are running:
- Backend: `http://localhost:5000/health` should return `{"status": "healthy"}`
- Frontend: `http://localhost:3000` should load

## Production Deployment

### Option 1: Both on Same Server

Use Nginx to proxy:

```nginx
server {
    listen 80;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
    }
    
    # Backend API
    location /api/ {
        proxy_pass http://localhost:5000/api/;
    }
}
```

### Option 2: Separate Servers

Update frontend to use production API URL:

```typescript
const API_URL = process.env.NODE_ENV === 'production' 
  ? 'https://api.deepsceneloc.com'
  : 'http://localhost:5000';
```

## Adding Ensemble Support

To use ensemble predictions (multiple models):

1. Load all 3 models in `backend_api.py`:

```python
class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'resnet': load_model('resnet50', 'models/checkpoints/resnet/...'),
            'efficientnet': load_model('efficientnet_b0', 'models/checkpoints/efficientnet/...'),
            'vit': load_model('vit_b16', 'models/checkpoints/vit/...')
        }
    
    def predict(self, image):
        predictions = []
        for name, model in self.models.items():
            pred = model(image)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
```

2. Update the predictor initialization to use ensemble

## Advanced Features

### Add Confidence Threshold

```python
@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    # ... existing code ...
    
    if confidence < 0.5:  # Low confidence
        return jsonify({
            'success': True,
            'data': {...},
            'warning': 'Low confidence prediction. Results may be uncertain.'
        })
```

### Add Heatmap Visualization

Use Grad-CAM to show what the model is looking at:

```python
from pytorch_grad_cam import GradCAM

def get_attention_map(model, image):
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    heatmap = cam(input_tensor=image)[0]
    return heatmap  # Send to frontend
```

## Testing Checklist

- [ ] Backend starts without errors
- [ ] Frontend connects to backend
- [ ] Image upload works
- [ ] Predictions appear in UI
- [ ] Map shows correct location
- [ ] Confidence scores display
- [ ] All 5 categories work
- [ ] Error handling works (try invalid image)

## Next Steps

1. **Train models on Kaggle** (see `docs/guides/KAGGLE_TRAINING_NOTEBOOK.md`)
2. **Download trained checkpoints** to `models/checkpoints/`
3. **Test backend** with real trained models
4. **Deploy** frontend + backend together

---

**Questions?** Check:
- `webapp/backend_api.py` - Backend code
- `frontend/server.ts` - Current Express server
- `demo_app.py` - Reference Gradio implementation
