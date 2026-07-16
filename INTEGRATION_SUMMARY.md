# Frontend ↔ Backend Integration Summary

## 🎯 Goal

Connect your React frontend's "Show Demo" section to use your **trained DeepSceneLoc models** instead of Gemini AI.

## 📊 Current Setup vs New Setup

### Before (Current)
```
┌──────────────┐
│   Frontend   │
│  (React +    │
│   Express)   │──────▶ Gemini AI API
│   Port 3000  │        (External)
└──────────────┘
```

### After (Integrated)
```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Frontend   │  HTTP   │    Flask     │ PyTorch │ DeepSceneLoc │
│   (React)    │────────▶│   Backend    │────────▶│    Models    │
│   Port 3000  │◀────────│   Port 5000  │◀────────│  (Trained)   │
└──────────────┘         └──────────────┘         └──────────────┘
```

## 📁 Files Created

### 1. `webapp/backend_api.py` ⭐
**Python Flask API server**

**What it does:**
- Loads your trained DeepSceneLoc models
- Accepts images from frontend (base64)
- Returns predictions in frontend-compatible format
- Provides location details (city, country, coordinates)

**Key features:**
- Model loading priority (EfficientNet → ViT → ResNet)
- Aspect-preserving transforms (matching training)
- Location database for rich responses
- CORS enabled for React frontend

### 2. `FRONTEND_INTEGRATION_GUIDE.md` 📚
**Complete setup instructions**

Includes:
- Architecture diagram
- Step-by-step setup
- API specification
- Troubleshooting guide
- Production deployment tips

### 3. `start_fullstack.bat` 🚀
**Windows launcher script**

Double-click to start:
- Python backend (Flask on port 5000)
- React frontend (Vite on port 3000)

## 🔧 How to Use

### Quick Start

1. **Install Flask dependencies:**
   ```bash
   .venv\Scripts\activate
   pip install flask flask-cors
   ```

2. **Start both servers:**
   ```bash
   start_fullstack.bat
   ```
   
   Or manually:
   ```bash
   # Terminal 1: Backend
   python webapp/backend_api.py
   
   # Terminal 2: Frontend
   cd frontend
   npm run dev
   ```

3. **Update Frontend API URL:**
   
   In `frontend/server.ts`, change the analyze-image endpoint to point to your Flask backend:
   
   ```typescript
   // Line ~100-ish
   const response = await fetch("http://localhost:5000/api/analyze-image", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify(req.body)
   });
   ```

4. **Test:**
   - Open: `http://localhost:3000`
   - Go to "Interactive Demo"
   - Upload image
   - See DeepSceneLoc predictions!

## 🔌 API Flow

### 1. User uploads image in React UI
```javascript
// React component
const uploadImage = async (file) => {
  const base64 = await fileToBase64(file);
  
  // Send to backend
  const response = await fetch('/api/analyze-image', {
    method: 'POST',
    body: JSON.stringify({ imageBase64: base64 })
  });
  
  const result = await response.json();
  // Display predictions
}
```

### 2. Express proxies to Flask (or direct)
```typescript
// server.ts
app.post("/api/analyze-image", async (req, res) => {
  const response = await fetch("http://localhost:5000/api/analyze-image", {
    method: "POST",
    body: JSON.stringify(req.body)
  });
  res.json(await response.json());
});
```

### 3. Flask processes with DeepSceneLoc
```python
# backend_api.py
@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    # Decode image
    image = decode_base64(request.json['imageBase64'])
    
    # Predict
    category, confidence = predictor.predict(image)
    
    # Get location details
    location = LOCATION_DATABASE[category][0]
    
    # Return result
    return jsonify({
        'success': True,
        'source': 'deepsceneloc_model',
        'data': {
            'sceneCategory': category,
            'confidence': confidence * 100,
            'landmarkName': location['landmarkName'],
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            ...
        }
    })
```

### 4. React displays results
```javascript
// Result displayed in UI
{
  sceneCategory: "Mountain",
  confidence: 96.48,
  landmarkName: "Rocky Mountain Range",
  city: "Colorado",
  country: "United States",
  latitude: 39.7392,
  longitude: -105.5130
}
```

## 📋 Model Loading

The backend automatically loads the best available model:

1. ✅ **EfficientNet-B0** (if exists)
   - Path: `models/checkpoints/efficientnet/EfficientNet-B0_best.pth`
   - Best accuracy (~87%)

2. ✅ **ViT-B/16** (if exists)
   - Path: `models/checkpoints/vit/ViT-B_16_best.pth`
   - Transformer-based (~87-88%)

3. ✅ **ResNet50** (if exists)
   - Path: `models/checkpoints/resnet/best_model.pth`
   - Baseline (~84-85%)

4. ⚠️ **Fallback:** Pretrained ResNet-50
   - Used if no trained models found
   - Demo mode only

## 🗺️ Location Database

The backend includes realistic location mappings for each category:

| Category | Example Locations |
|----------|-------------------|
| **Coastal** | Pacific Coastline (Monterey), Mediterranean Coast (Amalfi) |
| **Forest** | Pacific Northwest Rainforest, Black Forest (Germany) |
| **Mountain** | Rocky Mountains (Colorado), Swiss Alps |
| **Rural** | Iowa Farmland, Tuscan Countryside |
| **Urban** | San Francisco, Vienna |

Each location includes:
- GPS coordinates (lat/lng)
- Elevation
- Best season to visit
- Geological age
- Reasoning (why model predicted this)

## ✅ Checklist

### Setup
- [ ] Flask installed (`pip install flask flask-cors`)
- [ ] Backend starts (`python webapp/backend_api.py`)
- [ ] Frontend starts (`cd frontend && npm run dev`)
- [ ] Both accessible (backend: 5000, frontend: 3000)

### Testing
- [ ] Upload image in UI
- [ ] Predictions appear
- [ ] Map shows location
- [ ] Confidence scores display
- [ ] Works for all 5 categories

### Production
- [ ] Train models on Kaggle
- [ ] Download checkpoints to `models/checkpoints/`
- [ ] Test with real trained models
- [ ] Deploy frontend + backend together

## 🚀 Next Steps

### 1. Train Models (Priority)
```bash
# See: docs/guides/KAGGLE_TRAINING_NOTEBOOK.md
python scripts/training/run_training_efficientnet_b0.py --batch 128 --epochs 45 --full-finetune --swa
python scripts/training/run_training_vit_b16.py --batch 96 --epochs 45 --full-finetune --swa
python scripts/training/run_training_resnet50.py --batch 128 --epochs 40 --full-finetune
```

### 2. Test Backend with Trained Models
```bash
# Download checkpoints from Kaggle
# Place in models/checkpoints/

# Restart backend
python webapp/backend_api.py
# Should see: "✅ Successfully loaded: models/checkpoints/..."
```

### 3. Enhance Location Database
Edit `webapp/backend_api.py` → `LOCATION_DATABASE` to add more locations per category

### 4. Add Ensemble Support
Modify backend to load and average predictions from all 3 models

### 5. Deploy to Production
- Use Gunicorn for Flask
- Use Nginx for reverse proxy
- Deploy to cloud (AWS, Azure, Heroku)

## 🛠️ Troubleshooting

### Backend won't start
```bash
# Check Python path
python --version  # Should be 3.8+

# Check Flask
pip show flask

# Try different port
# Edit backend_api.py: port=5001
```

### Frontend can't connect
```bash
# Check backend is running
curl http://localhost:5000/health

# Check CORS
# Should see in backend logs: "127.0.0.1 - - [date] "POST /api/analyze-image HTTP/1.1" 200"
```

### Model not loading
```bash
# Check checkpoint exists
dir models\checkpoints\efficientnet\
dir models\checkpoints\vit\
dir models\checkpoints\resnet\

# Check error in backend logs
# Should see either:
#   ✅ "Successfully loaded: ..."
#   ⚠️  "Using pretrained ResNet-50 (demo mode)"
```

### Wrong predictions
- Make sure using trained models (not pretrained)
- Check transforms match training (Resize(256) → CenterCrop(224))
- Verify model architecture matches checkpoint

## 📚 Documentation

- **Setup guide:** `FRONTEND_INTEGRATION_GUIDE.md`
- **Backend code:** `webapp/backend_api.py`
- **Current frontend:** `frontend/server.ts`
- **Gradio demo:** `demo_app.py` (reference)
- **Training guide:** `docs/guides/TRAINING_README.md`

## 🎉 Success Criteria

Your integration is working when:

1. ✅ Backend loads trained model successfully
2. ✅ Frontend uploads image without errors
3. ✅ Predictions show in UI (scene category + confidence)
4. ✅ Map displays predicted location
5. ✅ All 5 categories work (Coastal, Forest, Mountain, Rural, Urban)
6. ✅ Response time < 2 seconds per image

## 💡 Pro Tips

1. **Use Ensemble:** Average predictions from all 3 models for best accuracy
2. **Cache Models:** Load models once on startup, not per request
3. **Add Queue:** Use Celery for async processing if many users
4. **Monitor:** Add logging for predictions and errors
5. **Validate:** Check image format/size before processing

---

**Ready to connect your frontend!** 🚀

Start with: `start_fullstack.bat` or follow `FRONTEND_INTEGRATION_GUIDE.md`
