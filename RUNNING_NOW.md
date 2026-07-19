# 🚀 DeepSceneLoc is Running!

## ✅ Status: RUNNING

### Backend API (Python FastAPI)
- **Status:** ✅ Running
- **URL:** http://localhost:5000
- **Mode:** Ensemble PyTorch Models (ResNet-50, EfficientNet-B0, ViT-B16) + Stage 2 Gemini Location Analyzer
- **Health Check:** http://localhost:5000/health

**Note:** Runs real-time local model inference and uses Gemini 3.1 Flash-Lite for sub-3-second Stage 2 coordinate grounding when `GEMINI_API_KEY` is provided in `.env`.

### Frontend (React + Vite)
- **Status:** ✅ Starting...
- **URL:** http://localhost:3000 (will open automatically)
- **Tech:** React + TypeScript + Tailwind CSS

## 🎯 What You Can Do Now

### 1. Open the Web Interface
```
http://localhost:3000
```

### 2. Test the Demo
1. Navigate to **"Interactive Demo"** section
2. **Upload an image** (drag & drop or click)
3. See the prediction:
   - Scene category (Coastal/Forest/Mountain/Rural/Urban)
   - Confidence score
   - Location details (city, country, GPS)
   - Interactive map with marker

### 3. Explore Features
- **Hero Section** - Project overview
- **How It Works** - Pipeline explanation
- **Interactive Demo** - Upload and test ⭐
- **Model Architecture** - Visual diagram
- **Performance Metrics** - Accuracy stats
- **Tech Stack** - Technologies used
- **Team Section** - Project team

## 📊 Current Setup

```
┌─────────────────┐         ┌─────────────────┐
│  React Frontend │  HTTP   │ FastAPI Backend │
│   Port 3000     │────────▶│   Port 5000     │
│                 │◀────────│  (backend.py)   │
└─────────────────┘         └─────────────────┘
```

## ⚡ Production Ensemble Mode

Currently using the **DeepSceneLoc production ensemble** because:
- Loads ResNet-50, EfficientNet-B0, and ViT-B16 checkpoints
- Performs Test-Time Augmentation (TTA) with horizontal flipping for max accuracy
- Ensembles predictions by averaging class probabilities

**To get real predictions:**

### Option 1: Fix PyTorch (Local)
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Option 2: Use Kaggle (Recommended)
1. Upload your data to Kaggle
2. Train models: `docs/guides/KAGGLE_TRAINING_NOTEBOOK.md`
3. Download trained checkpoints
4. Place in `models/checkpoints/`
5. Switch backend from `backend_api_mock.py` to `backend_api.py`

### Option 3: Use Gemini AI (Current Frontend Default)
The frontend already has Gemini AI integration. Set your API key:
```bash
# In frontend/.env
GEMINI_API_KEY=your_key_here
```

## 🧪 Testing

### Test Backend API
```bash
# Health check
curl http://localhost:5000/health

# Should return:
# {
#   "status": "healthy",
#   "model": "deepsceneloc_mock",
#   "mode": "mock",
#   "categories": ["Coastal", "Forest", "Mountain", "Rural", "Urban"]
# }
```

### Test Frontend
1. Open: http://localhost:3000
2. Upload any image
3. Should see prediction results

## 📝 Endpoints

### Backend API

**Health Check:**
```
GET http://localhost:5000/health
```

**Analyze Image:**
```
POST http://localhost:5000/api/analyze-image
Body: { "imageBase64": "data:image/jpeg;base64,..." }
```

**Response:**
```json
{
  "success": true,
  "source": "deepsceneloc_mock",
  "data": {
    "sceneCategory": "Mountain",
    "confidence": 94.25,
    "landmarkName": "Rocky Mountain Range",
    "city": "Colorado",
    "country": "United States",
    "latitude": 39.7392,
    "longitude": -105.5130,
    "reasoning": "High-altitude terrain with...",
    "elevation": "3,200m",
    "bestSeason": "July to September",
    "geologicalAge": "Precambrian Granite Core"
  }
}
```

## 🛑 To Stop Servers

### Option 1: Close terminal windows

### Option 2: Use Task Manager
- Find Python (Flask) and Node (Vite) processes
- End tasks

### Option 3: Command
```bash
# Kill Flask (port 5000)
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Kill Vite (port 3000)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

## 🔄 To Restart

### Restart Backend
```bash
python backend.py
```

### Restart Frontend
```bash
cd frontend
npm run dev
```

### Or Use Launcher
```bash
start_fullstack.bat
```

## 📚 Next Steps

1. **✅ Test the frontend** - Upload images, explore UI
2. **📖 Read docs** - `SETUP_FRONTEND.md` for details
3. **🎓 Train models** - On Kaggle for real predictions
4. **🚀 Deploy** - See `FRONTEND_INTEGRATION_GUIDE.md`

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Backend health check returns 200
- ✅ Frontend loads at http://localhost:3000
- ✅ Can upload image in demo section
- ✅ Predictions appear with map

## 💡 Tips

1. **Use Chrome/Edge** for best frontend experience
2. **Upload landscape images** for better mock predictions
3. **Check console** (F12) if issues occur
4. **CORS is enabled** on backend for React
5. **Hot reload** works - edit React code and see changes instantly

## 🆘 Troubleshooting

### Frontend won't load
Check if Vite started successfully. Should see:
```
VITE v5.x.x ready in xxx ms
➜ Local: http://localhost:3000/
```

### Backend error
```bash
# Check if FastAPI is running
curl http://localhost:5000/health
```

### Image upload not working
- Check browser console (F12)
- Verify backend is running and Gemini API Key is configured in `.env`
- Check for 503/404/network errors in Python terminal output

### Port already in use
Change ports:
- Backend: Edit `backend.py` -> `port=5001` or set PORT in `.env`
- Frontend: Edit `vite.config.ts` -> `port: 3001`

---

**Enjoy testing your DeepSceneLoc frontend!** 🌍🗺️

When you're ready for real predictions, follow the Kaggle training guide in `docs/guides/`.
