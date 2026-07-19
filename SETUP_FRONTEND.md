# Frontend Setup Guide

## Quick Start (3 Steps)

### Step 1: Install Python Backend Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate

# Install all backend requirements
pip install -r requirements.txt
```

### Step 2: Verify Frontend Proxy Configuration

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
    console.error("Backend proxy error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to connect to DeepSceneLoc backend"
    });
  }
});
```

### Step 3: Start Both Servers

**Windows (Easy):**
```bash
# Just double-click this file:
start_fullstack.bat
```

**Manual (Any OS):**
```bash
# Terminal 1 - Backend
python backend.py

# Terminal 2 - Frontend
cd frontend
npm install  # First time only
npm run dev
```
## ✅ Verify Setup

1. **Backend Check:**
   - Open: `http://localhost:5000/health`
   - Should see: `{"status": "healthy", "models_loaded": 3, "gemini_active": true, ...}`

2. **Frontend Check:**
   - Open: `http://localhost:3000`
   - Should see the DeepSceneLoc interface

3. **Integration Test:**
   - Navigate to "Interactive Demo" section
   - Upload any landscape image
   - Should see predictions from your model!

## 📊 What You'll See

After uploading an image:

```
Scene Classification: Mountain
Confidence: 96.48%

Location: Rocky Mountain Range
City: Colorado
Country: United States

Coordinates: 39.7392°N, 105.5130°W
Elevation: 3,200m
Best Season: July to September
```

Plus an interactive map showing the location!

## 🎨 Frontend Features You Get

Your React frontend has:
- ✨ **Beautiful UI** - Modern design with animations
- 📸 **Image Upload** - Drag & drop or click to upload
- 🗺️ **Interactive Map** - Shows predicted location with marker
- 📊 **Model Architecture** - Visualizes the AI pipeline
- 📈 **Performance Metrics** - Shows model accuracy
- 👥 **Team Section** - Your project team info
- 📱 **Responsive** - Works on mobile and desktop

## 🔧 Customization

### Change Location Database

Edit `webapp/backend_api.py` → `LOCATION_DATABASE` to add more locations:

```python
LOCATION_DATABASE = {
    'Mountain': [
        {
            'landmarkName': 'Your Mountain',
            'city': 'Your City',
            'country': 'Your Country',
            'latitude': 40.0,
            'longitude': -120.0,
            'reasoning': 'Why the model predicted this...',
            # ... more fields
        }
    ]
}
```

### Use Ensemble Models

Update `webapp/backend_api.py` to load multiple models and average predictions.

### Add More Categories

1. Update `CATEGORIES` in `webapp/backend_api.py`
2. Train models with new categories
3. Add location database entries

## 🚀 Production Deployment

When ready to deploy:

1. **Build Frontend:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Serve with Nginx:**
   ```nginx
   server {
       listen 80;
       
       # Serve React app
       location / {
           root /path/to/frontend/dist;
           try_files $uri /index.html;
       }
       
       # Proxy API to Flask
       location /api/ {
           proxy_pass http://localhost:5000/api/;
       }
   }
   ```

3. **Run Flask with Gunicorn:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 webapp.backend_api:app
   ```

## 📝 Comparison: Frontend vs Gradio

| Feature | React Frontend | Gradio |
|---------|----------------|--------|
| **UI Design** | Professional, custom | Simple, basic |
| **Branding** | Full customization | Limited |
| **Features** | Maps, metrics, architecture | Just upload/predict |
| **Mobile** | Fully responsive | Basic responsive |
| **Speed** | Very fast | Good |
| **Deployment** | Requires Node.js + Python | Just Python |
| **Team Section** | ✅ Built-in | ❌ None |
| **Architecture Viz** | ✅ Built-in | ❌ None |

**Your React frontend is production-ready and looks way better!** 🎉

## ⚡ Quick Commands

```bash
# Install backend deps
pip install flask flask-cors

# Start everything
start_fullstack.bat

# Or manually:
python webapp/backend_api.py          # Backend
cd frontend && npm run dev            # Frontend

# Check health
curl http://localhost:5000/health
```

## 🐛 Troubleshooting

### "Module not found: flask"
```bash
pip install flask flask-cors
```

### "Port 5000 already in use"
Edit `webapp/backend_api.py`, change `port=5000` to `port=5001`

### "Frontend can't connect to backend"
Check both servers are running:
- Backend: `http://localhost:5000/health` should work
- Frontend: `http://localhost:3000` should load

### "CORS error"
Make sure `flask-cors` is installed and imported in `backend_api.py`

### "No trained model found"
Backend will use pretrained model as fallback. To use trained models:
1. Train on Kaggle (see `docs/guides/KAGGLE_TRAINING_NOTEBOOK.md`)
2. Download checkpoints to `models/checkpoints/`
3. Restart backend

## 📚 More Info

- **Full guide:** `FRONTEND_INTEGRATION_GUIDE.md`
- **Integration summary:** `INTEGRATION_SUMMARY.md`
- **Backend code:** `webapp/backend_api.py`
- **Frontend code:** `frontend/src/App.tsx`

---

**Ready to go!** Just run `start_fullstack.bat` and open `http://localhost:3000` 🚀
