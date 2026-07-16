"""
DeepSceneLoc Backend API
Flask server to connect React frontend with DeepSceneLoc models

This replaces Gemini AI predictions with actual trained model predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import sys
from pathlib import Path
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model import create_model
try:
    from src.models.model_advanced import create_advanced_model
except:
    print("Advanced models not available, using ResNet-50 only")
    create_advanced_model = None

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
CATEGORIES = ['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mock location database for rich responses (matching frontend expectations)
LOCATION_DATABASE = {
    'Coastal': [
        {
            'landmarkName': 'Pacific Coastline',
            'city': 'Monterey',
            'country': 'United States',
            'latitude': 36.6002,
            'longitude': -121.8947,
            'reasoning': 'The image displays characteristic coastal features including ocean waves, sandy beaches, and maritime vegetation. The rock formations and water clarity suggest a Pacific coast location.',
            'elevation': '10m',
            'bestSeason': 'April to October',
            'geologicalAge': 'Quaternary Marine Deposits'
        },
        {
            'landmarkName': 'Mediterranean Coast',
            'city': 'Amalfi',
            'country': 'Italy',
            'latitude': 40.6340,
            'longitude': 14.6027,
            'reasoning': 'Coastal cliffs, terraced hillsides, and Mediterranean vegetation patterns visible in the image match the Amalfi Coast geological and architectural characteristics.',
            'elevation': '15m',
            'bestSeason': 'May to September',
            'geologicalAge': 'Mesozoic Limestone Cliffs'
        }
    ],
    'Forest': [
        {
            'landmarkName': 'Pacific Northwest Rainforest',
            'city': 'Olympic Peninsula',
            'country': 'United States',
            'latitude': 47.8021,
            'longitude': -123.8308,
            'reasoning': 'Dense coniferous forest canopy with characteristic Douglas fir and western hemlock trees. High moisture content and moss coverage indicate temperate rainforest conditions.',
            'elevation': '450m',
            'bestSeason': 'June to September',
            'geologicalAge': 'Quaternary Glacial Deposits'
        },
        {
            'landmarkName': 'Black Forest Region',
            'city': 'Baden-Württemberg',
            'country': 'Germany',
            'latitude': 48.3154,
            'longitude': 8.1599,
            'reasoning': 'Mixed deciduous and coniferous forest with typical Central European tree species. Forest density and understory vegetation patterns match Black Forest characteristics.',
            'elevation': '650m',
            'bestSeason': 'May to October',
            'geologicalAge': 'Triassic Sandstone Base'
        }
    ],
    'Mountain': [
        {
            'landmarkName': 'Rocky Mountain Range',
            'city': 'Colorado',
            'country': 'United States',
            'latitude': 39.7392,
            'longitude': -105.5130,
            'reasoning': 'High-altitude terrain with exposed rock formations, alpine vegetation, and characteristic Rocky Mountain geological features including granite outcrops and glacial valleys.',
            'elevation': '3,200m',
            'bestSeason': 'July to September',
            'geologicalAge': 'Precambrian Granite Core'
        },
        {
            'landmarkName': 'Alpine Mountain Region',
            'city': 'Grindelwald',
            'country': 'Switzerland',
            'latitude': 46.6246,
            'longitude': 8.0389,
            'reasoning': 'Dramatic alpine peaks with permanent snowfields and characteristic European Alpine architecture. Steep valleys and glacial features indicate Swiss Alpine region.',
            'elevation': '2,850m',
            'bestSeason': 'June to September',
            'geologicalAge': 'Mesozoic Limestone and Dolomite'
        }
    ],
    'Rural': [
        {
            'landmarkName': 'Agricultural Heartland',
            'city': 'Iowa',
            'country': 'United States',
            'latitude': 41.8780,
            'longitude': -93.0977,
            'reasoning': 'Expansive farmland with grid-pattern field divisions, agricultural buildings, and characteristic Midwest rural infrastructure. Flat topography and crop patterns indicate intensive agriculture.',
            'elevation': '280m',
            'bestSeason': 'May to October',
            'geologicalAge': 'Quaternary Loess Deposits'
        },
        {
            'landmarkName': 'Tuscan Countryside',
            'city': 'Siena',
            'country': 'Italy',
            'latitude': 43.3188,
            'longitude': 11.3308,
            'reasoning': 'Rolling hills with vineyards, olive groves, and cypress tree alleys. Traditional farmhouses and terraced agriculture match Tuscan rural landscape characteristics.',
            'elevation': '320m',
            'bestSeason': 'April to October',
            'geologicalAge': 'Pliocene Clay and Sandstone'
        }
    ],
    'Urban': [
        {
            'landmarkName': 'Metropolitan Downtown',
            'city': 'San Francisco',
            'country': 'United States',
            'latitude': 37.7749,
            'longitude': -122.4194,
            'reasoning': 'High-density urban environment with characteristic architecture, street patterns, and infrastructure. Building styles and urban layout indicate major metropolitan area.',
            'elevation': '52m',
            'bestSeason': 'September to November',
            'geologicalAge': 'Quaternary Bay Sediments'
        },
        {
            'landmarkName': 'European City Center',
            'city': 'Vienna',
            'country': 'Austria',
            'latitude': 48.2082,
            'longitude': 16.3738,
            'reasoning': 'Historic European architecture with characteristic building facades, street layouts, and urban infrastructure. Architectural styles indicate Central European city center.',
            'elevation': '171m',
            'bestSeason': 'April to October',
            'geologicalAge': 'Quaternary Alluvial Plain'
        }
    ]
}


class DeepSceneLocPredictor:
    """Model predictor for DeepSceneLoc"""
    
    def __init__(self):
        self.device = DEVICE
        self.categories = CATEGORIES
        
        # Transform matching training (aspect-preserving)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load models (try best first, fallback to pretrained)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self):
        """Load best available model"""
        # Try loading trained checkpoints
        checkpoint_paths = [
            'models/checkpoints/efficientnet/EfficientNet-B0_best.pth',
            'models/checkpoints/vit/ViT-B_16_best.pth',
            'models/checkpoints/resnet/best_model.pth',
            'model_repo/ResNet50/ResNet50_best_model.pth',
        ]
        
        for ckpt_path in checkpoint_paths:
            try:
                if Path(ckpt_path).exists():
                    print(f"Loading model from {ckpt_path}")
                    
                    # Determine architecture
                    if 'efficientnet' in ckpt_path.lower():
                        if create_advanced_model:
                            model = create_advanced_model('efficientnet_b0', num_classes=5, pretrained=False)
                        else:
                            continue
                    elif 'vit' in ckpt_path.lower():
                        if create_advanced_model:
                            model = create_advanced_model('vit_b16', num_classes=5, pretrained=False)
                        else:
                            continue
                    else:
                        model = create_model('resnet50', num_classes=5, pretrained=False)
                    
                    # Load weights
                    checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint.get('model_state') or checkpoint.get('model_state_dict') or checkpoint
                    
                    # Remove 'module.' prefix if present
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    
                    model.load_state_dict(state_dict)
                    print(f"✅ Successfully loaded: {ckpt_path}")
                    return model
                    
            except Exception as e:
                print(f"Failed to load {ckpt_path}: {e}")
                continue
        
        # Fallback: pretrained model
        print("⚠️  Using pretrained ResNet-50 (demo mode)")
        return create_model('resnet50', num_classes=5, pretrained=True)
    
    def predict(self, image):
        """Predict scene category from PIL image"""
        # Transform and predict
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top prediction
        pred_idx = probabilities.argmax().item()
        confidence = probabilities[pred_idx].item()
        category = self.categories[pred_idx]
        
        return category, confidence, probabilities.cpu().numpy()


# Initialize predictor
print("Initializing DeepSceneLoc predictor...")
predictor = DeepSceneLocPredictor()
print(f"Device: {DEVICE}")
print(f"Categories: {CATEGORIES}")


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """
    Analyze image and return location prediction
    Matches frontend API expectations
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Handle preset demos (fast path)
        preset_id = data.get('presetId')
        if preset_id:
            # Frontend presets are handled client-side
            # This is just for consistency
            return jsonify({
                'success': True,
                'source': 'preset',
                'message': 'Preset demos handled by frontend'
            })
        
        # Get image data
        image_base64 = data.get('imageBase64')
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Decode base64 image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Predict using DeepSceneLoc model
        category, confidence, all_probs = predictor.predict(image)
        
        # Get location details from database
        location_options = LOCATION_DATABASE.get(category, [])
        selected_location = location_options[0] if location_options else {}
        
        # Build response matching frontend expectations
        response = {
            'success': True,
            'source': 'deepsceneloc_model',
            'data': {
                'sceneCategory': category,
                'confidence': round(confidence * 100, 2),
                'landmarkName': selected_location.get('landmarkName', f'{category} Region'),
                'city': selected_location.get('city', 'Unknown'),
                'country': selected_location.get('country', 'Unknown'),
                'latitude': selected_location.get('latitude', 0.0),
                'longitude': selected_location.get('longitude', 0.0),
                'aiConfidence': round(confidence * 100, 2),
                'reasoning': selected_location.get('reasoning', f'The DeepSceneLoc model classified this image as {category} with {confidence*100:.1f}% confidence based on visual features including terrain patterns, vegetation characteristics, and structural elements visible in the scene.'),
                'elevation': selected_location.get('elevation', 'N/A'),
                'bestSeason': selected_location.get('bestSeason', 'All Year'),
                'geologicalAge': selected_location.get('geologicalAge', 'Variable')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Return available preset demos"""
    # Frontend has its own presets, this is just for consistency
    return jsonify({
        'success': True,
        'presets': []
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'deepsceneloc',
        'device': str(DEVICE),
        'categories': CATEGORIES
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DeepSceneLoc Backend API Server")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    print("="*60)
    print("\nStarting Flask server on http://localhost:5000")
    print("Frontend should connect to: http://localhost:5000/api/analyze-image")
    print("\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
