"""
DeepSceneLoc Backend API - Mock Version
For testing frontend integration when PyTorch is not available

This returns realistic mock predictions to test the frontend UI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import random
from PIL import Image

app = Flask(__name__)
CORS(app)

CATEGORIES = ['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']

# Mock location database
MOCK_LOCATIONS = {
    'Coastal': {
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
    'Forest': {
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
    'Mountain': {
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
    'Rural': {
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
    'Urban': {
        'landmarkName': 'Metropolitan Downtown',
        'city': 'San Francisco',
        'country': 'United States',
        'latitude': 37.7749,
        'longitude': -122.4194,
        'reasoning': 'High-density urban environment with characteristic architecture, street patterns, and infrastructure. Building styles and urban layout indicate major metropolitan area.',
        'elevation': '52m',
        'bestSeason': 'September to November',
        'geologicalAge': 'Quaternary Bay Sediments'
    }
}


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Mock image analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Handle presets
        preset_id = data.get('presetId')
        if preset_id:
            return jsonify({
                'success': True,
                'source': 'preset',
                'message': 'Preset handled by frontend'
            })
        
        # Get image
        image_base64 = data.get('imageBase64')
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Decode image (just to validate)
        try:
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            print(f"✓ Image received: {image.size}, {image.mode}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image: {e}'}), 400
        
        # Mock prediction (random category with high confidence)
        category = random.choice(CATEGORIES)
        confidence = random.uniform(85.0, 98.5)
        
        location = MOCK_LOCATIONS[category]
        
        response = {
            'success': True,
            'source': 'deepsceneloc_mock',
            'data': {
                'sceneCategory': category,
                'confidence': round(confidence, 2),
                'landmarkName': location['landmarkName'],
                'city': location['city'],
                'country': location['country'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'aiConfidence': round(confidence, 2),
                'reasoning': location['reasoning'],
                'elevation': location['elevation'],
                'bestSeason': location['bestSeason'],
                'geologicalAge': location['geologicalAge']
            }
        }
        
        print(f"✓ Prediction: {category} ({confidence:.2f}%)")
        return jsonify(response)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Return available presets"""
    return jsonify({
        'success': True,
        'presets': []
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'deepsceneloc_mock',
        'mode': 'mock',
        'note': 'Using mock predictions (PyTorch not available)',
        'categories': CATEGORIES
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DeepSceneLoc Backend API Server (MOCK MODE)")
    print("="*60)
    print("⚠️  Using mock predictions (PyTorch DLL not available)")
    print("Categories:", ', '.join(CATEGORIES))
    print("="*60)
    print("\nStarting Flask server on http://localhost:5000")
    print("Frontend should connect to: http://localhost:5000/api/analyze-image")
    print("\n✓ Ready to accept requests!")
    print("\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
