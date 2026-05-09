"""
Gemini AI Integration for Exact Location Detection
Provides landmark recognition and detailed location information
"""

import os
import base64
from io import BytesIO
from PIL import Image
import json
from typing import Dict, Optional, Tuple

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    # Only warn if explicitly using Google SDK

import requests
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class GeminiLocationAnalyzer:
    """
    Uses Gemini Vision API to identify exact locations, landmarks, and places
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini API (Native or OpenRouter)
        """
        # Load environment
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_model = os.getenv('OPENROUTER_MODEL', 'google/gemini-flash-1.5')
        self.gemini_api_key = api_key or os.getenv('GEMINI_API_KEY')

        if self.openrouter_key:
            print(f"[OK] OpenRouter initialized (Model: {self.openrouter_model})")
            self.mode = "openrouter"
        elif self.gemini_api_key:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Required for native Gemini mode.")
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.mode = "native"
            print("[OK] Native Gemini AI initialized successfully")
        else:
            raise ValueError(
                "No API key found. Please set OPENROUTER_API_KEY or GEMINI_API_KEY in .env"
            )
    
    def analyze_location(
        self, 
        image: Image.Image, 
        predicted_category: str = None,
        confidence: float = None
    ) -> Dict:
        """
        Analyze image to identify exact location, landmarks, and place details
        
        Args:
            image: PIL Image object
            predicted_category: Category predicted by DeepSceneLoc model
            confidence: Confidence of the prediction
            
        Returns:
            Dictionary with location analysis results:
            {
                'exact_location': 'Eiffel Tower, Paris, France',
                'latitude': 48.8584,
                'longitude': 2.2945,
                'confidence': 'high',  # high/medium/low
                'description': 'Detailed description...',
                'landmarks': ['Eiffel Tower'],
                'region': 'Western Europe',
                'country': 'France',
                'city': 'Paris',
                'place_type': 'landmark',  # landmark/city/natural/generic
                'additional_info': {...}
            }
        """
        
        # Create detailed prompt for location identification
        prompt = self._create_location_prompt(predicted_category, confidence)
        
        try:
            if self.mode == "openrouter":
                result_text = self._analyze_via_openrouter(image, prompt)
            else:
                # Call Native Gemini API
                response = self.model.generate_content([prompt, image])
                result_text = response.text
            
            # Parse response
            result = self._parse_gemini_response(result_text, predicted_category)
            result['provider'] = self.mode
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'exact_location': 'Unable to determine',
                'confidence': 'none',
                'description': f'Error analyzing image: {str(e)}'
            }
            
    def _analyze_via_openrouter(self, image: Image.Image, prompt: str) -> str:
        """Helper to call OpenRouter API via requests"""
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "HTTP-Referer": "https://github.com/KrishanYadav333/DeepSceneLoc",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.openrouter_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter Error ({response.status_code}): {response.text}")
            
        data = response.json()
        return data['choices'][0]['message']['content']

    def _create_location_prompt(self, category: str = None, confidence: float = None) -> str:
        """Create detailed prompt for Gemini"""
        
        prompt = """Analyze this image and provide the EXACT location identification.

Your task is to identify:
1. **EXACT LOCATION**: The specific place, landmark, or location name
2. **COORDINATES**: Best estimate of latitude and longitude (if identifiable)
3. **LOCATION DETAILS**: Country, city, region
4. **CONFIDENCE**: How confident are you? (high/medium/low)
5. **DESCRIPTION**: What makes you confident in this identification?

"""
        
        if category:
            prompt += f"\n**HINT**: A scene classifier predicted this is a '{category}' scene"
            if confidence:
                prompt += f" with {confidence:.1%} confidence."
        
        prompt += """

**RESPONSE FORMAT** (provide as structured text):

EXACT_LOCATION: [Specific place name, landmark, or "Cannot determine exact location"]
LATITUDE: [number or "unknown"]
LONGITUDE: [number or "unknown"]
COUNTRY: [country name or "unknown"]
CITY: [city name or "unknown"]
REGION: [geographic region or "unknown"]
CONFIDENCE: [high/medium/low]
PLACE_TYPE: [landmark/city/natural_feature/generic_scene]
LANDMARKS: [List any recognizable landmarks, comma-separated]

DESCRIPTION:
[Detailed explanation of what you see and why you identified this location. Include:
- Distinctive features that led to identification
- Architectural, natural, or cultural clues
- If you cannot determine exact location, explain what type of place it appears to be
- Your reasoning process]

Be specific if you recognize the place. If you cannot determine the exact location, provide as much detail as possible about the type of place, region, and characteristics."""

        return prompt
    
    def _parse_gemini_response(self, response_text: str, category: str = None) -> Dict:
        """Parse Gemini's structured response into dictionary"""
        
        result = {
            'exact_location': 'Unknown',
            'latitude': None,
            'longitude': None,
            'country': 'Unknown',
            'city': 'Unknown',
            'region': 'Unknown',
            'confidence': 'low',
            'place_type': 'generic',
            'landmarks': [],
            'description': '',
            'raw_response': response_text,
            'category_hint': category
        }
        
        try:
            lines = response_text.strip().split('\n')
            description_started = False
            description_lines = []
            
            for line in lines:
                line = line.strip()
                
                if description_started:
                    description_lines.append(line)
                    continue
                
                if line.startswith('EXACT_LOCATION:'):
                    result['exact_location'] = line.split(':', 1)[1].strip()
                elif line.startswith('LATITUDE:'):
                    lat_str = line.split(':', 1)[1].strip()
                    try:
                        result['latitude'] = float(lat_str)
                    except:
                        result['latitude'] = None
                elif line.startswith('LONGITUDE:'):
                    lon_str = line.split(':', 1)[1].strip()
                    try:
                        result['longitude'] = float(lon_str)
                    except:
                        result['longitude'] = None
                elif line.startswith('COUNTRY:'):
                    result['country'] = line.split(':', 1)[1].strip()
                elif line.startswith('CITY:'):
                    result['city'] = line.split(':', 1)[1].strip()
                elif line.startswith('REGION:'):
                    result['region'] = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    result['confidence'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('PLACE_TYPE:'):
                    result['place_type'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('LANDMARKS:'):
                    landmarks_str = line.split(':', 1)[1].strip()
                    if landmarks_str and landmarks_str.lower() != 'none':
                        result['landmarks'] = [l.strip() for l in landmarks_str.split(',')]
                elif line.startswith('DESCRIPTION:'):
                    description_started = True
            
            result['description'] = '\n'.join(description_lines).strip()
            
        except Exception as e:
            result['description'] = response_text
            result['parse_error'] = str(e)
        
        return result
    
    def get_location_summary(self, analysis_result: Dict) -> str:
        """
        Generate human-readable summary of location analysis
        
        Args:
            analysis_result: Result from analyze_location()
            
        Returns:
            Formatted string summary
        """
        if 'error' in analysis_result:
            return f"Error: {analysis_result['description']}"
        
        location = analysis_result['exact_location']
        confidence = analysis_result['confidence']
        
        # Confidence indicator
        conf_indicator = {
            'high': '[HIGH]',
            'medium': '[MEDIUM]',
            'low': '[LOW]'
        }.get(confidence, '[LOW]')
        
        summary = f"{conf_indicator} **Location Identified**\n\n"
        
        if location != 'Unknown' and 'cannot determine' not in location.lower():
            summary += f"**{location}**\n\n"
            
            if analysis_result.get('city') and analysis_result['city'] != 'Unknown':
                summary += f"City: {analysis_result['city']}\n"
            if analysis_result.get('country') and analysis_result['country'] != 'Unknown':
                summary += f"Country: {analysis_result['country']}\n"
            if analysis_result.get('region') and analysis_result['region'] != 'Unknown':
                summary += f"Region: {analysis_result['region']}\n"
            
            if analysis_result.get('latitude') and analysis_result.get('longitude'):
                lat = analysis_result['latitude']
                lon = analysis_result['longitude']
                summary += f"\nCoordinates: {lat:.4f}°, {lon:.4f}°\n"
                summary += f"[View on Google Maps](https://www.google.com/maps?q={lat},{lon})\n"
            
            if analysis_result.get('landmarks'):
                landmarks_str = ', '.join(analysis_result['landmarks'])
                summary += f"\nLandmarks: {landmarks_str}\n"
        else:
            summary += f"**Exact location could not be determined**\n\n"
            if analysis_result.get('region') != 'Unknown':
                summary += f"Estimated Region: {analysis_result['region']}\n"
        
        summary += f"\n**Confidence:** {confidence.upper()}\n"
        summary += f"\n**Analysis:**\n{analysis_result['description']}"
        
        return summary


def test_gemini_integration():
    """Test function to verify Gemini integration"""
    print("Testing Gemini Integration...")
    print("="*60)
    
    # Check if API key is available
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set in environment variables")
        print("\nTo use Gemini integration:")
        print("1. Get API key from: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: set GEMINI_API_KEY=your_key_here")
        return False
    
    try:
        analyzer = GeminiLocationAnalyzer(api_key)
        print("[OK] Gemini API initialized successfully")
        print("[OK] Ready to analyze images for exact locations")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


if __name__ == "__main__":
    test_gemini_integration()
