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
            genai.configure(api_key=self.gemini_api_key, transport='rest')
            self.model = genai.GenerativeModel('gemini-3.1-flash-lite')
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
        
        # Create compact JSON prompt
        prompt = self._create_location_prompt(predicted_category, confidence)
        
        # Compress image to reduce payload size (max 800px wide)
        img_to_send = image.copy()
        max_size = 800
        if img_to_send.width > max_size:
            ratio = max_size / img_to_send.width
            img_to_send = img_to_send.resize(
                (max_size, int(img_to_send.height * ratio)),
                Image.LANCZOS
            )
        
        try:
            if self.mode == "openrouter":
                result_text = self._analyze_via_openrouter(img_to_send, prompt)
                result = self._parse_gemini_response(result_text, predicted_category)
            else:
                # Call Native Gemini API with JSON mode and 120s safety timeout
                response = self.model.generate_content(
                    [prompt, img_to_send],
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                        "max_output_tokens": 300,
                    },
                    request_options={"timeout": 120}
                )
                import json as _json
                raw = response.text.strip()
                parsed = _json.loads(raw)
                result = {
                    'exact_location': parsed.get('exact_location', 'Unknown'),
                    'latitude': parsed.get('latitude'),
                    'longitude': parsed.get('longitude'),
                    'country': parsed.get('country', 'Unknown'),
                    'city': parsed.get('city', 'Unknown'),
                    'region': parsed.get('region', 'Unknown'),
                    'confidence': parsed.get('confidence', 'low'),
                    'place_type': parsed.get('place_type', 'generic'),
                    'landmarks': parsed.get('landmarks', []),
                    'description': parsed.get('description', ''),
                    'category_hint': predicted_category
                }
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
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenRouter Error ({response.status_code}): {response.text}")
            
        data = response.json()
        return data['choices'][0]['message']['content']

    def _create_location_prompt(self, category: str = None, confidence: float = None) -> str:
        """Create a compact JSON-mode prompt that minimises output tokens for speed."""
        
        hint = ""
        if category:
            hint = f" The scene has already been classified as '{category}'"
            if confidence:
                hint += f" with {confidence:.0%} confidence."
        
        prompt = (
            f"You are a precise geospatial extraction assistant.{hint}\n"
            "Analyze this image and return ONLY a valid JSON object — no markdown, no extra text.\n"
            "If the image is too generic to pinpoint, use 0 for lat/lng and 'Unknown' for names.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "exact_location": "string — landmark or location name",\n'
            '  "latitude": number,\n'
            '  "longitude": number,\n'
            '  "country": "string",\n'
            '  "city": "string",\n'
            '  "region": "string",\n'
            '  "confidence": "high|medium|low",\n'
            '  "place_type": "landmark|city|natural_feature|generic_scene",\n'
            '  "landmarks": ["list", "of", "visible", "landmarks"],\n'
            '  "description": "max 2 sentences describing key visual evidence for identification"\n'
            "}"
        )
        return prompt
    
    def _parse_gemini_response(self, response_text: str, category: str = None) -> Dict:
        """Parse Gemini's structured response into dictionary"""
        import re
        
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
                line_stripped = line.strip()
                
                if description_started:
                    description_lines.append(line)
                    continue
                
                # Clean up line for matching (remove asterisks, bolding, leading bullet chars)
                clean_match = line_stripped.replace('**', '').replace('*', '').strip()
                clean_match = re.sub(r'^[-\+\#\s\d\.]+\s*', '', clean_match).strip()
                
                if clean_match.upper().startswith('DESCRIPTION:'):
                    description_started = True
                    desc_part = clean_match.split(':', 1)
                    if len(desc_part) > 1 and desc_part[1].strip():
                        description_lines.append(desc_part[1].strip())
                    continue
                
                if ':' in clean_match:
                    key, val = clean_match.split(':', 1)
                    key = key.strip().upper().replace(' ', '_')
                    val = val.strip()
                    
                    if key == 'EXACT_LOCATION':
                        result['exact_location'] = val
                    elif key == 'LATITUDE':
                        try:
                            coord_match = re.search(r'[-+]?\d*\.\d+|\d+', val)
                            if coord_match:
                                num = float(coord_match.group())
                                if 'S' in val.upper():
                                    num = -abs(num)
                                result['latitude'] = num
                        except:
                            result['latitude'] = None
                    elif key == 'LONGITUDE':
                        try:
                            coord_match = re.search(r'[-+]?\d*\.\d+|\d+', val)
                            if coord_match:
                                num = float(coord_match.group())
                                if 'W' in val.upper():
                                    num = -abs(num)
                                result['longitude'] = num
                        except:
                            result['longitude'] = None
                    elif key == 'COUNTRY':
                        result['country'] = val
                    elif key == 'CITY':
                        result['city'] = val
                    elif key == 'REGION':
                        result['region'] = val
                    elif key == 'CONFIDENCE':
                        result['confidence'] = val.lower()
                    elif key == 'PLACE_TYPE':
                        result['place_type'] = val.lower()
                    elif key == 'LANDMARKS':
                        if val and val.lower() != 'none':
                            result['landmarks'] = [l.strip() for l in val.split(',')]
            
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
