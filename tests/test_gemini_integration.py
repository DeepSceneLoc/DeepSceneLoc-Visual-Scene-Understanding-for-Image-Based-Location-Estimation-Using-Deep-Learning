import pytest
import base64
from unittest.mock import MagicMock, patch
from PIL import Image
import io

from src.utils.gemini_integration import GeminiLocationAnalyzer

@pytest.fixture
def mock_genai():
    with patch('src.utils.gemini_integration.genai') as mock:
        yield mock

@pytest.fixture
def sample_image():
    # Create a 10x10 black image for testing
    img = Image.new('RGB', (10, 10), color='black')
    return img

def test_initialization_no_api_key(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(ValueError, match="Gemini API key not provided"):
        GeminiLocationAnalyzer(api_key=None)

def test_initialization_with_api_key(mock_genai, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    analyzer = GeminiLocationAnalyzer(api_key="test_key")
    mock_genai.configure.assert_called_once_with(api_key="test_key")
    mock_genai.GenerativeModel.assert_called_once_with('gemini-1.5-flash')

def test_analyze_location_success(mock_genai, sample_image):
    analyzer = GeminiLocationAnalyzer(api_key="test_key")
    
    # Mock the response from Gemini matching the structured text expected
    mock_response = MagicMock()
    mock_response.text = """EXACT_LOCATION: Times Square
LATITUDE: 40.7580
LONGITUDE: -73.9855
COUNTRY: USA
CITY: New York
REGION: NY
CONFIDENCE: high
PLACE_TYPE: landmark
LANDMARKS: TKTS Booth

DESCRIPTION:
Clear view of Times Square"""
    analyzer.model = MagicMock()
    analyzer.model.generate_content.return_value = mock_response

    result = analyzer.analyze_location(sample_image, "Urban", 0.95)

    assert result["exact_location"] == "Times Square"
    assert result["country"] == "USA"
    assert result["city"] == "New York"
    assert result["confidence"] == "high"
    assert "latency_ms" in result
    
    # Check that model.generate_content was called
    analyzer.model.generate_content.assert_called_once()
    
def test_analyze_location_fallback_on_invalid_text(mock_genai, sample_image):
    analyzer = GeminiLocationAnalyzer(api_key="test_key")
    
    # Mock a response that is NOT the valid structured text
    mock_response = MagicMock()
    mock_response.text = 'I cannot determine the location.'
    analyzer.model = MagicMock()
    analyzer.model.generate_content.return_value = mock_response

    result = analyzer.analyze_location(sample_image, "Urban", 0.95)

    # Should gracefully handle failure and return a generic fallback dict
    assert result["exact_location"] == "Unknown"
    assert result["confidence"] == "low"
    assert result["error"] is None # It doesn't actually flag it as an error, just parses what it can
