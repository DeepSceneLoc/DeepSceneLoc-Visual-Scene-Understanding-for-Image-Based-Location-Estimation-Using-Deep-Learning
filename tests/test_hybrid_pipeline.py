import pytest
from unittest.mock import MagicMock
from PIL import Image

from src.utils.pipeline_optimizer import TwoStagePipelineOptimizer, PipelineConfig, Stage1Result, Stage2Result

@pytest.fixture
def sample_image():
    return Image.new('RGB', (10, 10), color='black')

def test_pipeline_optimizer_skips_stage2_on_low_confidence(sample_image):
    config = PipelineConfig(stage1_confidence_threshold=0.5)
    mock_model = MagicMock()
    mock_gemini = MagicMock()
    
    optimizer = TwoStagePipelineOptimizer(model=mock_model, gemini_analyzer=mock_gemini, config=config)
    
    # Mock Stage 1 to return a low confidence result
    stage1_res = Stage1Result(
        predicted_class="Urban",
        confidence=0.4,
        top3=[("Urban", 0.4), ("Rural", 0.3), ("Forest", 0.2)],
        latency_ms=10.0,
        success=True
    )
    optimizer._run_stage1 = MagicMock(return_value=stage1_res)
    
    result = optimizer.run(sample_image)
    
    # Stage 2 (API) should NOT be called
    mock_gemini.analyze_location.assert_not_called()
    
    # Result should have a fallback stage2 result
    assert result.stage2 is not None
    assert result.stage2.source == "fallback"
    assert "low confidence" in result.stage2.description

def test_pipeline_optimizer_runs_stage2_on_high_confidence(sample_image):
    config = PipelineConfig(stage1_confidence_threshold=0.5)
    mock_model = MagicMock()
    mock_gemini = MagicMock()
    
    optimizer = TwoStagePipelineOptimizer(model=mock_model, gemini_analyzer=mock_gemini, config=config)
    
    # Mock Stage 1 to return a high confidence result
    stage1_res = Stage1Result(
        predicted_class="Urban",
        confidence=0.8,
        top3=[("Urban", 0.8), ("Rural", 0.1), ("Forest", 0.1)],
        latency_ms=10.0,
        success=True
    )
    optimizer._run_stage1 = MagicMock(return_value=stage1_res)
    
    # Mock Gemini response
    stage2_res = {"exact_location": "Eiffel Tower", "confidence": "high"}
    mock_gemini.analyze_location.return_value = stage2_res
    
    result = optimizer.run(sample_image)
    
    # Stage 2 should be called
    mock_gemini.analyze_location.assert_called_once()
    
    assert result.stage2 is not None
    assert result.stage2.exact_location == "Eiffel Tower"
    assert result.stage2.source == "gemini"
