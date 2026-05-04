"""
Hybrid System Evaluation Script
DeepSceneLoc — Semester 2, Week 13

Author: Jensi Paneliya (Evaluation & Docs Lead)
Contributor: Anuj Kondawar

Evaluates the Hybrid Model (EfficientNet-B0 + Gemini API)
on a specific dataset of landmarks and generic scenes.
"""

import os
import json
import time
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch

from src.utils.pipeline_optimizer import TwoStagePipelineOptimizer, PipelineConfig
from src.utils.cache_manager import CachedGeminiAnalyzer
from src.utils.gemini_integration import GeminiLocationAnalyzer
from src.models.model_advanced import create_advanced_model

def load_evaluation_dataset(data_dir: str):
    """
    Load images from the evaluation dataset.
    Expected structure:
    data_dir/
       landmarks/
       generic/
    """
    dataset_path = Path(data_dir)
    images = []
    
    if not dataset_path.exists():
        print(f"[WARN] Evaluation dataset not found at {data_dir}")
        return images
        
    for category in ["landmarks", "generic"]:
        cat_dir = dataset_path / category
        if cat_dir.exists():
            for img_path in cat_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images.append({
                        "path": str(img_path),
                        "type": category,
                        "filename": img_path.name
                    })
    return images

def run_hybrid_evaluation(data_dir: str, model_path: str, output_file: str):
    print("="*60)
    print(" DeepSceneLoc - Hybrid System Evaluation ")
    print("="*60)
    
    # 1. Initialize Stage 1 Model
    print(f"[1/3] Loading Stage 1 Model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_advanced_model("efficientnet_b0", num_classes=5)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 2. Initialize Stage 2 API
    print("[2/3] Initializing Gemini API...")
    try:
        base_analyzer = GeminiLocationAnalyzer()
        # Use cache to save API quota during repeated evaluations
        analyzer = CachedGeminiAnalyzer(base_analyzer, db_path="results/eval_gemini_cache.db")
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Please set GEMINI_API_KEY in your .env file before running evaluation.")
        return
        
    optimizer = TwoStagePipelineOptimizer(
        model=model,
        gemini_analyzer=analyzer,
        config=PipelineConfig(stage1_confidence_threshold=0.5, enable_profiling=True)
    )
    
    # 3. Load Dataset
    print(f"[3/3] Loading dataset from {data_dir}...")
    images = load_evaluation_dataset(data_dir)
    if not images:
        print("No images found to evaluate. Please populate the evaluation dataset.")
        return
        
    print(f"\nEvaluating {len(images)} images...")
    results = []
    
    for item in tqdm(images):
        img_path = item["path"]
        try:
            img = Image.open(img_path).convert("RGB")
            res = optimizer.run(img)
            
            # Store flattened result for easy CSV export
            record = {
                "filename": item["filename"],
                "type": item["type"],
                "stage1_class": res.stage1.predicted_class,
                "stage1_conf": res.stage1.confidence,
                "stage2_location": res.stage2.exact_location if res.stage2 else "N/A",
                "stage2_confidence": res.stage2.confidence if res.stage2 else "N/A",
                "stage2_city": res.stage2.city if res.stage2 else "N/A",
                "stage2_country": res.stage2.country if res.stage2 else "N/A",
                "total_latency_ms": res.total_latency_ms,
                "success": res.pipeline_success
            }
            results.append(record)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    # Save Results
    df = pd.DataFrame(results)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print("\n" + "="*60)
    print(" Evaluation Complete ")
    print("="*60)
    print(f"Results saved to: {out_path}")
    
    # Print basic metrics
    success_rate = (df['success'].sum() / len(df)) * 100
    avg_latency = df['total_latency_ms'].mean()
    print(f"Overall Success Rate: {success_rate:.1f}%")
    print(f"Average End-to-End Latency: {avg_latency:.1f} ms")
    
    # Let the optimizer print its profiling
    optimizer.print_latency_report()

if __name__ == "__main__":
    run_hybrid_evaluation(
        data_dir="data/evaluation",
        model_path="models/checkpoints/efficientnet/EfficientNet-B0_best.pth",
        output_file="results/evaluation/hybrid_evaluation_results.csv"
    )
