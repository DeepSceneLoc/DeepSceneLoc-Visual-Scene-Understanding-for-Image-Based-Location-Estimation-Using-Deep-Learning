"""
DeepSceneLoc - Interactive Demo App
Gradio interface for visual demonstration
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.models.model import create_model
except:
    print("Note: Running in standalone mode")


# Define location categories
CATEGORIES = ['Coastal', 'Forest', 'Mountain', 'Rural', 'Urban']

# Category descriptions
CATEGORY_INFO = {
    'Coastal': '🌊 Beaches, harbors, seaside, ocean views',
    'Forest': '🌲 Dense vegetation, woodland, jungle areas',
    'Mountain': '⛰️ Peaks, highlands, mountain terrain, valleys',
    'Rural': '🌾 Farmland, countryside, villages, agricultural areas',
    'Urban': '🏙️ Cities, streets, buildings, urban infrastructure'
}


class DemoModel:
    """Wrapper for model with demo-ready prediction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.categories = CATEGORIES
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model (or create untrained for demo)
        try:
            self.model = create_model('resnet50', num_classes=5, pretrained=True)
            # Try to load trained weights if available
            try:
                checkpoint = torch.load('models/checkpoints/best_model.pth', 
                                       map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_status = "✅ Trained Model Loaded"
            except:
                self.model_status = "⚠️ Using Pretrained Backbone (Demo Mode)"
        except:
            # Fallback: create simple model for demo
            import torchvision.models as models
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 5)
            self.model_status = "⚠️ Demo Model (Architecture Preview)"
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image):
        """Predict location category from image"""
        if image is None:
            return None
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Format results
        results = {}
        for i, category in enumerate(self.categories):
            results[category] = float(probabilities[i])
        
        return results


# Initialize model
print("Loading DeepSceneLoc model...")
demo_model = DemoModel()
print(f"Model status: {demo_model.model_status}")


def predict_location(image):
    """Main prediction function for Gradio"""
    if image is None:
        return None, "Please upload an image", ""
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Get predictions
    predictions = demo_model.predict(image)
    
    if predictions is None:
        return None, "Error in prediction", ""
    
    # Get top prediction
    top_category = max(predictions, key=predictions.get)
    top_confidence = predictions[top_category]
    
    # Format result text
    result_text = f"""
## 🎯 Predicted Location: **{top_category}**
**Confidence:** {top_confidence:.2%}

### Category Info:
{CATEGORY_INFO[top_category]}

---

### All Predictions:
"""
    
    # Add all predictions sorted
    for category in sorted(predictions, key=predictions.get, reverse=True):
        bar_length = int(predictions[category] * 20)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        result_text += f"\n**{category}:** {bar} {predictions[category]:.1%}"
    
    return predictions, result_text, demo_model.model_status


# Create Gradio Interface
with gr.Blocks(title="DeepSceneLoc - Location Estimator", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🌍 DeepSceneLoc
    ## Visual Scene Understanding for Image-Based Location Estimation
    
    Upload an image to predict its semantic location category using deep learning.
    
    **Categories:** Coastal 🌊 | Forest 🌲 | Mountain ⛰️ | Rural 🌾 | Urban 🏙️
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            image_input = gr.Image(
                label="Upload Image",
                type="pil",
                sources=["upload", "webcam", "clipboard"]
            )
            
            predict_btn = gr.Button("🔍 Predict Location", variant="primary", size="lg")
            
            gr.Markdown("""
            ### About This Project
            DeepSceneLoc uses transfer learning with ResNet-50 to classify images 
            into 5 semantic location categories based purely on visual features.
            
            **No GPS or metadata required!**
            
            **Team:** Krishan Yadav, Aditi Sah, Anuj Kondawar, Jensi Paneliya
            """)
        
        with gr.Column(scale=1):
            # Outputs
            model_status = gr.Textbox(
                label="Model Status",
                value=demo_model.model_status,
                interactive=False
            )
            
            result_text = gr.Markdown(label="Prediction Results")
            
            prediction_plot = gr.Label(
                label="Confidence Scores",
                num_top_classes=5
            )
    
    # Examples
    gr.Markdown("### 📸 Try These Examples:")
    gr.Examples(
        examples=[
            # These would be paths to sample images
            # For now, users can upload their own
        ],
        inputs=image_input,
        label="Sample Images (Upload your own to test)"
    )
    
    # Event handlers
    predict_btn.click(
        fn=predict_location,
        inputs=[image_input],
        outputs=[prediction_plot, result_text, model_status]
    )
    
    image_input.change(
        fn=predict_location,
        inputs=[image_input],
        outputs=[prediction_plot, result_text, model_status]
    )
    
    gr.Markdown("""
    ---
    
    ### 🔬 Technical Details
    - **Architecture:** ResNet-50 with custom classification head (2048 → 512 → 5)
    - **Training:** Transfer learning from ImageNet pretrained weights
    - **Dataset:** Places365 outdoor subset mapped to 5 categories
    - **Framework:** PyTorch 2.0+
    
    **GitHub:** [DeepSceneLoc Repository](https://github.com/KrishanYadav333/DeepSceneLoc)
    """)


if __name__ == "__main__":
    print("="*60)
    print("DeepSceneLoc - Interactive Demo")
    print("="*60)
    print(f"Model Status: {demo_model.model_status}")
    print(f"Device: {demo_model.device}")
    print(f"Categories: {', '.join(CATEGORIES)}")
    print("="*60)
    print("\n🚀 Launching Gradio interface...")
    print("📱 Access the demo at the URL shown below\n")
    
    # Launch with share=True to get public URL for mentor demo
    demo.launch(
        share=True,  # Creates public URL for 72 hours
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
