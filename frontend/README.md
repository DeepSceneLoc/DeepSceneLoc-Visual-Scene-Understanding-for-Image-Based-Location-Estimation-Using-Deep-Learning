# DeepSceneLoc

**A Deep Learning-based Geolocation Prediction System**

DeepSceneLoc is an academic research project that combines transfer learning, vision transformers, and multimodal AI reasoning to predict geographic coordinates from a single image. The system achieves high-precision geolocation estimates without GPS metadata using ResNet-50, EfficientNet, Vision Transformer (ViT), and Google's Gemini API.

## 🚀 Live Demo

**Try it now:** https://deepsceneloc.vercel.app/

## Features

- **Scene Classification**: ResNet-50 trained on Places365 dataset to categorize environments (Mountain, Urban, Coastal, Forest, Rural)
- **Feature Extraction**: EfficientNet-B0 for fine-grained spatial feature analysis
- **Vision Transformer**: Self-attention mechanisms for landmark and structural pattern recognition
- **Semantic Reasoning**: Gemini AI for multimodal geolocation analysis
- **High Accuracy**: 94.21% validation classifier accuracy
- **Interactive Demo**: Real-time image analysis with visualization tools

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Gemini API key

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd deepsceneloc
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env.local`
   - Add your `GEMINI_API_KEY` to `.env.local`

4. **Run the development server:**
   ```bash
   npm run dev
   ```

5. **Build for production:**
   ```bash
   npm run build
   npm start
   ```

## Project Structure

```
src/
├── components/          # React components (Map, Performance, Architecture)
├── assets/             # Images and static assets
├── App.tsx             # Main application component
├── types.ts            # TypeScript type definitions
└── index.css           # Global styles
```

## Technology Stack

- **Frontend**: React 19, TypeScript, Tailwind CSS, Vite
- **ML Models**: ResNet-50, EfficientNet-B0, Vision Transformer
- **AI API**: Google Gemini API
- **Backend**: Express.js
- **Deployment**: Node.js

## Team

**Final Year Computer Science Engineering Students:**
- **Krishan Yadav** - Technical Lead
- **Aditi Sah** - Data Lead
- **Anuj Kondawar** - Preprocessing Lead
- **Jensi Paneliya** - Documentation Lead

**Academic Sponsor**: Rashmi Pal

**Institution**: Parul University

## License

This project is part of an academic final-year research initiative. All rights reserved © 2026 DeepSceneLoc.

## Support

For inquiries or support, contact the academic team through Parul University.
