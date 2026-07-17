import React, { useState } from "react";
import { 
  Network, 
  Layers, 
  ArrowRight, 
  Cpu, 
  Database, 
  Check, 
  Sparkles,
  Info 
} from "lucide-react";

interface NodeData {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  color: string;
  badge: string;
  details: string[];
}

export default function ArchitectureGraph() {
  const [selectedNode, setSelectedNode] = useState<string>("resnet");

  const nodes: Record<string, NodeData> = {
    places365: {
      id: "places365",
      title: "Places365 Dataset",
      subtitle: "Source Corpus",
      badge: "DATA CORPUS",
      color: "border-slate-300 text-slate-700 bg-slate-50",
      description: "A standard benchmark designed for high-level scene classification containing over 1.8 million images across 365 scene categories.",
      details: [
        "Pre-trained category layers optimized for contextual indicators",
        "Assisted in teaching ResNet-50 the primary broad environments",
        "Features balanced subsets for Urban, Rural, and Wilderness biomes"
      ]
    },
    resnet: {
      id: "resnet",
      title: "ResNet-50",
      subtitle: "Transfer Learning Model",
      badge: "STAGE 1 MODEL",
      color: "border-blue-300 text-blue-700 bg-blue-50/50",
      description: "Deep convolutional model leveraged for Stage 1 coarse scene classification (Urban, Rural, Mountain, Coastal, Forest). Triggers prior knowledge from Places365.",
      details: [
        "Skip / Residual connections protect against vanishing gradients",
        "Deep 50-layer feature extraction for broad scene context mapping",
        "Finetuned in PyTorch with customized classification heads (94.21% accuracy)"
      ]
    },
    efficientnet: {
      id: "efficientnet",
      title: "EfficientNet-B0",
      subtitle: "Backbone Extractor",
      badge: "STAGE 2 FEED",
      color: "border-emerald-300 text-emerald-700 bg-emerald-50/50",
      description: "Lightweight, powerful scale-optimized convolution network to extract high-resolution botanical, architectural, and topological pixel features.",
      details: [
        "Compound scaling optimizes depth, width, and resolution simultaneously",
        "Extracts fine details like street furniture, flora, or structural style",
        "Feeds direct tensor weights to the Transformer module"
      ]
    },
    vit: {
      id: "vit",
      title: "Vision Transformer",
      subtitle: "Attention Map Module",
      badge: "STAGE 2 ATTENTION",
      color: "border-indigo-300 text-indigo-700 bg-indigo-50/50",
      description: "Accepts compound image patch tensors to compute spatial self-attention matrices. Resolves long-range geographic correlation without losing coordinate resolution.",
      details: [
        "16x16 pixel patch projection layer translates image blocks as tokens",
        "Self-attention mechanism registers high-priority architectural landmarks",
        "Preserves geographical relationship and pixel distances"
      ]
    },
    gemini: {
      id: "gemini",
      title: "Gemini AI Inference",
      subtitle: "Global GIS Grounding Engine",
      badge: "STAGE 2 FUSION",
      color: "border-violet-300 text-violet-700 bg-violet-50/50",
      description: "Integrates deep learning scene context and vision transformer coordinates to perform academic geolocation estimate, spatial deduction, and elevation verification.",
      details: [
        "Evaluates contextual metadata from visual foliage and soil shading",
        "Generates highly-resolved coordinates with descriptive geological analysis",
        "Supports fallback models for zero-shot geographical alignment"
      ]
    }
  };

  const nodeItem = nodes[selectedNode];

  return (
    <div className="w-full space-y-8">
      {/* Top Description bar */}
      <div className="border-b border-gray-150 pb-5">
        <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full uppercase tracking-wider">
          System Core Pipeline
        </span>
        <h3 className="text-xl font-bold font-sans text-gray-950 mt-1.5">
          Dual Stage Neural Architecture
        </h3>
        <p className="text-sm text-gray-500 mt-0.5">
          DeepSceneLoc operates by sequentially resolving a global scene envelope (Stage 1) before extracting fine spatial features and conducting deep semantic analysis (Stage 2).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Side: Dynamic Pipeline Blocks (7 of 12) */}
        <div className="lg:col-span-7 space-y-8">
          <div className="bg-gray-50/80 border border-gray-150 rounded-2xl p-6 relative overflow-hidden">
            
            {/* Stage 1 Block */}
            <div>
              <div className="text-[11px] font-mono font-bold text-gray-400 mb-3 tracking-widest uppercase">
                STAGE 1 — COARSE SCENE CLASSIFICATION (PyTorch)
              </div>
              <div className="flex flex-col sm:flex-row items-center gap-4">
                {/* Block: Places365 */}
                <button
                  id="btn-arch-places356"
                  onClick={() => setSelectedNode("places365")}
                  className={`w-full sm:w-2/5 p-4 border rounded-xl text-left transition-all cursor-pointer ${
                    selectedNode === "places365"
                      ? "bg-slate-900 text-white border-slate-900 shadow-lg scale-102"
                      : "bg-white text-gray-800 hover:border-gray-400 border-gray-200"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <Database className="w-4 h-4 text-blue-550 shrink-0" />
                    <span className="text-[10px] font-mono tracking-wider text-slate-500 uppercase">Input Corpus</span>
                  </div>
                  <h5 className="font-bold text-sm leading-tight">Places365</h5>
                  <p className="text-[11px] opacity-75 mt-0.5">1.8M classification dataset</p>
                </button>

                <div className="rotate-90 sm:rotate-0 text-slate-400">
                  <ArrowRight className="w-5 h-5" />
                </div>

                {/* Block: ResNet-50 */}
                <button
                  id="btn-arch-resnet"
                  onClick={() => setSelectedNode("resnet")}
                  className={`w-full sm:w-3/5 p-4 border rounded-xl text-left transition-all cursor-pointer ${
                    selectedNode === "resnet"
                      ? "bg-blue-600 text-white border-blue-600 shadow-lg scale-102"
                      : "bg-white text-gray-800 hover:border-gray-400 border-gray-200"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <Layers className="w-4 h-4 text-emerald-500 shrink-0" />
                    <span className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">Model Backbone</span>
                  </div>
                  <h5 className="font-bold text-sm leading-tight">ResNet-50 Transfer</h5>
                  <p className="text-[11px] opacity-80 mt-0.5">Preds: Urban, Coastal, Mountain...</p>
                </button>
              </div>
            </div>

            {/* Stage divider line */}
            <div className="my-6 border-t border-dashed border-gray-200 relative flex justify-center">
              <span className="absolute -top-3 px-3 py-1 bg-white border border-gray-150 rounded-full text-[9px] font-mono text-gray-400 tracking-wider">
                PIPELINE ENRICHMENT CONTEXT FEED (STAGES BRIDGE)
              </span>
            </div>

            {/* Stage 2 Block */}
            <div>
              <div className="text-[11px] font-mono font-bold text-gray-400 mb-3 tracking-widest uppercase">
                STAGE 2 — FINE SPATIAL LOCALIZATION & FUSION
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-stretch">
                {/* Block: EfficientNet-B0 */}
                <button
                  id="btn-arch-efficientnet"
                  onClick={() => setSelectedNode("efficientnet")}
                  className={`p-4 border rounded-xl text-left transition-all cursor-pointer flex flex-col justify-between ${
                    selectedNode === "efficientnet"
                      ? "bg-emerald-600 text-white border-emerald-600 shadow-lg scale-102"
                      : "bg-white text-gray-800 hover:border-gray-400 border-gray-200"
                  }`}
                >
                  <div>
                    <div className="flex items-center gap-1.5 mb-2">
                      <Cpu className="w-3.5 h-3.5 text-emerald-500" />
                      <span className="text-[9px] font-mono tracking-wider text-slate-400 uppercase">Conv2D</span>
                    </div>
                    <h5 className="font-bold text-xs leading-snug">EfficientNet-B0</h5>
                  </div>
                  <p className="text-[10px] opacity-80 mt-2">Compound coordinate scaling</p>
                </button>

                {/* Block: ViT */}
                <button
                  id="btn-arch-vit"
                  onClick={() => setSelectedNode("vit")}
                  className={`p-4 border rounded-xl text-left transition-all cursor-pointer flex flex-col justify-between ${
                    selectedNode === "vit"
                      ? "bg-indigo-650 text-white border-indigo-650 shadow-lg scale-102"
                      : "bg-white text-gray-800 hover:border-gray-400 border-gray-200"
                  }`}
                >
                  <div>
                    <div className="flex items-center gap-1.5 mb-2">
                      <Network className="w-3.5 h-3.5 text-indigo-500" />
                      <span className="text-[9px] font-mono tracking-wider text-slate-400 uppercase">Attention</span>
                    </div>
                    <h5 className="font-bold text-xs leading-snug">Vision Transformer (ViT)</h5>
                  </div>
                  <p className="text-[10px] opacity-80 mt-2">16x16 Self-Attention maps</p>
                </button>

                {/* Block: Gemini */}
                <button
                  id="btn-arch-gemini"
                  onClick={() => setSelectedNode("gemini")}
                  className={`p-4 border rounded-xl text-left transition-all cursor-pointer flex flex-col justify-between ${
                    selectedNode === "gemini"
                      ? "bg-violet-600 text-white border-violet-600 shadow-lg scale-102"
                      : "bg-white text-gray-800 hover:border-gray-400 border-gray-200"
                  }`}
                >
                  <div>
                    <div className="flex items-center gap-1.5 mb-2">
                      <Sparkles className="w-3.5 h-3.5 text-violet-400" />
                      <span className="text-[9px] font-mono tracking-wider text-slate-400 uppercase">LMM</span>
                    </div>
                    <h5 className="font-bold text-xs leading-snug">Gemini AI Engine</h5>
                  </div>
                  <p className="text-[10px] opacity-80 mt-2">Global GIS reasoning & elevation</p>
                </button>
              </div>
            </div>

            {/* Outgoing target prediction marker arrow */}
            <div className="mt-6 flex justify-end items-center gap-2">
              <span className="text-[10px] font-mono text-gray-400">FINAL CLASSIFICATION REPORT OUTPUT</span>
              <div className="bg-emerald-500 text-white p-1 rounded-full">
                <Check className="w-3.5 h-3.5" />
              </div>
            </div>

          </div>
        </div>

        {/* Right Side: Detailed Block Meta Card (5 of 12) */}
        <div id="arch-details-sidebar" className="lg:col-span-5 bg-white border border-gray-100 rounded-2xl p-6 shadow-xs relative">
          <div className="absolute top-4 right-4 text-gray-300">
            <Info className="w-5 h-5 text-gray-400" />
          </div>

          <div className="space-y-4">
            <div>
              <span className="text-[10px] font-mono font-bold bg-blue-50 text-blue-600 px-2 py-0.5 rounded uppercase">
                {nodeItem.badge}
              </span>
              <h4 className="text-lg font-bold text-gray-900 mt-2">{nodeItem.title}</h4>
              <p className="text-xs text-slate-400 mt-0.5 font-mono">{nodeItem.subtitle}</p>
            </div>

            <p className="text-sm text-gray-600 leading-relaxed font-sans border-t border-gray-100 pt-4">
              {nodeItem.description}
            </p>

            <div className="space-y-2.5 pt-2">
              <span className="text-[10px] font-mono font-semibold text-gray-400 block tracking-widest uppercase mb-1">
                KEY SPECIFICATIONS:
              </span>
              {nodeItem.details.map((detail, idx) => (
                <div key={idx} className="flex items-start gap-2.5 text-xs text-gray-700">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-650 mt-1.5 shrink-0" />
                  <p className="leading-snug">{detail}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
