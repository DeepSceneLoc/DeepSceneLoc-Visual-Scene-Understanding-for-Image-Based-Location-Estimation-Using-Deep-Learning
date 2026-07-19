import React, { useState, useRef, useEffect } from "react";
const appLogoDsl = "/src/assets/images/app_logo_dsl_1781883330018.jpg";
const teamAnujImg = "/src/assets/images/anuj.jpeg";
const teamKrishanImg = "/src/assets/images/krishan.jpeg";
const teamJensiImg = "/src/assets/images/jensi.jpeg";
const teamAditiImg = "/src/assets/images/Aditi.jpeg";
import { 
  Upload, 
  Map, 
  Cpu, 
  Info, 
  Users, 
  Activity, 
  ArrowRight, 
  Search, 
  Sparkles, 
  Code, 
  Compass, 
  Globe, 
  ShieldAlert, 
  CheckCircle2, 
  RefreshCw,
  AlertCircle,
  FileText,
  MapPin,
  ChevronRight,
  Database,
  X
} from "lucide-react";
import InteractiveMap from "./components/InteractiveMap";
import ModelPerformance from "./components/ModelPerformance";
import ArchitectureGraph from "./components/ArchitectureGraph";
import DeepSceneLocLogo from "./components/DeepSceneLocLogo";
import { PipelineOutput, PresetDemo, SceneCategory } from "./types";

export default function App() {
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [currentPrediction, setCurrentPrediction] = useState<PipelineOutput | null>(null);
  const [pipelineStep, setPipelineStep] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [presets, setPresets] = useState<PresetDemo[]>([]);
  const [showRealAPIAlert, setShowRealAPIAlert] = useState<boolean>(false);
  const [isFallbackActive, setIsFallbackActive] = useState<boolean>(false);
  const [isDemoInView, setIsDemoInView] = useState<boolean>(false);
  const [activeNav, setActiveNav] = useState<string>("project-overview");
  const [showAnujPhotoModal, setShowAnujPhotoModal] = useState<boolean>(false);
  const [showKrishanPhotoModal, setShowKrishanPhotoModal] = useState<boolean>(false);
  const [showJensiPhotoModal, setShowJensiPhotoModal] = useState<boolean>(false);
  const [showAditiPhotoModal, setShowAditiPhotoModal] = useState<boolean>(false);

  const isScrollingRef = useRef<boolean>(false);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Fetch presets on load
    fetch("/api/presets")
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setPresets(data.presets);
        }
      })
      .catch(err => console.error("Could not fetch analytical presets", err));

    // Initialize with default state (empty now)
  }, []);

  useEffect(() => {
    const sectionIds = [
      "project-overview",
      "how-it-works",
      "interactive-demo",
      "model-architecture",
      "performance-metrics",
      "tech-stack",
      "academic-team"
    ];

    const visibleRatios: Record<string, number> = {};

    const observerOption = {
      root: null,
      rootMargin: "-20% 0px -30% 0px", // focus on central-upper part of screen
      threshold: [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    };

    const handleIntersection = (entries: IntersectionObserverEntry[]) => {
      entries.forEach(entry => {
        const id = entry.target.id;
        if (entry.isIntersecting) {
          visibleRatios[id] = entry.intersectionRatio;
        } else {
          visibleRatios[id] = 0;
        }

        // Also update standard isDemoInView variable for standard button pulsing
        if (id === "interactive-demo") {
          setIsDemoInView(entry.isIntersecting);
        }
      });

      if (isScrollingRef.current) return;

      let highestRatio = -1;
      let activeId = "";

      sectionIds.forEach(id => {
        const ratio = visibleRatios[id] || 0;
        if (ratio > highestRatio && ratio > 0) {
          highestRatio = ratio;
          activeId = id;
        }
      });

      if (activeId) {
        setActiveNav(activeId);
      } else if (window.scrollY < 120) {
        setActiveNav("project-overview");
      }
    };

    const observer = new IntersectionObserver(handleIntersection, observerOption);

    sectionIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    const handleScroll = () => {
      if (isScrollingRef.current) return;
      if (window.scrollY < 120) {
        setActiveNav("project-overview");
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", handleScroll);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Handle Preset Selection Click
  const handlePresetSelect = async (presetId: string) => {
    setSelectedPresetId(presetId);
    setImageBase64(null);
    setImageFile(null);
    setIsAnalyzing(true);
    setPipelineStep(1);
    setErrorMessage(null);

    // Run pipeline step indicators sequential intervals
    const timers = [
      setTimeout(() => setPipelineStep(2), 250), // ResNet-50
      setTimeout(() => setPipelineStep(3), 500), // EfficientNet
      setTimeout(() => setPipelineStep(4), 750), // ViT
      setTimeout(() => setPipelineStep(5), 1000), // Gemini AI
    ];

    try {
      const response = await fetch("/api/analyze-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ presetId })
      });
      const result = await response.json();
      
      timers.forEach(t => clearTimeout(t));

      if (result.success) {
        setPipelineStep(6);
        setCurrentPrediction(result.data);
        setIsFallbackActive(false);
      } else {
        setErrorMessage(result.error);
        setPipelineStep(0);
      }
    } catch (e) {
      timers.forEach(t => clearTimeout(t));
      setErrorMessage("Network error during image evaluation step.");
      setPipelineStep(0);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Convert uploaded image to base64
  const processImageFile = (file: File) => {
    if (file.size > 12 * 1024 * 1024) {
      setErrorMessage("Image file exceeds the 12MB GIS inspection threshold limit.");
      return;
    }
    
    setImageFile(file);
    setSelectedPresetId("");
    setErrorMessage(null);

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      setImageBase64(base64String);
      // Automatically trigger active pipeline prediction
      evaluateUploadedImage(base64String);
    };
    reader.onerror = () => {
      setErrorMessage("Failed to decode uploaded image bits.");
    };
    reader.readAsDataURL(file);
  };

  // Submit custom upload to Express endpoint
  const evaluateUploadedImage = async (base64Data: string) => {
    setIsAnalyzing(true);
    setPipelineStep(1);
    setErrorMessage(null);

    // Visual indicators loop for the neural pipeline
    const timers = [
      setTimeout(() => setPipelineStep(2), 400),
      setTimeout(() => setPipelineStep(3), 900),
      setTimeout(() => setPipelineStep(4), 1400),
      setTimeout(() => setPipelineStep(5), 2000),
    ];

    try {
      const response = await fetch("/api/analyze-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageBase64: base64Data })
      });
      const result = await response.json();

      timers.forEach(t => clearTimeout(t));

      if (result.success) {
        setPipelineStep(6);
        setCurrentPrediction(result.data);
        setIsFallbackActive(!!result.fallbackActive);
        if (result.isDemoKey) {
          setShowRealAPIAlert(true);
        }
      } else {
        setErrorMessage(result.error);
        setPipelineStep(0);
      }
    } catch (err: any) {
      timers.forEach(t => clearTimeout(t));
      setErrorMessage("Failed to connect to the custom GIS evaluation service.");
      setPipelineStep(0);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Drop Handler
  const handleImageDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processImageFile(e.dataTransfer.files[0]);
    }
  };

  // Select File manually
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processImageFile(e.target.files[0]);
    }
  };

  // Scroll Helpers
  const scrollToDemo = () => {
    setActiveNav("interactive-demo");
    isScrollingRef.current = true;
    if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    document.getElementById("interactive-demo")?.scrollIntoView({ behavior: "smooth" });
    scrollTimeoutRef.current = setTimeout(() => {
      isScrollingRef.current = false;
    }, 800);
  };

  const scrollToCoreArch = () => {
    setActiveNav("model-architecture");
    isScrollingRef.current = true;
    if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    document.getElementById("model-architecture")?.scrollIntoView({ behavior: "smooth" });
    scrollTimeoutRef.current = setTimeout(() => {
      isScrollingRef.current = false;
    }, 800);
  };

  const handleNavClick = (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    setActiveNav(id);
    
    const el = document.getElementById(id);
    if (el) {
      isScrollingRef.current = true;
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
      
      el.scrollIntoView({ behavior: "smooth" });
      
      scrollTimeoutRef.current = setTimeout(() => {
        isScrollingRef.current = false;
      }, 800);
    }
  };

  const navItems = [
    { id: "project-overview", label: "Overview" },
    { id: "how-it-works", label: "How it Works" },
    { id: "interactive-demo", label: "Demo Console" },
    { id: "model-architecture", label: "Architecture" },
    { id: "performance-metrics", label: "Evaluation" },
    { id: "tech-stack", label: "Tech-Stack" },
    { id: "academic-team", label: "Research Team" }
  ];

  return (
    <div className="min-h-screen bg-white text-gray-800 font-sans antialiased selection:bg-blue-105 selection:text-blue-900">
      
      {/* ──────────────────── NAVIGATION BAR ──────────────────── */}
      <header className="sticky top-0 z-40 bg-white/95 border-b border-gray-100 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-14 h-10 rounded-md overflow-hidden bg-transparent border border-gray-200 shadow-xs flex items-center justify-center">
              <img
                src={appLogoDsl}
                alt="DeepSceneLoc Logo"
                className="w-full h-full object-cover"
                referrerPolicy="no-referrer"
              />
            </div>
            <div>
              <span className="font-extrabold font-sans text-gray-950 tracking-tight text-base block">
                DeepSceneLoc
              </span>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-6 text-sm font-semibold">
            {navItems.map((item) => {
              const isActive = activeNav === item.id;
              return (
                <a
                  key={item.id}
                  href={`#${item.id}`}
                  onClick={(e) => handleNavClick(item.id, e)}
                  className={`transition-all duration-200 pb-0.5 border-b-2 ${
                    isActive
                      ? "text-blue-600 border-blue-600 font-bold"
                      : "text-gray-500 border-transparent hover:text-blue-600"
                  }`}
                >
                  {item.label}
                </a>
              );
            })}
          </nav>

          <div className="flex items-center gap-3">
            <button 
              id="nav-btn-evaluate"
              onClick={scrollToDemo}
              className={`font-semibold text-xs py-2 px-4 rounded-lg tracking-tight transition-all duration-200 ${
                isDemoInView
                  ? "bg-blue-600 hover:bg-blue-700 text-white animate-subtle-pulse shadow-md"
                  : "bg-gray-950 hover:bg-blue-650 text-white shadow-xs"
              }`}
            >
              Analyze Image
            </button>
          </div>
        </div>
      </header>

      {/* ──────────────────── SECTION 1 — HERO ──────────────────── */}
      <section className="relative overflow-hidden pt-12 pb-20 border-b border-gray-100 bg-[radial-gradient(circle_at_top_right,rgba(59,130,246,0.04),transparent_50%)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            
            {/* Left Headline blocks */}
            <div className="lg:col-span-6 space-y-6">
              <span className="inline-flex items-center gap-1.5 bg-blue-50 text-blue-700 font-mono text-xs font-bold px-3 py-1.5 rounded-full tracking-wide uppercase">
                <Sparkles className="w-3.5 h-3.5" />
                Transfer learning + Transformer + Gemini AI
              </span>
              
              <h1 className="text-4xl sm:text-5xl lg:text-5.5xl font-extrabold tracking-tight text-gray-950 leading-tight">
                Find Any Place on Earth From a <span className="text-blue-600">Single Image</span>
              </h1>

              <p className="text-base text-gray-500 leading-relaxed max-w-xl">
                DeepSceneLoc combines fast neural scene classifiers, robust vision transformers, and global visual reasoning models to predict highly accurate geographic coordinates without GPS metadata.
              </p>

              <div className="flex flex-wrap items-center gap-4 pt-2">
                <button
                  id="hero-btn-upload"
                  onClick={scrollToDemo}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl shadow-lg transition-all flex items-center gap-2 transform hover:-translate-y-0.5 cursor-pointer"
                >
                  <Upload className="w-4 h-4" />
                  Upload & Estimate
                </button>
                <button
                  id="hero-btn-explore-arch"
                  onClick={scrollToCoreArch}
                  className="border border-gray-200 bg-white hover:bg-gray-50 text-gray-700 font-bold py-3 px-6 rounded-xl transition-all flex items-center gap-2 cursor-pointer"
                >
                  Explore Architecture
                  <ArrowRight className="w-4 h-4 text-gray-400" />
                </button>
              </div>

              {/* Research Footnotes */}
              <div className="pt-6 border-t border-gray-150 flex items-center gap-6 text-xs text-gray-400">
                <div>
                  <span className="block text-gray-800 font-bold">ResNet-50</span>
                  Places365 Contextual Classifier
                </div>
                <div className="h-6 w-[1.5px] bg-gray-200" />
                <div>
                  <span className="block text-gray-800 font-bold">EfficientNet + ViT</span>
                  Spatial Feature Attention 
                </div>
                <div className="h-6 w-[1.5px] bg-gray-200" />
                <div>
                  <span className="block text-gray-800 font-bold">94.21%</span>
                  Validation Classifier Accuracy
                </div>
              </div>
            </div>

            {/* Right Side: Hero Illustration (Interactive visual workflow block) */}
            <div id="hero-illustration-wrapper" className="lg:col-span-6 bg-gray-50 border border-gray-200 p-8 rounded-2xl relative shadow-xs">
              
              <div className="absolute top-4 right-4 bg-white/95 border border-gray-200 rounded-lg p-2 flex items-center gap-1.5 text-[10px] font-mono text-gray-500 shadow-xs z-10">
                <span className="w-2 h-2 rounded-full bg-blue-600 animate-pulse" />
                ACTIVE INTERACTIVE WORKFLOW
              </div>

              <div className="space-y-6 relative">
                
                {/* Step 1 Illustration Block */}
                <div className="bg-white border border-gray-150 rounded-xl p-4 shadow-sm flex items-center gap-4 hover:border-blue-400 transition-all">
                  <div className="w-12 h-12 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center font-bold font-mono">
                    01
                  </div>
                  <div className="flex-1">
                    <span className="text-[10px] font-mono font-extrabold text-blue-500 block uppercase">Input Raster</span>
                    <h4 className="font-bold text-gray-900 text-sm leading-tight">Image Upload / Capture</h4>
                    <p className="text-xs text-gray-400 mt-0.5">Accepts JPEG, PNG with local or satellite perspective</p>
                  </div>
                </div>

                {/* Connection Arrow */}
                <div className="flex justify-center -my-2">
                  <div className="h-5 w-0.5 bg-gradient-to-b from-blue-550 to-emerald-500" />
                </div>

                {/* Step 2 Illustration Block */}
                <div className="bg-white border border-gray-150 rounded-xl p-4 shadow-sm flex items-center gap-4 hover:border-emerald-450 transition-all">
                  <div className="w-12 h-12 rounded-lg bg-emerald-50 text-emerald-700 flex items-center justify-center font-bold font-mono">
                    02
                  </div>
                  <div className="flex-1">
                    <span className="text-[10px] font-mono font-extrabold text-emerald-500 block uppercase">STAGE 1: PRIOR ENVIRONMENT</span>
                    <h4 className="font-bold text-gray-900 text-sm leading-tight">Scene Classification</h4>
                    <p className="text-xs text-gray-400 mt-0.5">Places365 ResNet-50 weights: Mountain, Urban, Coastal...</p>
                  </div>
                </div>

                {/* Connection Arrow */}
                <div className="flex justify-center -my-2">
                  <div className="h-5 w-0.5 bg-gradient-to-b from-emerald-500 to-indigo-500" />
                </div>

                {/* Step 3 Illustration Block */}
                <div className="bg-white border border-gray-150 rounded-xl p-4 shadow-sm flex items-center gap-4 hover:border-indigo-400 transition-all">
                  <div className="w-12 h-12 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center font-bold font-mono">
                    03
                  </div>
                  <div className="flex-1">
                    <span className="text-[10px] font-mono font-extrabold text-indigo-500 block uppercase">STAGE 2: SPATIAL FOCUS</span>
                    <h4 className="font-bold text-gray-900 text-sm leading-tight">Feature & Attention Extraction</h4>
                    <p className="text-xs text-gray-400 mt-0.5">EfficientNet-B0 and Vision Transformer self-attention mapping</p>
                  </div>
                </div>

                {/* Connection Arrow */}
                <div className="flex justify-center -my-2">
                  <div className="h-5 w-0.5 bg-gradient-to-b from-indigo-500 to-violet-500" />
                </div>

                {/* Step 4 Illustration Block */}
                <div className="bg-white border border-gray-150 rounded-xl p-4 shadow-sm flex items-center gap-4 hover:border-violet-400 transition-all">
                  <div className="w-12 h-12 rounded-lg bg-violet-50 text-violet-700 flex items-center justify-center font-bold font-mono">
                    04
                  </div>
                  <div className="flex-1">
                    <span className="text-[10px] font-mono font-extrabold text-violet-500 block uppercase">STAGE 3: SEMANTIC MATCHING</span>
                    <h4 className="font-bold text-gray-900 text-sm leading-tight">Exact Geolocation Resolve</h4>
                    <p className="text-xs text-gray-400 mt-0.5">Gemini AI evaluates physical foliage, elevation and local markers</p>
                  </div>
                </div>

              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ──────────────────── SECTION 2 — PROJECT OVERVIEW ──────────────────── */}
      <section id="project-overview" className="py-20 bg-gray-50/50 border-b border-gray-100 relative">
        {/* Subtle background world map vector mesh simulation */}
        <div className="absolute inset-0 opacity-5 pointer-events-none overflow-hidden">
          <svg className="w-full h-full" viewBox="0 0 1000 600" fill="none" stroke="currentColor">
            {/* Outline grid shapes that mimic global coordinates */}
            <path d="M100,50 Q200,150 400,200 T800,250 M50,200 Q250,220 500,100 T950,150 M50,450 Q300,300 700,500 T900,100" strokeWidth="2" strokeDasharray="5 5" />
            <circle cx="200" cy="150" r="3" fill="#3b82f6" />
            <circle cx="500" cy="220" r="4.5" fill="#3b82f6" />
            <circle cx="780" cy="350" r="3" fill="#3b82f6" />
          </svg>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="max-w-3xl mx-auto text-center space-y-4">
            <span className="text-xs font-mono font-bold tracking-widest text-blue-600 bg-blue-100/50 px-3 py-1 rounded-full uppercase">
              Project Context & Goals
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-gray-950">
              About DeepSceneLoc
            </h2>
            <p className="text-base text-gray-500 leading-relaxed">
              DeepSceneLoc is an academic research and engineering pipeline targeting high-precision geolocation estimates from photographic visual inputs alone. Designed as a final-year CS project, the platform demonstrates how deep image understanding bypasses GPS spoofing issues.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
            
            {/* Box 1 */}
            <div id="overview-card-scene" className="bg-white border border-gray-150 rounded-2xl p-6 shadow-xs hover:border-blue-500 transition-all hover:scale-[1.01]">
              <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center font-bold mb-5">
                <Globe className="w-5 h-5" />
              </div>
              <h4 className="font-bold text-gray-900 text-lg leading-tight">Scene Understanding</h4>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Stage 1 employs transfer learning models to characterize rural landscapes, concrete urban layouts, ocean coastlines, and mountain parameters to restrict focus zones globally.
              </p>
            </div>

            {/* Box 2 */}
            <div id="overview-card-vision" className="bg-white border border-gray-150 rounded-2xl p-6 shadow-xs hover:border-blue-500 transition-all hover:scale-[1.01]">
              <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-750 flex items-center justify-center font-bold mb-5">
                <Cpu className="w-5 h-5" />
              </div>
              <h4 className="font-bold text-gray-900 text-lg leading-tight">Computer Vision</h4>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                EfficientNet compound scaling model analyzes high-resolution sub-pixel artifacts, flora types, building architecture styles, and traffic signs orientation in milliseconds.
              </p>
            </div>

            {/* Box 3 */}
            <div id="overview-card-landmark" className="bg-white border border-gray-150 rounded-2xl p-6 shadow-xs hover:border-blue-500 transition-all hover:scale-[1.01]">
              <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold mb-5">
                <MapPin className="w-5 h-5" />
              </div>
              <h4 className="font-bold text-gray-900 text-lg leading-tight">Landmark Recognition</h4>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Self-attention mechanics from our Vision Transformer target structural symmetries and famous historic elements to identify landmark candidates securely on the map.
              </p>
            </div>

            {/* Box 4 */}
            <div id="overview-card-geolocation" className="bg-white border border-gray-150 rounded-2xl p-6 shadow-xs hover:border-blue-500 transition-all hover:scale-[1.01]">
              <div className="w-10 h-10 rounded-xl bg-violet-50 text-violet-600 flex items-center justify-center font-bold mb-5">
                <Compass className="w-5 h-5" />
              </div>
              <h4 className="font-bold text-gray-900 text-lg leading-tight">Geolocation Prediction</h4>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Multi-stage fusion merges pixel analysis with the Gemini AI reasoning model for localized coordinate outputs and comprehensive spatial reports.
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* ──────────────────── SECTION 3 — HOW IT WORKS ──────────────────── */}
      <section id="how-it-works" className="py-20 border-b border-gray-100 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-12">
            <div>
              <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full uppercase tracking-wider block w-fit">
                Sequential Workflow Steps
              </span>
              <h2 className="text-3xl font-extrabold text-gray-950 mt-2">
                Operational Image Search Pipeline
              </h2>
            </div>
            <p className="text-xs text-gray-400 font-mono mt-2 md:mt-0 max-w-sm">
              Each frame is parsed linearly across neural and language layers to output exact GIS coordinate structures.
            </p>
          </div>

          {/* Horizontal Step Slider Container */}
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-6 relative">
            
            {/* Step 1 */}
            <div id="step-01" className="bg-gray-50 border border-gray-150 p-5 rounded-2xl relative">
              <div className="absolute top-4 right-4 text-[10px] text-gray-450 font-bold font-mono">
                01
              </div>
              <h4 className="font-bold text-gray-900 text-sm mt-4">Upload Image</h4>
              <p className="text-[11px] text-gray-500 mt-2">
                Drag-and-drop imagery. Supports high resolution files with varying formats.
              </p>
              <div className="text-[9px] font-mono text-blue-600 mt-4 uppercase font-bold tracking-wider">
                📥 INPUT
              </div>
            </div>

            {/* Step 2 */}
            <div id="step-02" className="bg-gray-50 border border-gray-150 p-5 rounded-2xl relative">
              <div className="absolute top-4 right-4 text-[10px] text-gray-450 font-bold font-mono">
                02
              </div>
              <h4 className="font-bold text-gray-900 text-sm mt-4">Scene Classify</h4>
              <p className="text-[11px] text-gray-500 mt-2">
                ResNet-50 sorts environment into: Urban, Rural, Coastal, Mountain, or Forest.
              </p>
              <div className="text-[9px] font-mono text-emerald-600 mt-4 uppercase font-bold tracking-wider">
                🌿 RESNET-50
              </div>
            </div>

            {/* Step 3 */}
            <div id="step-03" className="bg-gray-50 border border-gray-150 p-5 rounded-2xl relative">
              <div className="absolute top-4 right-4 text-[10px] text-gray-450 font-bold font-mono">
                03
              </div>
              <h4 className="font-bold text-gray-900 text-sm mt-4">Feature Extract</h4>
              <p className="text-[11px] text-gray-500 mt-2">
                EfficientNet-B0 maps fine geometries and structural style boundaries.
              </p>
              <div className="text-[9px] font-mono text-indigo-600 mt-4 uppercase font-bold tracking-wider">
                ⚙️ EFF-NET-B0
              </div>
            </div>

            {/* Step 4 */}
            <div id="step-04" className="bg-gray-50 border border-gray-150 p-5 rounded-2xl relative">
              <div className="absolute top-4 right-4 text-[10px] text-gray-450 font-bold font-mono">
                04
              </div>
              <h4 className="font-bold text-gray-900 text-sm mt-4">Vision Transformer</h4>
              <p className="text-[11px] text-gray-500 mt-2">
                Computes spatial attention layers across local raster pixel clusters.
              </p>
              <div className="text-[9px] font-mono text-indigo-650 mt-4 uppercase font-bold tracking-wider">
                ⛓️ VIT-ATTN
              </div>
            </div>

            {/* Step 5 */}
            <div id="step-05" className="bg-gray-50 border border-gray-150 p-5 rounded-2xl relative">
              <div className="absolute top-4 right-4 text-[10px] text-gray-450 font-bold font-mono">
                05
              </div>
              <h4 className="font-bold text-gray-900 text-sm mt-4">Gemini AI Analysis</h4>
              <p className="text-[11px] text-gray-500 mt-2">
                Executes semantic geographical reasoning on architectural styles & flora.
              </p>
              <div className="text-[9px] font-mono text-violet-600 mt-4 uppercase font-bold tracking-wider">
                🔮 LMM GROUNDING
              </div>
            </div>

            {/* Step 6 */}
            <div id="step-06" className="bg-gray-50 border border-gray-150 p-5 rounded-2xl relative">
              <div className="absolute top-4 right-4 text-[10px] text-gray-450 font-bold font-mono">
                06
              </div>
              <h4 className="font-bold text-gray-900 text-sm mt-4">Loc Prediction</h4>
              <p className="text-[11px] text-gray-500 mt-2">
                Outputs highly accurate landmark name, city, country, and direct coordinates.
              </p>
              <div className="text-[9px] font-mono text-emerald-500 mt-4 uppercase font-bold tracking-wider">
                🎯 GIS REPORT
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ──────────────────── SECTION 4 — INTERACTIVE DEMO ──────────────────── */}
      <section id="interactive-demo" className="py-20 bg-gray-50/50 border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="border-b border-gray-200 pb-6 mb-10">
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full uppercase tracking-wider block w-fit">
              Demonstration Arena
            </span>
            <h2 className="text-3xl font-extrabold text-gray-950 mt-2">
              Model Inference Console
            </h2>
            <p className="text-sm text-gray-500 mt-1">
              Select one of our preset research images to run the pipeline instantly, or drop your own image to test live detection.
            </p>
          </div>





          {errorMessage && (
            <div id="error-banner" className="mb-6 bg-red-50 border border-red-200 rounded-xl p-4 flex gap-3 text-sm text-red-800">
              <ShieldAlert className="w-5 h-5 text-red-650 shrink-0" />
              <div>
                <p className="font-bold">Evaluation process exception</p>
                <p className="text-xs pt-0.5">{errorMessage}</p>
              </div>
            </div>
          )}

          {/* TWO-COLUMN LAYOUT: Left (Upload / Active Preview) vs Right (Prediction HUD) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            
            {/* LEFT COLUMN: Upload/Preview Interactive Zone */}
            <div className="space-y-6">
              
              <div 
                id="drop-zone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleImageDrop}
                className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all bg-white flex flex-col items-center justify-center min-h-[340px] relative overflow-hidden group ${imageBase64 || selectedPresetId ? "border-blue-400" : "border-gray-200 hover:border-blue-450"}`}
              >
                {/* File Upload Hidden Input */}
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileInputChange} 
                  accept="image/*" 
                  className="hidden" 
                />

                {/* If image is selected/predefined, show layout preview, else show empty states */}
                {selectedPresetId || imageBase64 ? (
                  <div className="w-full h-full flex flex-col items-center">
                    {/* Image Preview Window */}
                    <div className="relative rounded-lg overflow-hidden border border-gray-100 max-h-72 w-full flex items-center justify-center bg-gray-50">
                      {selectedPresetId ? (
                        // Render stylish preset vector layouts
                        <div id="preset-ill-avatar" className="w-full h-48 sm:h-56 flex flex-col items-center justify-center bg-slate-900 text-white relative">
                          <Compass className="w-10 h-10 text-blue-500 animate-spin-slow mb-3" />
                          <span className="font-mono text-xs text-blue-400 uppercase tracking-widest font-semibold">PRESET DEMO SOURCE ACTIVE</span>
                          <span className="text-lg font-bold text-gray-100 mt-1.5 capitalize">{selectedPresetId.replace("_", " ")}</span>
                          <span className="text-[10px] text-gray-500 font-mono mt-0.5">Static coordinate bounds injected</span>
                        </div>
                      ) : (
                        <img 
                          id="uploaded-image-preview"
                          src={imageBase64 || ""} 
                          alt="Inspection source" 
                          className="object-contain max-h-72 w-full"
                        />
                      )}

                      {/* Top Overlay HUD over image */}
                      <div className="absolute top-2 left-2 bg-slate-900/80 text-white font-mono text-[9px] py-1 px-2.5 rounded backdrop-blur-md">
                        {selectedPresetId ? "PRESET SEED" : `${imageFile ? imageFile.name.substring(0, 15) + "..." : "CUSTOM IMAGE"}`}
                      </div>

                      {/* Active classification pipeline overlays */}
                      {isAnalyzing && (
                        <div className="absolute inset-0 bg-slate-950/85 flex flex-col items-center justify-center text-white backdrop-blur-xs p-4">
                          <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mb-3" />
                          <span className="font-mono text-xs text-blue-400 tracking-wider">ROUTING INFERENCE MATRIX...</span>
                          
                          {/* Stepper details */}
                          <p className="text-[11px] text-gray-400 mt-2 font-mono h-4">
                            {pipelineStep === 1 && "▶ Loading Image Buffer into memory"}
                            {pipelineStep === 2 && "▶ [Stage 1] Invoking Places365 ResNet-50 head"}
                            {pipelineStep === 3 && "▶ [Stage 2] Submitting to EfficientNet-B0"}
                            {pipelineStep === 4 && "▶ [Stage 2] Computing Self-Attention via ViT"}
                            {pipelineStep === 5 && "▶ [Stage 2] Performing multimodal coordinate matching"}
                            {pipelineStep === 6 && "▶ Report compiled successfully"}
                          </p>

                          <div className="w-48 bg-slate-800 h-1 rounded-full mt-4 overflow-hidden">
                            <div 
                              className="bg-blue-600 h-full transition-all duration-300"
                              style={{ 
                                width: `${(pipelineStep / 6) * 100}%` 
                              }}
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Quick Trigger change button */}
                    <div className="mt-4 flex items-center gap-3">
                      <button
                        id="btn-trigger-upload-change"
                        onClick={() => fileInputRef.current?.click()}
                        className="bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold py-2 px-4 rounded-lg flex items-center gap-2 transition-colors cursor-pointer"
                      >
                        <Upload className="w-3.5 h-3.5" />
                        Replace Image
                      </button>

                      {(selectedPresetId || imageBase64) && (
                        <button
                          id="btn-re-analyze"
                          onClick={() => {
                            if (selectedPresetId) handlePresetSelect(selectedPresetId);
                            else if (imageBase64) evaluateUploadedImage(imageBase64);
                          }}
                          className="bg-gray-100 hover:bg-gray-200 text-gray-800 text-xs font-semibold py-2 px-4 rounded-lg flex items-center gap-1.5 transition-colors"
                          title="Re-run image processing matrix"
                        >
                          <RefreshCw className="w-3.5 h-3.5" />
                          Re-run
                        </button>
                      )}
                    </div>
                  </div>
                ) : (
                  // Initial Drop Area Empty State
                  <div className="py-6">
                    <div className="w-16 h-16 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center mb-4 mx-auto transition-transform group-hover:scale-105">
                      <Upload className="w-8 h-8" />
                    </div>
                    <h4 className="font-extrabold text-gray-900 text-lg leading-tight">Drag and drop visual scene assets</h4>
                    <p className="text-xs text-gray-400 mt-1 max-w-sm mx-auto">
                      Submit landscape, street-view, coastal or geological frames to estimate exact coordinates. Max file size: 12MB.
                    </p>
                    <button
                      id="btn-browse-image"
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-6 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs py-2.5 px-5 rounded-lg inline-flex items-center gap-1.5 transition-colors cursor-pointer"
                    >
                      Browse Image File
                    </button>
                  </div>
                )}
              </div>



            </div>

            {/* RIGHT COLUMN: Prediction HUD */}
            <div id="demopredictions-section" className="bg-white border border-gray-150 rounded-2xl p-6 shadow-xs space-y-6">
              
              <div className="flex items-center justify-between border-b border-gray-100 pb-4">
                <div>
                  <span className="text-[10px] font-mono font-bold uppercase text-gray-400">Telemetry Out</span>
                  <h4 className="font-extrabold text-lg text-gray-950 mt-0.5">Prediction Results Report</h4>
                </div>
                {isAnalyzing ? (
                  <span className="bg-blue-50 text-blue-600 font-mono text-xs px-2 py-1 rounded inline-flex items-center gap-1.5">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    EVALUATING...
                  </span>
                ) : (
                  <span className="bg-emerald-50 text-emerald-700 font-mono text-[10px] font-bold px-2 py-1 rounded border border-emerald-100 uppercase">
                    Steady State
                  </span>
                )}
              </div>

              {/* Outputs Summary */}
              {currentPrediction && (
                <div className="space-y-6">
                  
                  {/* Category & Confidence Grid */}
                  <div className="grid grid-cols-2 gap-4">
                    <div id="result-scene-category" className="bg-gray-50 p-4 rounded-xl border border-gray-100">
                      <span className="text-[10px] font-mono text-gray-400 block uppercase">Scene Category</span>
                      <strong className="text-2xl text-gray-900 block mt-1">{currentPrediction.sceneCategory}</strong>
                      <span className="text-xs text-gray-400 font-mono">ResNet-50 Prior</span>
                    </div>

                    <div id="result-confidence" className="bg-gray-50 p-4 rounded-xl border border-gray-100">
                      <span className="text-[10px] font-mono text-gray-400 block uppercase">Confidence Ratio</span>
                      <strong className="text-2xl text-emerald-600 block mt-1">{currentPrediction.confidence}%</strong>
                      <span className="text-xs text-gray-400 font-mono">Classifier Probability</span>
                    </div>
                  </div>

                  {/* Highlight Location Card details */}
                  <div className="space-y-3.5">
                    <span className="text-[10px] font-mono font-bold text-gray-450 block tracking-wider uppercase">
                      RESOLVED GEOGRAPHICAL SPECIFICATIONS:
                    </span>

                    <div className="border border-gray-150 rounded-xl divide-y divide-gray-100 overflow-hidden text-sm">
                      
                      <div className="grid grid-cols-3 p-3 hover:bg-gray-50 transition-colors">
                        <span className="text-gray-450 font-medium">Landmark Name</span>
                        <span id="output-landmark" className="col-span-2 font-semibold text-gray-900">{currentPrediction.landmarkName}</span>
                      </div>

                      <div className="grid grid-cols-3 p-3 hover:bg-gray-50 transition-colors">
                        <span className="text-gray-450 font-medium">City / State</span>
                        <span id="output-city" className="col-span-2 font-semibold text-gray-900">{currentPrediction.city}</span>
                      </div>

                      <div className="grid grid-cols-3 p-3 hover:bg-gray-50 transition-colors">
                        <span className="text-gray-450 font-medium">Country</span>
                        <span id="output-country" className="col-span-2 font-semibold text-gray-900">{currentPrediction.country}</span>
                      </div>

                      <div className="grid grid-cols-3 p-3 hover:bg-gray-50 transition-colors">
                        <span className="text-gray-450 font-medium">Estimated GPS Coordinates</span>
                        <span id="output-coords" className="col-span-2 font-mono font-bold text-blue-600">
                          {currentPrediction.latitude.toFixed(6)}, {currentPrediction.longitude.toFixed(6)}
                        </span>
                      </div>

                      <div className="grid grid-cols-3 p-3 hover:bg-gray-50 transition-colors">
                        <span className="text-gray-450 font-medium">Vision Transformer accuracy</span>
                        <span id="output-transformer-acc" className="col-span-2 font-mono font-bold text-emerald-600">
                          {currentPrediction.aiConfidence}% Combined Match Score
                        </span>
                      </div>

                    </div>
                  </div>

                  {/* Scholarly reasoning report from pipeline */}
                  <div id="output-scholarly-reasoning" className="bg-slate-900 text-slate-100 rounded-xl p-4 font-sans text-xs relative overflow-hidden leading-relaxed shadow-sm">
                    <div className="absolute top-0 right-0 w-16 h-16 bg-blue-950 rounded-full blur-xl pointer-events-none" />
                    <span className="text-[9px] font-mono font-semibold tracking-wider text-blue-400 block uppercase mb-1">
                      Pipeline Deductive Reasoning (Academic Note):
                    </span>
                    <p className="italic">
                      "{currentPrediction.reasoning}"
                    </p>
                  </div>

                </div>
              )}

            </div>

          </div>

        </div>
      </section>

      {/* ──────────────────── SECTION 5 — MAP VISUALIZATION ──────────────────── */}
      <section id="map-visualization" className="py-20 bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-8">
            <div>
              <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full uppercase tracking-wider block w-fit">
                Mapbox GIS Integration
              </span>
              <h2 className="text-3xl font-extrabold text-gray-950 mt-2">
                Geospatial Signal Mapping
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                Visualizing physical coordinate tags in three different layer modes: Topography contours, Satellite, and Vector grid structures.
              </p>
            </div>
            
            <div className="text-left md:text-right mt-4 md:mt-0">
              <span className="text-xs font-semibold text-gray-500 block">Current Signal Node:</span>
              <span className="text-sm font-bold text-gray-900 block font-mono">
                {currentPrediction ? `${currentPrediction.latitude.toFixed(4)}, ${currentPrediction.longitude.toFixed(4)}` : "AWAITING SIGNAL"}
              </span>
            </div>
          </div>

          {/* Interactive Core viewport */}
          <InteractiveMap prediction={currentPrediction} isAnalyzing={isAnalyzing} />

        </div>
      </section>

      {/* ──────────────────── SECTION 6 — MODEL ARCHITECTURE ──────────────────── */}
      <section id="model-architecture" className="py-20 bg-gray-50/50 border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          {/* Include our modular core architecture schema */}
          <ArchitectureGraph />

        </div>
      </section>

      {/* ──────────────────── SECTION 7 — PERFORMANCE ──────────────────── */}
      <section id="performance-metrics" className="py-20 bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          {/* Include our model validation interactive metrics graphs */}
          <ModelPerformance />

        </div>
      </section>

      {/* ──────────────────── SECTION 8 — TECHNOLOGY STACK ──────────────────── */}
      <section id="tech-stack" className="py-20 bg-gray-50/50 border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-3">
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-55 px-2.5 py-1 rounded-full uppercase tracking-wider">
              Platform Components
            </span>
            <h2 className="text-3xl font-extrabold text-gray-950">
              Technology Stack Block Structure
            </h2>
            <p className="text-sm text-gray-500">
              DeepSceneLoc utilizes premium open-source toolsets engineered specifically for deep convolutional extraction and global topological predictions.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            
            {/* PyTorch */}
            <div id="tech-pytorch" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-orange-650 bg-orange-50 flex items-center justify-center font-bold text-base mb-4 font-mono">
                Py
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">PyTorch</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Primary neural framework for weight training, residual backpropagation optimizations, and deep tensor evaluations.
              </p>
            </div>

            {/* ResNet-50 */}
            <div id="tech-resnet" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-blue-650 bg-blue-50 flex items-center justify-center font-bold text-xs mb-4 font-mono">
                R50
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">ResNet-50</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Fifty layers deep residual block modeling leveraged to define prior envelopes across structural environments.
              </p>
            </div>

            {/* EfficientNet-B0 */}
            <div id="tech-effnet" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-emerald-650 bg-emerald-50 flex items-center justify-center font-bold text-xs mb-4 font-mono">
                EnB0
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">EfficientNet-B0</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Sub-pixels compound resolution multiplier resolving specific architectural, vehicle, and botany clusters.
              </p>
            </div>

            {/* Vision Transformer */}
            <div id="tech-vit" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-indigo-650 bg-indigo-50 flex items-center justify-center font-bold text-xs mb-4 font-mono">
                ViT
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">Vision Transformer (ViT)</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Applies standard self-attention matrices on subdivided 16x16 pixel blocks, preserving coordinate integrity.
              </p>
            </div>

            {/* Gemini API */}
            <div id="tech-gemini" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-violet-650 bg-violet-50 flex items-center justify-center font-bold text-xs mb-4 font-mono">
                Gem
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">Gemini API</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Performs deep multimodal spatial reasoning, extracting elevation, weather statistics and geopolitical boundaries.
              </p>
            </div>

            {/* Places365 Corpus */}
            <div id="tech-places" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-gray-650 bg-slate-100 flex items-center justify-center font-bold text-xs mb-4 font-mono">
                P365
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">Places365 Corpus</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Standard context suite containing millions of structured images for broad-latitude classification tasks.
              </p>
            </div>

            {/* Google Colab */}
            <div id="tech-colab" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-red-500 bg-red-50 flex items-center justify-center font-bold text-xs mb-4 font-mono">
                Col
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">Google Colab Pro</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Utilized high-RAM T4 & A100 GPU compute layers to complete PyTorch training epochs and fine-tuning blocks.
              </p>
            </div>

            {/* Python Stack */}
            <div id="tech-python" className="bg-white border border-gray-150 p-5 rounded-2xl shadow-xs hover:shadow-md transition-shadow">
              <div className="h-8 w-8 rounded text-blue-500 bg-blue-50 flex items-center justify-center font-bold text-xs mb-4 font-mono">
                Py3
              </div>
              <h5 className="font-semibold text-gray-900 text-sm">Python Stack</h5>
              <p className="text-xs text-gray-500 mt-2 leading-relaxed">
                Primary software stack for dataset preprocessing pipelines, array matrix calculations, and deep learning libraries.
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* ──────────────────── SECTION 9 — TEAM ──────────────────── */}
      <section id="academic-team" className="py-20 bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <span className="text-xs font-mono font-bold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full uppercase tracking-wider">
              Research Contributors
            </span>
            <h2 className="text-3xl font-extrabold text-gray-950">
              Academic Project Team
            </h2>
            <p className="text-sm text-gray-500">
              Final Year Computer Science Engineering students who spearheaded dataset orchestration, model fine-tuning and pipeline validation.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            
            {/* Maker 1 */}
            <div id="team-krishan" className="bg-gray-50 border border-gray-200/60 rounded-2xl p-5 text-center shadow-xs h-[320px] max-h-[320px] flex flex-col justify-between items-center transition-all duration-300 hover:shadow-xs">
              <button 
                onClick={() => setShowKrishanPhotoModal(true)}
                className="w-16 h-16 rounded-full overflow-hidden shadow-sm border-2 border-slate-200 bg-blue-50 flex items-center justify-center relative group/team cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 shrink-0"
                title="Click to view full image"
              >
                <img 
                  src={teamKrishanImg} 
                  alt="Krishan Yadav" 
                  className="w-full h-full object-cover transition-transform duration-300 group-hover/team:scale-110" 
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/team:opacity-100 flex items-center justify-center transition-opacity duration-200">
                  <Search className="w-4 h-4 text-white" />
                </div>
              </button>
              <div className="flex-1 flex flex-col justify-center my-1">
                <h4 className="font-bold text-gray-900 text-base">Krishan Yadav</h4>
                <p className="text-[11px] text-blue-600 font-mono font-bold uppercase mt-0.5">Technical Lead</p>
                <p className="text-xs text-gray-500 mt-2.5 leading-relaxed">
                  Engineered the hybrid transfer model connections, fine-tuned skip weights, and resolved multimodal API layers.
                </p>
              </div>
              <button 
                onClick={() => setShowKrishanPhotoModal(true)}
                className="text-[11px] text-blue-600 font-semibold hover:underline flex items-center gap-1 cursor-pointer shrink-0"
              >
                View Full Photo
              </button>
            </div>

            {/* Maker 2 */}
            <div id="team-aditi" className="bg-gray-50 border border-gray-200/60 rounded-2xl p-5 text-center shadow-xs h-[320px] max-h-[320px] flex flex-col justify-between items-center transition-all duration-300 hover:shadow-xs">
              <button 
                onClick={() => setShowAditiPhotoModal(true)}
                className="w-16 h-16 rounded-full overflow-hidden shadow-sm border-2 border-slate-200 bg-emerald-50 flex items-center justify-center relative group/team cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 shrink-0"
                title="Click to view full image"
              >
                <img 
                  src={teamAditiImg} 
                  alt="Aditi Sah" 
                  className="w-full h-full object-cover transition-transform duration-300 group-hover/team:scale-110" 
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/team:opacity-100 flex items-center justify-center transition-opacity duration-200">
                  <Search className="w-4 h-4 text-white" />
                </div>
              </button>
              <div className="flex-1 flex flex-col justify-center my-1">
                <h4 className="font-bold text-gray-900 text-base">Aditi Sah</h4>
                <p className="text-[11px] text-emerald-600 font-mono font-bold uppercase mt-0.5">Data Lead</p>
                <p className="text-xs text-gray-500 mt-2.5 leading-relaxed">
                  Orchestrated the Places365 target classes, audited training bias, and validated evaluation accuracy bounds.
                </p>
              </div>
              <button 
                onClick={() => setShowAditiPhotoModal(true)}
                className="text-[11px] text-blue-600 font-semibold hover:underline flex items-center gap-1 cursor-pointer shrink-0"
              >
                View Full Photo
              </button>
            </div>

            {/* Maker 3 */}
            <div id="team-anuj" className="bg-gray-50 border border-gray-200/60 rounded-2xl p-5 text-center shadow-xs h-[320px] max-h-[320px] flex flex-col justify-between items-center transition-all duration-300 hover:shadow-xs">
              <button 
                onClick={() => setShowAnujPhotoModal(true)}
                className="w-16 h-16 rounded-full overflow-hidden shadow-sm border-2 border-slate-200 bg-indigo-50 flex items-center justify-center relative group/team cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 shrink-0"
                title="Click to view full image"
              >
                <img 
                  src={teamAnujImg} 
                  alt="Anuj Kondawar" 
                  className="w-full h-full object-cover transition-transform duration-300 group-hover/team:scale-110" 
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/team:opacity-100 flex items-center justify-center transition-opacity duration-200">
                  <Search className="w-4 h-4 text-white" />
                </div>
              </button>
              <div className="flex-1 flex flex-col justify-center my-1">
                <h4 className="font-bold text-gray-900 text-base">Anuj Kondawar</h4>
                <p className="text-[11px] text-indigo-650 font-mono font-bold uppercase mt-0.5">Preprocessing Lead</p>
                <p className="text-xs text-gray-500 mt-2.5 leading-relaxed">
                  Programmed high-throughput image scaling utilities, adjusted color distributions, and isolated coordinate noises.
                </p>
              </div>
              <button 
                onClick={() => setShowAnujPhotoModal(true)}
                className="text-[11px] text-blue-600 font-semibold hover:underline flex items-center gap-1 cursor-pointer shrink-0"
              >
                View Full Photo
              </button>
            </div>

            {/* Maker 4 */}
            <div id="team-jensi" className="bg-gray-50 border border-gray-200/60 rounded-2xl p-5 text-center shadow-xs h-[320px] max-h-[320px] flex flex-col justify-between items-center transition-all duration-300 hover:shadow-xs">
              <button 
                onClick={() => setShowJensiPhotoModal(true)}
                className="w-16 h-16 rounded-full overflow-hidden shadow-sm border-2 border-slate-200 bg-violet-50 flex items-center justify-center relative group/team cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 shrink-0"
                title="Click to view full image"
              >
                <img 
                  src={teamJensiImg} 
                  alt="Jensi Paneliya" 
                  className="w-full h-full object-cover transition-transform duration-300 group-hover/team:scale-110" 
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/team:opacity-100 flex items-center justify-center transition-opacity duration-200">
                  <Search className="w-4 h-4 text-white" />
                </div>
              </button>
              <div className="flex-1 flex flex-col justify-center my-1">
                <h4 className="font-bold text-gray-900 text-base">Jensi Paneliya</h4>
                <p className="text-[11px] text-violet-650 font-mono font-bold uppercase mt-0.5">Documentation Lead</p>
                <p className="text-xs text-gray-500 mt-2.5 leading-relaxed">
                  Compiled thesis parameters, mapped verification indices, and documented API performance records.
                </p>
              </div>
              <button 
                onClick={() => setShowJensiPhotoModal(true)}
                className="text-[11px] text-blue-600 font-semibold hover:underline flex items-center gap-1 cursor-pointer shrink-0"
              >
                View Full Photo
              </button>
            </div>

          </div>
        </div>
      </section>

      {/* ──────────────────── SECTION 10 — FOOTER ──────────────────── */}
      <footer className="bg-slate-950 text-slate-400 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 border-b border-slate-900 pb-12 mb-12">
            
            {/* Column 1 info */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <div className="w-11 h-8 rounded-md overflow-hidden bg-transparent border border-slate-800 shadow-xs flex items-center justify-center">
                  <img
                    src={appLogoDsl}
                    alt="DeepSceneLoc Logo"
                    className="w-full h-full object-cover"
                    referrerPolicy="no-referrer"
                  />
                </div>
                <span className="font-bold text-white text-base">DeepSceneLoc</span>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                Visual Scene Understanding for Image-Based Location Estimation Using Deep Learning. An academic research milestone.
              </p>
            </div>

            {/* Column 2 - Academy details */}
            <div className="space-y-3 text-xs">
              <h5 className="font-bold text-white uppercase tracking-wider text-[11px] font-mono">Academic Institution</h5>
              <p className="text-slate-400">Department of Computer Science & Engineering</p>
              <p className="text-slate-400">Final Year Capstone Project Exhibition</p>
              <p className="text-slate-500">Academic Year: 2023 - 2027</p>
            </div>

            {/* Column 3 - Respository */}
            <div className="space-y-3 text-xs">
              <h5 className="font-bold text-white uppercase tracking-wider text-[11px] font-mono">Core Documentation</h5>
              <p className="flex items-center gap-1.5 hover:text-white transition-colors cursor-pointer">
                <FileText className="w-3.5 h-3.5" /> Project Thesis (PDF)
              </p>
              <p className="flex items-center gap-1.5 text-slate-500">
                <Code className="w-3.5 h-3.5" /> PyTorch Training Notebooks
              </p>
              <p className="flex items-center gap-1.5 text-slate-500">
                <Database className="w-3.5 h-3.5" /> Places365 subset manifests
              </p>
            </div>

            {/* Column 4 - Contact */}
            <div className="space-y-3 text-xs text-slate-400">
              <h5 className="font-bold text-white uppercase tracking-wider text-[11px] font-mono">Inquiries & Support</h5>
              <p>Academic Sponsor: Rashmi Pal</p>
            </div>

          </div>

          <div className="flex flex-col sm:flex-row items-center justify-between text-xs text-slate-600 font-mono">
            <span>© 2026 DeepSceneLoc. All rights reserved. </span>
          </div>

        </div>
      </footer>

      {/* ──────────────────── PHOTO LIGHTBOX MODAL ──────────────────── */}
      {showAnujPhotoModal && (
        <div 
          className="fixed inset-0 bg-black/95 z-50 flex flex-col items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
          onClick={() => setShowAnujPhotoModal(false)}
        >
          {/* Close trigger on top right */}
          <button 
            onClick={() => setShowAnujPhotoModal(false)}
            className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 text-white rounded-full p-2.5 transition-colors cursor-pointer focus:outline-none"
            title="Close Lightbox"
          >
            <X className="w-6 h-6" />
          </button>

          {/* Modal Container */}
          <div 
            className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden max-w-sm sm:max-w-md md:max-w-lg w-full flex flex-col shadow-2xl animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header info */}
            <div className="bg-zinc-950 p-4 border-b border-zinc-800 flex items-center justify-between">
              <div>
                <h4 className="text-white font-bold text-sm">Anuj Kondawar</h4>
                <p className="text-[10px] text-zinc-400 font-mono">Preprocessing Lead • Team Contributor</p>
              </div>
              <span className="text-[10px] font-mono font-semibold bg-blue-950 text-blue-400 border border-blue-900 px-2 py-0.5 rounded-full uppercase">
                Parul University
              </span>
            </div>

            {/* Live image */}
            <div className="relative bg-zinc-950 overflow-hidden flex justify-center items-center p-2">
              <img 
                src={teamAnujImg} 
                alt="Anuj Kondawar Original Image" 
                className="max-h-[70vh] w-auto object-contain rounded-lg shadow-inner"
                referrerPolicy="no-referrer"
              />
            </div>

            {/* Footer - Close Button */}
            <div className="bg-zinc-950 p-4 border-t border-zinc-800 text-center">
              <button
                onClick={() => setShowAnujPhotoModal(false)}
                className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 hover:text-white font-medium text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer"
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Krishan Photo Modal */}
      {showKrishanPhotoModal && (
        <div 
          className="fixed inset-0 bg-black/95 z-50 flex flex-col items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
          onClick={() => setShowKrishanPhotoModal(false)}
        >
          {/* Close trigger on top right */}
          <button 
            onClick={() => setShowKrishanPhotoModal(false)}
            className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 text-white rounded-full p-2.5 transition-colors cursor-pointer focus:outline-none"
            title="Close Lightbox"
          >
            <X className="w-6 h-6" />
          </button>

          {/* Modal Container */}
          <div 
            className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden max-w-sm sm:max-w-md md:max-w-lg w-full flex flex-col shadow-2xl animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header info */}
            <div className="bg-zinc-950 p-4 border-b border-zinc-800 flex items-center justify-between">
              <div>
                <h4 className="text-white font-bold text-sm">Krishan Yadav</h4>
                <p className="text-[10px] text-zinc-400 font-mono">Technical Lead • Team Contributor</p>
              </div>
              <span className="text-[10px] font-mono font-semibold bg-blue-950 text-blue-400 border border-blue-900 px-2 py-0.5 rounded-full uppercase">
                Parul University
              </span>
            </div>

            {/* Live image */}
            <div className="relative bg-zinc-950 overflow-hidden flex justify-center items-center p-2">
              <img 
                src={teamKrishanImg} 
                alt="Krishan Yadav Original Image" 
                className="max-h-[70vh] w-auto object-contain rounded-lg shadow-inner"
                referrerPolicy="no-referrer"
              />
            </div>

            {/* Footer - Close Button */}
            <div className="bg-zinc-950 p-4 border-t border-zinc-800 text-center">
              <button
                onClick={() => setShowKrishanPhotoModal(false)}
                className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 hover:text-white font-medium text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer"
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Jensi Photo Modal */}
      {showJensiPhotoModal && (
        <div 
          className="fixed inset-0 bg-black/95 z-50 flex flex-col items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
          onClick={() => setShowJensiPhotoModal(false)}
        >
          {/* Close trigger on top right */}
          <button 
            onClick={() => setShowJensiPhotoModal(false)}
            className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 text-white rounded-full p-2.5 transition-colors cursor-pointer focus:outline-none"
            title="Close Lightbox"
          >
            <X className="w-6 h-6" />
          </button>

          {/* Modal Container */}
          <div 
            className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden max-w-sm sm:max-w-md md:max-w-lg w-full flex flex-col shadow-2xl animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header info */}
            <div className="bg-zinc-950 p-4 border-b border-zinc-800 flex items-center justify-between">
              <div>
                <h4 className="text-white font-bold text-sm">Jensi Paneliya</h4>
                <p className="text-[10px] text-zinc-400 font-mono">Documentation Lead • Team Contributor</p>
              </div>
              <span className="text-[10px] font-mono font-semibold bg-blue-950 text-blue-400 border border-blue-900 px-2 py-0.5 rounded-full uppercase">
                Parul University
              </span>
            </div>

            {/* Live image */}
            <div className="relative bg-zinc-950 overflow-hidden flex justify-center items-center p-2">
              <img 
                src={teamJensiImg} 
                alt="Jensi Paneliya Original Image" 
                className="max-h-[70vh] w-auto object-contain rounded-lg shadow-inner"
                referrerPolicy="no-referrer"
              />
            </div>

            {/* Footer / Caption */}
            {/* Footer - Close Button */}
            <div className="bg-zinc-950 p-4 border-t border-zinc-800 text-center">
              <button
                onClick={() => setShowJensiPhotoModal(false)}
                className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 hover:text-white font-medium text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer"
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Aditi Photo Modal */}
      {showAditiPhotoModal && (
        <div 
          className="fixed inset-0 bg-black/95 z-50 flex flex-col items-center justify-center p-4 backdrop-blur-sm animate-fade-in"
          onClick={() => setShowAditiPhotoModal(false)}
        >
          {/* Close trigger on top right */}
          <button 
            onClick={() => setShowAditiPhotoModal(false)}
            className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 text-white rounded-full p-2.5 transition-colors cursor-pointer focus:outline-none"
            title="Close Lightbox"
          >
            <X className="w-6 h-6" />
          </button>

          {/* Modal Container */}
          <div 
            className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden max-w-sm sm:max-w-md md:max-w-lg w-full flex flex-col shadow-2xl animate-scale-in"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="bg-zinc-950 border-b border-zinc-800 p-4 flex items-center justify-between">
              <h3 className="text-white font-bold text-sm">Aditi Sah</h3>
              <span className="text-[10px] font-mono font-semibold bg-emerald-950 text-emerald-400 border border-emerald-900 px-2 py-0.5 rounded-full uppercase">
                Parul University
              </span>
            </div>

            {/* Live image */}
            <div className="relative bg-zinc-950 overflow-hidden flex justify-center items-center p-2">
              <img 
                src={teamAditiImg} 
                alt="Aditi Sah Original Image" 
                className="max-h-[70vh] w-auto object-contain rounded-lg shadow-inner"
                referrerPolicy="no-referrer"
              />
            </div>

            {/* Footer - Close Button */}
            <div className="bg-zinc-950 p-4 border-t border-zinc-800 text-center">
              <button
                onClick={() => setShowAditiPhotoModal(false)}
                className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 hover:text-white font-medium text-xs px-4 py-1.5 rounded-lg transition-colors cursor-pointer"
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
