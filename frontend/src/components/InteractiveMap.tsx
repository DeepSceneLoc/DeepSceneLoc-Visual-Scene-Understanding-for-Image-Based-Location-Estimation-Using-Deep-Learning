import React, { useState, useEffect } from "react";
import { 
  Compass, 
  Layers, 
  MapPin, 
  Globe, 
  Cpu, 
  Maximize2, 
  Navigation,
  Sparkles,
  Info
} from "lucide-react";
import { PipelineOutput } from "../types";

interface InteractiveMapProps {
  prediction: PipelineOutput | null;
  isAnalyzing: boolean;
}

export default function InteractiveMap({ prediction, isAnalyzing }: InteractiveMapProps) {
  const [mapType, setMapType] = useState<"terrain" | "satellite" | "vector">("terrain");
  const [zoomLevel, setZoomLevel] = useState<number>(14);
  const [scanning, setScanning] = useState<boolean>(false);
  const [scanProgress, setScanProgress] = useState<number>(0);

  // Default coordinate if no prediction is selected (Moraine Lake)
  const lat = prediction ? prediction.latitude : 51.3215;
  const lng = prediction ? prediction.longitude : -116.1860;
  const areaName = prediction ? `${prediction.landmarkName}, ${prediction.city}` : "Moraine Lake, Banff";
  const countryName = prediction ? prediction.country : "Canada";

  useEffect(() => {
    if (isAnalyzing) {
      setScanning(true);
      setScanProgress(0);
      const interval = setInterval(() => {
        setScanProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setTimeout(() => setScanning(false), 500);
            return 100;
          }
          return prev + 10;
        });
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isAnalyzing, prediction]);

  // Generate responsive terrain visual markers based on the coordinates
  const generateContourPaths = () => {
    const seed = Math.abs(lat * lng) % 1;
    const paths = [];
    const centerX = 300;
    const centerY = 200;
    
    for (let i = 1; i <= 5; i++) {
      const radiusX = (60 * i) + (seed * 15);
      const radiusY = (40 * i) + (seed * 10);
      // Slight wobbles to look like natural contour lines
      paths.push({
        radiusX,
        radiusY,
        opacity: 0.15 - (i * 0.02),
        strokeWidth: 1.2,
        dashArray: i === 4 ? "4 4" : "none"
      });
    }
    return paths;
  };

  const contours = generateContourPaths();

  return (
    <div id="gis-map-viewport" className="relative w-full h-[480px] bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl group">
      {/* Dynamic Background Map Simulation */}
      <div className="absolute inset-0 transition-all duration-700 overflow-hidden">
        {/* Layer 1: Satellite imagery styling */}
        {mapType === "satellite" && (
          <div className="absolute inset-0 bg-slate-900 mix-blend-screen opacity-90">
            {/* Grid Texture */}
            <div className="absolute inset-0" style={{
              backgroundImage: "radial-gradient(circle, rgba(148, 163, 184, 0.05) 1px, transparent 1px)",
              backgroundSize: "24px 24px"
            }} />
            {/* Dark green-blue terrain circles */}
            <div className="absolute top-[30%] left-[40%] w-96 h-96 rounded-full bg-emerald-950/30 blur-3xl animate-pulse" />
            <div className="absolute bottom-[20%] right-[30%] w-[450px] h-[450px] rounded-full bg-blue-950/20 blur-3xl" />
          </div>
        )}

        {/* Layer 2: Vector Grid map styling */}
        {mapType === "vector" && (
          <div className="absolute inset-0 bg-slate-950">
            {/* Fine coordinate grid */}
            <div className="absolute inset-0" style={{
              backgroundImage: `
                linear-gradient(to right, rgba(59, 130, 246, 0.05) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(59, 130, 246, 0.05) 1px, transparent 1px)
              `,
              backgroundSize: "40px 40px"
            }} />
            {/* Radial glow */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_30%,rgba(2,6,23,0.8)_80%)]" />
          </div>
        )}

        {/* Layer 3: Topographic Terrain styling */}
        {mapType === "terrain" && (
          <div className="absolute inset-0 bg-slate-900">
            <div className="absolute inset-0" style={{
              backgroundImage: "radial-gradient(circle, rgba(59, 130, 246, 0.05) 1.5px, transparent 1.5px)",
              backgroundSize: "32px 32px"
            }} />
            {/* Topography vector contours overlay */}
            <div className="absolute inset-0 flex items-center justify-center opacity-80">
              <svg className="w-full h-full text-blue-500/15" viewBox="0 0 600 400" fill="none">
                {contours.map((contour, index) => (
                  <ellipse
                    key={index}
                    cx="300"
                    cy="200"
                    rx={contour.radiusX}
                    ry={contour.radiusY}
                    stroke="currentColor"
                    strokeWidth={contour.strokeWidth}
                    strokeDasharray={contour.dashArray}
                    style={{ opacity: contour.opacity }}
                  />
                ))}
                {/* Geological Fault lines or rivers */}
                <path 
                  d="M0,150 Q150,180 300,200 T600,220" 
                  stroke="rgba(59, 130, 246, 0.12)" 
                  strokeWidth="1.5" 
                  strokeDasharray="5 5" 
                />
                <path 
                  d="M100,0 Q220,120 300,200 T500,400" 
                  stroke="rgba(59, 130, 246, 0.08)" 
                  strokeWidth="2" 
                />
              </svg>
            </div>
          </div>
        )}

        {/* Target Focal Ping (Center of Map) */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center pointer-events-none">
          <div className="absolute w-[300px] h-[300px] rounded-full border border-blue-500/10 scale-95 animate-pulse" />
          <div className="absolute w-[180px] h-[180px] rounded-full border border-dashed border-blue-500/20" />
          
          {/* Active Target Marker */}
          <div className="relative z-10 flex flex-col items-center">
            {/* Blinking wave rings */}
            <div className="absolute w-12 h-12 rounded-full bg-blue-500/20 animate-ping" />
            <div className="absolute w-6 h-6 rounded-full bg-blue-500/30 blur-xs" />
            
            <div className="bg-blue-600 text-white p-2 rounded-full shadow-lg border border-white/20 transform transition-transform hover:scale-115 duration-300">
              <MapPin className="w-5 h-5 animate-bounce" />
            </div>

            {/* Micro HUD next to flag */}
            <div className="mt-2 bg-slate-900/95 border border-slate-700 text-[10px] text-slate-300 font-mono py-1 px-2.5 rounded-md shadow-xl whitespace-nowrap flex items-center gap-1.5 backdrop-blur-md">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              GPS LOCATED: {lat.toFixed(4)}, {lng.toFixed(4)}
            </div>
          </div>
        </div>

        {/* Scan line effect */}
        {scanning && (
          <div className="absolute inset-x-0 h-[30%] bg-gradient-to-b from-transparent via-blue-500/20 to-transparent pointer-events-none animate-scan transform -translate-y-full" style={{
            animation: "scan 3s linear infinite"
          }} />
        )}
      </div>

      {/* Grid Pattern Coordinates Overlay Markers around border */}
      <div className="absolute inset-x-0 top-0 h-6 bg-slate-950/80 border-b border-slate-800 text-[9px] font-mono text-slate-400 flex items-center justify-between px-4 select-none backdrop-blur-xs">
        <span>GRID RANGE: DEEP-WGS84</span>
        <span className="text-blue-400 font-semibold">{scanning ? `SCANNING COORDS [${scanProgress}%]` : "STATUS: STEADY INFERENCE"}</span>
        <span>LAT LIMIT: {lat > 0 ? "NORTH" : "SOUTH"} GLOBE</span>
      </div>

      {/* Right HUD: Compass & Map Tuning */}
      <div className="absolute right-4 top-10 flex flex-col gap-2 z-10">
        <div className="bg-slate-900/95 border border-slate-800 p-2.5 rounded-xl shadow-lg flex flex-col gap-2.5 backdrop-blur-md">
          <button 
            id="map-compass"
            onClick={() => setScanning(true)}
            className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-blue-400 transition-colors flex items-center justify-center"
            title="Realignment Compass"
          >
            <Compass className="w-5 h-5 animate-spin-slow text-slate-400" />
          </button>
          
          <div className="h-[1px] bg-slate-800" />
          
          {/* Zoom Indicator */}
          <div className="flex flex-col items-center gap-1">
            <button 
              id="map-zoom-in"
              onClick={() => setZoomLevel(prev => Math.min(prev + 1, 18))}
              className="w-7 h-7 flex items-center justify-center bg-slate-800 hover:bg-blue-600 hover:text-white rounded text-sm text-slate-300 font-mono transition-all"
            >
              +
            </button>
            <span className="text-[10px] font-mono text-slate-400">{zoomLevel}x</span>
            <button 
              id="map-zoom-out"
              onClick={() => setZoomLevel(prev => Math.max(prev - 1, 5))}
              className="w-7 h-7 flex items-center justify-center bg-slate-800 hover:bg-blue-600 hover:text-white rounded text-sm text-slate-300 font-mono transition-all"
            >
              -
            </button>
          </div>
        </div>

        {/* Layers Select Panel */}
        <div className="bg-slate-900/95 border border-slate-800 p-1.5 rounded-xl shadow-lg flex flex-col gap-1.5 backdrop-blur-md">
          <button
            id="btn-layer-terrain"
            onClick={() => setMapType("terrain")}
            className={`p-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${mapType === "terrain" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
          >
            <Layers className="w-3.5 h-3.5" />
            TERRAIN
          </button>
          <button
            id="btn-layer-satellite"
            onClick={() => setMapType("satellite")}
            className={`p-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${mapType === "satellite" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
          >
            <Globe className="w-3.5 h-3.5" />
            SAT
          </button>
          <button
            id="btn-layer-vector"
            onClick={() => setMapType("vector")}
            className={`p-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${mapType === "vector" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
          >
            <Cpu className="w-3.5 h-3.5" />
            VECTOR
          </button>
        </div>
      </div>

      {/* Left HUD: Coordinates & Compass telemetry details */}
      <div className="absolute left-4 bottom-4 z-10 max-w-[280px]">
        <div className="bg-slate-900/95 border border-slate-800 rounded-xl p-3.5 shadow-xl text-slate-300 backdrop-blur-md flex flex-col gap-2.5">
          <div className="flex items-center gap-2 justify-between border-b border-slate-800 pb-1.5">
            <span className="text-[11px] font-mono font-bold text-blue-400 flex items-center gap-1.5">
              <Navigation className="w-3 h-3 text-blue-400 fill-current" />
              RESOLVED SIGNAL
            </span>
            <span className="text-[9px] font-mono text-slate-500 bg-slate-800/80 px-1.5 py-0.5 rounded uppercase">
              {prediction ? prediction.sceneCategory : "Mountain"}
            </span>
          </div>

          <div className="flex flex-col gap-1 text-xs">
            <p className="font-semibold text-white truncate text-sm">{areaName}</p>
            <p className="text-slate-400 text-[11px] truncate">{countryName}</p>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[10px] font-mono border-t border-slate-800 pt-2 text-slate-400">
            <div>
              <p className="text-slate-500 text-[9px]">LATITUDE</p>
              <p className="text-white font-medium">{lat.toFixed(5)}° {lat > 0 ? "N" : "S"}</p>
            </div>
            <div>
              <p className="text-slate-500 text-[9px]">LONGITUDE</p>
              <p className="text-white font-medium">{lng.toFixed(5)}° {lng > 0 ? "E" : "W"}</p>
            </div>
            {prediction && (
              <>
                <div className="mt-1">
                  <p className="text-slate-500 text-[9px]">ELEVATION</p>
                  <p className="text-emerald-400 font-medium truncate">{prediction.elevation || "1,885m"}</p>
                </div>
                <div className="mt-1">
                  <p className="text-slate-500 text-[9px]">BEST CAMERA SEASON</p>
                  <p className="text-blue-400 font-medium truncate">{prediction.bestSeason || "June to Sept"}</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Right Corner Telemetry: Pipeline Confidence status */}
      <div className="absolute right-4 bottom-4 z-10 flex flex-col gap-2">
        <div className="bg-slate-900/95 border border-slate-800 text-[10px] font-mono py-2 px-3 rounded-lg shadow-md text-slate-300 flex items-center gap-2 backdrop-blur-md">
          <Sparkles className="w-3.5 h-3.5 text-blue-400" />
          <span>MATCH QUALITY:</span>
          <span className="font-bold text-emerald-400">
            {prediction ? `${prediction.aiConfidence}%` : "91.8%"}
          </span>
        </div>

        {prediction && prediction.geologicalAge && (
          <div className="bg-slate-900/95 border border-slate-800 text-[10px] font-mono py-2 px-3 rounded-lg shadow-md text-slate-300 flex items-center gap-2 backdrop-blur-md">
            <Info className="w-3.5 h-3.5 text-slate-500" />
            <span className="truncate max-w-[130px]">{prediction.geologicalAge}</span>
          </div>
        )}
      </div>

      {/* Scale Indicator bottom rail */}
      <div className="absolute left-[310px] bottom-4 hidden md:flex items-center gap-1.5 text-[9.5px] font-mono text-slate-500">
        <div className="w-12 h-1 border-x border-b border-slate-500" />
        <span>100m</span>
        <span className="text-[8px] bg-slate-900 px-1 text-slate-400 rounded">SCALE ESTIMATE</span>
      </div>
    </div>
  );
}
