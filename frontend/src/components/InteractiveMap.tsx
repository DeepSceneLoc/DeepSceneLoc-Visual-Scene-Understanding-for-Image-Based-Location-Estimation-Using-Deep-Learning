import React, { useState, useEffect, useRef } from "react";
import { 
  Compass, 
  Layers, 
  MapPin, 
  Globe as GlobeIcon, 
  Cpu, 
  Maximize2, 
  Minimize2,
  Navigation,
  Sparkles,
  Info,
  Map as MapIcon,
  Eye
} from "lucide-react";
import { PipelineOutput } from "../types";
import Globe from "react-globe.gl";
import Map, { Marker } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

// Declare global types for Google Maps SDK
declare global {
  interface Window {
    google: any;
    googleMapScriptLoaded?: boolean;
  }
}

interface InteractiveMapProps {
  prediction: PipelineOutput | null;
  isAnalyzing: boolean;
}

export default function InteractiveMap({ prediction, isAnalyzing }: InteractiveMapProps) {
  const [viewMode, setViewMode] = useState<"2d" | "3d" | "streetview">("3d");
  const [mapType, setMapType] = useState<"terrain" | "satellite" | "vector">("terrain");
  const [scanning, setScanning] = useState<boolean>(false);
  const [scanProgress, setScanProgress] = useState<number>(0);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [dimensions, setDimensions] = useState({ width: 600, height: 480 });

  const containerRef = useRef<HTMLDivElement>(null);
  
  // Open-source fallback refs
  const globeRef = useRef<any>(null);
  const mapRef = useRef<any>(null);

  // Google Maps refs
  const googleMapContainerRef = useRef<HTMLDivElement>(null);
  const googleStreetViewContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markerInstanceRef = useRef<any>(null);
  const panoramaInstanceRef = useRef<any>(null);

  // Load API Key safely
  const googleMapsApiKey = (import.meta as any).env?.VITE_GOOGLE_MAPS_API_KEY || "";
  const [isGoogleMapsLoaded, setIsGoogleMapsLoaded] = useState<boolean>(false);

  // Default coordinate if no prediction is selected (Eiffel Tower, Paris)
  const lat = prediction ? prediction.latitude : 48.8584;
  const lng = prediction ? prediction.longitude : 2.2945;
  const isUnknown = prediction ? (prediction.latitude === 0 && prediction.longitude === 0) : false;

  const areaName = prediction ? `${prediction.landmarkName}, ${prediction.city}` : "Eiffel Tower, Paris";
  const countryName = prediction ? prediction.country : "France";

  // Dynamic Google Maps JS Loader
  useEffect(() => {
    if (!googleMapsApiKey) return;
    if (window.google && window.google.maps) {
      setIsGoogleMapsLoaded(true);
      return;
    }

    // Check if script is already present in page DOM to avoid duplicate injects
    const existingScript = document.getElementById("google-maps-script");
    if (existingScript) {
      const checkLoaded = setInterval(() => {
        if (window.google && window.google.maps) {
          setIsGoogleMapsLoaded(true);
          clearInterval(checkLoaded);
        }
      }, 100);
      return () => clearInterval(checkLoaded);
    }

    const script = document.createElement("script");
    script.id = "google-maps-script";
    script.src = `https://maps.googleapis.com/maps/api/js?key=${googleMapsApiKey}&libraries=geometry`;
    script.async = true;
    script.defer = true;
    script.onload = () => setIsGoogleMapsLoaded(true);
    document.head.appendChild(script);
  }, [googleMapsApiKey]);

  // Sync scan overlay progress indicator
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

  // Track parent viewport dimensions
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };

    handleResize();

    const observer = new ResizeObserver(handleResize);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Listen to native fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      const isFull = !!document.fullscreenElement && document.fullscreenElement === containerRef.current;
      setIsFullscreen(isFull);
      
      setTimeout(() => {
        if (mapRef.current) {
          mapRef.current.resize();
        }
        if (isGoogleMapsLoaded && mapInstanceRef.current && window.google) {
          window.google.maps.event.trigger(mapInstanceRef.current, "resize");
        }
      }, 150);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [isGoogleMapsLoaded]);

  // Google Maps Instance Controller
  useEffect(() => {
    if (!isGoogleMapsLoaded || !googleMapContainerRef.current || viewMode === "streetview" || !window.google) return;

    let googleMapType = window.google.maps.MapTypeId.ROADMAP;
    if (mapType === "satellite") googleMapType = window.google.maps.MapTypeId.HYBRID;
    if (mapType === "terrain") googleMapType = window.google.maps.MapTypeId.TERRAIN;

    const mapCenter = { lat: isUnknown ? 20 : lat, lng: isUnknown ? 0 : lng };

    if (!mapInstanceRef.current) {
      mapInstanceRef.current = new window.google.maps.Map(googleMapContainerRef.current, {
        center: mapCenter,
        zoom: isUnknown ? 2 : (viewMode === "3d" ? 17 : 14),
        mapTypeId: googleMapType,
        disableDefaultUI: true,
        tilt: viewMode === "3d" ? 45 : 0,
      });

      markerInstanceRef.current = new window.google.maps.Marker({
        position: mapCenter,
        map: isUnknown ? null : mapInstanceRef.current,
        animation: window.google.maps.Animation.DROP,
        icon: {
          path: "M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",
          fillColor: "#ea4335",
          fillOpacity: 1,
          strokeColor: "#ffffff",
          strokeWeight: 2,
          scale: 1.5,
          anchor: new window.google.maps.Point(12, 22),
        }
      });
    } else {
      mapInstanceRef.current.setMapTypeId(googleMapType);
      mapInstanceRef.current.setTilt(viewMode === "3d" ? 45 : 0);
      
      if (!isUnknown) {
        mapInstanceRef.current.panTo(mapCenter);
        mapInstanceRef.current.setZoom(viewMode === "3d" ? 17 : 14);
        markerInstanceRef.current?.setPosition(mapCenter);
        markerInstanceRef.current?.setMap(mapInstanceRef.current);
      } else {
        mapInstanceRef.current.panTo({ lat: 20, lng: 0 });
        mapInstanceRef.current.setZoom(2);
        markerInstanceRef.current?.setMap(null);
      }
    }
  }, [isGoogleMapsLoaded, lat, lng, viewMode, mapType, isUnknown, prediction]);

  // Google Maps Street View Controller
  useEffect(() => {
    if (!isGoogleMapsLoaded || !googleStreetViewContainerRef.current || viewMode !== "streetview" || !window.google) return;

    const targetPos = { lat, lng };

    const initializeOrUpdatePanorama = (positionToUse: { lat: number, lng: number }) => {
      if (!panoramaInstanceRef.current) {
        panoramaInstanceRef.current = new window.google.maps.StreetViewPanorama(googleStreetViewContainerRef.current, {
          position: positionToUse,
          pov: { heading: 165, pitch: 0 },
          zoom: 1,
          addressControl: false,
          linksControl: true,
          panControl: true,
          zoomControl: false,
          enableCloseButton: false,
        });
      } else {
        panoramaInstanceRef.current.setPosition(positionToUse);
      }
    };

    // Use StreetViewService to find the nearest panorama within 1000m
    try {
      const svService = new window.google.maps.StreetViewService();
      svService.getPanorama(
        {
          location: targetPos,
          radius: 1000, // 1km search radius
          source: window.google.maps.StreetViewSource.OUTDOOR
        },
        (data: any, status: any) => {
          if (status === window.google.maps.StreetViewStatus.OK && data && data.location && data.location.latLng) {
            console.log("[StreetView] Snapping to nearest panorama:", data.location.latLng.toString());
            initializeOrUpdatePanorama({
              lat: data.location.latLng.lat(),
              lng: data.location.latLng.lng()
            });
          } else {
            console.warn("[StreetView] No panorama found within 1km, falling back to exact coordinate.");
            initializeOrUpdatePanorama(targetPos);
          }
        }
      );
    } catch (err) {
      console.error("[StreetView] StreetViewService query failed, loading default coordinate:", err);
      initializeOrUpdatePanorama(targetPos);
    }
  }, [isGoogleMapsLoaded, viewMode, lat, lng]);

  // Open-Source Fallback: Map Type styles & textures
  const getGlobeImageUrl = () => {
    if (mapType === "satellite") return "https://unpkg.com/three-globe/example/img/earth-night.jpg";
    if (mapType === "vector") return "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg";
    return "https://unpkg.com/three-globe/example/img/earth-day.jpg";
  };

  const get2DMapStyle = () => {
    if (mapType === "satellite") {
      return {
        version: 8,
        sources: {
          "nasa-tiles": {
            type: "raster",
            tiles: [
              "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/2024-01-01/GoogleMapsCompatible/{z}/{y}/{x}.jpg"
            ],
            tileSize: 256
          }
        },
        layers: [
          {
            id: "nasa-layer",
            type: "raster",
            source: "nasa-tiles",
            minzoom: 0,
            maxzoom: 9
          }
        ]
      } as any;
    }
    if (mapType === "vector") return "https://tiles.openfreemap.org/styles/liberty";
    return "https://tiles.openfreemap.org/styles/positron";
  };

  // Open-Source Fallback: Navigation trigger
  useEffect(() => {
    if (isGoogleMapsLoaded) return;

    if (viewMode === "3d" && globeRef.current) {
      if (prediction && !isUnknown) {
        globeRef.current.pointOfView({ lat, lng, altitude: 1.8 }, 1500);
      } else {
        globeRef.current.pointOfView({ lat: 20, lng: 0, altitude: 2.5 }, 1500);
      }
    } else if (viewMode === "2d" && mapRef.current) {
      if (prediction && !isUnknown) {
        mapRef.current.flyTo({ center: [lng, lat], zoom: 12, duration: 1800 });
      } else {
        mapRef.current.flyTo({ center: [0, 20], zoom: 1.5, duration: 1800 });
      }
    }
  }, [prediction, viewMode, isUnknown, isGoogleMapsLoaded, lat, lng]);

  // Open-Source Fallback: Auto rotation control
  useEffect(() => {
    if (isGoogleMapsLoaded) return;
    if (viewMode === "3d" && globeRef.current) {
      const controls = globeRef.current.controls();
      if (controls) {
        controls.autoRotate = !prediction || isUnknown;
        controls.autoRotateSpeed = 0.3;
      }
    }
  }, [prediction, viewMode, isUnknown, isGoogleMapsLoaded]);

  // Handle Fullscreen request
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => {
        console.error("Error attempting to enable fullscreen:", err);
      });
    } else {
      document.exitFullscreen();
    }
  };

  // Compass view realignment resets
  const handleCompassClick = () => {
    setScanning(true);
    setTimeout(() => setScanning(false), 800);

    const mapCenter = { lat: isUnknown ? 20 : lat, lng: isUnknown ? 0 : lng };

    if (isGoogleMapsLoaded && window.google) {
      if (viewMode === "streetview" && panoramaInstanceRef.current) {
        panoramaInstanceRef.current.setPosition(mapCenter);
      } else if (mapInstanceRef.current) {
        mapInstanceRef.current.panTo(mapCenter);
        mapInstanceRef.current.setZoom(isUnknown ? 2 : (viewMode === "3d" ? 17 : 14));
        mapInstanceRef.current.setHeading(0);
        mapInstanceRef.current.setTilt(viewMode === "3d" ? 45 : 0);
      }
    } else {
      if (viewMode === "3d" && globeRef.current) {
        globeRef.current.pointOfView({ lat: isUnknown ? 20 : lat, lng: isUnknown ? 0 : lng, altitude: isUnknown ? 2.5 : 1.8 }, 1200);
      } else if (viewMode === "2d" && mapRef.current) {
        mapRef.current.flyTo({ center: isUnknown ? [0, 20] : [lng, lat], zoom: isUnknown ? 1.5 : 12, bearing: 0, pitch: 0, duration: 1200 });
      }
    }
  };

  // Custom marker drawing factory for backup globe.gl
  const createGlobeMarkerElement = () => {
    const el = document.createElement("div");
    el.className = "relative flex items-center justify-center pointer-events-none";
    
    const shadow = document.createElement("div");
    shadow.className = "absolute top-4 w-4 h-1.5 bg-black/40 rounded-full blur-xs";
    el.appendChild(shadow);

    const pin = document.createElement("div");
    pin.className = "bg-blue-600 text-white p-2 rounded-full shadow-lg border border-white/20 transform transition-transform duration-300 scale-100 hover:scale-120";
    pin.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0Z"/>
        <circle cx="12" cy="10" r="3"/>
      </svg>
    `;
    el.appendChild(pin);
    return el;
  };

  const globeMarkers = isUnknown ? [] : [{ lat, lng }];

  return (
    <div 
      ref={containerRef}
      id="gis-map-viewport" 
      className={`relative w-full bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl group transition-all duration-300 ${
        isFullscreen ? "h-screen w-screen border-none rounded-none z-50 fixed inset-0" : "h-[480px]"
      }`}
    >
      {/* 1. GOOGLE MAPS RENDERING VIEWPORT */}
      {isGoogleMapsLoaded && !isUnknown ? (
        <div className="absolute inset-0 w-full h-full">
          <div 
            ref={googleStreetViewContainerRef} 
            className={`absolute inset-0 w-full h-full ${viewMode === "streetview" ? 'opacity-100 z-10' : 'opacity-0 pointer-events-none -z-10'}`} 
          />
          <div 
            ref={googleMapContainerRef} 
            className={`absolute inset-0 w-full h-full ${viewMode !== "streetview" ? 'opacity-100 z-10' : 'opacity-0 pointer-events-none -z-10'}`} 
          />
        </div>
      ) : (
        /* 2. OPEN-SOURCE FALLBACK VIEWPORT */
        <div className="absolute inset-0 w-full h-full">
          {viewMode === "3d" ? (
            <Globe
              ref={globeRef}
              width={dimensions.width}
              height={dimensions.height}
              globeImageUrl={getGlobeImageUrl()}
              backgroundColor="#000000"
              showAtmosphere={true}
              atmosphereColor="#4466cc"
              atmosphereAltitude={0.2}
              htmlElementsData={globeMarkers}
              htmlLat={(d: any) => d.lat}
              htmlLng={(d: any) => d.lng}
              htmlElement={createGlobeMarkerElement}
            />
          ) : viewMode === "2d" ? (
            <Map
              ref={mapRef}
              initialViewState={{
                longitude: lng,
                latitude: lat,
                zoom: isUnknown ? 1.5 : 12
              }}
              mapStyle={get2DMapStyle()}
              style={{ width: "100%", height: "100%" }}
            >
              {!isUnknown && (
                <Marker longitude={lng} latitude={lat} anchor="bottom">
                  <div className="relative flex items-center justify-center pointer-events-none">
                    <div className="absolute top-4 w-4 h-1.5 bg-black/40 rounded-full blur-xs" />
                    <div className="bg-blue-600 text-white p-2 rounded-full shadow-lg border border-white/20 transform transition-transform hover:scale-115 duration-300">
                      <MapPin className="w-5 h-5 animate-bounce" />
                    </div>
                  </div>
                </Marker>
              )}
            </Map>
          ) : (
            /* Fallback Warning UI for missing API Key in Street View mode */
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 bg-slate-950 text-slate-400 select-none z-10">
              <Eye className="w-12 h-12 text-slate-600 mb-3 animate-pulse" />
              <h4 className="font-bold text-slate-200 font-mono text-sm uppercase tracking-wide">Google Street View Locked</h4>
              <p className="max-w-[340px] text-xs text-slate-500 font-mono mt-1.5 leading-relaxed">
                Add your <code className="bg-slate-900 border border-slate-800 text-blue-400 px-1 rounded">VITE_GOOGLE_MAPS_API_KEY</code> in your <code className="bg-slate-900 border border-slate-800 px-1 rounded">.env</code> configurations to launch high-fidelity interactive Street View.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Top HUD Header Status Strip */}
      <div className="absolute inset-x-0 top-0 h-6 bg-slate-950/80 border-b border-slate-800 text-[9px] font-mono text-slate-400 flex items-center justify-between px-4 select-none backdrop-blur-xs z-10">
        <span>GRID RANGE: DEEP-WGS84</span>
        <span className="text-blue-400 font-semibold">{scanning ? `SCANNING COORDS [${scanProgress}%]` : "STATUS: STEADY INFERENCE"}</span>
        <span>LAT LIMIT: {lat > 0 ? "NORTH" : "SOUTH"} GLOBE</span>
      </div>

      {/* Right Controls HUD HUD */}
      <div className="absolute right-4 top-10 flex flex-col gap-2 z-10 w-14">
        {/* Fullscreen & Realignment Block */}
        <div className="bg-slate-900/95 border border-slate-800 p-1.5 rounded-xl shadow-lg flex flex-col gap-2.5 backdrop-blur-md items-center w-full">
          {/* Full Screen */}
          <button 
            onClick={toggleFullscreen}
            className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-blue-400 transition-colors flex items-center justify-center cursor-pointer"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Map"}
          >
            {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </button>
          
          <div className="h-[1px] bg-slate-800 w-full" />
          
          {/* Realignment Compass */}
          <button 
            id="map-compass"
            onClick={handleCompassClick}
            className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-blue-400 transition-colors flex items-center justify-center cursor-pointer"
            title="Realignment Compass"
          >
            <Compass className="w-5 h-5 animate-spin-slow text-slate-400" />
          </button>
        </div>

        {/* 2D / 3D / Street View Mode Switcher Panel */}
        <div className="bg-slate-900/95 border border-slate-800 p-1 rounded-xl shadow-lg flex flex-col gap-1 backdrop-blur-md items-center w-full">
          <button
            onClick={() => setViewMode("2d")}
            className={`py-2 rounded-lg text-[8px] font-mono font-bold leading-none transition-all flex flex-col items-center justify-center gap-1 cursor-pointer w-full text-center ${viewMode === "2d" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
            title="2D Map View"
          >
            <MapIcon className="w-4 h-4" />
            <span>2D</span>
          </button>
          <button
            onClick={() => setViewMode("3d")}
            className={`py-2 rounded-lg text-[8px] font-mono font-bold leading-none transition-all flex flex-col items-center justify-center gap-1 cursor-pointer w-full text-center ${viewMode === "3d" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
            title="3D View"
          >
            <GlobeIcon className="w-4 h-4" />
            <span>3D</span>
          </button>
          <button
            onClick={() => setViewMode("streetview")}
            className={`py-2 rounded-lg text-[8px] font-mono font-bold leading-none transition-all flex flex-col items-center justify-center gap-1 cursor-pointer w-full text-center ${viewMode === "streetview" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
            title="Street View View"
          >
            <Eye className="w-4 h-4" />
            <span>STREET</span>
          </button>
        </div>

        {/* Map Layers (Styles) Selector Panel */}
        {viewMode !== "streetview" && (
          <div className="bg-slate-900/95 border border-slate-800 p-1 rounded-xl shadow-lg flex flex-col gap-1 backdrop-blur-md items-center w-full">
            <button
              id="btn-layer-terrain"
              onClick={() => setMapType("terrain")}
              className={`py-2 rounded-lg text-[8px] font-mono font-bold leading-none transition-all flex flex-col items-center justify-center gap-1 cursor-pointer w-full text-center ${mapType === "terrain" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
            >
              <Layers className="w-4 h-4" />
              <span>TERRAIN</span>
            </button>
            <button
              id="btn-layer-satellite"
              onClick={() => setMapType("satellite")}
              className={`py-2 rounded-lg text-[8px] font-mono font-bold leading-none transition-all flex flex-col items-center justify-center gap-1 cursor-pointer w-full text-center ${mapType === "satellite" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
            >
              <GlobeIcon className="w-4 h-4" />
              <span>SAT</span>
            </button>
            <button
              id="btn-layer-vector"
              onClick={() => setMapType("vector")}
              className={`py-2 rounded-lg text-[8px] font-mono font-bold leading-none transition-all flex flex-col items-center justify-center gap-1 cursor-pointer w-full text-center ${mapType === "vector" ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}
            >
              <Cpu className="w-4 h-4" />
              <span>VECTOR</span>
            </button>
          </div>
        )}
      </div>

      {/* Left HUD: Telemetry Details */}
      <div className="absolute left-4 bottom-4 z-10 max-w-[280px]">
        <div className="bg-slate-900/95 border border-slate-800 rounded-xl p-3.5 shadow-xl text-slate-300 backdrop-blur-md flex flex-col gap-2.5">
          <div className="flex items-center gap-2 justify-between border-b border-slate-800 pb-1.5">
            <span className="text-[11px] font-mono font-bold text-blue-400 flex items-center gap-1.5">
              <Navigation className="w-3 h-3 text-blue-400 fill-current" />
              RESOLVED SIGNAL
            </span>
            <span className="text-[9px] font-mono text-slate-500 bg-slate-800/80 px-1.5 py-0.5 rounded uppercase">
              {prediction ? prediction.sceneCategory : "Awaiting"}
            </span>
          </div>

          <div className="flex flex-col gap-1 text-xs">
            <p className="font-semibold text-white truncate text-sm">{isUnknown ? "Unknown Location" : areaName}</p>
            <p className="text-slate-400 text-[11px] truncate">{isUnknown ? "Coordinates not resolvable" : countryName}</p>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[10px] font-mono border-t border-slate-800 pt-2 text-slate-400">
            <div>
              <p className="text-slate-500 text-[9px]">LATITUDE</p>
              <p className="text-white font-medium">{isUnknown ? "0.00000" : lat.toFixed(5)}° {isUnknown ? "" : (lat > 0 ? "N" : "S")}</p>
            </div>
            <div>
              <p className="text-slate-500 text-[9px]">LONGITUDE</p>
              <p className="text-white font-medium">{isUnknown ? "0.00000" : lng.toFixed(5)}° {isUnknown ? "" : (lng > 0 ? "E" : "W")}</p>
            </div>
            {prediction && (
              <>
                <div className="mt-1">
                  <p className="text-slate-500 text-[9px]">ELEVATION</p>
                  <p className="text-emerald-400 font-medium truncate">{prediction.elevation || "N/A"}</p>
                </div>
                <div className="mt-1">
                  <p className="text-slate-500 text-[9px]">BEST CAMERA SEASON</p>
                  <p className="text-blue-400 font-medium truncate">{prediction.bestSeason || "N/A"}</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Right Telemetry Badges */}
      <div className="absolute right-20 bottom-4 z-10 flex flex-col gap-2 items-end">
        <div className="bg-slate-900/95 border border-slate-800 text-[10px] font-mono py-2 px-3 rounded-lg shadow-md text-slate-300 flex items-center gap-2 backdrop-blur-md">
          <Sparkles className="w-3.5 h-3.5 text-blue-400" />
          <span>MATCH QUALITY:</span>
          <span className="font-bold text-emerald-400">
            {prediction ? `${prediction.aiConfidence}%` : "---%"}
          </span>
        </div>

        {prediction && prediction.geologicalAge && (
          <div className="bg-slate-900/95 border border-slate-800 text-[10px] font-mono py-2 px-3 rounded-lg shadow-md text-slate-300 flex items-center gap-2 backdrop-blur-md">
            <Info className="w-3.5 h-3.5 text-slate-500" />
            <span className="truncate max-w-[130px]">{prediction.geologicalAge}</span>
          </div>
        )}
      </div>

      {/* Scale estimate indicator */}
      <div className="absolute left-[300px] bottom-4 hidden md:flex items-center gap-1.5 text-[9.5px] font-mono text-slate-500 z-10">
        <div className="w-12 h-1 border-x border-b border-slate-500" />
        <span>100m</span>
        <span className="text-[8px] bg-slate-900 px-1 text-slate-400 rounded">SCALE ESTIMATE</span>
      </div>

      {/* HUD Info Warning if no Google Maps Key is found */}
      {!googleMapsApiKey && (
        <div className="absolute left-4 top-10 bg-amber-500/10 border border-amber-500/25 px-2.5 py-1 rounded-md text-[9px] font-mono text-amber-400 select-none z-10 backdrop-blur-xs flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
          Connect Google Maps Key in .env to unlock Street View & HD SAT
        </div>
      )}

      {/* 3D mode gesture instructions overlay */}
      {isGoogleMapsLoaded && viewMode === "3d" && (
        <div className="absolute left-4 top-10 bg-blue-500/10 border border-blue-500/25 px-2.5 py-1 rounded-md text-[9px] font-mono text-blue-400 select-none z-10 backdrop-blur-xs flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          Hold Shift + drag mouse to tilt/rotate 3D view
        </div>
      )}

      {/* Scanline Effect Overlay */}
      {scanning && (
        <div className="absolute inset-x-0 h-[30%] bg-gradient-to-b from-transparent via-blue-500/20 to-transparent pointer-events-none animate-scan transform -translate-y-full z-20" style={{
          animation: "scan 3s linear infinite"
        }} />
      )}
    </div>
  );
}
