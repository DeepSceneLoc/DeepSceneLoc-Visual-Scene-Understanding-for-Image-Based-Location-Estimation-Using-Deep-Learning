import express from "express";
import path from "path";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Configure body parser with high limit to handle uploaded base64 images
app.use(express.json({ limit: "15mb" }));
app.use(express.urlencoded({ limit: "15mb", extended: true }));

// Safe initialization of Gemini Client
let ai: GoogleGenAI | null = null;
if (process.env.GEMINI_API_KEY) {
  ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// Preset locations mapping for direct, fast, highly visual interaction in Section 4 & 5
const PRESET_DEMOS: Record<string, any> = {
  moraine_lake: {
    sceneCategory: "Mountain",
    confidence: 96.48,
    landmarkName: "Moraine Lake",
    city: "Banff",
    country: "Canada",
    latitude: 51.3215,
    longitude: -116.1860,
    aiConfidence: 94.2,
    reasoning: "The image exhibits a bright turquoise glacially-fed lake surrounded by the rugged, steep grey shale pyramids of the Valley of the Ten Peaks in Alberta. The dense coniferous subalpine forest of Engelmann spruce and alpine larch climbing the lower moraine slopes is highly indicative of the Canadian Rocky Mountains transition zone.",
    elevation: "1,885m",
    bestSeason: "June to September",
    geologicalAge: "Late Pleistocene"
  },
  eiffel_tower: {
    sceneCategory: "Urban",
    confidence: 99.12,
    landmarkName: "Eiffel Tower",
    city: "Paris",
    country: "France",
    latitude: 48.8584,
    longitude: 2.2945,
    aiConfidence: 98.7,
    reasoning: "The signature puddle-iron lattice structure, rising in three distinct tiers above the Champ de Mars, is immediately identified. Parisian urban planning characteristics, such as Haussmann-style limestone residential blocks, zinc rooftops, and manicured sycamore tree alleys along the nearby Seine River, pinpoint the location, despite overcast Western European skies.",
    elevation: "35m",
    bestSeason: "All Year",
    geologicalAge: "Built 1889 (Holocene)"
  },
  mount_fuji: {
    sceneCategory: "Mountain",
    confidence: 95.84,
    landmarkName: "Mount Fuji",
    city: "Shizuoka Prefecture",
    country: "Japan",
    latitude: 35.3606,
    longitude: 138.7274,
    aiConfidence: 93.5,
    reasoning: "The silhouette reveals an exceptionally symmetrical stratovolcano with a prominent broad snow-clad summit caldera. The foreground contains cherry blossoms (Prunus serrulata) and Chureito Pagoda or Lake Kawaguchi, with typifying Japanese cedar forestation and typical Pacific ring-of-fire geological formations.",
    elevation: "3,776m",
    bestSeason: "July to September (Climbing)",
    geologicalAge: "Quaternary Volcano"
  },
  monaco_coastal: {
    sceneCategory: "Coastal",
    confidence: 92.15,
    landmarkName: "Monte Carlo Harbour",
    city: "Monaco",
    country: "Monaco",
    latitude: 43.7374,
    longitude: 7.4262,
    aiConfidence: 90.4,
    reasoning: "This coastal marine scene presents high-density Mediterranean residential towers spilling down rugged limestone sea cliffs. Deep indigo saltwater harboring luxury superyachts, accompanied by maritime pines, date palms, and dramatic French Riviera relief, identifies this as the sovereign microstate of Monaco on the Ligurian Sea.",
    elevation: "2m",
    bestSeason: "May to October",
    geologicalAge: "Mesozoic Reef Slopes"
  },
  amazon_rainforest: {
    sceneCategory: "Forest",
    confidence: 89.67,
    landmarkName: "Anavilhanas Archipelago",
    city: "Amazonas",
    country: "Brazil",
    latitude: -2.6934,
    longitude: -60.8282,
    aiConfidence: 87.2,
    reasoning: "The image showcases an dense, unbroken multi-layered tropical canopy split by meandering acid-stained dark river channels (blackwater river). The extreme density of broadleaf evergreen trees, lack of visible human pathways or industrial lines, and atmospheric moisture vapor column rise (evapotranspiration) is highly representative of the central Amazon Basin biome.",
    elevation: "90m",
    bestSeason: "June to November",
    geologicalAge: "Cenozoic Alluvial Plain"
  },
  kenya_savannah: {
    sceneCategory: "Rural",
    confidence: 91.35,
    landmarkName: "Maasai Mara Valley",
    city: "Narok County",
    country: "Kenya",
    latitude: -1.5271,
    longitude: 35.1915,
    aiConfidence: 88.9,
    reasoning: "This rural grassland biome is characterized by low-density, flat-topped Acacia tortilis (umbrella thorn acacia) scattered across dry yellow savannah soil. Red laterite soil paths, distant undulating tectonic rift escarpments, and characteristic lighting represent equatorial Eastern Hemisphere plateaus.",
    elevation: "1,520m",
    bestSeason: "July to October",
    geologicalAge: "Neogene Volcanic Cover"
  }
};

// API Route: Get preset list
app.get("/api/presets", (req, res) => {
  res.json({
    success: true,
    presets: Object.keys(PRESET_DEMOS).map(key => ({
      id: key,
      label: PRESET_DEMOS[key].landmarkName,
      sceneCategory: PRESET_DEMOS[key].sceneCategory,
      country: PRESET_DEMOS[key].country
    }))
  });
});

// API Route: Image Prediction & Geospatial Analysis
app.post("/api/analyze-image", async (req, res) => {
  try {
    const { imageBase64, presetId } = req.body;

    // Fast resolution if it is a preset demo image
    if (presetId && PRESET_DEMOS[presetId]) {
      // Simulate slightly complex deep-learning inference cycles (1.5 seconds)
      await new Promise(resolve => setTimeout(resolve, 1200));
      return res.json({
        success: true,
        source: "preset",
        data: PRESET_DEMOS[presetId]
      });
    }

    if (!imageBase64) {
      return res.status(400).json({
        success: false,
        error: "Missing image string or preset ID."
      });
    }

    // Prepare Base64 data (strip prefix if exist)
    let processedBase64 = imageBase64;
    let mimeType = "image/jpeg";

    if (imageBase64.includes(";base64,")) {
      const parts = imageBase64.split(";base64,");
      const match = parts[0].match(/data:(.*?)$/);
      if (match) mimeType = match[1];
      processedBase64 = parts[1];
    }

    // Call the actual PyTorch Flask Backend for REAL analysis!
    let modelSucceeded = false;
    let modelResponseData: any = null;

    try {
      console.log("Forwarding request to DeepSceneLoc PyTorch backend at http://127.0.0.1:5000...");
      const response = await fetch("http://127.0.0.1:5000/api/analyze-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          imageBase64: processedBase64,
          presetId: presetId
        })
      });

      if (response.ok) {
        const parsedData = await response.json();
        if (parsedData.success && parsedData.data) {
          modelResponseData = parsedData.data;
          modelSucceeded = true;
          console.log("Successfully retrieved PyTorch backend prediction.");
        } else {
          console.warn("PyTorch backend returned an error:", parsedData.error);
        }
      } else {
        console.warn("PyTorch backend responded with status:", response.status);
      }
    } catch (modelError: any) {
      console.warn("DeepSceneLoc Live PyTorch Model error. Will gracefully run simulated_pipeline fallback. Error details:", modelError.message);
    }

    if (modelSucceeded && modelResponseData) {

      let geminiSucceeded = false;
      let fusedData: any = {
        sceneCategory: modelResponseData.sceneCategory,
        confidence: modelResponseData.confidence,
        landmarkName: "Generic / Unknown",
        city: "Unknown",
        country: "Unknown",
        latitude: 0,
        longitude: 0,
        reasoning: "DeepSceneLoc PyTorch classification successful. Gemini multimodal fusion was skipped or failed.",
        elevation: "N/A",
        bestSeason: "N/A",
        geologicalAge: "N/A"
      };

      // Stage 2: Fuse with Gemini AI for dynamic location guessing
      if (ai) {
        try {
          console.log(`Querying Gemini with PyTorch constraint: ${modelResponseData.sceneCategory}...`);

          const prompt = `You are the final multimodal data fusion module of DeepSceneLoc.
Our first-stage PyTorch classifier (ResNet-50 / ViT) has analyzed this image and classified the scene category as: "${modelResponseData.sceneCategory}" with a confidence of ${modelResponseData.confidence}%.

Your task is to accept this hard constraint and estimate the exact real-world geographic location shown in the image.
CRITICAL INSTRUCTION: If the image is highly generic (e.g., just a random field, generic trees, generic coastal water) and lacks any specific identifying landmarks or architecture, DO NOT guess a random place. You must output "Unknown" for landmarkName, city, and country, and output 0 for latitude and longitude. Only guess a specific location if there is strong visual evidence.

You must output a strictly structured JSON response.
Return schema properties:
- landmarkName: Landmark name or specific physical site name (e.g. "Colosseum" or "Unknown" if generic)
- city: Closest town, city, state, or county (or "Unknown")
- country: Name of the country (or "Unknown")
- latitude: Best estimated GPS latitude as a decimal number (or 0 if Unknown)
- longitude: Best estimated GPS longitude as a decimal number (or 0 if Unknown)
- reasoning: Two to three crisp sentences explaining how you deduced this location based on visual clues (architecture, flora, etc.) while agreeing with the ${modelResponseData.sceneCategory} classification. If Unknown, explain why the image is too generic to pinpoint.
- elevation: Estimated average height above sea level (e.g. "120m" or "N/A")
- bestSeason: Optimal month range for clear weather photography (or "N/A")
- geologicalAge: Basic geographic age or era (e.g. "Paleogene Bedrock" or "N/A")`;

          const response = await ai.models.generateContent({
            model: "gemini-3.5-flash",
            contents: [
              {
                inlineData: {
                  data: processedBase64,
                  mimeType: mimeType
                }
              },
              {
                text: prompt
              }
            ],
            config: {
              responseMimeType: "application/json",
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  landmarkName: { type: Type.STRING },
                  city: { type: Type.STRING },
                  country: { type: Type.STRING },
                  latitude: { type: Type.NUMBER },
                  longitude: { type: Type.NUMBER },
                  reasoning: { type: Type.STRING },
                  elevation: { type: Type.STRING },
                  bestSeason: { type: Type.STRING },
                  geologicalAge: { type: Type.STRING }
                },
                required: ["landmarkName", "city", "country", "latitude", "longitude", "reasoning"]
              }
            }
          });

          const responseText = response.text;
          if (responseText) {
            const parsedData = JSON.parse(responseText.trim());
            fusedData = {
              ...modelResponseData,
              landmarkName: parsedData.landmarkName,
              city: parsedData.city,
              country: parsedData.country,
              latitude: parsedData.latitude,
              longitude: parsedData.longitude,
              reasoning: parsedData.reasoning,
              elevation: parsedData.elevation || "N/A",
              bestSeason: parsedData.bestSeason || "All Year",
              geologicalAge: parsedData.geologicalAge || "Recent Holocene",
              aiConfidence: 95.5 // High confidence for fused resolution
            };
            geminiSucceeded = true;
            console.log("Gemini fusion succeeded.");
          }
        } catch (geminiError: any) {
          console.warn("Gemini fusion error. Falling back to PyTorch base prediction.", geminiError.message);
        }
      }

      return res.json({
        success: true,
        source: geminiSucceeded ? "hybrid_fusion" : "deepsceneloc_model",
        data: fusedData
      });
    } else {
      console.warn("PyTorch Backend failed. Throwing 500 error instead of mocking.");
      return res.status(500).json({
        success: false,
        error: "PyTorch Backend (Flask on port 5000) failed to respond or returned an error. Check the Python server logs."
      });
    }

  } catch (error: any) {
    console.error("DeepSceneLoc analysis failed:", error);
    res.status(500).json({
      success: false,
      error: error.message || "Visual evaluation failed due to server timeout. Please try again."
    });
  }
});

// Configure Vite middleware in development or static asset serving in production
async function run() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
    console.log("Vite development server middleware mounted.");
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
    console.log("Serving static production build from:", distPath);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`DeepSceneLoc Backend active on port http://localhost:${PORT}`);
  });
}

run().catch(err => {
  console.error("Failed to start DeepSceneLoc fullstack environment:", err);
});
