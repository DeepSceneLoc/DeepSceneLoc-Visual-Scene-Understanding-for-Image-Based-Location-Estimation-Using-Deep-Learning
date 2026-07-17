import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function runTest() {
  console.log("Testing Gemini API key:", process.env.GEMINI_API_KEY ? "Loaded" : "Missing");
  try {
    const res = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: "Respond with exactly the word 'working'."
    });
    console.log("API Response:", res.text);
  } catch (err) {
    console.error("API Error:", err);
  }
}

runTest();
