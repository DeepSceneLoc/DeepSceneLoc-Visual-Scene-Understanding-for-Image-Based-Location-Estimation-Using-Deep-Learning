export type SceneCategory = "Urban" | "Rural" | "Coastal" | "Mountain" | "Forest";

export interface PipelineOutput {
  sceneCategory: SceneCategory;
  confidence: number;
  landmarkName: string;
  city: string;
  country: string;
  latitude: number;
  longitude: number;
  aiConfidence: number;
  reasoning: string;
  elevation?: string;
  bestSeason?: string;
  geologicalAge?: string;
}

export interface PresetDemo {
  id: string;
  label: string;
  sceneCategory: SceneCategory;
  country: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface EpochRecord {
  epoch: number;
  trainingLoss: number;
  valAccuracy: number;
  valLoss: number;
}

export interface ConfusionMatrixRow {
  actual: SceneCategory;
  Urban: number;
  Rural: number;
  Coastal: number;
  Mountain: number;
  Forest: number;
}

export interface TeamMember {
  name: string;
  role: string;
  contribution: string;
  imgUrl?: string;
}
