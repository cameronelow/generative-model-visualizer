/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type Grid = number[][]; // 2D array of values, typically between 0 and 1

export interface ImagePreset {
  id: string;
  name: string;
  category: string;
  data: Grid;
  description: string;
}

export type GenerativeModelType = 
  | 'diffusion' 
  | 'vae' 
  | 'gan' 
  | 'auto-encoder' 
  | 'vector-quantization' 
  | 'auto-regressive';

// --- Hyperparameters per Model ---

export interface DiffusionParams {
  schedule: 'linear' | 'cosine';
  timesteps: number;
  currentStep: number;
  noiseLevel: number; // Beta_t
  cfgScale: number; // Classifier Free Guidance
}

export interface VaeParams {
  latentDims: number;
  beta: number; // KL divergence penalty weight (disentanglement)
  temperature: number; // epsilon sampling noise scale
  selectedLatentX: number; // Coordinate X in 2D latent space
  selectedLatentY: number; // Coordinate Y in 2D latent space
}

export interface GanParams {
  learningRateG: number;
  learningRateD: number;
  discriminatorStrictness: number; // Threshold for binary decision
  trainingEpochs: number;
  generatorStrength: number;
}

export interface AeParams {
  bottleneckSize: number; // size of compression layer
  activation: 'relu' | 'sigmoid' | 'linear';
  sparsity: number; // L1 regularization factor
}

export interface VqParams {
  codebookSize: number; // K vectors
  vectorDimension: number; // D size
  commitmentWeight: number; // Beta multiplier
  gridResolution: number; // number of pixels (e.g. 8x8 vs 16x16 representation)
}

export interface ArParams {
  temperature: number; // Softmax scaling
  topK: number; // Limit vocabulary sampling
  topP: number; // Nucleus sampling threshold
  scanOrder: 'raster' | 'snake' | 'spiral';
  currentPixelIndex: number; // progress through sequence
}
