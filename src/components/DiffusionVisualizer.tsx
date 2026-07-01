/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Sparkles, TrendingUp, Info } from 'lucide-react';
import { Grid, ImagePreset } from '../types';
import { IMAGE_PRESETS, addNoise, lerpGrids, createNoiseGrid } from '../data/presets';

interface DiffusionVisualizerProps {
  selectedPreset: ImagePreset;
}

export default function DiffusionVisualizer({ selectedPreset }: DiffusionVisualizerProps) {
  const [schedule, setSchedule] = useState<'linear' | 'cosine'>('linear');
  const [timesteps, setTimesteps] = useState<number>(50);
  const [currentStep, setCurrentStep] = useState<number>(25);
  const [cfgScale, setCfgScale] = useState<number>(7.5);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Stop playback when preset changes
  useEffect(() => {
    setIsPlaying(false);
  }, [selectedPreset]);

  // Handle play/pause animation for backward denoise loop
  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev <= 0) {
            setIsPlaying(false);
            return 0;
          }
          return prev - 1;
        });
      }, 150);
    } else {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying]);

  // Compute beta_t and alpha_bar_t based on the step and schedule
  const getBetaAndAlphaBar = (step: number, total: number, type: 'linear' | 'cosine') => {
    const tRatio = step / total;
    if (type === 'linear') {
      const betaMin = 0.0001;
      const betaMax = 0.02;
      const beta_t = betaMin + tRatio * (betaMax - betaMin);
      // Approximate alpha_bar as cumulative product
      const alphaBar_t = Math.pow(1 - (betaMin + betaMax) / 2, step);
      return { beta_t, alphaBar_t };
    } else {
      // Cosine schedule (improved DDPM)
      const s = 0.008;
      const f = (t: number) => Math.pow(Math.cos(((t / total + s) / (1 + s)) * Math.PI / 2), 2);
      const alphaBar_t = f(step) / f(0);
      const alphaBar_prev = f(Math.max(0, step - 1)) / f(0);
      const beta_t = Math.min(0.999, 1 - alphaBar_t / alphaBar_prev);
      return { beta_t, alphaBar_t };
    }
  };

  const { beta_t, alphaBar_t } = getBetaAndAlphaBar(currentStep, timesteps, schedule);

  // Generate noisy image at current step
  // In diffusion, x_t = sqrt(alphaBar_t) * x_0 + sqrt(1 - alphaBar_t) * epsilon
  const getNoisyGrid = (): Grid => {
    const noiseAmount = Math.sqrt(1 - alphaBar_t);
    const signalAmount = Math.sqrt(alphaBar_t);
    
    // Create base noisy grid
    const cleanGrid = selectedPreset.data;
    const noisy = cleanGrid.map((row, r) =>
      row.map((val, c) => {
        // Deterministic noise based on position and step for smooth scrubbing
        const u1 = Math.abs(Math.sin(r * 12.9898 + c * 78.233 + 42) * 43758.5453) % 1;
        const u2 = Math.abs(Math.cos(r * 37.123 + c * 54.321 + 84) * 23456.789) % 1;
        const z0 = Math.sqrt(-2.0 * Math.log(u1 || 0.0001)) * Math.cos(2.0 * Math.PI * u2);
        
        const combined = val * signalAmount + z0 * noiseAmount;
        return Math.max(0, Math.min(1, combined));
      })
    );
    return noisy;
  };

  const noisyGrid = getNoisyGrid();

  // Simulated predicted noise (what the U-Net outputs)
  // We simulate it as the residual difference between current noisy image and target image,
  // perturbed slightly based on the guidance scale.
  const getPredictedNoiseGrid = (): Grid => {
    const cleanGrid = selectedPreset.data;
    const pNoise = noisyGrid.map((row, r) =>
      row.map((noisyVal, c) => {
        const cleanVal = cleanGrid[r][c];
        const rawNoise = noisyVal - cleanVal;
        // U-Net predictions are imperfect, model with slight scaling
        const prediction = rawNoise * (0.8 + 0.1 * Math.sin(r * c + currentStep));
        return Math.max(-1, Math.min(1, prediction));
      })
    );
    return pNoise;
  };

  const predictedNoise = getPredictedNoiseGrid();

  // Reconstructed/Denoised output estimate at this step
  // x_0_pred = (x_t - sqrt(1 - alpha_bar) * epsilon_pred) / sqrt(alpha_bar)
  const getDenoisedEstimateGrid = (): Grid => {
    const signalAmount = Math.sqrt(alphaBar_t);
    const noiseAmount = Math.sqrt(1 - alphaBar_t);
    
    if (currentStep === 0) return selectedPreset.data;
    
    // Dynamic CFG adjustment: pushes values toward categorical endpoints (0 or 1)
    return noisyGrid.map((row, r) =>
      row.map((noisyVal, c) => {
        const predN = predictedNoise[r][c];
        let estimatedClean = (noisyVal - noiseAmount * predN) / (signalAmount || 0.01);
        
        // CFG scale push
        if (cfgScale > 1) {
          const center = 0.5;
          estimatedClean = center + (estimatedClean - center) * (1 + (cfgScale - 1) * 0.08);
        }
        
        return Math.max(0, Math.min(1, estimatedClean));
      })
    );
  };

  const estimatedDenoised = getDenoisedEstimateGrid();

  const handleStartLoop = () => {
    setCurrentStep(timesteps);
    setIsPlaying(true);
  };

  const renderCell = (val: number, colorPrefix = 'rgba(99, 102, 241, ') => {
    return (
      <div
        style={{ backgroundColor: colorPrefix + val + ')' }}
        className="w-full aspect-square rounded transition-all duration-100 relative group border border-slate-800/20"
      >
        <span className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-slate-900/90 text-[8px] text-slate-300 font-mono rounded cursor-default select-none transition-opacity">
          {val.toFixed(2)}
        </span>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="diffusion-playground">
      {/* Hyperparameter Controls */}
      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-5 h-5 text-indigo-600" />
            <h3 className="text-md font-bold text-slate-800">Diffusion Controls</h3>
          </div>

          <div className="space-y-6">
            {/* Schedule type */}
            <div>
              <label className="text-xs font-bold text-slate-500 block mb-2 uppercase tracking-wide">
                Noise Schedule
              </label>
              <div className="grid grid-cols-2 gap-2 bg-slate-50 p-1 rounded-lg border border-slate-200">
                <button
                  onClick={() => setSchedule('linear')}
                  className={`py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                    schedule === 'linear'
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  Linear Schedule
                </button>
                <button
                  onClick={() => setSchedule('cosine')}
                  className={`py-1.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                    schedule === 'cosine'
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  Cosine Schedule
                </button>
              </div>
              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-medium">
                {schedule === 'linear' 
                  ? 'Adds noise linearly. Fast initial destruction of structure.' 
                  : 'Adds noise gradually (cosine curve). Retains structure longer into process.'}
              </p>
            </div>

            {/* Total steps */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Total Timesteps (T)</span>
                <span className="text-indigo-600 font-bold font-mono">{timesteps} steps</span>
              </div>
              <input
                type="range"
                min="20"
                max="100"
                step="10"
                value={timesteps}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  setTimesteps(val);
                  if (currentStep > val) setCurrentStep(val);
                }}
                className="w-full accent-indigo-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Current step t */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Current Step (t)</span>
                <span className="text-indigo-600 font-bold font-mono">t = {currentStep}</span>
              </div>
              <input
                type="range"
                min="0"
                max={timesteps}
                value={currentStep}
                onChange={(e) => {
                  setCurrentStep(parseInt(e.target.value));
                  setIsPlaying(false);
                }}
                className="w-full accent-indigo-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-bold font-mono mt-1">
                <span>x0 (Clean)</span>
                <span>xT (Pure Noise)</span>
              </div>
            </div>

            {/* Classifier-Free Guidance (CFG) */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Guidance Scale (CFG)</span>
                <span className="text-indigo-600 font-bold font-mono">{cfgScale.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="15.0"
                step="0.5"
                value={cfgScale}
                onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                className="w-full accent-indigo-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-medium">
                Determines how strongly the model pushes towards generating high-contrast pixels matching the conditioned prompt class.
              </p>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="mt-8 pt-4 border-t border-slate-100">
          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg font-bold text-xs transition-colors cursor-pointer ${
                  isPlaying 
                    ? 'bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100' 
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm shadow-indigo-100'
                }`}
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4" /> Pause Denoising
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white" /> Resume Denoising
                  </>
                )}
              </button>
              
              <button
                onClick={() => {
                  setIsPlaying(false);
                  setCurrentStep(0);
                }}
                className="p-2.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
                title="Reset to clean x0"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
            
            <button
              onClick={handleStartLoop}
              className="w-full flex items-center justify-center gap-2 py-2 bg-slate-50 border border-indigo-200 hover:bg-slate-100 text-indigo-700 font-bold text-xs rounded-lg transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Start Denoise Loop (xT → x0)
            </button>
          </div>
        </div>
      </div>

      {/* Main Forward & Backward Pass Visualization */}
      <div className="lg:col-span-8 space-y-6">
        {/* Step Grid Layout */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between mb-6">
            <h4 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
              <span>Diffusion Process Pipeline</span>
              <span className="px-2 py-0.5 text-[10px] bg-indigo-50 text-indigo-700 rounded-full border border-indigo-100 font-semibold">
                Step {currentStep} / {timesteps}
              </span>
            </h4>
            <div className="flex gap-4 text-xs font-mono">
              <span className="text-slate-500">
                Beta (<span className="text-indigo-600">β_t</span>): <span className="text-slate-700 font-semibold">{beta_t.toFixed(4)}</span>
              </span>
              <span className="text-slate-500">
                Alpha Bar (<span className="text-indigo-600 font-semibold">ᾱ_t</span>): <span className="text-slate-700 font-semibold">{alphaBar_t.toFixed(4)}</span>
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            {/* 1. Current State x_t */}
            <div className="flex flex-col items-center">
              <span className="text-xs font-bold text-slate-600 mb-2 font-mono">Noisy Image x_t</span>
              <div className="w-full max-w-[160px] bg-slate-50 p-2.5 rounded-lg border border-slate-200 shadow-inner">
                <div className="grid grid-cols-8 gap-1">
                  {noisyGrid.map((row, r) =>
                    row.map((val, c) => renderCell(val))
                  )}
                </div>
              </div>
              <p className="text-[10px] text-slate-500 text-center mt-2.5 leading-normal max-w-[160px]">
                {currentStep === 0 
                  ? 'Pure target image x0.' 
                  : currentStep === timesteps 
                  ? 'Gaussian noise at step xT.' 
                  : `Mixture of signal (${(Math.sqrt(alphaBar_t)*100).toFixed(0)}%) and noise.`}
              </p>
            </div>

            {/* 2. Predicted Noise epsilon_pred (U-Net) */}
            <div className="flex flex-col items-center relative">
              {/* Connector Arrows */}
              <div className="hidden md:block absolute -left-4 top-1/2 -translate-y-1/2 text-slate-300 text-xl font-mono">➔</div>
              <div className="hidden md:block absolute -right-4 top-1/2 -translate-y-1/2 text-slate-300 text-xl font-mono">➔</div>
              
              <span className="text-xs font-bold text-slate-600 mb-2 font-mono">U-Net Noise Pred (ε_θ)</span>
              
              {/* Simulated U-Net Box */}
              <div className="w-full max-w-[180px] bg-slate-50 p-3 rounded-lg border border-slate-200 relative overflow-hidden group">
                {/* Visual grid representingPredicted noise (gradient of blues and purples) */}
                <div className="grid grid-cols-8 gap-1 mb-3">
                  {predictedNoise.map((row, r) =>
                    row.map((val, c) => renderCell(Math.abs(val), 'rgba(239, 68, 68, ')) // Visualized as red/orange noise gradients
                  )}
                </div>
                
                <div className="bg-white border border-slate-200 p-1.5 rounded text-center">
                  <span className="text-[9px] font-mono text-slate-500 block uppercase tracking-wider font-bold">UNet Architecture</span>
                  <div className="flex justify-center gap-1.5 my-1">
                    <div className={`h-1.5 w-3 rounded-full ${isPlaying ? 'bg-indigo-600 animate-pulse' : 'bg-indigo-300'}`}></div>
                    <div className={`h-1.5 w-4 rounded-full ${isPlaying ? 'bg-purple-600 animate-pulse delay-75' : 'bg-purple-300'}`}></div>
                    <div className={`h-1.5 w-3 rounded-full ${isPlaying ? 'bg-cyan-600 animate-pulse delay-150' : 'bg-cyan-300'}`}></div>
                  </div>
                </div>
              </div>
              <p className="text-[10px] text-slate-400 text-center mt-2.5 leading-normal max-w-[165px] font-medium">
                Predicts noise in <span className="text-slate-600 font-semibold">x_t</span> relative to the prompt direction.
              </p>
            </div>

            {/* 3. Denoised Target Estimate */}
            <div className="flex flex-col items-center">
              <span className="text-xs font-bold text-slate-600 mb-2 font-mono">Clean Estimate (x0_pred)</span>
              <div className="w-full max-w-[160px] bg-slate-50 p-2.5 rounded-lg border border-indigo-200 shadow-sm">
                <div className="grid grid-cols-8 gap-1">
                  {estimatedDenoised.map((row, r) =>
                    row.map((val, c) => renderCell(val, 'rgba(129, 140, 248, ')) // Indigo scale
                  )}
                </div>
              </div>
              <p className="text-[10px] text-slate-400 text-center mt-2.5 leading-normal max-w-[160px] font-medium">
                Current estimate of the reconstructed clean image. Sharper with higher CFG scales.
              </p>
            </div>
          </div>
        </div>

        {/* Schedule Graph Visualizer */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-4 h-4 text-indigo-600" />
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">Noise &amp; Signal Schedule Function</span>
          </div>
          
          <div className="relative h-28 bg-slate-50 rounded-lg p-3 border border-slate-200 flex flex-col justify-end">
            {/* Graph Grid Lines */}
            <div className="absolute inset-0 grid grid-rows-3 p-3 pointer-events-none">
              <div className="border-b border-slate-200 w-full h-full"></div>
              <div className="border-b border-slate-200 w-full h-full"></div>
              <div className="w-full h-full"></div>
            </div>

            {/* Math lines */}
            <svg className="absolute inset-0 w-full h-full p-3 overflow-visible pointer-events-none">
              {/* Alpha Bar (cumulative signal) Line */}
              <path
                d={Array.from({ length: 101 }).map((_, idx) => {
                  const stepVal = (idx / 100) * timesteps;
                  const { alphaBar_t: aBar } = getBetaAndAlphaBar(stepVal, timesteps, schedule);
                  const x = `${idx}%`;
                  const y = `${(1 - aBar) * 100}%`;
                  return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                }).join(' ')}
                fill="none"
                stroke="rgba(99, 102, 241, 0.6)"
                strokeWidth="2"
              />

              {/* Beta (noise increment) Line */}
              <path
                d={Array.from({ length: 101 }).map((_, idx) => {
                  const stepVal = (idx / 100) * timesteps;
                  const { beta_t: b } = getBetaAndAlphaBar(stepVal, timesteps, schedule);
                  // Scale up beta so it is visible (beta ranges 0 to 0.02)
                  const x = `${idx}%`;
                  const y = `${(1 - b * 40) * 100}%`; // artificial scale
                  return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                }).join(' ')}
                fill="none"
                stroke="rgba(239, 68, 68, 0.4)"
                strokeWidth="1.5"
                strokeDasharray="3 3"
              />

              {/* Glowing Indicator for Current Timestep */}
              <circle
                cx={`${(currentStep / timesteps) * 100}%`}
                cy={`${(1 - alphaBar_t) * 100}%`}
                r="6"
                className="fill-indigo-600 stroke-white stroke-2"
              />
            </svg>

            {/* Axis & Label Details */}
            <div className="flex justify-between w-full text-[9px] text-slate-400 font-mono mt-2 z-10 select-none">
              <span>x0 (Clean, ᾱ=1)</span>
              <span className="text-indigo-600 font-bold">Current Step: t={currentStep}</span>
              <span>xT (Noisy, ᾱ≈0)</span>
            </div>
          </div>
          <div className="flex justify-start gap-4 mt-3 text-[10px] text-slate-500 leading-relaxed font-sans border-l-2 border-slate-200 pl-3">
            <span className="flex items-center gap-1 font-medium">
              <span className="inline-block w-2.5 h-0.5 bg-indigo-500"></span>
              Signal strength (ᾱ_t)
            </span>
            <span className="flex items-center gap-1 font-medium">
              <span className="inline-block w-2.5 h-0.5 border-t border-dashed border-red-500"></span>
              Added step noise (β_t)
            </span>
          </div>
        </div>

        {/* Technical Explainer card */}
        <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex items-start gap-3">
          <Info className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h5 className="text-xs font-bold text-slate-800">How Diffusion Works</h5>
            <p className="text-[11px] text-slate-500 leading-relaxed font-medium">
              Diffusion models work in two phases. In the <strong>Forward Process</strong>, noise is added incrementally following a mathematical schedule until the image is pure white noise. In the <strong>Backward Process</strong>, a trained neural network (U-Net) predicts the noise components at each step, allowing the model to slowly denoise the image. This visualizer lets you scrub or play this process in real-time.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
