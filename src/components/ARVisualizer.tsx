/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, ArrowRight, Eye, Grid3X3, Info } from 'lucide-react';
import { Grid, ImagePreset } from '../types';
import LatexRenderer from './LatexRenderer';

interface ArVisualizerProps {
  selectedPreset: ImagePreset;
}

// Generate coordinate indexes in different scan orders for an 8x8 grid
const getScanCoordinates = (order: 'raster' | 'snake' | 'spiral'): [number, number][] => {
  const coords: [number, number][] = [];
  
  if (order === 'raster') {
    // Normal raster scan: Row-by-Row left-to-right
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        coords.push([r, c]);
      }
    }
  } else if (order === 'snake') {
    // Snake/Boustrophedon scan: alternates directions
    for (let r = 0; r < 8; r++) {
      if (r % 2 === 0) {
        for (let c = 0; c < 8; c++) coords.push([r, c]);
      } else {
        for (let c = 7; c >= 0; c--) coords.push([r, c]);
      }
    }
  } else {
    // Spiral scan
    let top = 0, bottom = 7, left = 0, right = 7;
    let dir = 0; // 0: right, 1: down, 2: left, 3: up
    
    while (top <= bottom && left <= right) {
      if (dir === 0) {
        for (let i = left; i <= right; i++) coords.push([top, i]);
        top++;
      } else if (dir === 1) {
        for (let i = top; i <= bottom; i++) coords.push([i, right]);
        right--;
      } else if (dir === 2) {
        for (let i = right; i >= left; i--) coords.push([bottom, i]);
        bottom--;
      } else if (dir === 3) {
        for (let i = bottom; i >= top; i--) coords.push([i, left]);
        left++;
      }
      dir = (dir + 1) % 4;
    }
  }
  return coords;
};

export default function ArVisualizer({ selectedPreset }: ArVisualizerProps) {
  const [temperature, setTemperature] = useState<number>(0.15); // softmax scaling
  const [topK, setTopK] = useState<number>(4); // limit vocabulary size
  const [topP, setTopP] = useState<number>(0.85); // nucleus probability threshold
  const [scanOrder, setScanOrder] = useState<'raster' | 'snake' | 'spiral'>('raster');
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Buffer state that holds the partially generated grid
  const [generatedGrid, setGeneratedGrid] = useState<Grid>(() => 
    Array.from({ length: 8 }, () => Array(8).fill(0))
  );

  const scanCoords = getScanCoordinates(scanOrder);
  const activeCoord = scanCoords[currentIndex] || [0, 0];
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Reset the generator on preset or scan order change
  const resetGeneration = () => {
    setIsPlaying(false);
    setCurrentIndex(0);
    setGeneratedGrid(Array.from({ length: 8 }, () => Array(8).fill(0)));
  };

  useEffect(() => {
    resetGeneration();
  }, [selectedPreset, scanOrder]);

  // Compute probability distribution for the current pixel value candidates
  // Simulates a softmax over possible gray-scale values [0.0, 0.25, 0.5, 0.75, 1.0]
  const getProbabilityDistribution = () => {
    const targetVal = selectedPreset.data[activeCoord[0]][activeCoord[1]];
    const candidates = [0.0, 0.25, 0.5, 0.75, 1.0];

    // Compute raw logits. Target candidate gets a high logit, others based on distance.
    const logits = candidates.map((cand) => {
      const distance = Math.abs(cand - targetVal);
      // Perfect match gets logit of 4, others fall off.
      return 4.0 - distance * 7.5;
    });

    // Apply Temperature scaling: logits / T (guard against T=0)
    const T = Math.max(0.01, temperature);
    const scaledLogits = logits.map(l => l / T);

    // Softmax calculation
    const maxLogit = Math.max(...scaledLogits);
    const exps = scaledLogits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((acc, e) => acc + e, 0);
    let probs = exps.map(e => e / sumExps);

    // Apply Top-K: Zero out anything outside top K candidates
    const indexedProbs = probs.map((p, idx) => ({ p, idx, val: candidates[idx] }));
    indexedProbs.sort((a, b) => b.p - a.p); // Sort descending
    
    // Set candidates past K to zero probability
    indexedProbs.forEach((item, sortedIdx) => {
      if (sortedIdx >= topK) {
        probs[item.idx] = 0;
      }
    });

    // Re-normalize after Top-K
    let sumK = probs.reduce((acc, p) => acc + p, 0) || 1;
    probs = probs.map(p => p / sumK);

    // Apply Top-P (Nucleus sampling): accumulates probabilities and zero out tail
    const sortedProbs = probs.map((p, idx) => ({ p, idx })).sort((a, b) => b.p - a.p);
    let cumSum = 0;
    let cutOffIdx = sortedProbs.length;

    sortedProbs.forEach((item, sortedIdx) => {
      cumSum += item.p;
      if (cumSum > topP && cutOffIdx === sortedProbs.length && sortedIdx > 0) {
        cutOffIdx = sortedIdx + 1; // Keep up to this index
      }
    });

    sortedProbs.forEach((item, sortedIdx) => {
      if (sortedIdx >= cutOffIdx) {
        probs[item.idx] = 0;
      }
    });

    // Re-normalize final outputs
    let sumP = probs.reduce((acc, p) => acc + p, 0) || 1;
    probs = probs.map(p => p / sumP);

    return candidates.map((val, idx) => ({
      val,
      prob: probs[idx]
    }));
  };

  const distribution = getProbabilityDistribution();

  // Draw one pixel step
  const makeStep = () => {
    if (currentIndex >= 64) {
      setIsPlaying(false);
      return;
    }

    const [r, c] = activeCoord;
    
    // Sample a value from the simulated softmax probability distribution
    const rand = Math.random();
    let cum = 0;
    let selectedValue = 0;

    for (const item of distribution) {
      cum += item.prob;
      if (rand <= cum) {
        selectedValue = item.val;
        break;
      }
    }

    // Set value in buffer
    setGeneratedGrid((prev) => {
      const copy = prev.map(row => [...row]);
      copy[r][c] = selectedValue;
      return copy;
    });

    // Advance index
    setCurrentIndex((prev) => {
      if (prev >= 63) {
        setIsPlaying(false);
        return 64;
      }
      return prev + 1;
    });
  };

  // Automated playback loop
  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = setInterval(() => {
        makeStep();
      }, 100);
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, currentIndex, activeCoord, distribution]);

  // Is a coordinate part of the causal past of the current active pixel?
  // (In causal masking, the model can only attend to pixels generated before current step)
  const isCausalPast = (r: number, c: number): boolean => {
    const indexInScan = scanCoords.findIndex(([sr, sc]) => sr === r && sc === c);
    const activeIndexInScan = currentIndex;
    return indexInScan < activeIndexInScan;
  };

  const renderCell = (val: number, isMasked: boolean, isActive: boolean) => {
    let bg = `rgba(244, 63, 94, ${val})`; // base red/rose color
    
    if (isActive) {
      bg = 'rgb(244, 63, 94)'; // active pixel is glowing solid rose
    } else if (isMasked) {
      // Shaded green/blue represents the causal past (attention mask active)
      bg = `rgba(16, 185, 129, ${val * 0.45 + 0.15})`;
    }

    return (
      <div
        style={{ backgroundColor: bg }}
        className={`w-full aspect-square rounded transition-all duration-75 relative group border ${
          isActive 
            ? 'border-white ring-4 ring-rose-500/30 scale-105 z-10 animate-pulse' 
            : isMasked 
            ? 'border-emerald-500/25' 
            : 'border-slate-200'
        }`}
      >
        <span className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-white/95 text-[8px] text-slate-850 font-mono font-bold rounded cursor-default select-none transition-opacity z-10 border border-slate-200 shadow-sm">
          {val.toFixed(2)}
        </span>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="ar-playground">
      {/* Parameters Panel */}
      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Grid3X3 className="w-5 h-5 text-rose-600" />
            <h3 className="text-md font-bold text-slate-800">Sampling Hyperparameters</h3>
          </div>

          <div className="space-y-6">
            {/* Scan Order */}
            <div>
              <label className="text-xs font-bold text-slate-500 block mb-2 uppercase tracking-wide">
                Autoregressive Scan Order
              </label>
              <div className="grid grid-cols-3 gap-1.5 bg-slate-50 p-1 rounded-lg border border-slate-200">
                {(['raster', 'snake', 'spiral'] as const).map((order) => (
                  <button
                    key={order}
                    onClick={() => {
                      setScanOrder(order);
                      setIsPlaying(false);
                    }}
                    className={`py-1 text-[10px] font-mono uppercase font-bold rounded transition-all cursor-pointer ${
                      scanOrder === order
                        ? 'bg-rose-600 text-white shadow'
                        : 'text-slate-500 hover:text-slate-850'
                    }`}
                  >
                    {order}
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-slate-400 mt-2 leading-relaxed font-medium">
                {scanOrder === 'raster' 
                  ? 'Left-to-right, row-by-row raster scan. Traditional ordering.' 
                  : scanOrder === 'snake'
                  ? 'Alternates scanning direction every row, reducing sharp transitions.'
                  : 'Fills the grid starting from corners in a wrapping spiral.'}
              </p>
            </div>

            {/* Softmax Temperature */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Softmax Temperature (T)</span>
                <span className="text-rose-600 font-bold font-mono">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="1.5"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-rose-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-medium">
                {temperature < 0.2 
                  ? 'Highly deterministic. Selects optimal outputs, matching template exactly.' 
                  : 'High random variance. Introduces creative and chaotic deviations.'}
              </p>
            </div>

            {/* Top-K limit */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Top-K Vocabulary Filter</span>
                <span className="text-rose-600 font-bold font-mono">K = {topK} candidates</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full accent-rose-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Top-P Nucleus */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Top-P Nucleus Threshold</span>
                <span className="text-rose-600 font-bold font-mono">p = {(topP * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="w-full accent-rose-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Start buttons */}
        <div className="mt-8 pt-4 border-t border-slate-100 flex flex-col gap-2">
          <div className="flex gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={currentIndex >= 64}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg font-bold text-xs transition-colors cursor-pointer ${
                currentIndex >= 64
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
                  : isPlaying 
                  ? 'bg-rose-50 text-rose-700 border border-rose-200 hover:bg-rose-100' 
                  : 'bg-rose-600 hover:bg-rose-700 text-white shadow-sm'
              }`}
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4" /> Pause Generation
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" /> Start Autoregressive scan
                </>
              )}
            </button>
            <button
              onClick={resetGeneration}
              className="p-2.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
              title="Reset grid"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
          
          <button
            onClick={makeStep}
            disabled={isPlaying || currentIndex >= 64}
            className="w-full py-2 bg-slate-50 hover:bg-slate-100 border border-rose-200 text-rose-750 text-xs font-bold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            Generate Next Pixel step
          </button>
        </div>
      </div>

      {/* Main visualization grid */}
      <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Active scan canvas */}
        <div className="md:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1">
                <Eye className="w-4 h-4 text-emerald-600" />
                Causal Attention Mask Receptive Field
              </h4>
              <span className="font-mono text-[10px] bg-slate-50 px-2 py-0.5 border border-slate-200 rounded text-rose-600 font-bold">
                Index: {currentIndex} / 64
              </span>
            </div>
            <div className="flex gap-4 text-[10px] text-slate-500 mb-4 font-bold leading-normal">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-emerald-500/40 rounded inline-block"></span>
                Causal Receptive Past (Masked)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-rose-500 rounded inline-block animate-pulse border border-white"></span>
                Current Prediction (Next Token)
              </span>
            </div>
          </div>

          <div className="flex items-center justify-center p-6 bg-slate-50 rounded-lg border border-slate-200 shadow-inner">
            <div className="w-full max-w-[190px] bg-white p-3 rounded-lg border border-rose-250/20 shadow-md">
              <div className="grid grid-cols-8 gap-1.5">
                {Array.from({ length: 8 }).map((_, r) =>
                  Array.from({ length: 8 }).map((_, c) => {
                    const isActive = activeCoord[0] === r && activeCoord[1] === c;
                    const isMasked = isCausalPast(r, c);
                    // Value shown is from buffer if it was already generated, or 0 otherwise
                    const val = generatedGrid[r][c];
                    return (
                      <div key={`ar-cell-${r}-${c}`}>
                        {renderCell(val, isMasked, isActive)}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Softmax probabilities panel */}
        <div className="md:col-span-5 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Next-Pixel Logit Softmax Probabilities
            </h4>
            <p className="text-[11px] text-slate-400 leading-normal mb-6 font-medium">
              Receptivity histogram displaying candidates matching the causal condition. Affected by Temperature, Top-K, and Top-P.
            </p>
          </div>

          {/* Probability Bars */}
          <div className="space-y-3 flex-1 flex flex-col justify-center">
            {distribution.map((item, idx) => {
              const percentage = item.prob * 100;
              const isMax = item.prob === Math.max(...distribution.map(d => d.prob));
              return (
                <div key={`prob-${idx}`} className="space-y-1">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-slate-500 font-bold">Value candidate: {item.val.toFixed(2)}</span>
                    <span className={`${isMax ? 'text-rose-600 font-bold' : 'text-slate-400 font-bold'}`}>
                      {percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-50 rounded border border-slate-200 overflow-hidden relative shadow-inner">
                    <div
                      style={{ width: `${percentage}%` }}
                      className={`h-full rounded transition-all duration-150 ${
                        isMax 
                          ? 'bg-rose-500 shadow-md shadow-rose-500/20' 
                          : 'bg-slate-300'
                      }`}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-[10px] text-slate-400 leading-normal mt-5 font-bold uppercase">
            <span className="text-slate-500 block mb-0.5 font-mono">Equation context:</span>
            <div className="text-slate-700 block py-1 font-sans text-xs lowercase">
              <LatexRenderer math="p(x_i \mid x_1, x_2, \dots, x_{i-1})" />
            </div>
          </div>
        </div>
      </div>

      {/* Technical guide banner */}
      <div className="lg:col-span-12 bg-slate-50 p-4 rounded-xl border border-slate-200 flex items-start gap-3 shadow-sm">
        <Info className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h5 className="text-xs font-bold text-slate-800">How Autoregressive Models Generate Data</h5>
          <p className="text-[11px] text-slate-400 leading-relaxed font-medium">
            Autoregressive models (such as GPT or PixelCNN) generate data strictly <strong>sequentially (one element at a time)</strong>. Each step is modeled as a categorical probability prediction conditioned strictly on the preceding context elements. The model uses a **causal mask** (which shades out all future grid elements) to ensure it only learns from historically valid context tokens.
          </p>
        </div>
      </div>
    </div>
  );
}
