/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Sliders, Network, ShieldAlert, Cpu, Layers } from 'lucide-react';
import { Grid, ImagePreset } from '../types';

interface AeVisualizerProps {
  selectedPreset: ImagePreset;
}

export default function AeVisualizer({ selectedPreset }: AeVisualizerProps) {
  const [bottleneckSize, setBottleneckSize] = useState<number>(6); // compressed latent neurons
  const [activation, setActivation] = useState<'relu' | 'sigmoid' | 'linear'>('sigmoid');
  const [sparsity, setSparsity] = useState<number>(0.0); // L1 sparsity weight

  // Simulated Auto-Encoder forward pass
  // Encodes 64 pixels into bottleneck neurons, applies activations and sparsity, decodes back.
  const getReconstructedAndLatents = () => {
    const inputFlattened = selectedPreset.data.flat();
    
    // Simulate Encoder Weight compression
    // We create mock weights that compress 64 inputs into bottleneckSize neurons
    // High sparsity forces weights/activations to zero
    const latents = Array.from({ length: bottleneckSize }, (_, idx) => {
      // Deterministic representation based on input profile
      const frequency = 0.5 + idx * 0.15;
      let rawActivation = inputFlattened.reduce((sum, val, pixelIdx) => {
        const weight = Math.sin(pixelIdx * frequency) * 0.15;
        return sum + val * weight;
      }, 0.2);

      // Apply sparsity threshold (L1 penalty)
      if (sparsity > 0) {
        const threshold = sparsity * 0.35;
        if (Math.abs(rawActivation) < threshold) {
          rawActivation = 0;
        } else {
          rawActivation = rawActivation > 0 
            ? rawActivation - threshold 
            : rawActivation + threshold;
        }
      }

      // Apply activation function
      let activated = rawActivation;
      if (activation === 'relu') {
        activated = Math.max(0, rawActivation);
      } else if (activation === 'sigmoid') {
        activated = 1 / (1 + Math.exp(-rawActivation * 4));
      } else {
        // Linear
        activated = rawActivation * 1.5;
      }

      return Math.max(-1, Math.min(1, activated));
    });

    // Simulate Decoder Weight expansion
    // Decode bottleneck activations back to 64 pixels
    const reconstructedFlattened = inputFlattened.map((originalVal, pixelIdx) => {
      let reconstructedVal = 0;
      latents.forEach((latVal, idx) => {
        const decodeWeight = Math.sin(pixelIdx * (0.5 + idx * 0.15)) * 0.2;
        reconstructedVal += latVal * decodeWeight;
      });

      // Normalize reconstruction based on bottleneck capacity
      // High bottleneck size = near-perfect match
      // Low bottleneck size = poor resolution / blurred features
      const capacityRatio = bottleneckSize / 16; // 16 is maximum capacity
      
      // Interpolate original values based on capacity and compression fidelity
      let finalVal = originalVal * capacityRatio + (reconstructedVal + 0.1) * (1 - capacityRatio);
      
      // Activation clipping
      finalVal = Math.max(0, Math.min(1, finalVal));
      return finalVal;
    });

    // Reshape back to 8x8 grid
    const reconstructedGrid: Grid = [];
    for (let i = 0; i < 8; i++) {
      reconstructedGrid.push(reconstructedFlattened.slice(i * 8, i * 8 + 8));
    }

    return { reconstructedGrid, latents };
  };

  const { reconstructedGrid, latents } = getReconstructedAndLatents();

  // Compute reconstruction error (absolute differences)
  const getErrorGrid = (): Grid => {
    return selectedPreset.data.map((row, r) =>
      row.map((val, c) => {
        const reconVal = reconstructedGrid[r]?.[c] ?? 0;
        return Math.abs(val - reconVal);
      })
    );
  };

  const errorGrid = getErrorGrid();

  const renderCell = (val: number, colorPrefix = 'rgba(14, 165, 233, ') => {
    return (
      <div
        style={{ backgroundColor: colorPrefix + val + ')' }}
        className="w-full aspect-square rounded border border-slate-800/10 relative group"
      >
        <span className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-slate-900/90 text-[8px] text-slate-300 font-mono rounded cursor-default select-none transition-opacity">
          {val.toFixed(2)}
        </span>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="ae-playground">
      {/* Parameters Panel */}
      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-sky-600" />
            <h3 className="text-md font-bold text-slate-800">AE Settings</h3>
          </div>

          <div className="space-y-6">
            {/* Bottleneck Size */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Bottleneck Capacity (z)</span>
                <span className="text-sky-600 font-bold font-mono">{bottleneckSize} neurons</span>
              </div>
              <input
                type="range"
                min="2"
                max="16"
                step="1"
                value={bottleneckSize}
                onChange={(e) => setBottleneckSize(parseInt(e.target.value))}
                className="w-full accent-sky-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-bold font-mono mt-1">
                <span>z = 2 (High compression)</span>
                <span>z = 16 (Pristine)</span>
              </div>
            </div>

            {/* Activations */}
            <div>
              <label className="text-xs font-bold text-slate-500 block mb-2 uppercase tracking-wide">
                Activation Function (σ)
              </label>
              <div className="grid grid-cols-3 gap-1.5 bg-slate-50 p-1 rounded-lg border border-slate-200">
                {(['sigmoid', 'relu', 'linear'] as const).map((act) => (
                  <button
                    key={act}
                    onClick={() => setActivation(act)}
                    className={`py-1 text-[10px] font-mono uppercase font-bold rounded transition-all cursor-pointer ${
                      activation === act
                        ? 'bg-sky-600 text-white shadow'
                        : 'text-slate-500 hover:text-slate-850'
                    }`}
                  >
                    {act}
                  </button>
                ))}
              </div>
            </div>

            {/* Sparsity Constraint (L1 Regularization) */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Sparsity Constraint (L1)</span>
                <span className="text-sky-600 font-bold font-mono">{(sparsity * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={sparsity}
                onChange={(e) => setSparsity(parseFloat(e.target.value))}
                className="w-full accent-sky-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-medium">
                {sparsity > 0 
                  ? 'Forces latent activations to zero. Turns the network into a Sparse Auto-Encoder (SAE) for feature learning.' 
                  : 'No sparsity. All bottleneck neurons participate in reconstruction.'}
              </p>
            </div>
          </div>
        </div>

        {/* Informative Stats */}
        <div className="mt-8 pt-4 border-t border-slate-100 space-y-3 text-xs font-mono">
          <div className="flex justify-between text-[11px]">
            <span className="text-slate-400 font-bold uppercase">Compression Ratio</span>
            <span className="text-slate-700 font-bold">{(64 / bottleneckSize).toFixed(1)} : 1</span>
          </div>
          <div className="flex justify-between text-[11px]">
            <span className="text-slate-400 font-bold uppercase">Active Neurons</span>
            <span className="text-sky-600 font-bold">
              {latents.filter(l => Math.abs(l) > 0.05).length} / {bottleneckSize}
            </span>
          </div>
        </div>
      </div>

      {/* Visual Diagrams */}
      <div className="lg:col-span-8 space-y-6">
        {/* Network Connections Block */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-4">
            Deterministic Auto-Encoder Architecture Graph
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
            {/* Input grid */}
            <div className="md:col-span-3 flex flex-col items-center">
              <span className="text-[10px] font-mono text-slate-500 font-bold uppercase mb-2">Input Layer (x)</span>
              <div className="w-28 bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="grid grid-cols-8 gap-0.5">
                  {selectedPreset.data.map((row, r) =>
                    row.map((val, c) => renderCell(val, 'rgba(14, 165, 233, '))
                  )}
                </div>
              </div>
            </div>

            {/* Connections SVG */}
            <div className="md:col-span-6 flex flex-col items-center justify-center">
              <span className="text-[10px] font-mono text-slate-500 font-bold mb-2 uppercase tracking-wide">Bottleneck Layer (z)</span>
              
              <div className="flex items-center gap-4 w-full justify-center">
                {/* 5 Dummy Input channels represent flattened input */}
                <div className="flex flex-col gap-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={`in-${i}`} className="w-2.5 h-2.5 rounded-full bg-sky-500/60 shadow-sm shadow-sky-500/20"></div>
                  ))}
                </div>

                {/* SVG connection lines */}
                <div className="w-20 h-32 relative">
                  <svg className="absolute inset-0 w-full h-full">
                    {Array.from({ length: 5 }).map((_, inIdx) =>
                      Array.from({ length: Math.min(6, bottleneckSize) }).map((_, botIdx) => {
                        const startY = 10 + inIdx * 25;
                        const endY = 10 + botIdx * (110 / Math.min(5, bottleneckSize - 1 || 1));
                        const active = latents[botIdx] !== 0;
                        return (
                          <line
                            key={`l-${inIdx}-${botIdx}`}
                            x1="0"
                            y1={`${startY}`}
                            x2="100%"
                            y2={`${endY}`}
                            className={`stroke-2 transition-all duration-300 ${
                              active 
                                ? 'stroke-sky-500/30' 
                                : 'stroke-slate-200'
                            }`}
                          />
                        );
                      })
                    )}
                  </svg>
                </div>

                {/* Latents Circles list */}
                <div className="flex flex-col gap-1.5 justify-center h-28 overflow-y-auto pr-1">
                  {latents.map((val, idx) => {
                    const isActive = Math.abs(val) > 0.05;
                    const rSize = 10 + Math.abs(val) * 8; // Size modulated by activation
                    return (
                      <div key={`lat-${idx}`} className="flex items-center gap-1.5">
                        <div
                          style={{
                            width: `${rSize}px`,
                            height: `${rSize}px`,
                            backgroundColor: isActive ? 'rgba(14, 165, 233, 0.8)' : '#cbd5e1',
                            boxShadow: isActive ? '0 0 8px rgba(14, 165, 233, 0.4)' : 'none',
                          }}
                          className="rounded-full transition-all duration-300 border border-slate-300"
                          title={`Activation: ${val.toFixed(2)}`}
                        ></div>
                        <span className="text-[9px] font-mono text-slate-500 font-bold select-none">{idx}</span>
                      </div>
                    );
                  })}
                </div>

                {/* SVG connection lines Right */}
                <div className="w-20 h-32 relative">
                  <svg className="absolute inset-0 w-full h-full">
                    {Array.from({ length: Math.min(6, bottleneckSize) }).map((_, botIdx) =>
                      Array.from({ length: 5 }).map((_, outIdx) => {
                        const startY = 10 + botIdx * (110 / Math.min(5, bottleneckSize - 1 || 1));
                        const endY = 10 + outIdx * 25;
                        const active = latents[botIdx] !== 0;
                        return (
                          <line
                            key={`lr-${botIdx}-${outIdx}`}
                            x1="0"
                            y1={`${startY}`}
                            x2="100%"
                            y2={`${endY}`}
                            className={`stroke-2 transition-all duration-300 ${
                              active 
                                ? 'stroke-sky-500/30' 
                                : 'stroke-slate-200'
                            }`}
                          />
                        );
                      })
                    )}
                  </svg>
                </div>

                {/* Reconstruction channels represent outputs */}
                <div className="flex flex-col gap-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={`out-${i}`} className="w-2.5 h-2.5 rounded-full bg-sky-500/40"></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Output reconstruction */}
            <div className="md:col-span-3 flex flex-col items-center">
              <span className="text-[10px] font-mono text-slate-500 font-bold uppercase mb-2">Reconstruction (x̂)</span>
              <div className="w-28 bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="grid grid-cols-8 gap-0.5">
                  {reconstructedGrid.map((row, r) =>
                    row.map((val, c) => renderCell(val, 'rgba(14, 165, 233, '))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Reconstruction Heatmap error comparison */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4 text-sky-600" />
            Pixel-Wise Compression Loss Heatmap | |x - x̂|
          </h4>
          <p className="text-[11px] text-slate-400 leading-normal mb-4 font-medium">
            Displays absolute difference between original and reconstructed outputs. Red pixels indicate areas of high loss due to compression constraints.
          </p>

          <div className="flex flex-col md:flex-row items-center gap-6 justify-center">
            {/* Heatmap Grid */}
            <div className="w-32 bg-slate-50 p-2.5 rounded-lg border border-slate-250">
              <div className="grid grid-cols-8 gap-0.5">
                {errorGrid.map((row, r) =>
                  row.map((val, c) => renderCell(val, 'rgba(239, 68, 68, ')) // red for loss
                )}
              </div>
            </div>

            {/* Explanatory notes */}
            <div className="text-[11px] text-slate-500 leading-relaxed font-sans max-w-sm border-l-2 border-slate-200 pl-4 space-y-1.5 font-medium">
              <p>
                <strong>Information Loss Analysis:</strong>
              </p>
              <p>
                As you lower the <strong>Bottleneck Capacity (z)</strong>, the network is forced to squeeze the 64 input values into fewer variables, discarding high-frequency edge details and resulting in a blurred output.
              </p>
              <p>
                Adding <strong>Sparsity constraints</strong> shuts down non-essential pathways, creating localized, disentangled, and interpretable feature extractors.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
