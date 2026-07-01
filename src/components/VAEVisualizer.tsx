/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { Sparkles, Dice5, HelpCircle, RefreshCw, Layers } from 'lucide-react';
import { Grid, ImagePreset } from '../types';
import { IMAGE_PRESETS, lerpGrids } from '../data/presets';
import LatexRenderer from './LatexRenderer';

interface VaeVisualizerProps {
  selectedPreset: ImagePreset;
}

// 2D Prototype placements in latent space
const CLASS_PROTOTYPES = [
  { id: 'digit3', x: -1.2, y: 1.2, color: '#ef4444' }, // Red
  { id: 'heart', x: 1.2, y: 1.2, color: '#ec4899' },  // Pink
  { id: 'smiley', x: -1.2, y: -1.2, color: '#eab308' }, // Yellow
  { id: 'alien', x: 1.2, y: -1.2, color: '#22c55e' },  // Green
  { id: 'tree', x: 0.0, y: 0.0, color: '#06b6d4' },    // Cyan
];

export default function VaeVisualizer({ selectedPreset }: VaeVisualizerProps) {
  const [beta, setBeta] = useState<number>(1.0); // KL regularizer
  const [temperature, setTemperature] = useState<number>(0.3); // Noise scale
  const [latentX, setLatentX] = useState<number>(0.0);
  const [latentY, setLatentY] = useState<number>(0.0);
  const [isSampling, setIsSampling] = useState<boolean>(false);
  
  // Random sampling offset for reparameterization trick visualization
  const [epsX, setEpsX] = useState<number>(0.0);
  const [epsY, setEpsY] = useState<number>(0.0);

  // Sync latent position with selected preset on load/change
  useEffect(() => {
    const matched = CLASS_PROTOTYPES.find(p => p.id === selectedPreset.id);
    if (matched) {
      // Adjust positions slightly based on beta (KL divergence pulls centroids to origin)
      const shrinkFactor = 1 / (1 + (beta - 1) * 0.4);
      setLatentX(matched.x * shrinkFactor);
      setLatentY(matched.y * shrinkFactor);
    }
    setEpsX(0);
    setEpsY(0);
  }, [selectedPreset]);

  // Handle updates to Beta - adjust position representation accordingly
  const handleBetaChange = (newBeta: number) => {
    // As Beta increases (heavier KL penalty), distribution collapses to origin
    const prevShrink = 1 / (1 + (beta - 1) * 0.4);
    const nextShrink = 1 / (1 + (newBeta - 1) * 0.4);
    setLatentX((prevX) => (prevX / prevShrink) * nextShrink);
    setLatentY((prevY) => (prevY / prevShrink) * nextShrink);
    setBeta(newBeta);
  };

  // VAE Decoder simulation
  // Decodes an 8x8 image based on coordinate (latentX + epsX, latentY + epsY)
  const getDecodedGrid = (): Grid => {
    const samplingX = latentX + epsX;
    const samplingY = latentY + epsY;

    // Rescale coordinates back to uncompressed coordinates to map to class prototypes
    const shrinkFactor = 1 / (1 + (beta - 1) * 0.4);
    const uncompressedX = samplingX / shrinkFactor;
    const uncompressedY = samplingY / shrinkFactor;

    // Calculate distance weights to each prototype
    const weights = CLASS_PROTOTYPES.map((p) => {
      const dx = uncompressedX - p.x;
      const dy = uncompressedY - p.y;
      const distanceSq = dx * dx + dy * dy;
      
      // Standard Gaussian RBF Kernel for interpolation: exp(-d^2 / (2 * variance))
      // High beta means smooth but blurry boundaries (wider Gaussian)
      // Low beta means sharp boundaries but dead zones
      const variance = 0.8 + (beta - 1) * 0.3;
      return Math.exp(-distanceSq / (2 * variance));
    });

    const totalWeight = weights.reduce((acc, w) => acc + w, 0) || 0.001;
    const normalizedWeights = weights.map((w) => w / totalWeight);

    // Reconstruct grid as weighted average of prototypes
    const decoded = Array.from({ length: 8 }, (_, r) =>
      Array.from({ length: 8 }, (_, c) => {
        let val = 0;
        normalizedWeights.forEach((w, idx) => {
          const protoPreset = IMAGE_PRESETS.find(p => p.id === CLASS_PROTOTYPES[idx].id);
          if (protoPreset) {
            val += w * protoPreset.data[r][c];
          }
        });
        
        // Lower beta yields high reconstruction fidelity (sharper contrasts)
        // High beta regularizes, blurring the interpolation
        if (beta < 1.0) {
          // Push away from intermediate values toward 0 or 1
          val = val > 0.4 ? Math.min(1, val * 1.2) : val * 0.8;
        } else if (beta > 1.5) {
          // Smooth out contrast, make fuzzy
          val = val * 0.8 + 0.1;
        }
        
        return Math.max(0, Math.min(1, val));
      })
    );

    return decoded;
  };

  const decodedGrid = getDecodedGrid();

  // Trigger reparameterization trick animation (sampling z = mu + sigma * epsilon)
  const handleSample = () => {
    if (isSampling) return;
    setIsSampling(true);
    
    // Compute pseudo variance (σ) based on temperature and KL-Beta
    // Higher beta = lower variance in latent space (regularized distributions have small spread)
    const sigma = temperature * (1 / (0.5 + beta * 0.5));
    
    // Simulate multiple fast updates of epsilon for a shaking effect, ending on a random coordinate
    let count = 0;
    const interval = setInterval(() => {
      // Box-Muller normal sample
      const u1 = Math.random() || 0.001;
      const u2 = Math.random() || 0.001;
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

      setEpsX(z0 * sigma);
      setEpsY(z1 * sigma);

      count++;
      if (count >= 8) {
        clearInterval(interval);
        // Settle on final sample offset
        setIsSampling(false);
      }
    }, 80);
  };

  const renderCell = (val: number, colorPrefix = 'rgba(16, 185, 129, ') => {
    return (
      <div
        style={{ backgroundColor: colorPrefix + val + ')' }}
        className="w-full aspect-square rounded transition-all duration-70 border border-slate-800/10 relative group"
      >
        <span className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-slate-900/90 text-[8px] text-slate-300 font-mono rounded cursor-default select-none transition-opacity">
          {val.toFixed(2)}
        </span>
      </div>
    );
  };

  // Convert coordinate back to grid percentage positions for plotting
  const getPlotPos = (val: number) => {
    // Map -2.0 -> 2.0 to 0% -> 100%
    const min = -2.0;
    const max = 2.0;
    const percentage = ((val - min) / (max - min)) * 100;
    return Math.max(0, Math.min(100, percentage));
  };

  // Handle click inside 2D scatter plot to select mu
  const handleScatterClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Map 0 -> rect.width to -2.0 -> 2.0
    const valX = ((x / rect.width) * 4.0) - 2.0;
    // Map 0 -> rect.height to 2.0 -> -2.0 (Y axis inverted in screens)
    const valY = 2.0 - ((y / rect.height) * 4.0);

    setLatentX(valX);
    setLatentY(valY);
    setEpsX(0);
    setEpsY(0);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="vae-playground">
      {/* Parameters */}
      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-600" />
            <h3 className="text-md font-bold text-slate-800">VAE Bottleneck Options</h3>
          </div>

          {/* Beta term */}
          <div>
            <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
              <span className="text-slate-500 flex items-center gap-1">
                Beta Parameter (β-VAE)
                <span className="group relative">
                  <HelpCircle className="w-3.5 h-3.5 text-slate-400 cursor-help" />
                  <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-white text-[10px] text-slate-500 leading-normal p-2 rounded border border-slate-200 shadow-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-25">
                    Controls KL divergence regularization weight. High Beta compresses distributions together, maximizing disentanglement at the cost of blurrier reconstructions.
                  </span>
                </span>
              </span>
              <span className="text-emerald-600 font-bold font-mono">β = {beta.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="3.0"
              step="0.1"
              value={beta}
              onChange={(e) => handleBetaChange(parseFloat(e.target.value))}
              className="w-full accent-emerald-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400 font-bold mt-1">
              <span>Low (Sharp features)</span>
              <span>High (Symmetric, Blurred)</span>
            </div>
          </div>

          {/* Temperature Noise */}
          <div>
            <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
              <span className="text-slate-500">Sampling Temperature (σ)</span>
              <span className="text-emerald-600 font-bold font-mono">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-emerald-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
            <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-medium">
              Standard deviation of the sampling noise (ε). Higher values increase output variance when reparameterizing.
            </p>
          </div>

          {/* Latent Coordinate values */}
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-2">
            <span className="text-[11px] text-slate-500 font-bold uppercase block tracking-wider">Latent Coordinates</span>
            <div className="grid grid-cols-2 gap-4 text-xs font-mono">
              <div className="flex flex-col">
                <span className="text-slate-400 text-[10px] font-bold uppercase">Mean μ_x</span>
                <span className="text-slate-700 font-bold">{latentX.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-slate-400 text-[10px] font-bold uppercase">Mean μ_y</span>
                <span className="text-slate-700 font-bold">{latentY.toFixed(3)}</span>
              </div>
            </div>
            
            {/* Sampling offset */}
            <div className="border-t border-slate-200 pt-2 grid grid-cols-2 gap-4 text-xs font-mono">
              <div className="flex flex-col">
                <span className="text-slate-400 text-[10px] font-bold uppercase">Noise (ε · σ)_x</span>
                <span className={`text-emerald-600 font-bold transition-all duration-75 ${Math.abs(epsX) > 0 ? 'scale-105' : ''}`}>
                  {epsX >= 0 ? '+' : ''}{epsX.toFixed(3)}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-slate-400 text-[10px] font-bold uppercase">Noise (ε · σ)_y</span>
                <span className={`text-emerald-600 font-bold transition-all duration-75 ${Math.abs(epsY) > 0 ? 'scale-105' : ''}`}>
                  {epsY >= 0 ? '+' : ''}{epsY.toFixed(3)}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-4 border-t border-slate-100">
          <button
            onClick={handleSample}
            disabled={isSampling || temperature === 0}
            className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-lg font-bold text-xs transition-colors cursor-pointer ${
              temperature === 0
                ? 'bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed'
                : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm shadow-emerald-100'
            }`}
          >
            <RefreshCw className={`w-4 h-4 ${isSampling ? 'animate-spin' : ''}`} />
            {isSampling ? 'Reparameterizing...' : 'Sample Latent Space (Reparameterize)'}
          </button>
        </div>
      </div>

      {/* Main Visual Panels */}
      <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Latent Space 2D Plot Grid */}
        <div className="md:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Continuous 2D Latent manifold
            </h4>
            <p className="text-[11px] text-slate-400 leading-normal mb-4 font-medium">
              Click and drag inside this plane to move the mean coordinate. The prototypes pull together under high Beta.
            </p>
          </div>

          <div 
            onClick={handleScatterClick}
            className="w-full aspect-square bg-slate-50 rounded-lg border border-slate-200 relative cursor-crosshair overflow-hidden group select-none shadow-inner"
          >
            {/* Grid axis lines */}
            <div className="absolute inset-y-0 left-1/2 border-l border-dashed border-slate-200 pointer-events-none"></div>
            <div className="absolute inset-x-0 top-1/2 border-t border-dashed border-slate-200 pointer-events-none"></div>

            {/* Interpolated background representation */}
            <div className="absolute inset-0 opacity-15 blur-2xl pointer-events-none">
              {CLASS_PROTOTYPES.map((p) => {
                const shrinkFactor = 1 / (1 + (beta - 1) * 0.4);
                const px = p.x * shrinkFactor;
                const py = p.y * shrinkFactor;
                return (
                  <div
                    key={p.id}
                    style={{
                      left: `${getPlotPos(px)}%`,
                      top: `${getPlotPos(py)}%`,
                      backgroundColor: p.color,
                    }}
                    className="absolute w-36 h-36 rounded-full -translate-x-1/2 -translate-y-1/2"
                  ></div>
                );
              })}
            </div>

            {/* Prototype coordinates markers */}
            {CLASS_PROTOTYPES.map((p) => {
              const shrinkFactor = 1 / (1 + (beta - 1) * 0.4);
              const px = p.x * shrinkFactor;
              const py = p.y * shrinkFactor;
              const isCurrent = p.id === selectedPreset.id;
              
              return (
                <div
                  key={p.id}
                  style={{
                    left: `${getPlotPos(px)}%`,
                    top: `${getPlotPos(py)}%`,
                    borderColor: p.color,
                  }}
                  className={`absolute w-3 h-3 rounded-full -translate-x-1/2 -translate-y-1/2 border-2 bg-white flex items-center justify-center transition-all ${
                    isCurrent ? 'scale-125 border-slate-800 ring-4 ring-emerald-500/20' : ''
                  }`}
                  title={`${p.id} centroid`}
                >
                  <div 
                    style={{ backgroundColor: p.color }}
                    className="w-1 h-1 rounded-full"
                  ></div>
                  <span className="absolute left-4 text-[9px] text-slate-600 font-mono font-bold tracking-tight whitespace-nowrap bg-white/90 border border-slate-100 px-1 py-0.5 rounded pointer-events-none shadow-sm">
                    {p.id}
                  </span>
                </div>
              );
            })}

            {/* Selected Mean coordinate μ (large circle) */}
            <div
              style={{
                left: `${getPlotPos(latentX)}%`,
                top: `${getPlotPos(latentY)}%`,
              }}
              className="absolute w-5 h-5 rounded-full border-2 border-emerald-600 bg-emerald-500/20 shadow-md -translate-x-1/2 -translate-y-1/2 pointer-events-none transition-all flex items-center justify-center"
            >
              <div className="w-1.5 h-1.5 bg-emerald-600 rounded-full"></div>
            </div>

            {/* Reparameterized Sample coordinate z = μ + σ * ε (flashing dot) */}
            {(Math.abs(epsX) > 0 || Math.abs(epsY) > 0) && (
              <div
                style={{
                  left: `${getPlotPos(latentX + epsX)}%`,
                  top: `${getPlotPos(latentY + epsY)}%`,
                }}
                className="absolute w-4.5 h-4.5 rounded-full border border-indigo-500 bg-indigo-500/40 shadow-xl -translate-x-1/2 -translate-y-1/2 pointer-events-none animate-ping duration-150"
              ></div>
            )}
            
            <div
              style={{
                left: `${getPlotPos(latentX + epsX)}%`,
                top: `${getPlotPos(latentY + epsY)}%`,
              }}
              className="absolute w-3 h-3 rounded-full border-2 border-indigo-600 bg-white shadow-md -translate-x-1/2 -translate-y-1/2 pointer-events-none transition-all"
            ></div>
          </div>
        </div>

        {/* Decoder Output Visualizer */}
        <div className="md:col-span-5 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              VAE Decoded output
            </h4>
            <p className="text-[11px] text-slate-400 leading-normal mb-6 font-medium">
              Reconstructed 8x8 image decoded from coordinate <span className="font-mono text-emerald-600 font-bold text-xs">z</span>.
            </p>
          </div>

          <div className="flex flex-col items-center justify-center flex-1 py-4">
            <div className="w-full max-w-[170px] bg-slate-50 p-3 rounded-lg border border-slate-200 shadow-inner">
              <div className="grid grid-cols-8 gap-1">
                {decodedGrid.map((row, r) =>
                  row.map((val, c) => renderCell(val))
                )}
              </div>
            </div>
            
            <div className="mt-6 text-center space-y-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest block font-bold">Decoder Equation</span>
              <div className="text-sm text-slate-700 font-bold block">
                <LatexRenderer math="p_\theta(x \mid z)" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
