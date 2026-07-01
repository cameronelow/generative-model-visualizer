/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { Layers, Grid2X2, ArrowRight, BookOpen, Compass } from 'lucide-react';
import { Grid, ImagePreset } from '../types';
import { IMAGE_PRESETS } from '../data/presets';
import LatexRenderer from './LatexRenderer';

interface VqVisualizerProps {
  selectedPreset: ImagePreset;
}

interface Centroid {
  id: number;
  x: number;
  y: number;
  color: string;
}

// Fixed codebook configurations for size K = 2, 4, 8, 16
const CODEBOOK_PRESETS: Record<number, Centroid[]> = {
  2: [
    { id: 0, x: -0.8, y: 0.0, color: '#ec4899' }, // Pink
    { id: 1, x: 0.8, y: 0.0, color: '#3b82f6' }  // Blue
  ],
  4: [
    { id: 0, x: -0.7, y: 0.7, color: '#ec4899' }, // Pink
    { id: 1, x: 0.7, y: 0.7, color: '#3b82f6' },  // Blue
    { id: 2, x: -0.7, y: -0.7, color: '#10b981' }, // Green
    { id: 3, x: 0.7, y: -0.7, color: '#eab308' }  // Yellow
  ],
  8: [
    { id: 0, x: -0.8, y: 0.8, color: '#ec4899' },
    { id: 1, x: 0.0, y: 0.9, color: '#3b82f6' },
    { id: 2, x: 0.8, y: 0.8, color: '#10b981' },
    { id: 3, x: -0.9, y: 0.0, color: '#eab308' },
    { id: 4, x: 0.9, y: 0.0, color: '#8b5cf6' }, // Purple
    { id: 5, x: -0.8, y: -0.8, color: '#f97316' }, // Orange
    { id: 6, x: 0.0, y: -0.9, color: '#06b6d4' }, // Cyan
    { id: 7, x: 0.8, y: -0.8, color: '#a855f7' }  // Violet
  ],
  16: [
    { id: 0, x: -1.1, y: 1.1, color: '#ec4899' }, { id: 1, x: -0.4, y: 1.2, color: '#3b82f6' },
    { id: 2, x: 0.4, y: 1.2, color: '#10b981' }, { id: 3, x: 1.1, y: 1.1, color: '#eab308' },
    { id: 4, x: -1.2, y: 0.4, color: '#8b5cf6' }, { id: 5, x: -0.4, y: 0.4, color: '#f97316' },
    { id: 6, x: 0.4, y: 0.4, color: '#06b6d4' }, { id: 7, x: 1.2, y: 0.4, color: '#a855f7' },
    { id: 8, x: -1.2, y: -0.4, color: '#f43f5e' }, { id: 9, x: -0.4, y: -0.4, color: '#14b8a6' },
    { id: 10, x: 0.4, y: -0.4, color: '#6366f1' }, { id: 11, x: 1.2, y: -0.4, color: '#d946ef' },
    { id: 12, x: -1.1, y: -1.1, color: '#84cc16' }, { id: 13, x: -0.4, y: -1.2, color: '#fb7185' },
    { id: 14, x: 0.4, y: -1.2, color: '#38bdf8' }, { id: 15, x: 1.1, y: -1.1, color: '#c084fc' }
  ]
};

export default function VqVisualizer({ selectedPreset }: VqVisualizerProps) {
  const [codebookSize, setCodebookSize] = useState<2 | 4 | 8 | 16>(8);
  const [commitmentWeight, setCommitmentWeight] = useState<number>(0.25);
  
  // Continuous encoder output (user-draggable cursor coordinate)
  const [encoderX, setEncoderX] = useState<number>(0.15);
  const [encoderY, setEncoderY] = useState<number>(0.35);

  const centroids = CODEBOOK_PRESETS[codebookSize];

  // Map Selected preset to a starting encoder coordinate
  useEffect(() => {
    // Deterministic starting spot based on selected preset
    const presetIndex = IMAGE_PRESETS.findIndex(p => p.id === selectedPreset.id);
    const angle = (presetIndex / IMAGE_PRESETS.length) * 2 * Math.PI;
    const r = 0.6;
    setEncoderX(Math.cos(angle) * r);
    setEncoderY(Math.sin(angle) * r);
  }, [selectedPreset]);

  // Compute nearest Codebook Centroid index
  const getNearestCentroid = (): Centroid => {
    let nearest = centroids[0];
    let minDistSq = Infinity;

    centroids.forEach((c) => {
      const dx = encoderX - c.x;
      const dy = encoderY - c.y;
      const distSq = dx * dx + dy * dy;
      if (distSq < minDistSq) {
        minDistSq = distSq;
        nearest = c;
      }
    });

    return nearest;
  };

  const nearestCentroid = getNearestCentroid();

  // Simulated VQ decoder reconstruction
  // Snapping each block of pixels in the 8x8 input to the nearest vector template.
  // We divide the 8x8 grid into 4 sub-quadrants (representing patch tokens), 
  // and snap each patch to codebook vectors to construct the final image.
  const getQuantizedGrid = (): Grid => {
    const orig = selectedPreset.data;
    const nearestIdx = nearestCentroid.id;

    return orig.map((row, r) =>
      row.map((val, c) => {
        // Divide grid into four quadrants (patches)
        const quadrantX = c >= 4 ? 1 : 0;
        const quadrantY = r >= 4 ? 1 : 0;
        const quadIndex = quadrantY * 2 + quadrantX;

        // Snapping effect:
        // High codebook size maps pixel details continuously
        // Low codebook size posterizes details into heavy stylized block shapes
        if (codebookSize === 2) {
          // Severely quantized to only 2 styles (on/off thresholds based on nearest centroid)
          const threshold = nearestIdx === 0 ? 0.4 : 0.6;
          return val > threshold ? 0.9 : 0.15;
        } else if (codebookSize === 4) {
          // Posterized to discrete thresholds
          if (val < 0.25) return 0.0;
          if (val < 0.6) return nearestIdx % 2 === 0 ? 0.35 : 0.55;
          return 0.95;
        } else if (codebookSize === 8) {
          // Medium fidelity: mixes quadrant indices with nearestCentroid templates
          const patchMultiplier = 0.65 + 0.05 * quadIndex;
          let quantizedVal = val * patchMultiplier;
          if (val > 0.1 && val < 0.75) {
            quantizedVal = nearestIdx % 4 === 0 ? quantizedVal * 0.8 : quantizedVal * 1.1;
          }
          return Math.max(0, Math.min(1, quantizedVal));
        } else {
          // K = 16 (High fidelity, minimal discrete artifacts)
          return Math.max(0, Math.min(1, val * 0.96));
        }
      })
    );
  };

  const quantizedGrid = getQuantizedGrid();

  // Help position coordinate value on SVG plot
  const getPlotPos = (val: number) => {
    // Map -1.5 -> 1.5 to 0% -> 100%
    const min = -1.5;
    const max = 1.5;
    return Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
  };

  const handlePlaneClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const valX = ((x / rect.width) * 3.0) - 1.5;
    const valY = 1.5 - ((y / rect.height) * 3.0); // flip screen Y

    setEncoderX(valX);
    setEncoderY(valY);
  };

  const renderCell = (val: number, colorPrefix = 'rgba(139, 92, 246, ') => {
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
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="vq-playground">
      {/* Parameters Panel */}
      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-violet-600" />
            <h3 className="text-md font-bold text-slate-800">Codebook Configuration</h3>
          </div>

          <div className="space-y-6">
            {/* Codebook Size K */}
            <div>
              <label className="text-xs font-bold text-slate-500 block mb-2 uppercase tracking-wide">
                Codebook Vocabulary Size (K)
              </label>
              <div className="grid grid-cols-4 gap-1.5 bg-slate-50 p-1 rounded-lg border border-slate-200">
                {([2, 4, 8, 16] as const).map((size) => (
                  <button
                    key={size}
                    onClick={() => setCodebookSize(size)}
                    className={`py-1 text-xs font-mono font-bold rounded transition-all cursor-pointer ${
                      codebookSize === size
                        ? 'bg-violet-600 text-white shadow'
                        : 'text-slate-500 hover:text-slate-850'
                    }`}
                  >
                    K={size}
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-slate-400 mt-2 leading-relaxed font-medium">
                Total number of discrete anchor vectors available in the codebook. Smaller K forces harsher discrete clustering.
              </p>
            </div>

            {/* Commitment weight */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">Commitment Loss (β)</span>
                <span className="text-violet-600 font-bold font-mono">{commitmentWeight.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={commitmentWeight}
                onChange={(e) => setCommitmentWeight(parseFloat(e.target.value))}
                className="w-full accent-violet-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed font-medium">
                Penalizes the encoder for outputting features too far from selected codebook anchors. High values force immediate snapping.
              </p>
            </div>

            {/* Current coordinates info */}
            <div className="bg-slate-50 p-3.5 rounded-lg border border-slate-200 space-y-3 font-mono text-xs shadow-inner">
              <span className="text-[10px] text-slate-400 font-bold uppercase block tracking-wider">Quantization Lookup</span>
              <div className="flex justify-between items-center">
                <span className="text-slate-500 font-medium">Continuous z_e(x):</span>
                <span className="text-slate-800 font-bold">({encoderX.toFixed(2)}, {encoderY.toFixed(2)})</span>
              </div>
              <div className="flex justify-between items-center border-t border-slate-200 pt-2">
                <span className="text-slate-500 font-medium">Nearest Anchor ID:</span>
                <span 
                  style={{ color: nearestCentroid.color }}
                  className="font-bold flex items-center gap-1.5"
                >
                  <span 
                    style={{ backgroundColor: nearestCentroid.color }}
                    className="w-2 h-2 rounded-full inline-block"
                  ></span>
                  e_{nearestCentroid.id}
                </span>
              </div>
              <div className="flex justify-between items-center border-t border-slate-200 pt-2">
                <span className="text-slate-500 font-medium">Quantized z_q(x):</span>
                <span style={{ color: nearestCentroid.color }} className="font-bold">
                  ({nearestCentroid.x.toFixed(2)}, {nearestCentroid.y.toFixed(2)})
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-[10px] text-slate-400 leading-normal mt-4 font-bold uppercase">
          <span className="text-slate-500 block mb-0.5 font-mono">Quantization Operator:</span>
          <div className="text-slate-700 block py-1 font-sans text-xs lowercase">
            <LatexRenderer math="z_q(x) = \operatorname{argmin}_i \|z_e(x) - e_i\|" />
          </div>
        </div>
      </div>

      {/* Main visual diagrams */}
      <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Codebook Space diagram */}
        <div className="md:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1">
              <Compass className="w-4 h-4 text-violet-600" />
              Discrete Codebook Space (Voronoi Cells)
            </h4>
            <p className="text-[11px] text-slate-400 leading-normal mb-4 font-medium">
              Click or drag the continuous cursor <span className="text-slate-800 font-mono font-bold">z_e</span>. Watch it snap to the nearest color anchor centroid <span className="font-mono text-violet-600 font-bold">e_i</span>.
            </p>
          </div>

          <div 
            onClick={handlePlaneClick}
            className="w-full aspect-square bg-slate-50 rounded-lg border border-slate-200 relative cursor-crosshair overflow-hidden select-none shadow-inner"
          >
            {/* Grid references */}
            <div className="absolute inset-y-0 left-1/2 border-l border-slate-200 pointer-events-none"></div>
            <div className="absolute inset-x-0 top-1/2 border-t border-slate-200 pointer-events-none"></div>

            {/* Simulated Voronoi Boundary Highlights */}
            {centroids.map((c) => (
              <div
                key={`cell-${c.id}`}
                style={{
                  left: `${getPlotPos(c.x)}%`,
                  top: `${getPlotPos(-c.y)}%`,
                  backgroundColor: c.color,
                }}
                className="absolute w-44 h-44 rounded-full -translate-x-1/2 -translate-y-1/2 opacity-10 blur-3xl pointer-events-none"
              ></div>
            ))}

            {/* Codebook centroids list */}
            {centroids.map((c) => {
              const isNearest = c.id === nearestCentroid.id;
              return (
                <div
                  key={`centroid-${c.id}`}
                  style={{
                    left: `${getPlotPos(c.x)}%`,
                    top: `${getPlotPos(-c.y)}%`, // invert Y
                    borderColor: c.color,
                  }}
                  className={`absolute w-3.5 h-3.5 rounded-full border-2 bg-white flex items-center justify-center -translate-x-1/2 -translate-y-1/2 transition-all ${
                    isNearest ? 'scale-125 border-slate-800 ring-4 ring-violet-500/20' : 'opacity-80 border-slate-200'
                  }`}
                >
                  <div style={{ backgroundColor: c.color }} className="w-1.5 h-1.5 rounded-full"></div>
                  <span className="absolute left-4 text-[9px] text-slate-600 font-mono font-bold bg-white/90 border border-slate-100 px-1 py-0.5 rounded pointer-events-none select-none shadow-sm">
                    e_{c.id}
                  </span>
                </div>
              );
            })}

            {/* SVG Connection line from Encoder cursor to Snapped Centroid */}
            <svg className="absolute inset-0 w-full h-full p-4 overflow-visible pointer-events-none">
              <line
                x1={`${getPlotPos(encoderX)}%`}
                y1={`${getPlotPos(-encoderY)}%`}
                x2={`${getPlotPos(nearestCentroid.x)}%`}
                y2={`${getPlotPos(-nearestCentroid.y)}%`}
                className="stroke-violet-500 stroke-2 stroke-dasharray-[4 4]"
              />
            </svg>

            {/* Draggable encoder output cursor z_e */}
            <div
              style={{
                left: `${getPlotPos(encoderX)}%`,
                top: `${getPlotPos(-encoderY)}%`,
              }}
              className="absolute w-5 h-5 rounded-full border border-slate-400 bg-white flex items-center justify-center -translate-x-1/2 -translate-y-1/2 shadow-md pointer-events-none transition-transform"
            >
              <div className="w-1.5 h-1.5 bg-slate-800 rounded-full"></div>
              <span className="absolute -top-6 text-[9px] font-mono font-bold bg-white border border-slate-200 px-1 py-0.5 rounded text-slate-800 shadow-sm">
                z_e
              </span>
            </div>
          </div>
        </div>

        {/* Quantized output display */}
        <div className="md:col-span-5 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Grid2X2 className="w-4 h-4 text-violet-600" />
              VQ Reconstructed Output
            </h4>
            <p className="text-[11px] text-slate-400 leading-normal mb-6 font-medium">
              Discrete grid reconstruction showing compression blocks matching the selected vocabulary.
            </p>
          </div>

          <div className="flex flex-col items-center justify-center flex-1 py-4">
            <div className="w-full max-w-[170px] bg-slate-50 p-3 rounded-lg border border-slate-200 shadow-inner">
              <div className="grid grid-cols-8 gap-1">
                {quantizedGrid.map((row, r) =>
                  row.map((val, c) => renderCell(val))
                )}
              </div>
            </div>

            <div className="mt-6 flex items-center gap-3 bg-slate-50 px-3 py-2 rounded-lg border border-slate-200 text-[11px] font-mono font-bold">
              <span className="text-slate-400 uppercase text-[9px]">Continuous</span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-violet-600">Discrete Codes</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
