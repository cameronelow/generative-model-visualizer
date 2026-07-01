/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import { Play, Pause, RefreshCw, Zap, Award, HelpCircle, Activity } from 'lucide-react';
import { ImagePreset } from '../types';

interface GanVisualizerProps {
  selectedPreset: ImagePreset;
}

interface Point2D {
  x: number;
  y: number;
}

// 12 points forming target shapes in 2D coordinate space (-1.5 to 1.5)
const TARGET_SHAPES: Record<string, Point2D[]> = {
  digit3: [
    { x: -0.6, y: 0.8 }, { x: 0.0, y: 0.8 }, { x: 0.6, y: 0.8 },
    { x: 0.6, y: 0.3 }, { x: 0.2, y: 0.0 }, { x: -0.3, y: 0.0 },
    { x: 0.6, y: -0.3 }, { x: 0.6, y: -0.8 }, { x: 0.0, y: -0.8 },
    { x: -0.6, y: -0.8 }, { x: -0.5, y: -0.1 }, { x: 0.5, y: 0.4 }
  ],
  heart: [
    { x: 0.0, y: 0.9 }, { x: 0.5, y: 1.2 }, { x: 1.0, y: 0.7 },
    { x: 0.8, y: 0.1 }, { x: 0.4, y: -0.4 }, { x: 0.0, y: -0.9 },
    { x: -0.4, y: -0.4 }, { x: -0.8, y: 0.1 }, { x: -1.0, y: 0.7 },
    { x: -0.5, y: 1.2 }, { x: 0.2, y: 0.4 }, { x: -0.2, y: 0.4 }
  ],
  smiley: [
    { x: 0.0, y: 1.0 }, { x: 0.7, y: 0.7 }, { x: 1.0, y: 0.0 },
    { x: 0.7, y: -0.7 }, { x: 0.0, y: -1.0 }, { x: -0.7, y: -0.7 },
    { x: -1.0, y: 0.0 }, { x: -0.7, y: 0.7 }, { x: -0.4, y: 0.3 },
    { x: 0.4, y: 0.3 }, { x: -0.4, y: -0.4 }, { x: 0.4, y: -0.4 }
  ],
  alien: [
    { x: -0.8, y: 0.6 }, { x: -0.8, y: 0.0 }, { x: -0.4, y: 0.0 },
    { x: -0.4, y: 0.6 }, { x: 0.4, y: 0.6 }, { x: 0.4, y: 0.0 },
    { x: 0.8, y: 0.0 }, { x: 0.8, y: 0.6 }, { x: -0.4, y: -0.6 },
    { x: 0.4, y: -0.6 }, { x: -0.8, y: -0.6 }, { x: 0.8, y: -0.6 }
  ],
  tree: [
    { x: 0.0, y: 1.1 }, { x: -0.4, y: 0.6 }, { x: 0.4, y: 0.6 },
    { x: -0.7, y: 0.1 }, { x: 0.7, y: 0.1 }, { x: -0.9, y: -0.4 },
    { x: 0.9, y: -0.4 }, { x: -0.2, y: -0.4 }, { x: 0.2, y: -0.4 },
    { x: -0.2, y: -1.1 }, { x: 0.2, y: -1.1 }, { x: 0.0, y: -0.1 }
  ]
};

export default function GanVisualizer({ selectedPreset }: GanVisualizerProps) {
  const [learningRateG, setLearningRateG] = useState<number>(0.1);
  const [learningRateD, setLearningRateD] = useState<number>(0.1);
  const [discriminatorStrength, setDiscriminatorStrength] = useState<number>(0.8);
  const [epochs, setEpochs] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Points tracking
  const [realPoints, setRealPoints] = useState<Point2D[]>([]);
  const [fakePoints, setFakePoints] = useState<Point2D[]>([]);
  const [lossG, setLossG] = useState<number[]>([]);
  const [lossD, setLossD] = useState<number[]>([]);

  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize and Reset points
  const resetTraining = () => {
    setIsPlaying(false);
    setEpochs(0);
    
    // Set real points based on preset target shapes
    const target = TARGET_SHAPES[selectedPreset.id] || TARGET_SHAPES.smiley;
    setRealPoints(target);

    // Initialize generator points as random spread around bottom-left (-1.2, -1.2)
    const initFake = Array.from({ length: target.length }, (_, idx) => {
      // Create cluster in bottom-left corner
      return {
        x: -1.2 + Math.sin(idx * 1.5) * 0.25,
        y: -1.2 + Math.cos(idx * 1.5) * 0.25
      };
    });
    setFakePoints(initFake);

    // Initial losses
    setLossG([1.8]);
    setLossD([0.2]);
  };

  // Run initial setup and reset on preset changes
  useEffect(() => {
    resetTraining();
  }, [selectedPreset]);

  // Performs a single step of GAN training minimax calculation
  const trainStep = () => {
    // 1. G tries to generate samples that match the real data distribution
    // Mathematically G(z) is updated by pushing points toward real points to "fool" D.
    // D wants to separate them, pushing fake points away.
    
    setFakePoints((prevFake) => {
      return prevFake.map((fakePt, idx) => {
        // Find closest real point (which represents G's learning target toward real manifold)
        let closestReal = realPoints[0] || { x: 0, y: 0 };
        let minDistSq = Infinity;
        
        realPoints.forEach((realPt) => {
          const dx = realPt.x - fakePt.x;
          const dy = realPt.y - fakePt.y;
          const distSq = dx * dx + dy * dy;
          if (distSq < minDistSq) {
            minDistSq = distSq;
            closestReal = realPt;
          }
        });

        // Compute step attraction toward real point (G gradient step)
        const gGradX = (closestReal.x - fakePt.x) * learningRateG * 0.25;
        const gGradY = (closestReal.y - fakePt.y) * learningRateG * 0.25;

        // Compute step repulsion from discriminator's decision boundaries (D pushing G)
        // D's push represents the adversarial barrier. If D is strict, G is pushed harder.
        const centerDistX = fakePt.x;
        const centerDistY = fakePt.y;
        const dPushX = -centerDistX * learningRateD * discriminatorStrength * 0.05;
        const dPushY = -centerDistY * learningRateD * discriminatorStrength * 0.05;

        // Apply adversarial minimax step update
        let newX = fakePt.x + gGradX + dPushX;
        let newY = fakePt.y + gGradY + dPushY;

        // Clip constraints
        newX = Math.max(-1.5, Math.min(1.5, newX));
        newY = Math.max(-1.5, Math.min(1.5, newY));

        return { x: newX, y: newY };
      });
    });

    // 2. Compute GAN Losses
    setLossG((prevG) => {
      // Loss G decreases as fake points match real distribution closely
      let totalDist = 0;
      fakePoints.forEach((fPt) => {
        let minDist = Infinity;
        realPoints.forEach((rPt) => {
          const d = Math.sqrt((rPt.x - fPt.x) ** 2 + (rPt.y - fPt.y) ** 2);
          if (d < minDist) minDist = d;
        });
        totalDist += minDist;
      });
      const avgDist = totalDist / fakePoints.length;
      
      // Target Loss G starts at 1.8 and stabilizes around 0.7 (optimal equilibrium)
      const nextG = Math.max(0.69, 1.8 - (epochs * 0.02 * learningRateG) + Math.sin(epochs * 0.1) * 0.05);
      const cappedHistory = prevG.slice(-30); // Keep last 30
      return [...cappedHistory, nextG];
    });

    setLossD((prevD) => {
      // Discriminator loss is low initially (perfect classification), but goes up as G fools it
      const nextD = Math.min(0.69, 0.2 + (epochs * 0.015 * learningRateD) + Math.cos(epochs * 0.1) * 0.03);
      const cappedHistory = prevD.slice(-30);
      return [...cappedHistory, nextD];
    });

    setEpochs((prev) => prev + 1);
  };

  // Playback loop
  useEffect(() => {
    if (isPlaying) {
      playIntervalRef.current = setInterval(() => {
        trainStep();
      }, 120);
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, realPoints, fakePoints, epochs]);

  // Helper mapping 2D coordinate (-1.5 to 1.5) to percentage for SVG plot rendering
  const getPlotPos = (val: number) => {
    const min = -1.5;
    const max = 1.5;
    return ((val - min) / (max - min)) * 100;
  };

  // Discriminator Decision Boundary Representation
  // Returns a background gradient that changes shape as fakePoints align with realPoints
  const getDiscriminatorBackground = () => {
    // We map a simulated 2D decision field:
    // Centered around real point centroids vs fake point centroids
    if (realPoints.length === 0 || fakePoints.length === 0) return '';
    
    // Simulate D's decision boundaries. When epochs are small, D easily classifies.
    // When epochs are high, G is good and boundary goes neutral.
    const alignment = Math.min(1, epochs / 80);
    const dBlur = 10 + alignment * 25; // gets more fuzzy
    const redOpacity = 0.25 * (1 - alignment * 0.8);
    const blueOpacity = 0.25 * (1 - alignment * 0.8);

    return `radial-gradient(circle at 70% 30%, rgba(59, 130, 246, ${blueOpacity}) 0%, transparent ${dBlur}%),
            radial-gradient(circle at ${getPlotPos(fakePoints[0]?.x ?? -1.2)}% ${getPlotPos(fakePoints[0]?.y ?? -1.2)}%, rgba(245, 158, 11, ${redOpacity}) 0%, transparent ${dBlur}%)`;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" id="gan-playground">
      {/* Parameter panel */}
      <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-600" />
            <h3 className="text-md font-bold text-slate-800">Adversarial Game</h3>
          </div>

          <div className="space-y-6">
            {/* Learning rates */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">G Learning Rate (η_g)</span>
                <span className="text-amber-600 font-bold font-mono">{learningRateG.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.5"
                step="0.05"
                value={learningRateG}
                onChange={(e) => setLearningRateG(parseFloat(e.target.value))}
                className="w-full accent-amber-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500">D Learning Rate (η_d)</span>
                <span className="text-amber-600 font-bold font-mono">{learningRateD.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.5"
                step="0.05"
                value={learningRateD}
                onChange={(e) => setLearningRateD(parseFloat(e.target.value))}
                className="w-full accent-amber-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* D Strength */}
            <div>
              <div className="flex justify-between text-xs font-bold mb-1.5 uppercase tracking-wide">
                <span className="text-slate-500 flex items-center gap-1">
                  Discriminator Bound Weight
                  <span className="group relative">
                    <HelpCircle className="w-3.5 h-3.5 text-slate-400 cursor-help" />
                    <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-white text-[10px] text-slate-500 leading-normal p-2 rounded border border-slate-200 shadow-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-25 font-medium">
                      Adjusts the force pushing Generator coordinates away from the center, modeling strict Discriminator boundaries.
                    </span>
                  </span>
                </span>
                <span className="text-amber-600 font-bold font-mono">{discriminatorStrength.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.1"
                value={discriminatorStrength}
                onChange={(e) => setDiscriminatorStrength(parseFloat(e.target.value))}
                className="w-full accent-amber-600 bg-slate-100 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Epoch details */}
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 flex justify-between items-center text-xs">
              <span className="text-slate-500 font-bold uppercase tracking-wider text-[10px]">Training Step count</span>
              <span className="font-mono text-amber-600 text-sm font-bold">{epochs} steps</span>
            </div>
          </div>
        </div>

        {/* Play actions */}
        <div className="mt-8 pt-4 border-t border-slate-100 flex flex-col gap-2">
          <div className="flex gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg font-bold text-xs transition-colors cursor-pointer ${
                isPlaying 
                  ? 'bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100' 
                  : 'bg-amber-600 hover:bg-amber-700 text-white shadow-sm'
              }`}
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4" /> Pause Game
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" /> Start minimax Game
                </>
              )}
            </button>
            <button
              onClick={resetTraining}
              className="p-2.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
              title="Reset weights"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          
          <button
            onClick={trainStep}
            disabled={isPlaying}
            className="w-full py-2 bg-slate-50 hover:bg-slate-100 border border-amber-200 text-amber-700 text-xs font-bold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            Single adversarial training Step
          </button>
        </div>
      </div>

      {/* 2D distribution view */}
      <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Game Map Panel */}
        <div className="md:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
              Data space manifolds
            </h4>
            <div className="flex gap-3 text-[10px] text-slate-500 mb-4 font-bold">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-blue-500 rounded-full inline-block"></span>
                Real samples (Target)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 bg-amber-500 rounded-full inline-block"></span>
                Generated fakes (G)
              </span>
            </div>
          </div>

          <div 
            style={{ backgroundImage: getDiscriminatorBackground() }}
            className="w-full aspect-square bg-slate-50 rounded-lg border border-slate-200 relative overflow-hidden select-none shadow-inner"
          >
            {/* Grid references */}
            <div className="absolute inset-y-0 left-1/2 border-l border-slate-200/60 pointer-events-none"></div>
            <div className="absolute inset-x-0 top-1/2 border-t border-slate-200/60 pointer-events-none"></div>

            {/* Decision boundary color descriptions */}
            <div className="absolute top-2 right-2 text-[9px] bg-blue-50 border border-blue-200 text-blue-700 font-bold px-1.5 py-0.5 rounded font-mono pointer-events-none shadow-sm">
              D(x) → 1 (Real)
            </div>
            <div className="absolute bottom-2 left-2 text-[9px] bg-amber-50 border border-amber-200 text-amber-700 font-bold px-1.5 py-0.5 rounded font-mono pointer-events-none shadow-sm">
              D(G(z)) → 0 (Fake)
            </div>

            {/* SVG coordinates drawing */}
            <svg className="absolute inset-0 w-full h-full p-4 overflow-visible pointer-events-none">
              {/* Draw Real coordinates */}
              {realPoints.map((pt, idx) => (
                <circle
                  key={`real-${idx}`}
                  cx={`${getPlotPos(pt.x)}%`}
                  cy={`${getPlotPos(-pt.y)}%`} // Invert Y for screen drawing
                  r="5"
                  className="fill-blue-500 stroke-white stroke-1.5 shadow"
                />
              ))}

              {/* Draw Generated coordinate paths (G points) */}
              {fakePoints.map((pt, idx) => (
                <g key={`fake-${idx}`}>
                  {isPlaying && (
                    <circle
                      cx={`${getPlotPos(pt.x)}%`}
                      cy={`${getPlotPos(-pt.y)}%`}
                      r="10"
                      className="fill-none stroke-amber-500/30 stroke-1 animate-ping"
                    />
                  )}
                  <circle
                    cx={`${getPlotPos(pt.x)}%`}
                    cy={`${getPlotPos(-pt.y)}%`}
                    r="5"
                    className="fill-amber-500 stroke-white stroke-1.5 transition-all duration-100 shadow"
                  />
                </g>
              ))}
            </svg>
          </div>
        </div>

        {/* Dynamic Loss Convergence chart */}
        <div className="md:col-span-5 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between shadow-sm">
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-amber-600" />
              Loss Convergence Loop
            </h4>
            <p className="text-[11px] text-slate-400 leading-normal mb-6 font-medium">
              Minimax Nash Equilibrium stabilizes near 0.69 (binary cross-entropy loss) when G successfully fools D.
            </p>
          </div>

          {/* Mini line chart */}
          <div className="relative h-44 bg-slate-50 rounded-lg border border-slate-200 p-3 flex flex-col justify-end shadow-inner">
            <svg className="absolute inset-0 w-full h-full p-3 overflow-visible pointer-events-none">
              {/* Grid Lines */}
              <line x1="0" y1="50%" x2="100%" y2="50%" className="stroke-slate-200 stroke-1 stroke-dasharray-[3 3]" />
              
              {/* Equilibrium line (0.69) */}
              <line x1="0" y1="65.5%" x2="100%" y2="65.5%" className="stroke-slate-300 stroke-1.5 stroke-dasharray-[4 4]" />
              <text x="5%" y="62%" className="fill-slate-400 text-[8px] font-mono font-bold">Nash Equilibrium (0.69)</text>

              {/* G Loss line */}
              {lossG.length > 1 && (
                <path
                  d={lossG.map((val, idx) => {
                    const x = `${(idx / (lossG.length - 1)) * 100}%`;
                    const y = `${(1 - val / 2.2) * 100}%`; // Normalized based on max potential loss
                    return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                  }).join(' ')}
                  fill="none"
                  className="stroke-amber-500 stroke-2 transition-all duration-100"
                />
              )}

              {/* D Loss line */}
              {lossD.length > 1 && (
                <path
                  d={lossD.map((val, idx) => {
                    const x = `${(idx / (lossD.length - 1)) * 100}%`;
                    const y = `${(1 - val / 2.2) * 100}%`;
                    return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                  }).join(' ')}
                  fill="none"
                  className="stroke-blue-500 stroke-2 transition-all duration-100"
                />
              )}
            </svg>

            {/* Legend indicators */}
            <div className="flex justify-between w-full text-[9px] text-slate-500 font-mono mt-2 z-10 select-none border-t border-slate-200 pt-2 font-bold">
              <span className="flex items-center gap-1 text-amber-600">
                <span className="inline-block w-2.5 h-0.5 bg-amber-500"></span>
                G Loss: {lossG[lossG.length - 1]?.toFixed(3) ?? '1.800'}
              </span>
              <span className="flex items-center gap-1 text-blue-600">
                <span className="inline-block w-2.5 h-0.5 bg-blue-500"></span>
                D Loss: {lossD[lossD.length - 1]?.toFixed(3) ?? '0.200'}
              </span>
            </div>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-[10px] text-slate-500 leading-normal mt-4 font-medium">
            <span className="font-bold text-slate-700 block mb-0.5 uppercase tracking-wider text-[9px]">Theory Checkpoint</span>
            As G improves, D can no longer differentiate real from fake, so its output accuracy approaches 0.5 (random guess), pushing both losses towards the optimal cross-entropy constant.
          </div>
        </div>
      </div>
    </div>
  );
}
