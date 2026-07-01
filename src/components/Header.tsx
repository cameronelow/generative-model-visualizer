/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Sparkles, Image as ImageIcon } from 'lucide-react';
import { ImagePreset } from '../types';
import { IMAGE_PRESETS } from '../data/presets';

interface HeaderProps {
  selectedPreset: ImagePreset;
  onPresetChange: (preset: ImagePreset) => void;
}

export default function Header({ selectedPreset, onPresetChange }: HeaderProps) {
  return (
    <header className="border-b border-slate-200 bg-white/90 backdrop-blur-md sticky top-0 z-40 px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
      {/* Title block */}
      <div className="flex items-center gap-3">
        <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-indigo-700 rounded-xl shadow-lg shadow-indigo-500/10 flex items-center justify-center">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-md sm:text-lg font-bold text-slate-900 tracking-tight leading-tight">
            Generative Model Visualizer
          </h1>
          <p className="text-[11px] text-slate-500 font-medium">
            Interactive, real-time exploration of forward passes and latent spaces
          </p>
        </div>
      </div>

      {/* Preset Target selector */}
      <div className="flex flex-col sm:flex-row items-center gap-3 bg-slate-50 p-1.5 rounded-xl border border-slate-200 w-full sm:w-auto">
        <div className="flex items-center gap-1.5 px-2 text-slate-500 font-sans text-[11px] font-bold uppercase tracking-wider">
          <ImageIcon className="w-3.5 h-3.5 text-indigo-600" />
          <span>Input Target:</span>
        </div>

        <div className="flex flex-wrap gap-1 w-full sm:w-auto">
          {IMAGE_PRESETS.map((p) => {
            const isSelected = p.id === selectedPreset.id;
            return (
              <button
                key={p.id}
                onClick={() => onPresetChange(p)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-indigo-600 text-white border border-indigo-500 shadow-sm shadow-indigo-100'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white border border-transparent'
                }`}
              >
                {/* Micro Thumbnail Grid */}
                <div className="grid grid-cols-8 gap-[0.5px] w-3 h-3 shrink-0 opacity-80">
                  {p.data.map((row, r) =>
                    row.map((val, c) => (
                      <div
                        key={`thumb-${p.id}-${r}-${c}`}
                        style={{ backgroundColor: val > 0.4 ? 'currentColor' : 'transparent' }}
                        className="w-full aspect-square"
                      ></div>
                    ))
                  )}
                </div>
                <span>{p.name}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
