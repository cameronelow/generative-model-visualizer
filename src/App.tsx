/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { 
  Sparkles, 
  Layers, 
  Zap, 
  Cpu, 
  BookOpen, 
  Grid3X3, 
  Book, 
  HelpCircle,
  Code,
  Globe,
  Binary
} from 'lucide-react';
import { GenerativeModelType, ImagePreset } from './types';
import { IMAGE_PRESETS } from './data/presets';

// Core Components
import Header from './components/Header';
import DiffusionVisualizer from './components/DiffusionVisualizer';
import VaeVisualizer from './components/VAEVisualizer';
import GanVisualizer from './components/GANVisualizer';
import AeVisualizer from './components/AEVisualizer';
import VqVisualizer from './components/VQVisualizer';
import ArVisualizer from './components/ARVisualizer';
import LatexRenderer from './components/LatexRenderer';

// Model definitions with their meta descriptions and visual branding details
interface ModelTabDefinition {
  type: GenerativeModelType;
  title: string;
  subtitle: string;
  colorClass: string; // Tailwind borders/text colors
  badgeClass: string; // Background accents
  icon: any;
  equation: string;
  useCases: string[];
}

const MODEL_TABS: ModelTabDefinition[] = [
  {
    type: 'diffusion',
    title: 'Diffusion',
    subtitle: 'Iterative Denoising',
    colorClass: 'border-indigo-200 hover:border-indigo-400 text-indigo-700',
    badgeClass: 'bg-indigo-50 text-indigo-700 border border-indigo-200',
    icon: Sparkles,
    equation: 'q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t}x_{t-1}, \\beta_t I)',
    useCases: ['Stable Diffusion', 'Midjourney', 'DALL-E 3', 'Imagen', 'Audio Diffusion'],
  },
  {
    type: 'vae',
    title: 'VAE',
    subtitle: 'Probabilistic Latents',
    colorClass: 'border-emerald-200 hover:border-emerald-400 text-emerald-700',
    badgeClass: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
    icon: Layers,
    equation: '\\mathcal{L}_{VAE} = \\mathbb{E}[\\log p_\\theta(x|z)] - \\beta D_{KL}(q_\\phi(z|x) || p(z))',
    useCases: ['Anomalous Detection', 'Continuous Interpolation', 'Molecule Generation', 'Face Morphing'],
  },
  {
    type: 'gan',
    title: 'GAN',
    subtitle: 'Adversarial Minimax',
    colorClass: 'border-amber-200 hover:border-amber-400 text-amber-700',
    badgeClass: 'bg-amber-50 text-amber-700 border border-amber-200',
    icon: Zap,
    equation: '\\min_G \\max_D V(D, G) = \\mathbb{E}[\\log D(x)] + \\mathbb{E}[\\log (1 - D(G(z)))]',
    useCases: ['StyleGAN (Faces)', 'CycleGAN (Style Transfer)', 'Super Resolution', 'Deepfakes'],
  },
  {
    type: 'auto-encoder',
    title: 'Auto-Encoder',
    subtitle: 'Compression Bottleneck',
    colorClass: 'border-sky-200 hover:border-sky-400 text-sky-700',
    badgeClass: 'bg-sky-50 text-sky-700 border border-sky-200',
    icon: Cpu,
    equation: '\\mathcal{L}_{AE} = ||x - g_\\theta(f_\\phi(x))||^2 + \\lambda \\sum |a_i|',
    useCases: ['Dimensionality Reduction', 'Sparse Feature Learning (SAE)', 'Denoising', 'Image Pretraining'],
  },
  {
    type: 'vector-quantization',
    title: 'VQ-VAE',
    subtitle: 'Discrete Codebook',
    colorClass: 'border-violet-200 hover:border-violet-400 text-violet-700',
    badgeClass: 'bg-violet-50 text-violet-700 border border-violet-200',
    icon: BookOpen,
    equation: 'z_q(x) = e_k \\quad \\text{where} \\quad k = \\text{argmin}_i ||z_e(x) - e_i||',
    useCases: ['VQ-GAN (Image Tokenizers)', 'SoundStream (Audio)', 'VideoGen tokenizers', 'Discrete representation learning'],
  },
  {
    type: 'auto-regressive',
    title: 'Autoregressive',
    subtitle: 'Sequential Token Prediction',
    colorClass: 'border-rose-200 hover:border-rose-400 text-rose-700',
    badgeClass: 'bg-rose-50 text-rose-700 border border-rose-200',
    icon: Binary,
    equation: 'p(x) = \\prod_{i=1}^n p(x_i | x_1, x_2, ..., x_{i-1})',
    useCases: ['GPT Large Language Models', 'PixelCNN (Image)', 'WaveNet (Voice)', 'MusicLM', 'Llama / Gemini'],
  },
];

export default function App() {
  const [selectedPreset, setSelectedPreset] = useState<ImagePreset>(IMAGE_PRESETS[2]); // Default to Smiley Face
  const [activeModel, setActiveModel] = useState<GenerativeModelType>('diffusion');

  const activeModelMeta = MODEL_TABS.find((t) => t.type === activeModel)!;

  // Render correct visualizer based on active tabs
  const renderActiveVisualizer = () => {
    switch (activeModel) {
      case 'diffusion':
        return <DiffusionVisualizer selectedPreset={selectedPreset} />;
      case 'vae':
        return <VaeVisualizer selectedPreset={selectedPreset} />;
      case 'gan':
        return <GanVisualizer selectedPreset={selectedPreset} />;
      case 'auto-encoder':
        return <AeVisualizer selectedPreset={selectedPreset} />;
      case 'vector-quantization':
        return <VqVisualizer selectedPreset={selectedPreset} />;
      case 'auto-regressive':
        return <ArVisualizer selectedPreset={selectedPreset} />;
      default:
        return <DiffusionVisualizer selectedPreset={selectedPreset} />;
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 flex flex-col font-sans">
      
      {/* Universal Header */}
      <Header selectedPreset={selectedPreset} onPresetChange={setSelectedPreset} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-6 space-y-8">
        
        {/* Model Selector Bar */}
        <section className="space-y-3" id="model-selectors">
          <div className="flex items-center gap-2 mb-1">
            <Book className="w-4 h-4 text-indigo-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Select Generative Architecture
            </span>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {MODEL_TABS.map((tab) => {
              const Icon = tab.icon;
              const isSelected = activeModel === tab.type;
              
              let activeBorderColor = 'border-indigo-600 bg-indigo-50/40 ring-2 ring-indigo-600/10 text-indigo-700';
              if (tab.type === 'vae') activeBorderColor = 'border-emerald-600 bg-emerald-50/40 ring-2 ring-emerald-600/10 text-emerald-700';
              if (tab.type === 'gan') activeBorderColor = 'border-amber-600 bg-amber-50/40 ring-2 ring-amber-600/10 text-amber-700';
              if (tab.type === 'auto-encoder') activeBorderColor = 'border-sky-600 bg-sky-50/40 ring-2 ring-sky-600/10 text-sky-700';
              if (tab.type === 'vector-quantization') activeBorderColor = 'border-violet-600 bg-violet-50/40 ring-2 ring-violet-600/10 text-violet-700';
              if (tab.type === 'auto-regressive') activeBorderColor = 'border-rose-600 bg-rose-50/40 ring-2 ring-rose-600/10 text-rose-700';

              return (
                <button
                  key={tab.type}
                  onClick={() => setActiveModel(tab.type)}
                  className={`flex flex-col items-start p-4 rounded-xl border text-left transition-all duration-200 cursor-pointer ${
                    isSelected
                      ? activeBorderColor
                      : 'border-slate-200 bg-white text-slate-500 hover:text-slate-900 hover:bg-slate-50 shadow-sm'
                  }`}
                >
                  <div className={`p-1.5 rounded-lg mb-3 ${isSelected ? 'bg-white shadow-sm' : 'bg-slate-100'}`}>
                    <Icon className={`w-4 h-4 ${isSelected ? 'text-indigo-600' : 'text-slate-400'}`} />
                  </div>
                  <span className="text-xs font-bold block text-slate-800 truncate w-full">
                    {tab.title}
                  </span>
                  <span className="text-[10px] text-slate-400 mt-0.5 truncate w-full font-semibold">
                    {tab.subtitle}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Dynamic Sandbox Section */}
        <section className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm relative overflow-hidden">
          {/* Subtle Decorative Background Glow */}
          <div className="absolute top-0 right-0 w-80 h-80 bg-gradient-to-tr from-indigo-500/5 to-purple-500/5 rounded-full blur-3xl pointer-events-none"></div>
          
          {renderActiveVisualizer()}
        </section>

        {/* Technical Explainer Panel */}
        <section className="grid grid-cols-1 md:grid-cols-12 gap-6 bg-white border border-slate-200 rounded-2xl p-6" id="architectural-notes">
          <div className="md:col-span-4 space-y-4">
            <h4 className="text-sm font-semibold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Code className="w-4.5 h-4.5 text-indigo-600" />
              Theoretical Profile
            </h4>
            
            <div className="space-y-4 text-xs font-mono">
              <div className="bg-slate-50 p-3.5 border border-slate-200 rounded-lg space-y-1">
                <span className="text-slate-400 text-[9px] uppercase tracking-widest font-bold block">
                  Core Formulation
                </span>
                <div className="text-slate-700 overflow-x-auto block py-1 select-all font-medium leading-relaxed">
                  <LatexRenderer math={activeModelMeta.equation} block={true} />
                </div>
              </div>

              <div className="bg-slate-50 p-3.5 border border-slate-200 rounded-lg space-y-1.5">
                <span className="text-slate-400 text-[9px] uppercase tracking-widest font-bold block">
                  Industrial Use Cases
                </span>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {activeModelMeta.useCases.map((uc) => (
                    <span
                      key={uc}
                      className="px-2 py-0.5 rounded text-[10px] bg-white border border-slate-200 text-slate-600 font-sans font-medium shadow-sm"
                    >
                      {uc}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="md:col-span-8 flex flex-col justify-between border-t md:border-t-0 md:border-l border-slate-200 pt-6 md:pt-0 md:pl-6">
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <Globe className="w-4.5 h-4.5 text-indigo-600" />
                Comparative Generative Characteristics
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed font-sans font-medium">
                Generative machine learning models tackle the task of modeling the true distribution of data in highly distinctive ways.
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 text-xs">
                <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                  <span className="font-bold text-slate-700 block">Density Estimation</span>
                  <p className="text-[11px] text-slate-500 leading-relaxed font-sans font-medium">
                    {activeModel === 'diffusion' || activeModel === 'vae' || activeModel === 'auto-regressive'
                      ? 'Explicit (calculates mathematically structured distribution weights).'
                      : 'Implicit (learns patterns directly without calculating actual distribution formulas).'}
                  </p>
                </div>
                <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                  <span className="font-bold text-slate-700 block">Sampling Speed</span>
                  <p className="text-[11px] text-slate-500 leading-relaxed font-sans font-medium">
                    {activeModel === 'diffusion' 
                      ? 'Slow (requires multiple sequential denoising steps).' 
                      : activeModel === 'auto-regressive'
                      ? 'Slow (predicts token-by-token sequentially).'
                      : 'Fast (generates in a single feedforward pass).'}
                  </p>
                </div>
                <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                  <span className="font-bold text-slate-700 block">Training Stability</span>
                  <p className="text-[11px] text-slate-500 leading-relaxed font-sans font-medium">
                    {activeModel === 'gan'
                      ? 'Low (highly sensitive to minimax game imbalances and mode collapse).'
                      : 'High (stable convergence using maximum likelihood or standard MSE loss).'}
                  </p>
                </div>
              </div>
            </div>

            <div className="text-[10px] text-slate-400 font-mono mt-6 pt-3 border-t border-slate-200 select-none">
              Generative Model Visualizer • Developed for interactive browser-based machine learning education.
            </div>
          </div>
        </section>

      </main>

      {/* Universal Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 px-6 text-center text-xs text-slate-400 font-mono mt-auto">
        <span>© 2026 Generative Architecture visualizer • Designed with clean modular UI</span>
      </footer>
    </div>
  );
}
