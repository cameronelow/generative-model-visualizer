/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagePreset, Grid } from '../types';

// 8x8 Grids representing distinctive patterns.
// Values range from 0 (empty/background) to 1 (fully active/foreground).

export const PRESET_DIGIT_3: Grid = [
  [0, 1, 1, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 1, 0, 0],
  [0, 0, 1, 1, 1, 0, 0, 0],
  [0, 0, 0, 0, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 1, 1, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
];

export const PRESET_HEART: Grid = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 1, 0, 0, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [0, 1, 1, 1, 1, 1, 1, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
];

export const PRESET_SMILEY: Grid = [
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 1, 0, 0, 0, 0, 1, 0],
  [1, 0, 1, 0, 0, 1, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 0, 0, 1, 0, 1],
  [1, 0, 0, 1, 1, 0, 0, 1],
  [0, 1, 0, 0, 0, 0, 1, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
];

export const PRESET_ALIEN: Grid = [
  [0, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 1, 1, 0, 0, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 1, 1, 1, 1, 0, 1],
  [1, 0, 1, 0, 0, 1, 0, 1],
  [0, 0, 0, 1, 1, 0, 0, 0],
];

export const PRESET_TREE: Grid = [
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 1, 1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
];

export const IMAGE_PRESETS: ImagePreset[] = [
  {
    id: 'digit3',
    name: 'Digit 3',
    category: 'Numbers',
    data: PRESET_DIGIT_3,
    description: 'An MNIST-like handwritten digit representing class 3.',
  },
  {
    id: 'heart',
    name: 'Heart Symbol',
    category: 'Icons',
    data: PRESET_HEART,
    description: 'A symmetrical heart icon demonstrating smooth curves.',
  },
  {
    id: 'smiley',
    name: 'Smiley Face',
    category: 'Icons',
    data: PRESET_SMILEY,
    description: 'A cheerful emoji face showing high detail features.',
  },
  {
    id: 'alien',
    name: 'Space Invader',
    category: 'Retro Art',
    data: PRESET_ALIEN,
    description: 'An 8-bit arcade classic displaying symmetric block patterns.',
  },
  {
    id: 'tree',
    name: 'Pine Tree',
    category: 'Nature',
    data: PRESET_TREE,
    description: 'A pixelated pine tree showcasing vertical stacking structures.',
  },
];

// Helper functions for grid math

// Adds Gaussian noise to a grid
export function addNoise(grid: Grid, amount: number, seed: number = 0.5): Grid {
  return grid.map((row, r) =>
    row.map((val, c) => {
      // Basic pseudo-random normal approximation using Box-Muller transform
      const u1 = Math.abs(Math.sin(r * 12.9898 + c * 78.233 + seed) * 43758.5453) % 1;
      const u2 = Math.abs(Math.cos(r * 37.123 + c * 54.321 + seed * 2) * 23456.789) % 1;
      const z0 = Math.sqrt(-2.0 * Math.log(u1 || 0.0001)) * Math.cos(2.0 * Math.PI * u2);
      
      const noisy = val + z0 * amount;
      return Math.max(0, Math.min(1, noisy)); // Clip between 0 and 1
    })
  );
}

// Computes a linear interpolation between two grids
export function lerpGrids(gridA: Grid, gridB: Grid, t: number): Grid {
  return gridA.map((row, r) =>
    row.map((val, c) => {
      const bVal = gridB[r]?.[c] ?? 0;
      return val * (1 - t) + bVal * t;
    })
  );
}

// Creates an empty 8x8 grid of zeros
export function createEmptyGrid(rows = 8, cols = 8): Grid {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

// Generate random uniform noise grid
export function createNoiseGrid(rows = 8, cols = 8): Grid {
  return Array.from({ length: rows }, () => 
    Array.from({ length: cols }, () => Math.random())
  );
}
