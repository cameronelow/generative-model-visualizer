/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';
import katex from 'katex';

interface LatexRendererProps {
  math: string;
  block?: boolean;
}

export default function LatexRenderer({ math, block = false }: LatexRendererProps) {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        katex.render(math, containerRef.current, {
          displayMode: block,
          throwOnError: false,
        });
      } catch (err) {
        console.error('KaTeX rendering error:', err);
        containerRef.current.textContent = math;
      }
    }
  }, [math, block]);

  return (
    <span 
      id="latex-rendered-equation"
      ref={containerRef} 
      className={`inline-block select-all max-w-full overflow-x-auto ${
        block ? 'w-full py-2 text-indigo-700 font-medium font-sans' : ''
      }`} 
    />
  );
}
