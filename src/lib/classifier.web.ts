import type { InferenceSession } from 'onnxruntime-web';

import { type Bin, ITEMS, type ItemId, type Prediction } from './items';
import labels from './model-labels.json';

// The model runs right in the browser, so photos never leave the phone.
// The runtime and model are served from public/, and the browser caches them after the first load.

declare global {
  interface Window {
    ort?: typeof import('onnxruntime-web');
  }
}

// Must match training/train.py.
const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session: Promise<InferenceSession> | null = null;

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Couldn't load ${src}`));
    document.head.appendChild(script);
  });
}

export function loadModel() {
  session ??= (async () => {
    if (!window.ort) await loadScript('/ort/ort.wasm.min.js');
    const ort = window.ort!;
    ort.env.wasm.wasmPaths = '/ort/';
    // Multiple threads need special server headers; one thread is fast enough for one photo.
    ort.env.wasm.numThreads = 1;
    return ort.InferenceSession.create('/model/trashsort.onnx');
  })().catch((error) => {
    session = null;
    throw error;
  });
  return session;
}

// Center-crop to a square and shrink to 224x224, laid out the way the model expects.
async function toPixels(uri: string) {
  const img = new Image();
  img.src = uri;
  await img.decode();
  const side = Math.min(img.naturalWidth, img.naturalHeight);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, (img.naturalWidth - side) / 2, (img.naturalHeight - side) / 2, side, side, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

  const pixels = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    for (let c = 0; c < 3; c++) {
      pixels[c * SIZE * SIZE + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return pixels;
}

export async function classify(uri: string): Promise<Prediction> {
  const [model, pixels] = await Promise.all([loadModel(), toPixels(uri)]);
  const ort = window.ort!;
  const output = await model.run({ image: new ort.Tensor('float32', pixels, [1, 3, SIZE, SIZE]) });
  const logits = output.logits.data as Float32Array;

  const max = Math.max(...logits);
  const exps = Array.from(logits, (x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);

  // Several item types share a bin (cans, bottles, and paper are all recycling), so add up
  // each bin's share before picking. Then name the likeliest item within that bin.
  const binTotals: Partial<Record<Bin, number>> = {};
  (labels as ItemId[]).forEach((item, i) => {
    const bin = ITEMS[item].bin;
    binTotals[bin] = (binTotals[bin] ?? 0) + probs[i];
  });
  const [bin, confidence] = Object.entries(binTotals).sort((a, b) => b[1] - a[1])[0] as [Bin, number];
  const item = (labels as ItemId[])
    .map((id, i) => ({ id, p: probs[i] }))
    .filter(({ id }) => ITEMS[id].bin === bin)
    .sort((a, b) => b.p - a.p)[0].id;

  return { item, bin, confidence };
}
