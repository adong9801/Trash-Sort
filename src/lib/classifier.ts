import type { Prediction } from './items';

// The model currently runs only in the web version (see classifier.web.ts). Running it inside
// the native app needs onnxruntime-react-native and a development build, since Expo Go can't load it.

export function loadModel() {
  return Promise.resolve();
}

export async function classify(_uri: string): Promise<Prediction> {
  throw new Error('For now, the trash model only works in the web version. Open the app’s link in your phone’s browser.');
}
