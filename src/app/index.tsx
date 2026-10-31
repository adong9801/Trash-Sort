import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useColorScheme,
  View,
} from 'react-native';

import { classify, loadModel } from '@/lib/classifier';
import { type Bin, ITEMS, type Prediction } from '@/lib/items';

const BINS: Record<Bin, { label: string; emoji: string; color: string }> = {
  recycling: { label: 'Recycling', emoji: '♻️', color: '#1D6FD8' },
  compost: { label: 'Compost', emoji: '🌱', color: '#2E8540' },
  trash: { label: 'Trash', emoji: '🗑️', color: '#4B4F56' },
  special: { label: 'Special drop-off', emoji: '⚠️', color: '#C25E00' },
};

const PALETTES = {
  light: {
    background: '#F4F5F2',
    card: '#FFFFFF',
    text: '#16181D',
    muted: '#5F6470',
    border: '#DADDE3',
    error: '#B3261E',
  },
  dark: {
    background: '#0E0F11',
    card: '#1B1D21',
    text: '#F2F3F5',
    muted: '#A3A8B3',
    border: '#2E3138',
    error: '#F2B8B5',
  },
};

type Palette = (typeof PALETTES)['light'];

type State =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'done'; prediction: Prediction }
  | { status: 'error'; message: string };

export default function HomeScreen() {
  const colors = PALETTES[useColorScheme() === 'dark' ? 'dark' : 'light'];
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [state, setState] = useState<State>({ status: 'idle' });
  const loading = state.status === 'loading';

  // Start downloading the model right away so it's ready by the time the first photo is taken.
  useEffect(() => {
    loadModel().catch(() => {});
  }, []);

  async function pickPhoto(source: 'camera' | 'library') {
    // Browsers handle camera access themselves, and awaiting here would block the camera from opening.
    if (source === 'camera' && Platform.OS !== 'web') {
      const permission = await ImagePicker.requestCameraPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Camera access needed', 'Allow camera access in Settings to snap your item.', [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Open Settings', onPress: () => Linking.openSettings() },
        ]);
        return;
      }
    }

    const options: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.8 };
    const picked =
      source === 'camera'
        ? await ImagePicker.launchCameraAsync(options)
        : await ImagePicker.launchImageLibraryAsync(options);
    if (picked.canceled) return;

    const uri = picked.assets[0].uri;
    setPhotoUri(uri);
    setState({ status: 'loading' });
    try {
      setState({ status: 'done', prediction: await classify(uri) });
    } catch (error) {
      setState({
        status: 'error',
        message: error instanceof Error ? error.message : 'Something went wrong. Try again.',
      });
    }
  }

  return (
    <ScrollView
      style={{ backgroundColor: colors.background }}
      contentContainerStyle={styles.content}
      contentInsetAdjustmentBehavior="automatic">
      <Pressable
        style={({ pressed }) => [styles.button, styles.primary, (pressed || loading) && styles.dimmed]}
        onPress={() => pickPhoto('camera')}
        disabled={loading}>
        <Text style={styles.primaryText}>📸 Take a photo</Text>
      </Pressable>
      <Pressable
        style={({ pressed }) => [
          styles.button,
          { borderColor: colors.border, backgroundColor: colors.card },
          (pressed || loading) && styles.dimmed,
        ]}
        onPress={() => pickPhoto('library')}
        disabled={loading}>
        <Text style={[styles.secondaryText, { color: colors.text }]}>Choose from library</Text>
      </Pressable>

      {state.status === 'loading' && (
        <View style={styles.loadingRow}>
          <ActivityIndicator color={colors.muted} />
          <Text style={{ color: colors.muted }}>Checking…</Text>
        </View>
      )}
      {state.status === 'error' && (
        <Text style={[styles.errorText, { color: colors.error }]}>{state.message}</Text>
      )}
      {state.status === 'done' && <ResultCard prediction={state.prediction} colors={colors} />}

      {photoUri ? (
        <Image source={{ uri: photoUri }} style={styles.photo} contentFit="cover" />
      ) : (
        <View style={[styles.placeholder, { borderColor: colors.border }]}>
          <Text style={[styles.placeholderText, { color: colors.muted }]}>
            Snap something you’re about to throw away and I’ll tell you which bin it goes in. Works
            best with one item, up close.
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

function ResultCard({ prediction, colors }: { prediction: Prediction; colors: Palette }) {
  const bin = BINS[prediction.bin];
  const item = ITEMS[prediction.item];
  return (
    <View style={[styles.card, { backgroundColor: colors.card, borderColor: bin.color }]}>
      <View style={[styles.banner, { backgroundColor: bin.color }]}>
        <Text style={styles.bannerEmoji}>{bin.emoji}</Text>
        <Text style={styles.bannerLabel}>{bin.label}</Text>
      </View>
      <View style={styles.cardBody}>
        <Text style={[styles.itemName, { color: colors.text }]}>Looks like: {item.name}</Text>
        {item.tip ? <Text style={[styles.bodyText, { color: colors.text }]}>💡 {item.tip}</Text> : null}
        {prediction.confidence < 0.6 ? (
          <Text style={[styles.note, { color: colors.muted }]}>
            Not totally sure about this one. Try a closer photo with just the item in it.
          </Text>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  content: {
    padding: 16,
    gap: 12,
    paddingBottom: 48,
  },
  button: {
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'transparent',
  },
  primary: {
    backgroundColor: '#2E8540',
  },
  primaryText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  secondaryText: {
    fontSize: 16,
    fontWeight: '500',
  },
  dimmed: {
    opacity: 0.6,
  },
  // Square, since the model looks at the center square of the photo.
  photo: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 16,
  },
  placeholder: {
    borderWidth: 2,
    borderStyle: 'dashed',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 22,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 8,
  },
  errorText: {
    fontSize: 16,
    textAlign: 'center',
  },
  card: {
    borderWidth: 2,
    borderRadius: 16,
    overflow: 'hidden',
  },
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 14,
    paddingHorizontal: 16,
  },
  bannerEmoji: {
    fontSize: 28,
  },
  bannerLabel: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '700',
  },
  cardBody: {
    padding: 16,
    gap: 8,
  },
  itemName: {
    fontSize: 18,
    fontWeight: '600',
  },
  bodyText: {
    fontSize: 16,
    lineHeight: 22,
  },
  note: {
    fontSize: 14,
    lineHeight: 20,
  },
});
