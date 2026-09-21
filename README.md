# Find Your Way — Romantic Hand-Tracking Experience

A cinematic React + TypeScript + Three.js + MediaPipe Hands experience.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL Vite prints (usually http://localhost:5173).

### Camera permissions
Webcam access works on `localhost` during development. For deployment, use HTTPS.

### What is included
- MediaPipe hand landmark tracking with GPU delegate
- Index fingertip tracking (landmark 8)
- Smoothed, eased butterfly following
- Procedural Three.js butterfly with animated wings
- Atmospheric particles and motion trail
- Cinematic appearance/disappearance sequence
- Romantic reveal after the hand leaves
- Optional user-triggered chime audio
- Responsive mobile/desktop layout
- Graceful camera-denied state

The MediaPipe WASM runtime and model are loaded from public CDNs, so the first launch requires internet access.
