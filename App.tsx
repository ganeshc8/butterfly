import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import "./styles.css";

type Point = { x: number; y: number };
type Phase = "intro" | "tracking" | "message" | "denied";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function mapVideoPoint(p: Point, video: HTMLVideoElement) {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const W = window.innerWidth;
  const H = window.innerHeight;
  const scale = Math.max(W / vw, H / vh);
  const rw = vw * scale;
  const rh = vh * scale;
  const ox = (rw - W) / 2;
  const oy = (rh - H) / 2;
  // Camera is mirrored visually, so mirror the x coordinate too.
  return { x: (1 - p.x) * rw - ox, y: p.y * rh - oy };
}

function Butterfly({ x, y, visible, landing }: { x: number; y: number; visible: boolean; landing: boolean }) {
  return (
    <div
      className={`butterfly ${visible ? "is-visible" : ""} ${landing ? "is-landing" : ""}`}
      style={{ transform: `translate3d(${x}px, ${y}px, 0)` }}
      aria-hidden="true"
    >
      <div className="butterfly-aura" />
      <div className="butterfly-spark s1" /><div className="butterfly-spark s2" /><div className="butterfly-spark s3" />
      <svg viewBox="0 0 240 180" className="butterfly-art">
        <defs>
          <linearGradient id="wingA" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#ffe7ff" stopOpacity=".98" />
            <stop offset=".25" stopColor="#caa4ff" />
            <stop offset=".58" stopColor="#7e5cff" />
            <stop offset="1" stopColor="#ff62c7" />
          </linearGradient>
          <linearGradient id="wingB" x1="1" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#8ef1ff" stopOpacity=".95" />
            <stop offset=".42" stopColor="#8066ff" />
            <stop offset="1" stopColor="#ff65d2" />
          </linearGradient>
          <radialGradient id="body" cx="50%" cy="40%">
            <stop offset="0" stopColor="#ffd8ff" />
            <stop offset=".4" stopColor="#7d55a6" />
            <stop offset="1" stopColor="#160d26" />
          </radialGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <g className="wing-left">
          <path d="M120 87 C91 74 53 19 27 28 C4 36 14 72 36 94 C57 116 92 111 120 97Z" fill="url(#wingA)" filter="url(#glow)"/>
          <path d="M116 89 C88 103 47 137 27 125 C8 113 20 91 42 80 C66 68 95 76 116 89Z" fill="url(#wingB)" opacity=".9" filter="url(#glow)"/>
          <path d="M113 87 C81 72 52 42 29 35 M112 91 C74 96 47 103 25 119" className="vein"/>
          <circle cx="43" cy="55" r="5" className="wing-dot"/><circle cx="65" cy="74" r="3" className="wing-dot"/><circle cx="52" cy="105" r="4" className="wing-dot"/>
        </g>
        <g className="wing-right">
          <path d="M120 87 C149 74 187 19 213 28 C236 36 226 72 204 94 C183 116 148 111 120 97Z" fill="url(#wingA)" filter="url(#glow)"/>
          <path d="M124 89 C152 103 193 137 213 125 C232 113 220 91 198 80 C174 68 145 76 124 89Z" fill="url(#wingB)" opacity=".9" filter="url(#glow)"/>
          <path d="M127 87 C159 72 188 42 211 35 M128 91 C166 96 193 103 215 119" className="vein"/>
          <circle cx="197" cy="55" r="5" className="wing-dot"/><circle cx="175" cy="74" r="3" className="wing-dot"/><circle cx="188" cy="105" r="4" className="wing-dot"/>
        </g>
        <ellipse cx="120" cy="91" rx="8" ry="34" fill="url(#body)"/>
        <path d="M116 59 Q102 41 95 40 M124 59 Q138 41 145 40" stroke="#d7b6ff" strokeWidth="2" fill="none" strokeLinecap="round"/>
      </svg>
    </div>
  );
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const particleRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef(0);
  const state = useRef({
    target: { x: innerWidth / 2, y: innerHeight / 2 } as Point,
    smooth: { x: innerWidth / 2, y: innerHeight / 2 } as Point,
    lastTip: { x: .5, y: .5 } as Point,
    handSeen: false,
    indexUp: false,
    stillSince: 0,
    lastSeen: 0,
    firstDetected: false,
    departing: false,
  });

  const [phase, setPhase] = useState<Phase>("intro");
  const [hint, setHint] = useState("Allow camera access to begin");
  const [butterfly, setButterfly] = useState({ x: innerWidth / 2, y: innerHeight / 2, visible: false, landing: false });
  const [cameraReady, setCameraReady] = useState(false);
  const [sound, setSound] = useState(false);
  const soundRef = useRef(false);

  const chime = (freq: number) => {
    if (!soundRef.current) return;
    const AC = window.AudioContext || (window as any).webkitAudioContext;
    const ctx = new AC();
    const o = ctx.createOscillator(); const g = ctx.createGain();
    o.type = "sine"; o.frequency.setValueAtTime(freq, ctx.currentTime); o.frequency.exponentialRampToValueAtTime(freq * 1.45, ctx.currentTime + .38);
    g.gain.setValueAtTime(.0001, ctx.currentTime); g.gain.exponentialRampToValueAtTime(.05, ctx.currentTime + .03); g.gain.exponentialRampToValueAtTime(.0001, ctx.currentTime + .55);
    o.connect(g).connect(ctx.destination); o.start(); o.stop(ctx.currentTime + .58);
  };

  const startCamera = async () => {
    try {
      setHint("Waking the little magic…");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      setCameraReady(true);
      setPhase("tracking");
      setHint("Raise your index finger");

      const vision = await FilesetResolver.forVisionTasks(WASM);
      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
        runningMode: "VIDEO", numHands: 1,
        minHandDetectionConfidence: .55, minHandPresenceConfidence: .55, minTrackingConfidence: .5
      });
    } catch (e) {
      console.error(e);
      setPhase("denied");
      setHint("Camera access is needed for the magic to begin.");
    }
  };

  useEffect(() => {
    void startCamera();
    const resize = () => setButterfly(b => ({ ...b, x: innerWidth / 2, y: innerHeight / 2 }));
    addEventListener("resize", resize);
    return () => {
      removeEventListener("resize", resize);
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const canvas = particleRef.current!;
    const ctx = canvas.getContext("2d")!;
    const resize = () => { canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; canvas.style.width = innerWidth + "px"; canvas.style.height = innerHeight + "px"; ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0); };
    resize(); addEventListener("resize", resize);

    const particles = Array.from({ length: 85 }, () => ({ x: Math.random()*innerWidth, y: Math.random()*innerHeight, vx:(Math.random()-.5)*.18, vy:(Math.random()-.5)*.18, r:Math.random()*1.5+.25, a:Math.random()*.5+.1 }));
    const animate = (now: number) => {
      ctx.clearRect(0,0,innerWidth,innerHeight);
      const s = state.current;
      particles.forEach((p,i) => {
        const dx=s.smooth.x-p.x, dy=s.smooth.y-p.y, d=Math.hypot(dx,dy)||1;
        if (d < 180) { p.vx += dx/d*.002; p.vy += dy/d*.002; }
        p.x += p.vx + Math.sin(now*.0007+i)*.015; p.y += p.vy;
        if (p.x<0)p.x=innerWidth;if(p.x>innerWidth)p.x=0;if(p.y<0)p.y=innerHeight;if(p.y>innerHeight)p.y=0;
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fillStyle=`rgba(226,190,255,${p.a})`; ctx.fill();
      });
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => { cancelAnimationFrame(rafRef.current); removeEventListener("resize", resize); };
  }, []);

  useEffect(() => {
    let frame = 0;
    const tick = (now: number) => {
      const s = state.current;
      const video = videoRef.current;
      const lm = landmarkerRef.current;
      if (video && lm && video.readyState >= 2) {
        const result = lm.detectForVideo(video, now);
        const hand = result.landmarks?.[0];
        if (hand) {
          const tip = hand[8], pip = hand[6], mcp = hand[5];
          const indexUp = tip.y < pip.y - .025 && pip.y < mcp.y + .06;
          const mapped = mapVideoPoint({ x: tip.x, y: tip.y }, video);
          s.target = mapped; s.handSeen = true; s.lastSeen = now; s.indexUp = indexUp;
          if (Math.hypot(tip.x-s.lastTip.x, tip.y-s.lastTip.y) > .012) s.stillSince = now;
          s.lastTip = { x: tip.x, y: tip.y };
          if (indexUp && !s.firstDetected) { s.firstDetected = true; chime(640); }
        } else {
          s.handSeen = false; s.indexUp = false;
        }
      }

      const active = s.handSeen && s.indexUp && phase !== "message";
      const smooth = active ? .16 : .075;
      s.smooth.x = lerp(s.smooth.x, s.target.x, smooth);
      s.smooth.y = lerp(s.smooth.y, s.target.y, smooth);
      const idle = now - s.stillSince;
      const landing = active && idle > 1700;

      if (active) {
        if (!butterfly.visible) chime(640);
        setButterfly({ x:s.smooth.x, y:s.smooth.y-62, visible:true, landing });
        setHint(landing ? "She's landed…" : "Hold still… let her come closer");
      } else if (s.handSeen && !s.indexUp) {
        setButterfly(b => ({ ...b, visible:false, landing:false }));
        setHint("Raise just your index finger");
      } else if (!s.handSeen && s.firstDetected && now-s.lastSeen > 1500 && phase !== "message") {
        setButterfly(b => ({ ...b, visible:false, landing:false }));
        if (!s.departing) { s.departing=true; setTimeout(() => { setPhase("message"); setHint(""); }, 700); }
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [phase, butterfly.visible]);

  const reset = () => {
    state.current.firstDetected=false; state.current.departing=false; state.current.stillSince=performance.now();
    setPhase("tracking"); setHint("Raise your index finger");
  };

  return (
    <main className={`experience ${phase}`}>
      <video ref={videoRef} className="webcam" autoPlay muted playsInline />
      <div className="cinematic-color" />
      <canvas ref={particleRef} className="particles" />
      <div className="scanline" />
      <div className="topbar">
        <div className="brand"><span className="heart">♥</span><span>A LITTLE MAGIC<br/>JUST FOR YOU</span></div>
        <div className="controls">
          <button onClick={() => { soundRef.current=!sound; setSound(soundRef.current); }} className="tiny-btn">{sound ? "♪ Sound on" : "♪ Sound"}</button>
          <div className={`camera-status ${cameraReady ? "ready" : ""}`}><i /> {cameraReady ? "Camera active" : "Connecting"}</div>
        </div>
      </div>

      <div className="focus-frame"><i/><i/><i/><i/></div>

      {phase === "intro" || phase === "tracking" ? (
        <section className="hero-copy">
          <div className="eyebrow">an interactive little surprise</div>
          <h1>Show me<br/><em>your hand…</em></h1>
          <p>{hint}</p>
          <div className="instruction"><span className="finger">☝</span><span><b>Raise your index finger</b><small>The butterfly will find it.</small></span></div>
        </section>
      ) : null}

      {phase === "denied" && <section className="hero-copy denied"><div className="eyebrow">one tiny thing first</div><h1>Let me see<br/><em>you…</em></h1><p>{hint}</p><button onClick={startCamera}>Allow camera</button></section>}

      {phase === "message" && <section className="message"><div className="eyebrow">and then, somehow</div><h2>Some things just<br/><em>find their way to you.</em> <span>♥</span></h2><p>Just like you found your way into my life.</p><button onClick={reset}>Try again</button></section>}

      <Butterfly x={butterfly.x} y={butterfly.y} visible={butterfly.visible} landing={butterfly.landing} />
      <div className="bottom-note">Move slowly · she follows your fingertip</div>
      <div className="vignette" />
    </main>
  );
}
