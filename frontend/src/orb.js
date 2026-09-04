/* The voice orb.
 *
 * A call has no visible surface of its own: audio arrives, audio leaves, and
 * between the two there is nothing on screen to say whether the assistant is
 * listening, working or talking. On a phone that ambiguity is filled in by
 * hearing the other person breathe. Here it has to be drawn.
 *
 * So the orb is driven by the actual audio rather than by a timer: an
 * `AnalyserNode` on the microphone track while the caller speaks, and one on
 * the agent's subscribed track while it answers. A purely decorative animation
 * would look the same whether or not the audio path was working - and the audio
 * path failing silently, with the room connected and no sound, is precisely how
 * WebRTC through Docker goes wrong. If the orb moves, media is flowing.
 *
 * Colours are read from the stylesheet's custom properties, so the orb follows
 * the light/dark theme without a second palette to keep in sync.
 */

(function () {
  const TAU = Math.PI * 2;

  // Ring harmonics. Three incommensurate frequencies read as organic; one reads
  // as a pulsing circle, and four or more as noise at this size.
  const HARMONICS = [
    { k: 2, amp: 0.085, speed: 0.7 },
    { k: 3, amp: 0.05, speed: -0.45 },
    { k: 5, amp: 0.028, speed: 0.9 },
  ];

  const STATES = {
    idle: { hue: "--ink-3", gain: 0.0, label: "" },
    listening: { hue: "--ok", gain: 1.0, label: "Listening" },
    thinking: { hue: "--warn", gain: 0.0, label: "Working" },
    speaking: { hue: "--accent", gain: 1.0, label: "Speaking" },
  };

  function cssColor(name, fallback) {
    const value = getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
    return value || fallback;
  }

  /* Parse a #rgb/#rrggbb custom property into components, so the orb can draw
     the same colour at several alphas without needing a second token for each. */
  function rgb(hex) {
    let h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return Number.isNaN(n)
      ? [138, 90, 20]
      : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  class Orb {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.state = "idle";
      this.level = 0;      // smoothed amplitude, 0..1
      this.phase = 0;
      this.raf = null;
      this.audio = null;   // AudioContext, created on first attach
      this.sources = new Map();
      this.analysers = { mic: null, agent: null };
      this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      this._resize();
      window.addEventListener("resize", () => this._resize());
    }

    _resize() {
      // Draw at device resolution: a blurry orb on a retina screen reads as a
      // rendering bug rather than as a soft edge.
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const size = this.canvas.clientWidth || 240;
      this.canvas.width = size * dpr;
      this.canvas.height = size * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.size = size;
    }

    /** Feed a live audio track into the orb.
     *
     * @param {MediaStreamTrack} track  mic or agent audio
     * @param {"mic"|"agent"} which     which half of the call it is
     */
    attach(track, which) {
      if (!track) return;
      try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return;
        if (!this.audio) this.audio = new Ctx();
        // Autoplay policy suspends a context created before a gesture; the mic
        // button is that gesture, but resume() is still needed explicitly.
        if (this.audio.state === "suspended") this.audio.resume();

        const stream = new MediaStream([track]);
        const source = this.audio.createMediaStreamSource(stream);
        const analyser = this.audio.createAnalyser();
        analyser.fftSize = 512;
        analyser.smoothingTimeConstant = 0.75;
        source.connect(analyser);
        // Deliberately not connected to the destination: the agent's audio is
        // already played by the <audio> element LiveKit attaches, and routing
        // it here as well would double it. The mic must never be played back.

        this.sources.set(which, source);
        this.analysers[which] = { node: analyser, buf: new Uint8Array(analyser.fftSize) };
      } catch (err) {
        // A missing analyser costs the animation its reactivity, nothing more.
        // The call itself must not fail because a visualiser could not start.
        console.warn("orb: could not analyse", which, err);
      }
    }

    detach() {
      for (const source of this.sources.values()) {
        try {
          source.disconnect();
        } catch {
          /* already gone */
        }
      }
      this.sources.clear();
      this.analysers = { mic: null, agent: null };
      if (this.audio) {
        this.audio.close().catch(() => {});
        this.audio = null;
      }
    }

    setState(state) {
      if (STATES[state]) this.state = state;
    }

    /** Root-mean-square of the current window, as 0..1. */
    _amplitude(which) {
      const a = this.analysers[which];
      if (!a) return 0;
      a.node.getByteTimeDomainData(a.buf);
      let sum = 0;
      for (let i = 0; i < a.buf.length; i++) {
        const v = (a.buf[i] - 128) / 128;
        sum += v * v;
      }
      // Speech RMS sits around 0.05-0.2; scaling by 4 puts normal talking near
      // the top of the range without clipping every syllable.
      return Math.min(1, Math.sqrt(sum / a.buf.length) * 4);
    }

    start() {
      if (this.raf) return;
      const frame = () => {
        this._draw();
        this.raf = requestAnimationFrame(frame);
      };
      this.raf = requestAnimationFrame(frame);
    }

    stop() {
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = null;
      this.detach();
      this.state = "idle";
      this.level = 0;
      this._draw();
    }

    _draw() {
      const { ctx, size } = this;
      const spec = STATES[this.state] || STATES.idle;
      const mid = size / 2;

      ctx.clearRect(0, 0, size, size);

      // Which half of the call drives the motion depends on who is talking.
      const raw =
        spec.gain === 0
          ? 0
          : this.state === "speaking"
            ? this._amplitude("agent")
            : this._amplitude("mic");

      // Asymmetric smoothing: rise quickly so a syllable registers, fall slowly
      // so the orb does not strobe between words.
      const target = raw * spec.gain;
      this.level += (target - this.level) * (target > this.level ? 0.35 : 0.08);

      if (!this.reduced) this.phase += 0.016;

      const [r, g, b] = rgb(cssColor(spec.hue, "#8a5a14"));
      // A resting breath so an idle or thinking orb is alive but not busy.
      const breath = this.reduced ? 0 : Math.sin(this.phase * 1.6) * 0.015;
      const base = size * (0.3 + breath) + this.level * size * 0.075;

      // Outer glow. The outer stop is the inscribed circle, not a multiple of
      // the orb: anything larger is still partly opaque where it meets the
      // canvas edge, and the fill leaves a visible square halo around the orb.
      const glow = ctx.createRadialGradient(
        mid, mid, base * 0.4, mid, mid, Math.min(base * 1.9, mid)
      );
      glow.addColorStop(0, `rgba(${r},${g},${b},${0.22 + this.level * 0.3})`);
      glow.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, size, size);

      // Deformed rings. Each is the same blob at a different scale and phase,
      // which is what gives the surface its depth at three drawn paths.
      for (let ring = 0; ring < 3; ring++) {
        const scale = 1 - ring * 0.11;
        const alpha = 0.1 + ring * 0.09 + this.level * 0.18;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const theta = (i / 120) * TAU;
          let radius = base * scale;
          for (const h of HARMONICS) {
            // Near-circular when quiet: the deformation is what a voice does
            // to the shape, so at rest there should be almost none of it.
            radius +=
              base *
              h.amp *
              scale *
              (0.25 + this.level * 0.9) *
              Math.sin(h.k * theta + this.phase * h.speed + ring * 0.6);
          }
          const x = mid + Math.cos(theta) * radius;
          const y = mid + Math.sin(theta) * radius;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.fill();
      }

      // Core.
      const core = ctx.createRadialGradient(
        mid, mid - base * 0.12, base * 0.04, mid, mid, base * 0.82
      );
      core.addColorStop(0, `rgba(255,255,255,${0.72 + this.level * 0.25})`);
      core.addColorStop(0.45, `rgba(${r},${g},${b},0.92)`);
      core.addColorStop(1, `rgba(${r},${g},${b},0.35)`);
      ctx.beginPath();
      ctx.arc(mid, mid, base * 0.82, 0, TAU);
      ctx.fillStyle = core;
      ctx.fill();

      // The working state has no amplitude to show, so it gets a sweep instead
      // of a pulse - motion that reads as "busy" rather than as "quiet".
      if (this.state === "thinking" && !this.reduced) {
        ctx.beginPath();
        ctx.arc(mid, mid, base * 1.22, this.phase * 2, this.phase * 2 + 1.1);
        ctx.strokeStyle = `rgba(${r},${g},${b},0.8)`;
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.stroke();
      }
    }
  }

  window.OmniCareOrb = Orb;
})();
