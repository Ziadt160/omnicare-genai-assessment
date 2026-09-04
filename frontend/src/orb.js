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

  /** Mix a colour toward white. Used for the lit part of the body so the core
   *  stays the orb's own hue instead of washing out to grey. */
  function lighten(r, g, b, amount) {
    const mix = (c) => Math.round(c + (255 - c) * amount);
    return `${mix(r)},${mix(g)},${mix(b)}`;
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

    /** Stop drawing, keep the audio graph.
     *
     * Going back to the chat does not end the call, so the analysers must
     * survive it - tearing them down and rebuilding would need a second
     * AudioContext and, on the agent side, a track that has already been
     * subscribed and will not fire again. */
    pause() {
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = null;
    }

    stop() {
      this.pause();
      this.detach();
      this.state = "idle";
      this.level = 0;
      this._draw();
    }

    /** Re-read the CSS size. The canvas is laid out by the stylesheet and sized
     *  in `vmin`, so its box is 0 while the panel is hidden; measuring then and
     *  never again renders the orb at the fallback size. */
    resize() {
      this._resize();
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

      // A helper for the deformed outline, so the body and the halo are the
      // same shape rather than a blob inside a circle.
      const blob = (radius, phaseShift, strength) => {
        ctx.beginPath();
        for (let i = 0; i <= 140; i++) {
          const theta = (i / 140) * TAU;
          let rad = radius;
          for (const h of HARMONICS) {
            // Near-circular when quiet: the deformation is what a voice does to
            // the shape, so at rest there should be almost none of it.
            rad +=
              radius *
              h.amp *
              strength *
              Math.sin(h.k * theta + this.phase * h.speed + phaseShift);
          }
          const x = mid + Math.cos(theta) * rad;
          const y = mid + Math.sin(theta) * rad;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
      };

      const strength = 0.2 + this.level * 0.45;

      // Outer bloom. The outer stop is the inscribed circle, not a multiple of
      // the orb: anything larger is still partly opaque where it meets the
      // canvas edge, and the fill leaves a visible square halo around it.
      const glow = ctx.createRadialGradient(
        mid, mid, base * 0.55, mid, mid, Math.min(base * 2.0, mid)
      );
      glow.addColorStop(0, `rgba(${r},${g},${b},${0.2 + this.level * 0.26})`);
      glow.addColorStop(0.55, `rgba(${r},${g},${b},${0.07 + this.level * 0.1})`);
      glow.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, size, size);

      // Two faint outriders, slightly larger and out of phase, to suggest the
      // surface moving. Each is filled with its own fading gradient rather than
      // a flat alpha: a flat fill has a hard rim, and two hard rims just
      // outside the body read as concentric outlines drawn around the orb.
      for (let i = 1; i <= 2; i++) {
        const radius = base * (1 + i * 0.06);
        const halo = ctx.createRadialGradient(
          mid, mid, radius * 0.55, mid, mid, radius * 1.02
        );
        const a = 0.07 + this.level * 0.06;
        halo.addColorStop(0, `rgba(${r},${g},${b},${a})`);
        halo.addColorStop(0.7, `rgba(${r},${g},${b},${a * 0.6})`);
        halo.addColorStop(1, `rgba(${r},${g},${b},0)`);
        blob(radius, i * 1.9, strength);
        ctx.fillStyle = halo;
        ctx.fill();
      }

      // The body: one fill, feathering to nothing at the rim. Drawn as a
      // gradient rather than a flat shape with a highlight on top - a hard edge
      // with a white spot reads as a billiard ball, which is a solid object,
      // and this is meant to be a voice.
      const body = ctx.createRadialGradient(
        mid, mid - base * 0.2, base * 0.05, mid, mid, base * 1.02
      );
      body.addColorStop(0, `rgba(255,255,255,${0.85 + this.level * 0.12})`);
      body.addColorStop(0.18, `rgba(${lighten(r, g, b, 0.55)},${0.95})`);
      body.addColorStop(0.52, `rgba(${r},${g},${b},0.92)`);
      body.addColorStop(0.82, `rgba(${r},${g},${b},0.5)`);
      body.addColorStop(1, `rgba(${r},${g},${b},0)`);
      blob(base, 0, strength);
      ctx.fillStyle = body;
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
