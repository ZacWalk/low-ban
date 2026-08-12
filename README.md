# low-ban

Low bandwidth video experiment: a face-prioritised transform codec and a face-weighted patch
autoencoder, side by side on the same webcam frame. Plain C++17 on Win32.

During a hackathon at Microsoft I tried to invent a super low-bandwidth video codec optimised
for 2G networks. It got a lot of attention at the event, though it stayed an experimental
prototype. The core idea was to use facial landmarking to preserve the user's expressions, so
that non-verbal communication survives even when the video frame is heavily compressed and
blurry.

![screenshot](https://github.com/ZacWalk/low-ban/assets/181544/01160e1c-c3ce-40d4-895e-375e01c712d1)

Three panes show the raw feed, a traditional 8x8 DCT codec, and a learned 20x20 patch
autoencoder. Faces are detected and landmarked first, producing one importance map that both
codecs use to spend bits on brows, eyes, nose and mouth. Each codec runs its own 5 Hz packet
stream and its own compressed landmark stream, so neither gets geometry for free, and both
panes are drawn only from bytes that were actually serialized.

The traditional stream sends the complete scene once at startup, then delta packets capped at
1,200 bytes. A block is coded either against what the decoder already holds or against a
motion-compensated window of the previous decoded frame, whichever costs less. The neural
stream sends four bytes per patch - an exact brightness byte plus a three-unit latent - with
1,500-byte delta packets and a full keyframe every three seconds. Both decoders hold their
last reconstruction when nothing changed.

Both schedule their packets the same way: distortion removed per byte, with face importance as
a bounded multiplier and a deadline that stops any part of the picture being starved. That
last part matters more than it sounds - ranking by importance alone leaves copies of your head
behind when you move.

## Current results

| Approach | Complete keyframe | Mean PSNR | 5 Hz two-face reference |
| -------- | ----------------: | --------: | --------------------: |
| Traditional DCT | 322.4 kbit startup | 36.84 dB | 54.4 kbit/s after startup |
| Neural patches | 27.4 kbit | 25.89 dB | 71.1 kbit/s reference |

These are serialized in-memory codec payloads, not network throughput. The traditional
startup keyframe is a one-time cost and is deliberately not hidden inside its steady-state
figure. The rate references assume landmark deltas stay within 80 bytes per moving face;
larger-motion escapes can exceed them. Transport framing, audio, loss recovery and neural
model updates are excluded.

```
.\dd.ps1 build
.\dd.ps1 run        # Release
.\dd.ps1 test       # headless checks, no camera needed
.\dd.ps1 evaluate   # score samples/*.png offline
.\dd.ps1 clean
```

Drop `shape_predictor_68_face_landmarks.dat` into `exe/` for the landmark overlay. The
application runs without it.

The online autoencoder is still experimental: encoder and decoder share its changing weights
inside this process. A remote implementation needs a frozen model at both endpoints or an
explicit model-update stream whose bytes are included in the rate.

**See [docs/design.md](docs/design.md) for the architecture, the algorithms, the model file
format, the measured rates and the known limitations, and [docs/todo.md](docs/todo.md) for
what is not done yet.**

The face detector's network design and weights come from
[libfacedetection](https://github.com/ShiqiYu/libfacedetection) by Shiqi Yu, under a 3-clause
BSD licence. The landmark model file is the one published with dlib's implementation of
Kazemi and Sullivan's regression-tree face alignment; both the reader and the predictor here
were written from scratch.
