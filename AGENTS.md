# AGENTS

`low-ban` is an experimental Windows application for low bandwidth video communication. It
compares low-ban - a face-prioritised transform codec whose receiver warps the face onto
transmitted landmarks - with an H.264 baseline given the same bitrate and the same face boxes,
on the same webcam frame. Face detection and 68-point landmarking decide where the bits go,
predict the face's motion and measure whether expression survived. Both received pictures come
from independently serialized wires: low-ban's 15 Hz geometry and 5 Hz pictures, and H.264 at
15 frames a second.

**Read [docs/design.md](docs/design.md) first.** It is the single source of truth for the
architecture, the module boundaries, the algorithms and the model file format. Keep it
current when you change behaviour, and do not duplicate its content in other markdown files.
[docs/todo.md](docs/todo.md) records known gaps and further work; move an item out of it when
you close it.

## Essentials

- C++17, Win32, Media Foundation. Builds `Debug|x64` and `Release|x64` only.
- No Windows AI APIs and no third-party codec libraries; the H.264 baseline uses the encoder
  and decoder that ship with Windows. The only external work is the face detector's network
  and weights, adapted from libfacedetection; keep its BSD notice intact in
  `face.cpp`/`face-model.cpp`.
- Build, launch and test through `dd.ps1` (`build`, `launch`, `test`, `evaluate`, `clean`).
- The [dd](https://github.com/ZacWalk/dd) build driver is vendored unmodified: `dd.ps1`
  and `.dd/`, with hashes in [docs/dd-upstream.json](docs/dd-upstream.json). Project
  settings live in `dd.psd1`, dependency pins in `cmake/dd-dependencies.json`, and
  `.\dd.ps1 commands --json` lists this repository's project commands. dd's
  [two modes](https://github.com/ZacWalk/dd/blob/main/docs/mcp.md#two-modes) and
  [commands](https://github.com/ZacWalk/dd/blob/main/docs/cli.md#commands) are documented
  upstream.
- Add a check to `src/selftest.cpp` for anything that can be tested without a camera, and
  run `.\dd.ps1 test --label low-ban` before finishing; it runs `/test` in both
  configurations. Checks that need Media Foundation's H.264 encoder skip, not fail, where
  Windows lacks it or where the child-process probe finds it crashes (as on GitHub's Windows
  Server runners); keep it that way.
- `dd.ps1 evaluate` scores `samples/*.png` against H.264 at equal bytes and replays any
  `samples/*.mp4` through the call session. Use it when changing either wire, the importance
  map, the warp, or anything that moves a rate, distortion or landmark error number.
- Quote payload, quality and landmark error together. A PSNR gain bought with more bytes is
  not an improvement, and neither is the reverse.
- Keep the rate regimes distinct. Each wire sends one startup picture, then a steady rate; do
  not describe a startup size as a steady-state rate. Payload is not the link estimate, which
  adds packet headers and an audio allowance.
- Current reference figures, from the nine stills at the 320x240 coded size: low-ban's startup
  is 95.6 kbit at 32.94 dB with 1.80% landmark error; H.264 within the same bytes reaches
  36.03 dB and 1.45%, and 1.30% with the face offset. After startup low-ban is capped at 48
  kbit/s of pictures plus 15 Hz geometry, a 57.6 kbit/s one-face ceiling if each landmark delta
  fits 80 bytes (escapes are not hard-capped); H.264 runs at a constant 52 kbit/s.
- Stills only ever code a startup picture, so they cannot see the warp, motion compensation
  or scheduling. Temporal claims need a recorded clip through `evaluate`, or a sequence in
  `src/selftest.cpp`. Synthetic sequences whose motion is made by the warp only check the
  mechanism; never quote them as a result.

## Style

- Follow the existing style: lower_snake_case, tabs, standard containers, `const` by default.
- `image.h`, `draw.h` and `warp.h` must stay free of Windows headers so the self test can use
  them; `baseline.h` and `capture.h` keep Media Foundation behind their implementations.
- Keep image processing out of `main.cpp`; it owns the window and nothing else. Per-frame wire
  logic belongs in `session.cpp`, which the engine and `evaluate` share.
- Check `HRESULT` with `SUCCEEDED`/`FAILED`.
- Shared state crosses threads only as an immutable value swapped under a short mutex. The
  H.264 transforms belong to the analysis thread that created them.
- Encoder and decoder state must stay separate. Pictures shown on screen have to come from
  bytes that were actually serialized, never from the encoder's own copy, and that includes
  the H.264 pane.
- Low-ban's startup must reconstruct the complete scene before its pane is published. Never
  replace it with a partial progressive refresh or periodically reset established blocks to
  grey. Post-startup picture packets remain capped at 1,200 bytes.
- Preserve malformed-packet rejection and transactional decoding: a rejected packet must not
  partially mutate decoder state.
- The transform stream schedules packets on distortion per byte with face importance as a
  *bounded* multiplier plus an eight-packet starvation deadline. Importance may reorder
  blocks; it must never be able to stop one being served. Ranking by importance first is what
  made a moving head leave copies of itself on screen, and `src/selftest.cpp` guards against
  it.
- The transform codec warps the held face onto the newest transmitted geometry, then predicts
  either from what the decoder holds or from a motion compensated window of the previous
  decoded frame, and drops residual coefficients that do not pay for their bytes. Three
  consequences are load-bearing: the encoder must store what it *sent*, not the source; both
  endpoints must build every predictor with the same code path; and the encoder must warp from
  the geometry the receiver holds (`temporal_landmark_stream::transmitted()`), never from the
  tracker's landmarks.
- The warp stays integer arithmetic. It is committed only when it lowers distortion, and a
  refusal is announced to an animating receiver; a packet that is not sent commits nothing.
- Keyframes must stay exact: rate-distortion trimming runs with lambda zero there, and a self
  test compares the startup keyframe against a direct reconstruction byte for byte.
- Landmarking is a bit-allocation, prediction and measurement signal. The overlay is a separate
  View toggle that only suppresses drawing: hiding it must not change prioritisation, the warp
  or the geometry bytes charged to the wire. The jitter filter is sender-side only; importance
  and landmark error use raw landmarks.
- The H.264 baseline must stay a real wire at low-ban's budget, fed the same face boxes the
  importance map is built from. Do not tune it down to flatter low-ban.
