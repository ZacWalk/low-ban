# AGENTS

`low-ban` is an experimental Windows application for low bandwidth video communication;
[README.md](README.md) says what it does and how to build it.

**Read [docs/design.md](docs/design.md) first.** It is the single source of truth for the
architecture, the module boundaries, the algorithms, the wire and model file formats, the
reference figures and the known limitations. Keep it current when you change behaviour, and
do not duplicate its content in other markdown files.

## Essentials

- C++20 at the libdesktop presentation boundary in `main.cpp`; the processing modules stay
  C++17-compatible. Win32 and Media Foundation; `Debug|x64` and `Release|x64` only.
- No Windows AI APIs and no third-party codec libraries; the H.264 baseline uses the encoder
  and decoder that ship with Windows. The only external work in the source is the face
  detector's network and weights, adapted from libfacedetection; keep its BSD notice intact in
  `face.cpp`/`face-model.cpp`.
- Build, launch, test and evaluate through `dd.ps1`. The [dd](https://github.com/ZacWalk/dd)
  build driver is vendored unmodified - `dd.ps1` and `.dd/`, with hashes in
  [docs/dd-upstream.json](docs/dd-upstream.json) - so never edit it. Project settings and the
  `evaluate` command live in `dd.psd1`, and `.\dd.ps1 commands --json` lists project commands.
  dd's [two modes](https://github.com/ZacWalk/dd/blob/main/docs/mcp.md#two-modes) and
  [commands](https://github.com/ZacWalk/dd/blob/main/docs/cli.md#commands) are documented
  upstream.
- Add a check to `src/selftest.cpp` for anything that can be tested without a camera, and
  run `.\dd.ps1 test --label low-ban` before finishing; it runs `/test` in both
  configurations. Checks that need Media Foundation's H.264 encoder skip, not fail, where
  Windows lacks it or where the child-process probe finds it crashes; keep it that way. The
  probe fails on GitHub's Windows Server runners, so CI does not exercise H.264 and those
  checks run only on a machine whose encoder works.
- `dd.ps1 evaluate` scores `samples/*.png` against H.264 at equal bytes and replays any
  `samples/*.mp4` through the call session. Use it when changing either wire, the importance
  map, the warp, or anything that moves a rate, distortion or landmark error number, and
  compare with the reference figures in [docs/design.md](docs/design.md#quality-evaluation).
- Quote payload, quality and landmark error together. A PSNR gain bought with more bytes is
  not an improvement, and neither is the reverse.
- Keep the rate regimes distinct. Each wire sends one startup picture, then a steady rate; do
  not describe a startup size as a steady-state rate. Payload is not the link estimate, which
  adds packet headers and an audio allowance.
- Stills only ever code a startup picture, so they cannot see the warp, motion compensation
  or scheduling. Temporal claims need a recorded clip through `evaluate`, or a sequence in
  `src/selftest.cpp`. Synthetic sequences whose motion is made by the warp only check the
  mechanism; never quote them as a result.

## Style

- Follow the existing style: lower_snake_case, standard containers, `const` by default.
- `.clang-format` is the layout: tabs, Allman braces, 120 columns. `.\dd.ps1 fmt` applies it
  to `src` and `tests` (dd formats only `include`, `src` and `tests`), given clang-format on
  PATH, the only place dd looks; Visual Studio installs one in `VC\Tools\Llvm\x64\bin`. A
  table laid out by hand, such as the codec's scan order, is fenced with
  `// clang-format off` and `// clang-format on`.
- Both targets build at `/W4` with no warnings in Debug and Release, and `.clang-tidy` names
  the bug-finding and performance checks the code is kept clean against; a deliberate
  exception is a `// NOLINT(check): reason` on its line. `face-model.cpp`, the generated
  weight table, keeps libfacedetection's names and layout: `.clang-format-ignore` lists it,
  and clang-tidy runs leave it out.
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
  grey. Post-startup picture packets stay capped at 1,200 bytes.
- Preserve malformed-packet rejection and transactional decoding: a rejected packet must not
  partially mutate decoder state.
- The transform stream schedules packets on distortion per byte with face importance as a
  *bounded* multiplier plus an eight-packet starvation deadline. Importance may reorder
  blocks; it must never be able to stop one being served, or a moving head leaves copies of
  itself on screen; `src/selftest.cpp` guards against that.
- Three consequences of the [transform codec](docs/design.md#transform-codec) are
  load-bearing: the encoder must store what it *sent*, not the source; both endpoints must
  build every predictor with the same code path; and the encoder must warp from the geometry
  the receiver holds (`temporal_landmark_stream::transmitted()`), never from the tracker's
  landmarks.
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
