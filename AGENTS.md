# AGENTS

`low-ban` is an experimental Windows application for low bandwidth video communication. It
compares a face-prioritised transform codec with a face-weighted patch autoencoder on the
same webcam frame, using face detection and 68-point landmarking to decide where the bits go.
Both reconstructions come from independent serialized 5 Hz image and landmark streams.

**Read [docs/design.md](docs/design.md) first.** It is the single source of truth for the
architecture, the module boundaries, the algorithms and the model file format. Keep it
current when you change behaviour, and do not duplicate its content in other markdown files.
[docs/todo.md](docs/todo.md) records known gaps and further work; move an item out of it when
you close it.

## Essentials

- C++17, Win32, Media Foundation. Builds `Debug|x64` and `Release|x64` only.
- No Windows AI APIs. The only external work is the face detector's network and weights,
  adapted from libfacedetection; keep its BSD notice intact in `face.cpp`/`face-model.cpp`.
- Build, run and test through `dd.ps1` (`build`, `run`, `test`, `evaluate`, `clean`).
- Add a check to `src/selftest.cpp` for anything that can be tested without a camera, and
  run `.\dd.ps1 test` before finishing.
- `dd.ps1 evaluate` scores `samples/*.png` offline. Use it when changing either codec, the
  importance map, or anything that moves a rate or distortion number.
- Quote payload and quality together. A PSNR gain bought with more bytes is not an
  improvement, and neither is the reverse.
- Keep the two rate regimes distinct. Traditional sends one complete startup scene keyframe,
  then capped deltas; neural sends periodic full-latent keyframes. Do not describe a complete
  keyframe size as a steady-state rate.
- Current reference figures are 322.4 kbit at 36.84 dB for the traditional startup keyframe,
  27.4 kbit at 25.89 dB for a neural keyframe, and two-face 5 Hz payload references of 54.4
  kbit/s traditional and 71.1 kbit/s neural. The rate references assume each moving face's
  landmark delta fits within 80 bytes; geometry escapes are not hard-capped.
- `evaluate` only ever codes keyframes, so it cannot see motion compensation or scheduling.
  Anything temporal has to be measured by the sequences in `src/selftest.cpp`.

## Style

- Follow the existing style: lower_snake_case, tabs, standard containers, `const` by default.
- `image.h`/`draw.h` must stay free of Windows headers so the self test can use them.
- Keep image processing out of `main.cpp`; it owns the window and nothing else.
- Check `HRESULT` with `SUCCEEDED`/`FAILED`.
- Shared state crosses threads only as an immutable value swapped under a short mutex.
- Encoder and decoder state must stay separate. Reconstructions shown on screen have to come
  from bytes that were actually serialized, never from the encoder's own copy.
- Traditional startup must reconstruct the complete scene before panel 2 is published. Never
  replace it with a partial progressive refresh or periodically reset established blocks to
  grey. Post-startup transform packets remain capped at 1,200 bytes.
- Neural delta packets remain capped at 1,500 bytes, with a full latent keyframe every three
  seconds. Both codecs suppress unchanged payload and send their own compressed landmarks.
- Preserve malformed-packet rejection and transactional decoding: a rejected packet must not
  partially mutate decoder state.
- Both streams schedule packets on distortion per byte with face importance as a *bounded*
  multiplier plus an eight-packet starvation deadline. Importance may reorder units; it must
  never be able to stop one being served. Ranking by importance first is what made a moving
  head leave copies of itself on screen, and `src/selftest.cpp` guards against it.
- The transform codec predicts either from what the decoder holds or from a motion compensated
  window of the previous decoded frame, and drops residual coefficients that do not pay for
  their bytes. Two consequences are load-bearing: the encoder must store what it *sent*, not
  the source, and both endpoints must build the predictor with the same code path.
- Keyframes must stay exact: rate-distortion trimming runs with lambda zero there, and a self
  test compares the startup keyframe against a direct reconstruction byte for byte.
- Landmarking is a bit-allocation and training signal. The overlay is a separate View toggle
  that only suppresses drawing: hiding it must not change prioritisation, training or the
  geometry bytes charged to either wire.
- The online autoencoder currently shares changing weights in-process. Do not claim remote
  deployability until the model is frozen at both endpoints or model updates are serialized
  and included in the bitrate.
