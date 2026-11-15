# low-ban

[![Build](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml/badge.svg)](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml)

Low bandwidth video experiment: a face-prioritised transform codec whose receiver
animates the face from transmitted landmarks, side by side with H.264 given the
same bitrate and the same face boxes. The idea is to use facial landmarking to
preserve the user's expressions, so that non-verbal communication survives even
when the video frame is heavily compressed and blurry.

![screenshot](https://github.com/ZacWalk/low-ban/assets/181544/01160e1c-c3ce-40d4-895e-375e01c712d1)

## How it works

Three panes show the raw webcam feed, low-ban and the H.264 baseline, each
received from serialized payloads, with rate, quality and landmark error under
each. The limits that matter most: both codecs code luma only; packets are
serialized and decoded in memory, never sent over a network, and the encoder
assumes every one arrives; and motion can only be measured on a clip you record
yourself, since none is committed. The design's
[known limitations](docs/design.md#known-limitations) list the rest.

## Building

Requires Windows x64 and Visual Studio with the Desktop C++ workload. The
vendored [dd](https://github.com/ZacWalk/dd) runtime locates Visual Studio and
uses the CMake and Ninja that ship with it. The app builds against the
libdesktop revision pinned by `GIT_TAG` in `CMakeLists.txt`, or against a
sibling `..\libdesktop` checkout when there is one.

```powershell
.\dd.ps1 build                 # both configurations
.\dd.ps1 test --label low-ban  # headless checks, no camera needed
.\dd.ps1 test --label headless # the same no-visible-UI gate, with console check counts
.\dd.ps1 launch                # build Release, then start the app
.\dd.ps1 evaluate --yes        # score samples offline against H.264, after a Release build
```

Both labels keep the camera-independent suite and manifest check, and print
the actual check, failure and optional-component skip counts. Presets show
successful output too. After building,
`ctest --preset debug -L headless -V --no-tests=error` reruns without rebuilding;
use `release` for Release.
Await a direct GUI-executable test by piping both streams:
`& .\exe\low-ban-64.exe /test 2>&1 | Out-Host; $LASTEXITCODE`.

The three panes are 640x480 each, so the window needs a client area at least
1,968 pixels wide.

File > Record clip saves `samples/clip-*.mp4`, which `evaluate` replays to
measure motion. Clips are kept out of git because they show whoever recorded
them.

Landmarking needs `exe/shape_predictor_68_face_landmarks.dat`, which the
repository includes. Without it the application still runs and says so on its
status line, with no landmark weighting, warp or landmark error.

### Icon

`python tools\generate_icon.py` regenerates `src\app.ico`: a face and shoulders
with microphone and video-call controls, in nine desktop sizes. Add `--check`
to verify without writing, or `--output tmp\icon-128.png --size 128` for another
size. Only Python 3.9+'s standard library and the vendored `tools\iconlib.py` are
needed; no face photograph, graphics package or sibling checkout is used.

## Documentation

- [docs/design.md](docs/design.md) — architecture, algorithms, wire and model
  file formats, measured rates and known limitations
- [AGENTS.md](AGENTS.md) — rules for contributors and coding agents

## Credits

The face detector's network design and weights come from
[libfacedetection](https://github.com/ShiqiYu/libfacedetection) by Shiqi Yu. The
landmark model file is the one published with dlib's
implementation of Kazemi and Sullivan's regression-tree face alignment; both the
reader and the predictor here were written from scratch.

## License

[MIT](LICENSE). The executable includes
[libdesktop](https://github.com/ZacWalk/libdesktop), which is licensed under the
Mozilla Public License 2.0: anyone distributing the executable must tell its
recipients where to get libdesktop's source code, including any changes made to
it. `tools/iconlib.py` is vendored from libdesktop under the same licence.

The face detector keeps libfacedetection's 3-clause BSD licence, stated in
`src/face.cpp` and `src/face-model.cpp`. The licence does not cover
`exe/shape_predictor_68_face_landmarks.dat`, dlib's 68-point landmark model,
which dlib says cannot be used in a commercial product because the dataset it
was trained on excludes commercial use.
