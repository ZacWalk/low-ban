# low-ban

[![Build](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml/badge.svg)](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml)

Low bandwidth video experiment: a face-prioritised transform codec whose receiver
animates the face from transmitted landmarks, side by side with H.264 given the
same bitrate and the same face boxes. Plain C++17 on Win32.

During a hackathon at Microsoft I tried to invent a super low-bandwidth video codec optimised
for 2G networks. It got a lot of attention at the event, though it stayed an experimental
prototype. The core idea was to use facial landmarking to preserve the user's expressions, so
that non-verbal communication survives even when the video frame is heavily compressed and
blurry.

![screenshot](https://github.com/ZacWalk/low-ban/assets/181544/01160e1c-c3ce-40d4-895e-375e01c712d1)

## How it works

Three panes show the raw feed, low-ban and the H.264 baseline, each received
from serialized payloads, with rate, quality and landmark error under each. The
full architecture, packet scheduling rule, wire formats, measured rates and
current limitations live in [docs/design.md](docs/design.md);
[docs/todo.md](docs/todo.md) tracks the known gaps that are not part of the
current design.

## Building

Requires Windows x64 and Visual Studio with the Desktop C++ workload. The
vendored [dd](https://github.com/ZacWalk/dd) runtime locates Visual Studio and
uses the CMake and Ninja that ship with it.

```powershell
.\dd.ps1 build            # both configurations
.\dd.ps1 test --label low-ban  # headless checks, no camera needed
.\dd.ps1 launch           # build Release, then start the app
.\dd.ps1 evaluate --yes   # score samples offline against H.264
```

File > Record clip saves `samples/clip-*.mp4`, which `evaluate` replays to
measure motion. Clips are kept out of git because they show whoever recorded
them.

Landmarking needs `exe/shape_predictor_68_face_landmarks.dat`. Without it the
application still runs, with no landmark weighting, warp or landmark error.

## Documentation

- [docs/design.md](docs/design.md) — architecture, algorithms, model file format,
  measured rates and known limitations
- [docs/todo.md](docs/todo.md) — what is not done yet
- [AGENTS.md](AGENTS.md) — conventions for contributors and coding agents

## Credits

The face detector's network design and weights come from
[libfacedetection](https://github.com/ShiqiYu/libfacedetection) by Shiqi Yu, under
a 3-clause BSD licence. The landmark model file is the one published with dlib's
implementation of Kazemi and Sullivan's regression-tree face alignment; both the
reader and the predictor here were written from scratch.

## License

[MIT](LICENSE)
