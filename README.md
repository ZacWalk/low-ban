# low-ban

[![Build](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml/badge.svg)](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml)

Low bandwidth video experiment: a face-prioritised transform codec and a
face-weighted patch autoencoder, side by side on the same webcam frame. Plain
C++17 on Win32.

During a hackathon at Microsoft I tried to invent a super low-bandwidth video codec optimised
for 2G networks. It got a lot of attention at the event, though it stayed an experimental
prototype. The core idea was to use facial landmarking to preserve the user's expressions, so
that non-verbal communication survives even when the video frame is heavily compressed and
blurry.

![screenshot](https://github.com/ZacWalk/low-ban/assets/181544/01160e1c-c3ce-40d4-895e-375e01c712d1)

## How it works

Three panes show the raw feed and the two independent low-bandwidth
reconstructions produced from serialized payloads. The full architecture,
packet scheduling rule, wire formats, measured rates and current limitations
live in [docs/design.md](docs/design.md); [docs/todo.md](docs/todo.md) tracks
the known gaps that are not part of the current design.

## Building

Requires Windows x64 and Visual Studio with the Desktop C++ workload. The
vendored [dd](https://github.com/ZacWalk/dd) runtime locates Visual Studio and
uses the CMake and Ninja that ship with it.

```powershell
.\dd.ps1 build            # both configurations
.\dd.ps1 test --label low-ban  # headless checks, no camera needed
.\dd.ps1 launch           # build Release, then start the app
.\dd.ps1 evaluate --yes   # score samples/*.png offline
```

Drop `shape_predictor_68_face_landmarks.dat` into `exe/` for the landmark
overlay. The application runs without it.

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
