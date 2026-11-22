# low-ban

[![Build](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml/badge.svg)](https://github.com/ZacWalk/low-ban/actions/workflows/build.yml)

Low bandwidth video experiment: a face-prioritised codec that spends a 20 kbit/s
call's scarce bits on faces, side by side with H.264 at the same rate and given
the same face boxes. The idea is to preserve the user's expressions, so that
non-verbal communication survives even when the video frame is heavily
compressed: five landmarks a face move the face the receiver holds, and the
picture's bits go where a face moves beyond what those five points say - mouth,
brows and eyes - learned offline as an
[expression prior](docs/design.md#the-expression-prior). On standard
video-conferencing sequences at 20 kbit/s, low-ban's faces keep about two fifths of
H.264's landmark error, and its whole picture needs about as many bits as H.264's for
the same PSNR ([figures](docs/design.md#quality-evaluation)).

![screenshot](https://github.com/ZacWalk/low-ban/assets/181544/01160e1c-c3ce-40d4-895e-375e01c712d1)

## How it works

Three panes show the raw webcam feed, low-ban and the H.264 baseline, each
received from serialized payloads, with rate, quality and landmark error under
each. The limits that matter most: both codecs code luma only; packets are
serialized and decoded in memory, never sent over a network, and the encoder
assumes every one arrives. The design's
[known limitations](docs/design.md#known-limitations) list the rest.

## Building

Requires Windows x64, Visual Studio with the Desktop C++ workload, and Python 3,
which [libmug](https://github.com/ZacWalk/libmug) runs as it builds to convert
the face detector's weights. The vendored [dd](https://github.com/ZacWalk/dd)
runtime locates Visual Studio and uses the CMake and Ninja that ship with it.
The app builds against the libdesktop and libmug revisions pinned by `GIT_TAG`
in `CMakeLists.txt`, or against sibling `..\libdesktop` and `..\libmug`
checkouts when there are.

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
1,968 pixels wide. Two buttons sit above them: Record clip, and Face priority,
which switches face detection off to show what it buys.

`evaluate` replays any clips in `samples/`: `clip-*.mp4` recorded with the
Record clip button, or YUV4MPEG2 test sequences such as Xiph's
[video-conferencing set](https://media.xiph.org/video/derf/) (`Johnny`,
`KristenAndSara`, `akiyo`, `mother_daughter`, `news`). Neither is kept in git:
recordings show whoever made them, and the sequences are hundreds of megabytes.

`tools/learn_expression_prior.py` relearns the expression prior into
`src/expression-prior.cpp`; it needs `tools/requirements.txt` in a virtual
environment and downloads its data on first use.

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

Faces are found by [libmug](https://github.com/ZacWalk/libmug), which runs
[libfacedetection](https://github.com/ShiqiYu/libfacedetection)'s YuNet, by
Shiqi Yu, on its own kernels. The expression prior was learned from [Labeled
Faces in the Wild](https://vis-www.cs.umass.edu/lfw/) with dlib's 68-point
landmark model, an implementation of Kazemi and Sullivan's regression-tree face
alignment, as the teacher.

## License

[MIT](LICENSE). The executable includes
[libdesktop](https://github.com/ZacWalk/libdesktop), which is licensed under the
Mozilla Public License 2.0: anyone distributing the executable must tell its
recipients where to get libdesktop's source code, including any changes made to
it. `tools/iconlib.py` is vendored from libdesktop under the same licence.

The executable also includes libmug (MIT) and, through it, YuNet's weights,
which keep libfacedetection's 3-clause BSD licence: its notice is
[libmug's licences/libfacedetection-BSD-3-Clause.txt](https://github.com/ZacWalk/libmug/blob/main/licences/libfacedetection-BSD-3-Clause.txt).
The expression prior is a statistic of landmarks that dlib's 68-point model
found; that model, which dlib says cannot be used in a commercial product
because the dataset it was trained on excludes commercial use, is used only by
the learning tool and is not part of the repository or the executable.
