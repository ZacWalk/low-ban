# low-ban design

Low bandwidth video experiment. Three panes compare the raw webcam feed, low-ban - a
face-prioritised transform codec whose receiver animates the face from transmitted landmarks -
and an H.264 baseline given the same bitrate and the same face boxes. The premise is that
non-verbal communication survives extreme compression if scarce bits are concentrated around
expression-bearing regions and the face's geometry travels separately from its pixels.

This file describes what the code does now. [todo.md](todo.md) lists what it does not.

The application is C++20 at its presentation boundary and uses libdesktop's
`core`, `platform`, `render` and `ui` libraries. The UI app owns a native-caption frame, a drawn
menu bar and an element tree whose content element paints through `render::draw_context`. The
H.264 baseline and clip recording still use the encoder, decoder and file
transforms that ship with Windows' Media Foundation behind their existing implementations; there
is no ONNX or DirectML runtime and no Windows AI API. The one piece not written here is the face
detector's network design and weights, which come from
[libfacedetection](https://github.com/ShiqiYu/libfacedetection) under a 3-clause BSD licence;
the inference code was rewritten for this application. This is a demo: the goal is to be small
and legible, not to be robust in every lighting condition.

## Layout

```
src/
  main.cpp        libdesktop `ui::app`, frame, menu element and render painting. No image processing.
  engine.h/.cpp   Threads, capture and clip recording, publishing results.
  session.h/.cpp  One simulated call: face tracking, both wires, clocks and metrics.
  capture.h/.cpp  Media Foundation webcam capture, clip recording and replay.
  image.h/.cpp    Image containers, YUY2 and NV12 conversion, resize, blur, Sobel.
  face.h/.cpp     CNN face detector.
  face-model.cpp  Generated detector weights.
  landmarks.h/.cpp 68-point landmark regressor, model file reader, landmark wire, jitter
                  filter and landmark error.
  warp.h/.cpp     Integer Delaunay triangulation and piecewise-affine face warp.
  codec.h/.cpp    Low-ban transform codec, its wire stream and quality metrics.
  baseline.h/.cpp H.264 baseline wire over Media Foundation's encoder and decoder.
  evaluate.cpp    Offline evaluation of PNG stills and recorded clips.
  draw.h/.cpp     Anti-aliased line drawing over an 8-bit image.
  selftest.cpp    Headless checks (`/test`).
exe/
  shape_predictor_68_face_landmarks.dat   landmark model (not produced by this repo)
samples/          stills and recorded clips used by `dd.ps1 evaluate`
dd.ps1            build / run / test / evaluate / clean
```

Dependency direction is strictly one way: `main` depends on `engine`, `engine` on `session`
and `capture`, `session` on the processing modules, and the processing modules on `image`.
`main` depends on libdesktop's `ui` layer, with rendering through `render` and platform
services through `platform`; it does not name an `HWND`, GDI object or private Win32 facade. `image`, `draw` and `warp` have no Windows dependency at all,
which is what makes the self test possible without a camera or a window. `baseline` and `capture`
hide Media Foundation behind Windows-free headers.

## Threading

| Thread   | Responsibility |
| -------- | -------------- |
| capture  | Blocks on `IMFSourceReader::ReadSample`, decodes YUY2 into planes, publishes the frame, and writes it to a clip while recording. |
| analysis | Detects faces, runs landmarks, runs the call session, publishes a `render_state`. Owns the H.264 transforms, so it initialises COM. |
| UI       | The app's `ui::app` creates the frame with `ui::create_app_frame`, while `platform::run` drives the native window/message loop. Its 16 ms application frame tick checks the generation counter and queues one invalidation; the shared paint pass draws the three panes, captions, status and face boxes through one `render::draw_context`. |

Handover is by `std::shared_ptr` to an immutable value, swapped under a short mutex. A
consumer either sees the previous frame in full or the next frame in full, never a mixture,
and neither producer ever blocks on the consumer. The UI thread only converts the already-published panes to `core::surface` values and draws them; it
does no codec, camera, face or landmark processing, so a slow analysis pass degrades the frame rate
rather than freezing the window.

The drawn menu bar is built at runtime through `ui::command` objects. Its checked callbacks read the four
application toggles, while libdesktop dispatches menu accelerators and keyboard messages without
exposing native message parameters to the application. The UI frame closes through the app lifecycle and keeps the existing minimum client content size
through `ui::frame_style::minimum_client`.

The analysis thread compares `shared_ptr` identity to skip frames it has already handled,
which is also what keeps it from spinning when the camera stalls.

The `platform::run` UI thread is STA, but `/test`, `/evaluate` and `/probe-h264` run their
Media Foundation and WIC work through the capture backend's fresh native MTA boundary. This
keeps console-mode COM apartment behavior equal to the GUI path without putting COM
initialisation in the GUI layer.

## Capture

`webcam` requests YUY2 at 640x480. YUY2 is what webcams natively produce, so there is no
conversion cost, and it hands us chroma for free, which the face detector then uses.

`decode_yuy2` splits the packed `Y0 Cb Y1 Cr` macropixels into three planes: full resolution
luma and half width chroma. A negative `MF_MT_DEFAULT_STRIDE` means a bottom-up buffer and is
normalised to top-down during the split, so nothing downstream has to think about it.

The negotiated subtype is checked after `SetCurrentMediaType`, and opening fails with the
actual FOURCC if the camera substituted something else. Everything downstream unpacks YUY2 by
hand, so a silent substitution would appear as noise rather than as a failure.

A camera that opens and then stops delivering - held by another application, unplugged - fails
`ReadSample` or flags an error or the end of the stream. The capture thread then stops and
reports why on the window's status line, as it does when no camera opens at all, instead of
retrying a reader that is finished. Setting a status moves the generation counter as a frame
does: with no frames arriving, nothing else would make the window repaint, and the status
would stay off screen.

### Recorded clips

File > Record clip writes the camera to `samples/clip-<local time>.mp4` until it is chosen
again. Frames are written on an even 15 Hz cadence and stamped at exact intervals, so a clip
replays at precisely its frame rate whatever the camera's own timing was. The picture is packed
back into YUY2 and handed to a Media Foundation sink writer as H.264 at 4 Mbit/s - about 0.9
bits per pixel, high enough to be a faithful source for both wires (the self test requires
better than 30 dB on a round trip). A clip is finished when recording stops, when the camera
fails and when the app closes, because an unfinished MP4 cannot be played back.

`clip_reader` replays a clip as captured frames through the source reader's advanced video
processor, as YUY2 or, failing that, NV12, at 640x480 whatever size it was recorded at.
Recorded clips show whoever was in front of the camera, so `samples/*.mp4` is ignored by git.

## The call session

`call_session` is one simulated call. Every captured frame goes in with its timestamp in
seconds and the face analysis; out comes what both receivers show. The live engine passes
steady-clock time and `evaluate` passes clip timestamps, so the offline numbers are measured on
exactly the code the window runs.

The sender's analysis - `face_tracker` - detects faces at the full capture size, runs the
landmark model on each box and feeds each box the model implies back to the next frame. Its
68 points per face produce the importance map: background pixels have weight 32, the detected
face rectangle has weight 96, and discs around landmarks 17 through 67 (brows, eyes, nose and
mouth) have weight 255.

Pictures are coded at half the capture size in each direction, 320x240, and scaled back up for
display and measurement. Detection and landmarking keep the full 640x480, where the features
are. Every practical low-rate system drops resolution first: at 1,200 bytes a picture the full
frame is 0.03 bits per pixel, and the halved frame four times that.

Two clocks run, each on an even cadence with a few milliseconds of slack so a 30 fps camera
lands on every tick rather than slipping to every third frame:

| Clock | Rate | Carries |
| ----- | ---- | ------- |
| geometry | 15 Hz | a low-ban landmark packet, and one H.264 frame |
| picture | 5 Hz | a low-ban picture packet, capped at 1,200 bytes after the startup keyframe |

On a tick where both are due, geometry goes first, so the picture packet predicts from the
newest landmarks. The low-ban receiver redraws at the geometry rate: between picture packets it
shows its last decoded picture with the face warped onto the newest received landmarks, so a
mouth or a brow moves 15 times a second even though pictures arrive five times. H.264 runs at
the same 15 frames a second, so both receivers update equally often.

The window shows three caption lines under each received pane: rate, quality, and what the last
packet was. Rates are rolling ten-second averages of serialized bytes - picture and geometry
payload for low-ban, the encoder's output for H.264 - and include and then age out each wire's
startup picture.

### The link estimate

Payload is not what a link has to carry. Each pane also shows a link figure: payload plus 40
bytes of IPv4, UDP and RTP header on every packet of every stream, plus an allowance for the
audio a real call cannot do without - 6 kbit/s of low-rate speech in 60 ms packets, 11.3
kbit/s with its headers. Nothing is sent through a socket; this is accounting, not a network
measurement, and it excludes retransmission, error correction and congestion. At 15 Hz, a
moving face's geometry packets cost about as much again in headers as in landmarks (see
[todo.md](todo.md)).

## Low-ban

### Packet scheduling

Far more of the frame wants to change each interval than the packet budget can carry. The
transform stream resolves that with one rule, and the rule is the single most important thing
in this repository.

An 8x8 transform block competes on

```
priority = importance_gain * distortion_removed / bytes
```

`distortion_removed` is how much squared pixel error this record would actually retire,
measured against what the decoder holds rather than against the source. That is exact rather
than a proxy: the DCT here is orthonormal, so the sum of squared dequantised coefficient errors
is the block's pixel-domain squared error. `bytes` is the serialized length of the record, so
the ranking buys the most error reduction per byte. `importance_gain` runs from 1 to 8 across
the importance map.

The gain being *bounded* is the point. An importance term that can outweigh distortion
outright does not prioritise the face, it starves everything else forever, and the visible
symptom is a head that leaves copies of itself behind: the region the face moved off is now
low importance, so no matter how wrong it is it never outbids the face that keeps changing at
its new position.

A bounded gain is still not enough on its own, because the face regenerates high-priority
candidates every interval. The stream therefore also carries a per-block counter of how many
packets a block with error has been deferred. At eight packets it is promoted ahead of every
block still inside its deadline, and the promoted set is ordered among itself by the same
rate-distortion rule. Importance decides the order in which blocks are served; it can never
decide that a block is not served. That bound is what removes the ghosting, at the cost of a
face that visibly pauses for an interval or two while a large vacated region is repaired.

Selection and serialization are separate passes. Blocks are chosen by priority, then written
in index order so each record can carry the gap from the previous one instead of an absolute
index. A gap never needs more bytes than the absolute index it replaces, so a budget reserved
during selection always holds.

### Transform codec

Pictures are coded by a block transform codec in the same broad family as MPEG. Each 8x8 luma
block is level shifted, transformed by an orthonormal DCT, quantised and inverse transformed.
Higher spatial frequencies receive progressively coarser quantisation. The importance map
lowers a block's quantiser from 34 toward 8, preserving more coefficients in face-feature
blocks. The mapping is snapped to a ladder of eight steps, because a landmark disc drifting by
one pixel would otherwise re-quantise blocks whose pixels never moved, and every one of those
blocks then has to be re-sent for no visible gain.

#### The landmark warp

Before a single block is coded, both ends move the face they hold to where the newest
transmitted geometry says it is. Each face's 68 landmarks, plus ten points extrapolated above
the brows so the forehead moves with them, are triangulated where they are now, and every pixel
inside that mesh is sampled from the matching triangle where they were when the held picture
was decoded. Every block the warp changes is then re-predicted from the warped picture at the
quantiser it already holds; every other block keeps its coefficients exactly.

The warp costs nothing on the wire: both ends run `warp_faces` on the same reference picture
with the same transmitted geometry, the encoder's copy from `temporal_landmark_stream::
transmitted()` and the decoder's from what it decoded. It is integer arithmetic throughout - an
exact Bowyer-Watson Delaunay triangulation with 64-bit predicates, barycentric mapping in
fixed point, bilinear sampling in 1/256 pixel steps - so two endpoints agree whatever compiler
built them. Repeated points, such as closed lips on the two-pixel landmark grid, are left out
of the triangulation rather than producing zero-area triangles.

Landmarks are not always right: a hand over the face or a lost track moves the mesh without
moving the face. The encoder therefore commits the warp only when it brings the blocks it
touches closer to the source than leaving them alone would, and the packet says so in its
header. Otherwise it *refuses* the warp, and that is announced too - in a bare ten-byte header
if there is nothing else to send, but only to a receiver that is currently animating. Between
picture packets the receiver animates its picture onto newer geometry unless the encoder's last
word was a refusal; a picture that defies its geometry therefore costs one header and is then
shown as it is. A packet with no records and no refusal to announce is never sent, and its warp
is not committed: the next packet simply warps from the same reference to newer geometry.

On synthetic frames whose face moves by the same warp the codec predicts with - a mechanism
check, not a claim about real faces - the self test sends a drifting, mouth-opening face for
2,420 bytes at 31.3 dB with geometry against 7,871 bytes at 30.6 dB without.

#### Prediction

What remains after the warp is coded against a predictor, and there are two to choose from:

- what the decoder already holds for that block, warped or not, or
- a *motion compensated* window of the previous decoded frame, displaced by a vector.

Translation is the one thing a block transform cannot express, and a head in front of a webcam
is mostly translation. Without the second predictor every block a head passes through has to
be retransmitted in full, which no realistic packet budget can afford. Both endpoints keep the
last fully decoded frame and run the same quantised transform over the displaced window, so a
block that merely moved costs its vector and almost nothing else.

The encoder searches for the vector with a three-step search over +/-16 pixels, seeded with
the previous block's vector. Neighbouring blocks of one moving head share a vector, so the
seed usually lands on the answer and 32 comparisons are enough where an exhaustive search
would need 1,089.

#### Rate-distortion decisions

The predictor is compared against the source in quantised coefficients, and every residual
coefficient then has to earn its byte:

```
keep the coefficient if (delta * quantiser * frequency_scale)^2 > lambda * bytes
lambda = 2 * quantiser^2
```

This is the Lagrange multiplier of the usual rate-distortion formulation, tied to the block's
own quantiser so a coarsely coded background is allowed to be sloppier about its residual than
a face is. Without it, a well predicted block still costs twenty bytes of requantisation
noise: the difference between two independently rounded quantisations of nearly identical
pixels is a scatter of plus and minus ones across the whole block. With it, such a block sends
no residual at all and the record collapses to two bytes.

The same rule picks the predictor: whichever of the two minimises `distortion + lambda * bytes`
wins. Keyframes use `lambda = 0`, so a keyframe is still an exact intra coding of every block.

Because a residual coefficient can be judged not worth its byte, the encoder stores what it
actually sent rather than what the source was. Everything downstream - the distortion estimate,
the scheduler, the next frame's predictor - is then measured against the picture the decoder
is really holding.

#### On the wire

Transform packets use the `LT` signature, version 3, a flags byte - bit 0 keyframe, bit 1 warp
committed, bit 2 warp refused, never two at once - then the coded width and height and the
record count. Each record is

| Field | Encoding |
| ----- | -------- |
| index | variable-length gap from the previous record |
| flags | bit 0 motion, bit 1 quantiser follows, bit 2 reuse previous vector, bit 3 no residual |
| quantiser | one byte, if bit 1 |
| vector | two variable-length signed values, if bit 0 and not bit 2 |
| count | variable-length token count, if not bit 3 |
| tokens | that many run/level tokens |

Coefficients are visited in a low-frequency-first zigzag scan, so the zero runs between
surviving coefficients are long. One token byte carries a zero run up to 15 and a level in
-7..7, which is what almost every motion-compensated residual coefficient turns out to be;
a zero byte escapes to full variable-length values. Bits 2 and 3 are what make a block that
simply moved with its neighbours cost two bytes in total. The records are byte aligned and not
entropy coded, and that framing is about two fifths of the startup keyframe (see
[todo.md](todo.md)).

The live wire emits one complete keyframe at startup, so the receiver begins with the whole
scene rather than neutral-grey blocks that fill in gradually. On the sample set this one-time
startup picture, with its geometry keyframe, averages 95.6 kbit at 32.94 dB PSNR. It is
included in the displayed rolling bitrate and is not subject to the steady-state packet cap.
After startup, picture packets are capped at 1,200 bytes, 48 kbit/s at 5 Hz. There are no
periodic decoder resets, so an established scene is not replaced by grey every few seconds.

Decoding is transactional: malformed or truncated input, unknown flags and contradictory flags
are rejected before decoder state is replaced.

### Landmark wire

Landmarks have their own serialized temporal stream, sent at 15 Hz and charged to low-ban.
Coordinates are quantised to two-pixel units, which is one pixel of the coded picture.
Keyframes contain absolute variable-length coordinates. A delta packet describes each face as
one rigid shift - the median of its points' motion, which is what the head did - plus per-point
residuals, which are what the expression did. Residuals are packed as two nibbles per point,
-7 through 7, with variable-length escapes for larger motion. Each face takes whichever of four
layouts is shortest: with or without the shift, listing every point or only those whose
residual is not zero behind a 68-bit mask. A still or rigidly moving face costs a few bytes
rather than one per point, a talking mouth lists only the points that moved, and the worst
small-motion face still fits in 80 bytes. Unchanged geometry emits no packet.

Landmark packets use the `LM` signature, version 2, a keyframe flag and the face count, and are
decoded transactionally. Faces are currently paired by detector order rather than by a
persistent track identifier.

Before transmission the landmarks pass through a "one euro" filter (Casiez, Roussel and Vogel,
2012), whose cut-off rises with speed: 1.5 Hz for a still landmark, about 7.5 Hz for one moving
at 200 pixels a second. Raw landmarks jitter by a grid step from frame to frame, which at 15 Hz
costs geometry bytes and makes the receiver's warped face shimmer, while a fast cut-off during
real motion keeps lips and brows from lagging. The filter is sender-side only: the importance
map and the landmark error use the raw landmarks.

View > Landmark overlay draws the received geometry over the low-ban picture. Landmarking is a
bit-allocation, prediction and measurement signal, not a picture element, so hiding the overlay
leaves prioritisation, the warp and the geometry bitrate untouched and shows the picture as a
viewer would actually receive it. View > Face landmarking is the ablation: no landmark
weighting, no warp, no geometry stream, no face offset for H.264.

## H.264 baseline

The third pane is H.264 from the encoder and decoder transforms that ship with Windows. It is a
real serialized wire: the picture shown is what the decoder made of the encoder's bytes, and
every byte the decoder needs - including the parameter sets, charged to the first frame if the
encoder keeps them out of band - is counted. Access unit delimiters and SEI messages are
dropped before counting, because the decoder needs neither: this encoder opens every frame with
a delimiter and its first frame with about 300 bytes of SEI, and charging them would handicap
the baseline.

It codes the same 320x240 luma as low-ban, with flat grey chroma that costs it next to
nothing, at a constant 52 kbit/s: the low-ban picture cap plus a typical single-face geometry
stream, so both wires spend about the same. Main profile with CABAC, which is worth about a
quarter of the bytes here; High profile streams from this encoder in low-latency mode did not
survive the in-box decoder past the first few macroblocks of each slice. Low-latency mode on
both transforms, no B-frames, and a single key frame at the start like low-ban's.

It is fed the same face boxes the importance map is built from. The encoder takes one region
of interest per frame, so several faces share their union, and that region is coded six
quantiser steps finer than the rest - half the quantiser, comparable to low-ban's average face
emphasis. `evaluate` runs H.264 with and without the offset, so face priority can be judged on
a competitive codec on its own, and reports whether the encoder honoured the offset at all.

When the transforms are missing, the pane says why and the self test skips its H.264 checks
rather than failing. They can also be present and broken. GitHub's Windows Server 2025 runners
register an H.264 encoder (`mfh264enc.dll` 26100.7309) that is an older build than the rest of
their Media Foundation (26100.32995 and later), and it crashes on its own worker thread during
the first frame; installing Server's Media Foundation feature succeeds without a restart but
leaves the same encoder. No error code reports the crash, and no handler can recover from it,
so before the first H.264 wire or clip opens, the program runs one short round trip in a child
copy of itself (`/probe-h264`), which suppresses the crash dialog. If the child crashes or
hangs, the encoder is unavailable for the rest of the process: the pane, `evaluate` and clip
recording say so, and the self test skips. Ordinary failures, such as a missing transform, pass
the probe and are reported by the call that meets them.

H.264 frames are stamped at an even 15 frames a second. A camera that delivers fewer than 15
frames a second fires the geometry clock less often, and the encoder, budgeting per frame,
then spends proportionally less than 52 kbit/s; the pane's measured rate shows it.

## Quality evaluation

Both received pictures are compared with the source luma, at the capture size, before the
overlay is drawn. Each pane reports ordinary PSNR and importance-weighted PSNR, which uses the
same map as low-ban so an error around an eye or mouth counts more than the same error in the
background.

Neither says whether expression survived, so each pane also reports *landmark error*: the
landmark model is run on the received picture from the same box the source was landmarked
with, and the normalised mean error is the average distance between the two sets of points as
a percentage of the source's outer-eye-corner distance. The mouth's 20 points are also scored
alone in `evaluate`, since speech and expression move them most. Payload, quality and landmark
error must be read together; a higher PSNR at a much higher payload is not automatically
better.

`dd.ps1 evaluate` has two parts.

- **Stills.** Every `samples/*.png` is resized to 640x480, detected and landmarked, and coded
  as low-ban's startup: the geometry keyframe and one complete picture keyframe. H.264 then
  codes the same picture at the finest fixed quantiser whose single frame fits the same total
  byte count, with and without the face offset. On the current nine samples, low-ban's
  startup averages 95.6 kbit at 32.94 dB, 32.75 dB face-weighted and 1.80% landmark error;
  H.264 at 90.3 kbit reaches 36.03 dB, 35.56 dB and 1.45%, and with the face offset, at 90.7
  kbit, 36.00 dB, 35.60 dB and 1.30%. H.264 lands under the budget because whole quantiser
  steps are coarse; one finer step would overshoot it.
- **Clips.** Every `samples/*.mp4` is replayed through two call sessions that differ only in
  whether H.264 gets the face offset. Each clip reports low-ban's startup size, steady picture
  and geometry rates, link estimate, quality, landmark and mouth error and the number of warped
  packets, and the same for both H.264 variants. Rates after startup exclude the startup
  pictures, which are reported on their own. This is the only measurement of motion, warp and
  scheduling; it needs a clip recorded with File > Record clip.

## Face detection

A small convolutional network: a depthwise-separable backbone, a three level feature pyramid
built by upsampling and adding coarser levels into finer ones, and per-cell objectness,
classification and box regression heads at strides 8, 16 and 32.

The network design and the trained weights in `face-model.cpp` come from
[libfacedetection](https://github.com/ShiqiYu/libfacedetection) by Shiqi Yu, used under its
3-clause BSD licence. The inference code in `face.cpp` was rewritten for this application:

- No SIMD or OpenMP variants, and no aligned blob allocator to support them. Release builds
  target AVX2 and the compiler vectorises the inner loops.
- No C result-buffer API; detection returns boxes in the landmark model's convention.
- The five point keypoint branch is not evaluated. Landmarks come from the 68 point model, so
  those three convolution pairs per pyramid level are pure cost.
- Input comes straight from the captured YCbCr planes rather than from a BGR image.

A fixed YCbCr skin threshold was tried first and abandoned. The approach is not viable
indoors: a warm-lit white bookshelf sits inside every plausible skin window, so the mask
covered most of the room. Skin colour cannot separate a face from a warm-lit scene at all,
which is why this is a learned detector rather than a hand-tuned one.

Detection runs at half the capture resolution. A face in a webcam frame is hundreds of pixels
across, so this costs nothing in recall and a quarter of the arithmetic. Scores are
`sqrt(classification x objectness)`, thresholded at 0.3, then greedy non-maximum suppression
at 0.45 IoU. Priors are the cell centres with zero offset, matching the reference.

`detect` also returns the best score found anywhere in the frame before thresholding, which
is shown next to the face count in the window. That distinguishes the two failure modes: a
high peak with no boxes means the threshold or the suppression rejected them, while a peak
near zero means the network never fired at all.

The trained boxes span hairline to chin and are taller than they are wide; the landmark model
wants a square from brow to chin. The conversion takes the detected width as the side and
starts 20% of the detected height down from its top. It only has to be close, because of the
refinement step below.

## Landmarks

`shape_predictor` is a reimplementation of the inference half of *One Millisecond Face
Alignment with an Ensemble of Regression Trees* (Kazemi and Sullivan, 2014). It reads the
`shape_predictor_68_face_landmarks.dat` model that the reference implementation publishes.
Only inference is implemented; there is no training code.

### Model file format

The file is a stream of length-prefixed values:

| Value | Encoding |
| ----- | -------- |
| integer | control byte, then that many little-endian magnitude bytes. Low four bits are the byte count, `0x80` marks negative. |
| float | two integers: mantissa and exponent. The value is `mantissa * 2^exponent`. Exponents 32000, 32001 and 32002 mean `+inf`, `-inf` and `NaN`. |
| vector | an integer count, then the elements |
| column matrix | `-rows`, `-columns`, then the elements |

The file itself is then:

```
int      version, must be 1
matrix   initial_shape          interleaved x,y of the mean shape in unit box coordinates
vector   forests[cascade][tree] splits (idx1, idx2, threshold) and leaf shape deltas
vector   anchor_idx[cascade][i] which landmark each sampled pixel hangs off
vector   deltas[cascade][i]     that pixel's offset from its landmark
```

Everything is bounds checked while loading, so a truncated or foreign file is reported as an
error rather than trusted.

### Inference

The shape starts at the mean and is refined by each cascade in turn:

1. Fit the 2x2 linear part of the least squares similarity transform from the mean shape to
   the current shape. Closed form: with centred point sets `p` and `q`,
   `M = [[a, -b], [b, a]] / n` where `a = sum(p . q)`, `b = sum(p x q)`, `n = sum(|p|^2)`.
   This is the same optimum the usual SVD formulation produces, without the SVD.
2. Sample one pixel per feature, at `unnormalise(M * delta[i] + shape[anchor[i]])`, where
   `unnormalise` maps the unit box onto the detection rectangle. Pixels outside the frame
   read as zero.
3. Walk each tree by comparing pairs of sampled pixels against a threshold, and add the
   leaf's shape delta.

The box passed in matters: the model was trained against a particular detector's boxes, and
the coarse detector above does not reproduce them exactly.

`box_from_parts` closes that gap. The box-to-image mapping is `image = origin + span * model`
per axis, so given a predicted shape a least squares fit recovers the box the model itself
implies. The tracker feeds that refined box back on the next frame whenever it overlaps a
fresh detection, so after one frame the detector's calibration constants stop mattering and
the boxes track the model's own convention.

## Bitrate figures

After startup, low-ban is capped at 1,200-byte picture packets at 5 Hz, 48 kbit/s, plus 15 Hz
geometry. With each face's delta within 80 bytes, that is a 57.6 kbit/s ceiling for one face
and 67.2 kbit/s for two; geometry escapes for large motion are not hard-capped, and a still or
rigidly moving face costs far less than 80 bytes. H.264 runs at a constant 52 kbit/s. The
headless checks enforce the one-face ceiling below 80 kbit/s and every picture packet after
startup inside its cap. All of this is serialized payload; the link estimate above adds
headers and audio.

## Building and running

For quick commands, see [README.md#building](../README.md#building). The exact verification
gate for contributors is in [AGENTS.md#essentials](../AGENTS.md#essentials). `dd.ps1` is the
vendored dd driver, and the `evaluate` project command is declared in `dd.psd1`.

```
.\dd.ps1 test --label low-ban
.\dd.ps1 evaluate --yes
```

`dd.ps1 test --label low-ban` builds both configurations and runs the headless checks that do
not need a camera or a window. They cover image operations, YUY2 and NV12 conversion including
bottom-up and padded buffers, drawing and clipping, the transform codec and its wire, the
Delaunay triangulation and the warp, the landmark wire and filter, landmark error, the H.264
wire and its face offset, clip recording and replay, the call session's clocks and caps, face
detection, the embedded manifest and the landmark model reader. Checks that need Media
Foundation's H.264 encoder - the H.264 wire, clip recording and the session's H.264 receiver -
are skipped, not failed, where Windows does not have it or where the probe above finds it
crashes. On GitHub's Windows Server runner they are skipped, so CI does not exercise H.264.

The `/test` mode also runs deterministic GUI-boundary checks without creating a visible window:
the minimum client size, the centred pane rectangles and the precedence and case handling of
`/test`, `/evaluate` and `/probe-h264`. The GUI itself is headless for command modes because the
the console path in `platform::run` returns before creating the native frame.

`/test` and `/evaluate` write to standard output when it is redirected, so CTest's log and the
`evaluate` result carry their text; run from a console, they attach to it instead.

Four checks are worth calling out, because they catch the failures that are otherwise silent:

- A face that keeps changing at a new position must not permanently outbid the blocks it
  vacated. The check moves a volatile textured square across a static background under the
  production 1,200-byte cap and requires the abandoned region to come back to within one
  level of the truth. Importance-first ranking leaves it at over a hundred levels of error,
  which on a real camera is a second copy of your head. A companion check translates a
  detailed subject across a plain scene and requires the result to hold above 28.5 dB on a
  capped wire; with the motion predictor disabled the same sequence manages 26.9 dB while
  spending more bytes. A third holds a scene still and requires the wire to fall silent, which
  is what catches rate-distortion trimming mistaking its own rounding for new work.
- Geometry that moves over a still picture must be refused for no more than one header, with
  the receiver showing the picture it holds; and a rejected warped packet must leave the
  decoder exactly as a decoder that never saw it. Those are what keep a wrong landmark track
  from silently corrupting the picture at either end.
- The mean shape decoded from the landmark model has to actually look like a face, with the
  eyes above the nose above the mouth above the chin. That is hard to satisfy by accident if
  any part of the packed integer or float decoding is wrong.
- The detector's peak score has to be a finite probability, non-zero, unsaturated on noise
  and identical across repeated runs. A mis-copied weight table or an uninitialised
  accumulator saturates or produces NaN rather than simply scoring low, so negative cases
  alone would not notice.

The landmark model is not built by this repository. Without
`exe/shape_predictor_68_face_landmarks.dat` the application still runs; it reports the
missing model in the window, and there is no landmarking, warp or landmark error.

## Known limitations

What follows is what the current design cannot do by construction. Work that would remove
these limits, and everything else worth doing next, is in [todo.md](todo.md).

- Only the first camera is used and only 640x480 YUY2 is requested.
- The transform encoder advances its reference frame and geometry assuming each packet is
  delivered. Every packet `encode` returns must reach `decode`, or the two sides start
  predicting from different pictures.
- The warp is integer arithmetic, but the transform predictor it feeds - and the motion
  predictor - is a floating point DCT at both ends. In this process that is trivially
  identical; two endpoints built with different compilers could disagree by a rounding step.
- The warp mesh covers the face from chin to forehead. Hair, ears, neck, shoulders and hands
  are moved only by block motion compensation, and gestures get the background's weight.
- The warp is committed or refused for a whole packet, not per face or per triangle.
- Motion vectors are whole-pixel against a single reference frame, and there is no in-loop
  deblocking. The records are byte aligned rather than entropy coded.
- Both codecs code luma only.
- H.264 takes a single region of interest per frame, so several faces share their union.
- Packets are serialized and decoded in memory but are not sent through a socket. The link
  estimate adds headers and audio by accounting, not by measurement.
- Multiple faces are paired by detector order. If their order swaps, the next landmark delta
  can be larger until the following keyframe.
- No clip is committed, so motion results depend on whoever records one.
- The self test cannot prove the detector finds real faces. It checks the weight table is
  wired up, that the forward pass is numerically healthy and deterministic, and that it does
  not fire on synthetic input, but there is no reference face image in the repository.
  Detection quality has to be judged from the green box and the peak score on the source pane.
