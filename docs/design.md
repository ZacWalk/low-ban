# low-ban design

Low bandwidth video experiment. Three panes compare the raw webcam feed, a traditional
transform codec and a learned patch codec on the same frame. Both low bandwidth approaches
prioritise facial features and transmit the face geometry separately from the picture. The
premise is that non-verbal communication survives extreme compression if scarce bits and
training capacity are concentrated around expression-bearing regions.

This file describes what the code does now. [todo.md](todo.md) lists what it does not.

Everything is plain C++17 on the Win32 API. There is no ONNX or DirectML runtime and no
Windows AI API. The one piece not written here is the face detector's network design and
weights, which come from [libfacedetection](https://github.com/ShiqiYu/libfacedetection) under
a 3-clause BSD licence; the inference code was rewritten for this application. This is a demo:
the goal is to be small and legible, not to be robust in every lighting condition.

## Layout

```
src/
  main.cpp        Win32 window, menu, painting. No image processing.
  engine.h/.cpp   Threading and the per-frame pipeline.
  capture.h/.cpp  Media Foundation webcam capture.
  image.h/.cpp    Image containers, YUY2 decoding, resize, blur, Sobel, contrast stretch.
  face.h/.cpp     CNN face detector.
  face-model.cpp  Generated detector weights.
  landmarks.h/.cpp 68-point landmark regressor and model file reader.
  codec.h/.cpp    Prioritised DCT codec, patch autoencoder and quality metrics.
  evaluate.cpp    Offline PNG sample evaluation through Windows Imaging Component.
  draw.h/.cpp     Anti-aliased line drawing over an 8-bit image.
  selftest.cpp    Headless checks (`/test`).
exe/
  shape_predictor_68_face_landmarks.dat   landmark model (not produced by this repo)
samples/          still images used by `dd.ps1 evaluate`
dd.ps1            build / run / test / evaluate / clean
```

Dependency direction is strictly one way: `main` depends on `engine`, `engine` depends on the
processing modules, and the processing modules depend only on `image`. `image` and `draw`
have no Windows dependency at all, which is what makes the self test possible without a
camera or a window.

## Threading

| Thread   | Responsibility |
| -------- | -------------- |
| capture  | Blocks on `IMFSourceReader::ReadSample`, decodes YUY2 into planes, publishes the frame. |
| analysis | Detects faces, runs landmarks, produces both low bandwidth panes, publishes a `render_state`. |
| UI       | On a timer, repaints if the generation counter moved. Blits three bitmaps and some text. |

Handover is by `std::shared_ptr` to an immutable value, swapped under a short mutex. A
consumer either sees the previous frame in full or the next frame in full, never a mixture,
and neither producer ever blocks on the consumer. The UI thread does no image processing,
so a slow analysis pass degrades the frame rate rather than freezing the window.

The analysis thread compares `shared_ptr` identity to skip frames it has already handled,
which is also what keeps it from spinning when the camera stalls.

## Capture

`webcam` requests YUY2 at 640x480. YUY2 is what webcams natively produce, so there is no
conversion cost, and it hands us chroma for free, which the face detector then uses.

`decode_yuy2` splits the packed `Y0 Cb Y1 Cr` macropixels into three planes: full resolution
luma and half width chroma. A negative `MF_MT_DEFAULT_STRIDE` means a bottom-up buffer and is
normalised to top-down during the split, so nothing downstream has to think about it.

The negotiated subtype is checked after `SetCurrentMediaType`, and opening fails with the
actual FOURCC if the camera substituted something else. Everything downstream unpacks YUY2 by
hand, so a silent substitution would appear as noise rather than as a failure.

## Comparison

All three panes are produced simultaneously so a changing webcam scene cannot favour one
approach. Face detection and landmark prediction run first. They produce one 8-bit importance
map used by both codecs and by evaluation: background pixels have weight 32, the detected
face rectangle has weight 96, and discs around landmarks 17 through 67 (brows, eyes, nose and
mouth) have weight 255. Both approaches send their own copy of the compressed temporal
landmark stream, so neither receives geometry for free.

### Packet scheduling

Both codecs face the same problem: far more of the frame wants to change each interval than
the packet budget can carry. They resolve it the same way, and the rule is the single most
important thing in this repository.

A unit (an 8x8 transform block, or a 20x20 latent patch) competes on

```
priority = importance_gain * distortion_removed / bytes
```

`distortion_removed` is how much squared pixel error this record would actually retire,
measured against what was transmitted rather than against the source. For the transform codec
that is exact rather than a proxy: the DCT here is orthonormal, so the sum of squared
dequantised coefficient errors is the block's pixel-domain squared error. `bytes` is the
serialized length of the record, so the ranking buys the most error reduction per byte.
`importance_gain` runs from 1 to 8 across the importance map.

The gain being *bounded* is the point. An importance term that can outweigh distortion
outright does not prioritise the face, it starves everything else forever, and the visible
symptom is a head that leaves copies of itself behind: the region the face moved off is now
low importance, so no matter how wrong it is it never outbids the face that keeps changing at
its new position.

A bounded gain is still not enough on its own, because the face regenerates high-priority
candidates every interval. Each stream therefore also carries a per-unit counter of how many
packets a unit with error has been deferred. At eight packets it is promoted ahead of every
unit still inside its deadline, and the promoted set is ordered among itself by the same
rate-distortion rule. Importance decides the order in which units are served; it can never
decide that a unit is not served. That bound is what removes the ghosting, at the cost of a
face that visibly pauses for an interval or two while a large vacated region is repaired.

Selection and serialization are separate passes. Units are chosen by priority, then written in
index order so each record can carry the gap from the previous one instead of an absolute
index. A gap never needs more bytes than the absolute index it replaces, so a budget reserved
during selection always holds.

### Traditional transform codec

Panel 2 is a block transform codec in the same broad family as MPEG. Each 8x8 luma block is
level shifted, transformed by an orthonormal DCT, quantised and inverse transformed. Higher
spatial frequencies receive progressively coarser quantisation. The importance map lowers a
block's quantiser from 34 toward 8, preserving more coefficients in face-feature blocks. The
mapping is snapped to a ladder of eight steps, because a landmark disc drifting by one pixel
would otherwise re-quantise blocks whose pixels never moved, and every one of those blocks
then has to be re-sent for no visible gain.

#### Prediction

A block is coded against a predictor, and there are two to choose from:

- what the decoder already holds for that block, or
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

Transform packets use the `LT` signature followed by a version, keyframe flag, dimensions and
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
simply moved with its neighbours cost two bytes in total.

The live wire emits one complete keyframe at startup, so panel 2 begins with the whole scene
rather than neutral-grey blocks that fill in gradually. On the sample set this one-time
startup payload averages 322.4 kbit at 36.84 dB PSNR. It is included in the displayed rolling
bitrate and is not subject to the steady-state packet cap.

After startup, the wire runs at 5 Hz with packets capped at 1,200 bytes. Landmarks use the
compressed temporal geometry format described below. With the two-face maximum in the sample
set, the configured ongoing payload reference is 54.4 kbit/s when each landmark delta stays
within 80 bytes. The 1,200-byte image packet cap is hard; the combined rate is not, because
large landmark escapes can exceed that assumption. There are no periodic decoder resets, so
an established scene is not replaced by grey every few seconds.

Decoding is transactional: malformed or truncated input is rejected before decoder state is
replaced.

### Learned patch codec

The frame is cut into fixed 20x20 luma patches. Each patch's mean is measured, quantised to one
byte and subtracted; only the remaining texture goes through the network:

```
analysis    400 -> tanh(3)
synthesis     3 -> tanh(32) -> linear(400)
```

Brightness is by far the highest variance direction in a patch, and eight bits buy it outright
for less than a bottleneck unit spent trying to represent it. A patch therefore costs four
bytes exactly as before - one mean and three code units - but the bottleneck only has to carry
shape. On the sample set that swap alone is worth several dB at an unchanged rate.

The two halves are deliberately asymmetric. Analysis is a single layer because it runs once per
patch and only has to produce three numbers; synthesis carries the capacity because it is the
side that has to invent 400 pixels from them. That is the shape every practical learned codec
settles on.

Training is online on importance-weighted mean squared error, 256 randomly chosen patches per
captured frame. Each weight is stepped by its gradient divided by a running root mean square of
that gradient, so every layer moves at its own pace rather than at a rate chosen for the
worst-scaled one; the learning rate is consequently a normalised step, far smaller than a plain
SGD rate would be. Face features contribute up to eight times the gradient of background
pixels. Weights are initialised uniformly at `±1/sqrt(fan_in)`.

The wire runs independently of analysis at 5 Hz. Every three seconds it emits a keyframe with
a ten-byte header followed by the full four-bytes-per-patch raster: `32 x 24 x 4 = 3,072`
bytes for a 640x480 frame. Other packets contain only changed patches: a variable-length gap
from the previous patch index, then signed deltas for the mean and the three code units.
Candidates are scheduled by the same rate-distortion rule and the same eight-packet deadline
as the transform stream, with squared latent change standing in for pixel distortion, since
the decoder here is a non-linear network and Parseval does not apply. A mean step and a code
step are each worth roughly one grey level at the output, so they are summed unweighted. Delta
packets are capped at 1,500 bytes. If no quantised code changed, no image packet is emitted and
the decoder holds its previous reconstruction.

Latent packets use the `LB` signature followed by a version, keyframe flag, dimensions and
record count. Keyframes store the mean and code bytes in raster order. As with the transform
stream, decoding is transactional and reconstructed pixels come from decoder-owned codes.

Landmarks have their own serialized temporal stream. Coordinates are quantised to two-pixel
units. Keyframes contain absolute variable-length coordinates; ordinary packets pack changes
from -7 through 7 into four-bit nibbles and use variable-length escapes for larger motion.
A small-motion 68-point face delta fits within 80 bytes in the self test, and unchanged
geometry emits no packet. Larger-motion escapes can exceed 80 bytes because landmark packets
are not hard-capped. Panel 3 draws only geometry received by this decoder.

Landmark packets use the `LM` signature and carry their own keyframe flag and face count.
Each image codec has a separate landmark encoder and decoder, so geometry bytes are charged
to both approaches. Faces are currently paired by detector order rather than by a persistent
track identifier.

The autoencoder lives entirely on the analysis thread, so it needs no locking: the same
thread that trains it is the one that runs it.

The View menu can reset the autoencoder weights, which makes convergence easy to demonstrate.
It also separates *Face landmarking*, which decides whether the landmarker runs at all, from
*Landmark overlay*, which decides only whether the received geometry is drawn over panels 2
and 3. Landmarking is primarily a bit-allocation and training signal, not a picture element,
so hiding the overlay leaves prioritisation, training and the geometry bitrate untouched and
shows the reconstructions as a viewer would actually receive them. Turning landmarking itself
off is the ablation: no landmark weighting, no geometry stream.

### Quality evaluation

Both reconstructions are compared with source luma before landmark lines are drawn. Each pane
reports ordinary PSNR and importance-weighted PSNR. The weighted version uses the same map as
the codecs, so an error around an eye or mouth counts more than the same error in the
background. Payload and quality must be read together as a rate-distortion result; a higher
PSNR at a much higher payload is not automatically better.

`dd.ps1 evaluate` loads every `samples/*.png` through Windows Imaging Component, resizes it to
640x480, detects and landmarks faces, trains the autoencoder for 50 deterministic passes over
the set, then serializes and decodes independent complete keyframes for both approaches. It
prints per-image and mean CSV rows plus both configured 5 Hz temporal rates. On the current
nine samples, traditional averages a 322.4 kbit complete keyframe at 36.84 dB PSNR and neural
averages 27.4 kbit at 25.89 dB. The configured two-face references are 54.4 kbit/s traditional
and 71.1 kbit/s neural, assuming each landmark delta fits within 80 bytes. This is a
reproducible still-image check of reconstruction and prioritisation. It says nothing about
motion compensation, which only exists between frames, and nothing about generalisation to
people outside the training set.

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
implies. The engine feeds that refined box back on the next frame whenever it overlaps a
fresh detection, so after one frame the detector's calibration constants stop mattering and
the boxes track the model's own convention.

## Bitrate figures

The left pane is a reference figure for the raw feed at 24 bits per pixel under a conventional
codec at an assumed 5% of raw. Both low-bandwidth panes report rolling rates from serialized
image and landmark bytes actually emitted by independent 5 Hz wire clocks. Traditional uses
a ten-second window, which includes and then ages out its one-time complete startup keyframe;
neural uses a three-second window matching its full-latent keyframe interval.

The full-resolution headless checks enforce a 51.2 kbit/s traditional ceiling for one face
and measure 65.9 kbit/s for a changing neural sequence. This is serialized codec payload, not
a network measurement. Transport headers, audio, retransmission and error correction are
excluded.

## Building and running

`dd.ps1` locates MSBuild through `vswhere` and builds `low-ban.sln` for `x64`.

```
.\dd.ps1 build            # Debug
.\dd.ps1 run              # Release, so the demo runs at full speed
.\dd.ps1 test             # Debug
.\dd.ps1 evaluate         # Debug, evaluate samples/*.png
.\dd.ps1 clean
```

Pass `-Config Debug` or `-Config Release` to override.

`dd.ps1 test` runs the built executable with `/test`, which exercises everything that does
not need a camera or a window, in six groups: image operations, YUY2 decoding including
bottom-up buffers, drawing and clipping, the low bandwidth codecs, face detection, and the
landmark model. Between them they cover transform ROI quality, autoencoder convergence, exact
traditional and neural keyframe reconstruction, bounded sparse deltas,
latent/transform/landmark corruption handling, complete 4,800-block traditional startup,
unchanged-payload suppression, and full-resolution streams held under their configured rate
references.

Three checks are worth calling out, because they catch the failures that are otherwise silent:

- A face that keeps changing at a new position must not permanently outbid the blocks it
  vacated. The check moves a volatile textured square across a static background under the
  production 1,200-byte cap and requires the abandoned region to come back to within one
  level of the truth. Importance-first ranking leaves it at over a hundred levels of error,
  which on a real camera is a second copy of your head. A companion check translates a
  detailed subject across a plain scene and requires the result to hold above 28.5 dB on a
  capped wire; with the motion predictor disabled the same sequence manages 26.9 dB while
  spending more bytes. A third holds a scene still and requires the wire to fall silent, which
  is what catches rate-distortion trimming mistaking its own rounding for new work.
- The mean shape decoded from the landmark model has to actually look like a face, with the
  eyes above the nose above the mouth above the chin. That is hard to satisfy by accident if
  any part of the packed integer or float decoding is wrong.
- The detector's peak score has to be a finite probability, non-zero, unsaturated on noise
  and identical across repeated runs. A mis-copied weight table or an uninitialised
  accumulator saturates or produces NaN rather than simply scoring low, so negative cases
  alone would not notice.

The landmark model is not built by this repository. Without
`exe/shape_predictor_68_face_landmarks.dat` the application still runs; it reports the
missing model in the window and the landmark overlay is disabled.

## Known limitations

What follows is what the current design cannot do by construction. Work that would remove
these limits, and everything else worth doing next, is in [todo.md](todo.md).

- Only the first camera is used and only 640x480 YUY2 is requested.
- The autoencoder learns from whatever is currently in front of the camera. It is a small
  online demonstration, not a competitive pretrained neural video codec.
- The in-process encoder and decoder currently share the same changing model object. A real
  remote decoder cannot reproduce online weight updates from latent packets alone. Deployment
  requires a frozen pretrained model at both endpoints or an explicit, budgeted model-update
  protocol; model bytes are not included in the displayed bitrate.
- The transform encoder advances its reference frame assuming each packet is delivered. Every
  packet `encode` returns must reach `decode`, or the two sides start predicting from
  different pictures.
- The motion predictor is computed in floating point at both ends. In this process that is
  trivially identical, but two endpoints built with different compilers or optimisation
  settings could disagree by a rounding step. A deployable version would have to specify the
  predictor in integer arithmetic.
- Motion vectors are whole-pixel against a single reference frame, and there is no in-loop
  deblocking. The full startup keyframe is substantially larger than one steady-state packet
  and would take time to deliver over a real 2G link.
- Both codecs are luma only. A real call would want at least coarse chroma, which neither
  panel currently spends a byte on.
- Packets are serialized and decoded in memory but are not sent through a socket. Both
  displayed wire figures therefore exclude IP/transport framing, loss recovery, jitter and
  congestion.
- Multiple faces are paired by detector order. If their order swaps, the next landmark delta
  can be larger until the following keyframe.
- The self test cannot prove the detector finds real faces. It checks the weight table is
  wired up, that the forward pass is numerically healthy and deterministic, and that it does
  not fire on synthetic input, but there is no reference face image in the repository.
  Detection quality has to be judged from the green box and the peak score on the source pane.
