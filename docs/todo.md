# todo

Further work, roughly in the order that would move the demo the most. See
[design.md](design.md) for what is already there and why.

## Known gaps

### CI does not exercise H.264

GitHub's Windows Server 2025 runner registers an H.264 encoder that crashes on its first frame,
so the probe makes every H.264, clip and session-H.264 check skip there, and those checks only
run on developer machines. Installing Server's Media Foundation feature in the workflow
succeeds without a restart but leaves the same encoder, which still crashes. Exercising H.264
in CI needs a runner whose encoder works, such as a self-hosted Windows 11 machine.

### No reference clip

Every temporal property - the warp, the receiver's animation, motion compensation, scheduling,
the geometry rate - is measured only on clips recorded with File > Record clip, and clips are
kept out of git because they show a person. Until there is a clip everyone can use, two
contributors' motion numbers are not comparable. A short, consented, licensed talking-head clip
in `samples/`, recorded once with the app, would fix that.

### The encoder assumes every packet lands

`temporal_transform_stream` advances its reference frame and geometry at the end of `encode`,
on the assumption that the packet it just produced will be decoded. Nothing enforces that. In
this process the two calls are always paired, but a rejected or dropped packet would leave the
two sides predicting from different pictures and the error would never resolve on its own. The
landmark stream has the same assumption, and a lost geometry packet now also changes what the
next picture is warped from.

A real transport needs either an acknowledgement before the reference advances, or periodic
intra refresh so drift has a bounded lifetime. The latter is cheaper and is what conferencing
codecs actually do.

### The transform predictor is floating point

The warp is integer arithmetic, but `transform_codec::predict_block` runs a float DCT at both
ends for both the warped and the motion-compensated predictor. Identical in one process, and
identical between the Debug and Release builds only by luck. A deployable format has to specify
the predictor in integer arithmetic.

### Luma only

Neither codec spends a byte on colour. A grey call is a strange thing to demo as a
communication tool. Coarse chroma - say one 16x16 DC per block pair - would cost maybe 10% more
and would change the impression far more than another dB of luma PSNR. It touches the render
path too: the received panes are 8-bit palette DIBs today, and H.264 is fed flat grey chroma.

## Low-ban

- **Entropy coding.** Records are byte-aligned varints and run/level nibbles, and the framing
  alone - gap, flags, quantiser and count, four bytes for every one of 1,200 blocks, 38.4 kbit -
  is about two fifths of the startup keyframe before any coefficient is coded. That is a large
  part of why H.264 is 3 dB ahead at the same startup byte count; intra prediction and
  deblocking are the rest. An adaptive binary arithmetic coder with a few contexts, and an
  implicit block order in keyframes, are the standard next steps. It complicates the budget
  accounting, because record sizes stop being independent, so the two-pass selection would
  need a size estimate and a trim-and-retry.
- **Larger warp mesh.** The mesh covers chin to forehead. Hair, ears, neck and shoulders move
  with the head but are left to block motion compensation; points extrapolated from the jaw
  and the face box would carry them for free.
- **Finer warp decisions.** The warp is committed or refused for a whole packet. Deciding per
  face, or per triangle with a small mask, would keep a good face warping while a hand covers
  another.
- **Half-pixel motion.** Whole-pixel vectors leave a lot on the table for slow head motion.
  A bilinear half-pel predictor is cheap and is usually worth 1-2 dB on its own.
- **Overlapped or larger partitions.** One vector per 8x8 block is a lot of vectors for a rigid
  head that the warp did not cover. Coding a 16x16 vector with an optional per-8x8 refinement
  would cut the vector count fourfold.
- **In-loop deblocking.** At quantiser 31 the background shows block edges, and those edges are
  then fed back in as the motion and warp reference, so they persist.
- **Tune `rd_lambda` properly.** It is currently a single constant at 2.0, picked because it
  behaved well. The principled value is the slope of the operating point, which could be
  estimated from the previous packet's marginal candidate.

## Geometry

- **Ride in the audio packets.** At 15 Hz a moving face's geometry packets cost about as much
  again in IP, UDP and RTP headers as in landmarks. Speech already sends a packet every 60 ms;
  carrying the geometry in it would remove those headers entirely.
- **Tune the filter on real video.** The one euro filter's cut-offs were chosen, not measured.
  A recorded clip with its landmark error and geometry rate at a few settings would settle
  them.
- **Track identity.** Faces are paired by detector order, so if two people swap order the next
  landmark delta is large until the following keyframe, and the warp pairs the wrong faces for
  a packet. A persistent track identifier would fix both.

## Baseline

- **Real timestamps.** H.264 frames are stamped at an even 15 frames a second, so a camera
  delivering fewer leaves the baseline spending less than its 52 kbit/s. Stamping each frame
  with its capture time, if the encoder's rate control honours it, would keep the budgets
  equal in low light.
- **Several regions.** Media Foundation takes one region of interest per frame, so several
  faces share their union. A weighted quantiser map, where an encoder accepts one, would give
  H.264 the same three-level importance low-ban has.
- **Keypoint synthesis as a third receiver.** Keypoint-driven generative codecs (face-vid2vid,
  Gemino) are where the landmark idea has been taken furthest. They need a neural runtime this
  project does not have, but they are the comparison a reader will ask about.

## Measurement

- **Expression, not just landmarks.** Landmark error says whether the points survived, not
  whether a person reading the picture would see a smile. Expression or action unit agreement
  between source and received pictures, or a small viewing study, would test the premise
  directly.
- **No timing numbers anywhere.** Nothing measures how long an encode takes. The warp, motion
  search, two received-picture landmark runs and an H.264 round trip happen every frame
  without a budget.

## Application

- Only the first camera is used, and only 640x480 YUY2 is requested.
- The GDI fonts and pen in `main.cpp` are created once and never released. Harmless at process
  lifetime, but it is the kind of thing that gets copied into somewhere it matters.
