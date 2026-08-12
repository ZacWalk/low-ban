# todo

Further work, roughly in the order that would move the demo the most. See
[design.md](design.md) for what is already there and why.

## Known gaps

### The encoder assumes every packet lands

`temporal_transform_stream` advances its reference frame at the end of `encode`, on the
assumption that the packet it just produced will be decoded. Nothing enforces that. In this
process the two calls are always paired, but a rejected or dropped packet would leave the two
sides predicting from different pictures and the error would never resolve on its own.

A real transport needs either an acknowledgement before the reference advances, or periodic
intra refresh so drift has a bounded lifetime. The latter is cheaper and is what conferencing
codecs actually do.

### The motion predictor is floating point

`transform_codec::predict_block` runs a float DCT at both ends. Identical in one process, and
identical between the Debug and Release builds only by luck; the self test already sees a
0.5 dB difference between configurations on the moving-subject sequence, which is exactly this
effect showing up through the mode decision. A deployable format has to specify the predictor
in integer arithmetic.

### Luma only

Neither codec spends a byte on colour. A grey call is a strange thing to demo as a
communication tool. Coarse chroma - say one 16x16 DC per block pair - would cost maybe 10% more
and would change the impression of both panels far more than another dB of luma PSNR. It
touches the render path too: panels 2 and 3 are 8-bit palette DIBs today.

### Frame sizes that are not a multiple of the patch size

`patch_autoencoder::reconstruct_codes` leaves any partial right column or bottom row black.
640x480 divides exactly by 20 so nothing shows it, but a different capture size would.

## Traditional codec

- **Half-pixel motion.** Whole-pixel vectors leave a lot on the table for slow head motion,
  which is most of what a webcam sees. A bilinear half-pel predictor is cheap and is usually
  worth 1-2 dB on its own.
- **Overlapped or larger partitions.** One vector per 8x8 block is a lot of vectors for a rigid
  head. Coding a 16x16 vector with an optional per-8x8 refinement would cut the vector count
  fourfold on the parts of the frame that move together.
- **In-loop deblocking.** At quantiser 31 the background shows block edges, and those edges are
  then fed back in as the motion reference, so they persist. A deblocking filter applied before
  the frame becomes a reference would help both quality and prediction.
- **Entropy coding.** Records are byte-aligned varints and run/level nibbles. An adaptive binary
  arithmetic coder with a few contexts is the standard next step and is typically worth 15-25%.
  It complicates the budget accounting, because record sizes stop being independent, so the
  two-pass selection would need a size estimate and a trim-and-retry.
- **Tune `rd_lambda` properly.** It is currently a single constant at 2.0, picked because it
  behaved well. The principled value is the slope of the operating point, which could be
  estimated from the previous packet's marginal candidate.

## Learned codec

- **Freeze the model, or budget its updates.** The in-process encoder and decoder share one
  changing weight set. Until the model is frozen at both ends, or weight updates are serialized
  and charged to the bitrate, panel 3 is not a codec - it is a codec-shaped demonstration. This
  is the single biggest honesty gap in the project.
- **Pretrain on faces, then adapt.** Online training from a cold start means the first seconds
  are unusable and the model only ever knows one room. Shipping trained weights and letting the
  online loop fine-tune would give a fair comparison and a much better first impression.
- **Conditional latent prediction.** Delta-coding the latent is not the same as predicting it.
  A small predictor from the previous latent, with only the residual coded, is where learned
  codecs actually win, and it is where the current design is least like one.
- **Progressive codes.** Train with random truncation of the code vector so a prefix is
  meaningful on its own, then send more code units for face patches than for background ones.
  That is the neural analogue of the region-of-interest quantiser, and it would let importance
  buy resolution rather than just scheduling order.
- **Learn from the landmarks, not just near them.** Landmark positions are currently only a
  weight on the loss. Conditioning the decoder on the geometry it already receives would let it
  spend its three numbers on appearance rather than on where the mouth is.

## Measurement

- **`evaluate` only codes keyframes.** Every temporal property - motion compensation, packet
  scheduling, the starvation deadline - is invisible to it and is covered only by the sequences
  in `src/selftest.cpp`. A short recorded clip in `samples/` and a sequence mode for `evaluate`
  would make temporal changes measurable rather than argued.
- **PSNR is the wrong metric for this.** The whole premise is that error around an eye matters
  more than error on a wall, and importance-weighted PSNR only half captures that. A structural
  metric, or a landmark-position error measured on the reconstruction, would say more about
  whether expression survived.
- **No timing numbers anywhere.** Nothing measures how long an encode takes. Motion search and
  two full reconstructions per packet were added without a budget.

## Application

- Multiple faces are paired by detector order, so if two people swap order the next landmark
  delta is large until the following keyframe. A persistent track identifier would fix it.
- Only the first camera is used, and only 640x480 YUY2 is requested.
- The GDI fonts and pen in `main.cpp` are created once and never released. Harmless at process
  lifetime, but it is the kind of thing that gets copied into somewhere it matters.
