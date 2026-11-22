"""Learns low-ban's expression prior: where a face moves in ways its five landmarks do not say.

low-ban sends five landmarks a face - the eye centres, the tip of the nose and the corners of the
mouth - and the receiver warps the face it holds to follow them. Whatever a face does that those
five points do not predict - a mouth opening, a brow rising, an eye closing - can only reach the
receiver as picture, so that is where the picture's bits should go. This measures where it is.

A teacher that sees more than five points does the measuring: dlib's 68-point regression-tree
model, run here in numpy over Labeled Faces in the Wild. Every face is aligned by its five points
alone on libmug's template, the 112-pixel square low-ban places the prior in, and a linear model
predicts all 68 points from the five. What it cannot predict, taken within each person's own
pictures so that the shape of a face does not count as expression, is weighed by the picture's
gradient where it happens - the change a feature's move makes to the picture, which is what the
codec has to send - and spread over the square: the prior, written as a table of cells for
low-ban to compile in.

    python tools/learn_expression_prior.py [--cache tmp/expression-prior] [--output src/expression-prior.cpp]

Needs numpy and Pillow (tools/requirements.txt). LFW (180 MB) and dlib's model (64 MB) are fetched
into the cache at their pins on first use. dlib's model was trained on iBUG 300-W, whose licence
excludes commercial use; it only measures here, and nothing of it ships but the table.
"""

import argparse
import bz2
import hashlib
import io
import math
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

LFW_URL = "https://ndownloader.figshare.com/files/5976018"
LFW_SHA256 = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
DLIB_URL = ("https://github.com/davisking/dlib-models/raw/{commit}/shape_predictor_68_face_landmarks.dat.bz2")
DLIB_COMMIT = "6a82ae1d20e9f5ca2a68156344d927da4d4d45e7"
DLIB_SHA256 = "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"  # of the decompressed .dat

# libmug's template: ArcFace's five landmarks in the 112-pixel square SFace crops (libmug/detect.h, template_fit).
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                     [70.7299, 92.2041]])
EXTENT = 112.0
CELLS = 28

# dlib's 68 points, by feature. The jaw line frames a face rather than expressing anything, and is the
# teacher's least certain stretch, so it is left out of the prior as low-ban's old importance map left it out.
GROUPS = {"brows": range(17, 27), "nose": range(27, 36), "eyes": range(36, 48), "mouth": range(48, 68)}
EXPRESSIVE = np.arange(17, 68)

# The spread each point's motion is given over the square: about the height of an eye.
KERNEL = 4.0

# What the teacher's pass computes; a change to it moves this, so its saved results are not reused.
TEACHER_VERSION = 2


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, path, expected=None):
    if path.exists() and (expected is None or sha256(path) == expected):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}", flush=True)
    with urllib.request.urlopen(url) as response, open(path, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    if expected is not None and sha256(path) != expected:
        raise SystemExit(f"{path} does not match its pin")
    return path


def dlib_model(cache):
    """The 68-point model, decompressed into the cache and checked against its pin."""
    dat = cache / "shape_predictor_68_face_landmarks.dat"
    if dat.exists() and sha256(dat) == DLIB_SHA256:
        return dat
    packed = fetch(DLIB_URL.format(commit=DLIB_COMMIT), cache / "shape_predictor_68_face_landmarks.dat.bz2")
    dat.write_bytes(bz2.decompress(packed.read_bytes()))
    if sha256(dat) != DLIB_SHA256:
        raise SystemExit(f"{dat} does not match its pin")
    return dat


# ---- the teacher ------------------------------------------------------------------------------------------------

def read_model(path, cache):
    """dlib's serialisation, as low-ban's reader read it: an integer is a control byte - its low four bits the
    byte count, 0x80 negative - and little-endian bytes; a float is a mantissa and an exponent."""
    saved = cache / "shape_predictor_68.npz"
    if saved.exists():
        return dict(np.load(saved))

    data = path.read_bytes()
    pos = 0

    def integer():
        nonlocal pos
        control = data[pos]
        count = control & 0x0F
        value = int.from_bytes(data[pos + 1:pos + 1 + count], "little")
        pos += 1 + count
        return -value if control & 0x80 else value

    def real():
        mantissa = integer()
        exponent = integer()
        return math.ldexp(mantissa, exponent) if exponent < 32000 else math.nan

    def column():
        rows, cols = abs(integer()), abs(integer())
        return [real() for _ in range(rows * cols)]

    if integer() != 1:
        raise SystemExit("unsupported model version")
    initial = np.array(column(), np.float32)
    cascades = integer()
    idx1, idx2, thresh, leaves = [], [], [], []
    for c in range(cascades):
        trees = integer()
        a, b, t, leaf = [], [], [], []
        for _ in range(trees):
            splits = integer()
            row = [(integer(), integer(), real()) for _ in range(splits)]
            a.append([s[0] for s in row])
            b.append([s[1] for s in row])
            t.append([s[2] for s in row])
            leaf.append([column() for _ in range(integer())])
        idx1.append(a)
        idx2.append(b)
        thresh.append(t)
        leaves.append(leaf)
        print(f"  read cascade {c + 1} of {cascades}", flush=True)
    anchors = [[integer() for _ in range(integer())] for _ in range(integer())]
    deltas = [[(real(), real()) for _ in range(integer())] for _ in range(integer())]

    model = {"initial": initial, "idx1": np.array(idx1, np.int32), "idx2": np.array(idx2, np.int32),
             "thresh": np.array(thresh, np.float32), "leaves": np.array(leaves, np.float32),
             "anchors": np.array(anchors, np.int32), "deltas": np.array(deltas, np.float32)}
    np.savez(saved, **model)
    return model


def predict(model, images, boxes):
    """dlib's regression-tree cascade over a batch: images (n, h, w) of luma, boxes (n, 4) left, top, right, bottom.
    Returns (n, 68, 2) points."""
    n = len(images)
    initial = model["initial"].reshape(68, 2)
    shape = np.repeat(initial[None].astype(np.float64), n, axis=0)
    origin = boxes[:, None, :2]
    span = (boxes[:, 2:] - boxes[:, :2])[:, None, :]
    height, width = images.shape[1:]
    rows = np.arange(n)[:, None]
    mean = initial - initial.mean(axis=0)
    norm = (mean ** 2).sum()

    for c in range(len(model["anchors"])):
        # The similarity's linear part, from the mean shape onto the current one.
        centred = shape - shape.mean(axis=1, keepdims=True)
        dot = (mean[None] * centred).sum(axis=(1, 2)) / norm
        cross = (mean[None, :, 0] * centred[:, :, 1] - mean[None, :, 1] * centred[:, :, 0]).sum(axis=1) / norm
        d = model["deltas"][c]
        offset = np.stack([dot[:, None] * d[None, :, 0] - cross[:, None] * d[None, :, 1],
                           cross[:, None] * d[None, :, 0] + dot[:, None] * d[None, :, 1]], axis=2)
        at = np.floor(origin + (shape[:, model["anchors"][c]] + offset) * span + 0.5).astype(np.int64)
        inside = (at[..., 0] >= 0) & (at[..., 0] < width) & (at[..., 1] >= 0) & (at[..., 1] < height)
        features = np.where(inside, images[rows, at[..., 1].clip(0, height - 1), at[..., 0].clip(0, width - 1)],
                            0).astype(np.int32)

        trees, splits = model["idx1"].shape[1:]
        tree = np.arange(trees)[None, :]
        node = np.zeros((n, trees), np.int64)
        for _ in range((splits + 1).bit_length() - 1):
            a = model["idx1"][c][tree, node]
            b = model["idx2"][c][tree, node]
            diff = features[rows, a] - features[rows, b]
            node = 2 * node + np.where(diff > model["thresh"][c][tree, node], 1, 2)
        leaf = node - splits
        for start in range(0, n, 128):
            part = slice(start, start + 128)
            shape[part] += model["leaves"][c][tree, leaf[part]].sum(axis=1).reshape(-1, 68, 2)

    return origin + shape * span


def box_from_points(model, points):
    """The box the model implies for its own points: image = origin + span * mean shape, per axis."""
    initial = model["initial"].reshape(68, 2).astype(np.float64)
    m = initial - initial.mean(axis=0)
    p = points - points.mean(axis=1, keepdims=True)
    span = (m[None] * p).sum(axis=1) / (m ** 2).sum(axis=0)[None]
    origin = points.mean(axis=1) - span * initial.mean(axis=0)
    return np.concatenate([origin, origin + span], axis=1)


def lfw_faces(model, lfw, limit):
    """The teacher's 68 points on every LFW picture whose face it settles on, with the person's name and the
    picture's luma gradient at each point."""
    names, points, gradients = [], [], []
    batch_images, batch_names = [], []

    def flush():
        if not batch_images:
            return
        images = np.stack(batch_images)
        boxes = np.tile(np.array([70.0, 80.0, 180.0, 190.0]), (len(images), 1))
        for _ in range(4):
            found = predict(model, images, boxes)
            boxes = box_from_points(model, found)
        eyes = np.stack([found[:, 36:42].mean(axis=1), found[:, 42:48].mean(axis=1)], axis=1)
        spacing = np.linalg.norm(eyes[:, 1] - eyes[:, 0], axis=1)
        centre = eyes.mean(axis=1)
        settled = (spacing > 25) & (spacing < 80) & (np.abs(centre - [125, 115]).max(axis=1) < 40)

        # The gradient of the luma, smoothed over 3x3 so it is the feature's edge rather than the camera's noise.
        padded = np.pad(images.astype(np.float32), ((0, 0), (1, 1), (1, 1)), mode="edge")
        smooth = sum(padded[:, dy:dy + images.shape[1], dx:dx + images.shape[2]] for dy in range(3)
                     for dx in range(3)) / 9.0
        gy, gx = np.gradient(smooth, axis=(1, 2))
        at = np.rint(found).astype(np.int64)
        x = at[..., 0].clip(0, images.shape[2] - 1)
        y = at[..., 1].clip(0, images.shape[1] - 1)
        rows = np.arange(len(images))[:, None]
        slope = np.stack([gx[rows, y, x], gy[rows, y, x]], axis=2)

        for keep, name, p, g in zip(settled, batch_names, found, slope):
            if keep:
                names.append(name)
                points.append(p)
                gradients.append(g)
        batch_images.clear()
        batch_names.clear()

    with tarfile.open(lfw, "r:gz") as archive:
        for member in archive:
            if not member.name.endswith(".jpg"):
                continue
            image = Image.open(io.BytesIO(archive.extractfile(member).read())).convert("L")
            batch_images.append(np.asarray(image, np.uint8))
            batch_names.append(member.name.split("/")[-2])
            if len(batch_images) == 512:
                flush()
                print(f"  {len(points)} faces", flush=True)
            if limit and len(points) + len(batch_images) >= limit:
                break
        flush()
    return names, np.array(points), np.array(gradients)


# ---- the prior --------------------------------------------------------------------------------------------------

def fit_template(five):
    """Per face, the least-squares similarity carrying its five points onto the template, as libmug fits it:
    (n, 2, 2) linear parts and (n, 2) translations."""
    target = TEMPLATE - TEMPLATE.mean(axis=0)
    p = five - five.mean(axis=1, keepdims=True)
    spread = (p ** 2).sum(axis=(1, 2))
    a = (target[None, :, 0] * p[:, :, 0] + target[None, :, 1] * p[:, :, 1]).sum(axis=1) / spread
    b = (target[None, :, 1] * p[:, :, 0] - target[None, :, 0] * p[:, :, 1]).sum(axis=1) / spread
    linear = np.stack([np.stack([a, -b], axis=1), np.stack([b, a], axis=1)], axis=1)
    shift = TEMPLATE.mean(axis=0)[None] - np.einsum("nij,nj->ni", linear, five.mean(axis=1))
    return linear, shift


def learn(names, points, gradients):
    five = np.stack([points[:, 36:42].mean(axis=1), points[:, 42:48].mean(axis=1), points[:, 30], points[:, 48],
                     points[:, 54]], axis=1)
    linear, shift = fit_template(five)
    aligned = np.einsum("nij,npj->npi", linear, points) + shift[:, None]
    aligned_five = np.einsum("nij,npj->npi", linear, five) + shift[:, None]

    # The picture's gradient in the square's coordinates: the similarity's linear part is a scale s times a turn,
    # and a gradient turns with it and shrinks by s.
    scale_squared = linear[:, 0, 0] ** 2 + linear[:, 1, 0] ** 2
    slope = np.einsum("nij,npj->npi", linear, gradients) / scale_squared[:, None, None]

    # What the five points predict of the 68: a ridge regression on the aligned points.
    x = aligned_five.reshape(len(points), -1)
    y = aligned.reshape(len(points), -1)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    weights = np.linalg.solve(xc.T @ xc + 1e-3 * len(x) * np.eye(x.shape[1]), xc.T @ yc)
    residual = (yc - xc @ weights).reshape(len(points), 68, 2)

    # Within each person, so a face's own shape is not counted: pooled over people with several pictures. A point
    # that moves by d where the picture's gradient is g changes the picture by about g.d, so what the codec has to
    # send for it is the square of that, not of d: an eyelid barely moves but changes a lot of picture.
    people = defaultdict(list)
    for i, name in enumerate(names):
        people[name].append(i)
    motion = np.zeros(68)
    variation = np.zeros(68)
    change = np.zeros(68)
    degrees = 0
    for members in people.values():
        if len(members) < 2:
            continue
        r = residual[members] - residual[members].mean(axis=0)
        motion += (r ** 2).sum(axis=(0, 2))
        variation += ((aligned[members] - aligned[members].mean(axis=0)) ** 2).sum(axis=(0, 2))
        change += ((r * slope[members]).sum(axis=2) ** 2).sum(axis=0)
        degrees += len(members) - 1
    motion /= degrees
    variation /= degrees
    change /= degrees

    # Each expressive point's unexplained change spread over where that point lies: a Gaussian with the point's own
    # spread across faces, widened by the kernel.
    centres = (np.arange(CELLS) + 0.5) * EXTENT / CELLS
    gx, gy = np.meshgrid(centres, centres)
    cells = np.stack([gx, gy], axis=2)
    grid = np.zeros((CELLS, CELLS))
    for i in EXPRESSIVE:
        mu = aligned[:, i].mean(axis=0)
        sigma = np.cov(aligned[:, i].T) + KERNEL ** 2 * np.eye(2)
        d = cells - mu
        mahalanobis = np.einsum("...i,ij,...j->...", d, np.linalg.inv(sigma), d)
        grid += change[i] * np.exp(-0.5 * mahalanobis) / (2 * math.pi * math.sqrt(np.linalg.det(sigma)))

    def shares(energy):
        return {g: energy[list(r)].sum() / energy[EXPRESSIVE].sum() for g, r in GROUPS.items()}

    report = {"faces": len(points), "people": sum(1 for m in people.values() if len(m) >= 2),
              "explained": 1.0 - motion[EXPRESSIVE].sum() / variation[EXPRESSIVE].sum(),
              "motion": shares(motion), "groups": shares(change), "mean": aligned.mean(axis=0)}
    return grid / grid.max(), report


def write_table(grid, report, path):
    values = np.clip(np.rint(grid * 255), 0, 255).astype(int)
    shares = ", ".join(f"{g} {100 * s:.0f}%" for g, s in report["groups"].items())
    lines = [
        "// The expression prior, generated by tools/learn_expression_prior.py; do not edit.",
        "//",
        f"// {report['faces']} faces of Labeled Faces in the Wild, {report['people']} people with several pictures, landmarked",
        "// by dlib's 68-point model and aligned on libmug's template by five of those points. The five explain",
        f"// {100 * report['explained']:.0f}% of how a person's 51 expressive points move between pictures; the picture change",
        f"// the rest makes is {shares}.",
        '#include "face.h"',
        "",
        "// clang-format off",
        "const uint8_t expression_prior[expression_prior_cells * expression_prior_cells] = {",
    ]
    for row in values:
        lines.append("\t" + "".join(f"{v:3d}," for v in row))
    lines += ["};", "// clang-format on", ""]
    path.write_text("\n".join(lines), newline="\n")


def write_preview(grid, report, path, scale=8):
    """The prior over the template square, with the mean 68 points, for looking at."""
    heat = np.kron(grid, np.ones((scale, scale)))
    rgb = np.stack([heat * 255, heat ** 2 * 200, (1 - heat) * 60], axis=2).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    pixels = image.load()
    factor = CELLS * scale / EXTENT
    for x, y in report["mean"]:
        cx, cy = int(x * factor), int(y * factor)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if 0 <= cx + dx < image.width and 0 <= cy + dy < image.height:
                    pixels[cx + dx, cy + dy] = (255, 255, 255)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/expression-prior")
    parser.add_argument("--output", type=Path, default=ROOT / "src/expression-prior.cpp")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many faces")
    args = parser.parse_args()

    # The teacher's pass is the slow part, so what it found is kept; a change to it renames the file.
    found = args.cache / f"lfw-teacher-{TEACHER_VERSION}-{args.limit}.npz"
    if found.exists():
        saved = np.load(found)
        names, points, gradients = list(saved["names"]), saved["points"], saved["gradients"]
    else:
        lfw = fetch(LFW_URL, args.cache / "lfw.tgz", LFW_SHA256)
        model = read_model(dlib_model(args.cache), args.cache)
        names, points, gradients = lfw_faces(model, lfw, args.limit)
        np.savez(found, names=np.array(names), points=points, gradients=gradients)

    grid, report = learn(names, points, gradients)
    write_table(grid, report, args.output)
    write_preview(grid, report, args.cache / "expression-prior.png")

    print(f"{report['faces']} faces, {report['people']} people with several pictures")
    print(f"the five points explain {100 * report['explained']:.1f}% of how the expressive points move")
    print("what they leave:        motion   picture change")
    for group in GROUPS:
        print(f"  {group:6}              {100 * report['motion'][group]:5.1f}%   {100 * report['groups'][group]:5.1f}%")
    print(f"wrote {args.output} and {args.cache / 'expression-prior.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
