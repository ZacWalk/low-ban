"""Vector source for low-ban's face-on-a-video-call icon."""

import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from iconlib import Canvas, generate

ROOT = Path(__file__).resolve().parents[1]


def draw(canvas: Canvas) -> None:
    """A recognisable face and shoulders, with separate microphone and video call controls."""
    ink, skin, white = (32, 43, 58), (246, 194, 151), (255, 255, 255)
    canvas.circle(50, 40, 38, ink)
    canvas.circle(50, 40, 35, (225, 243, 238))
    canvas.polygon(((23, 61), (32, 55), (43, 52), (57, 52), (68, 55), (77, 61),
                    (70, 68), (60, 73), (40, 73), (30, 68)), (247, 178, 69))
    canvas.rect(45, 44, 10, 14, skin)
    canvas.polygon(((41, 52), (50, 58), (45, 64), (37, 55)), (230, 86, 83))
    canvas.polygon(((59, 52), (50, 58), (55, 64), (63, 55)), (230, 86, 83))
    canvas.circle(37, 35, 3, skin)
    canvas.circle(63, 35, 3, skin)
    canvas.polygon([(50 + 13 * math.cos(i * math.pi / 16),
                     33 + 17 * math.sin(i * math.pi / 16)) for i in range(32)], skin)
    canvas.polygon(((37, 31), (37, 22), (40, 16), (47, 13), (55, 15), (61, 19),
                    (64, 29), (59, 27), (55, 25), (49, 27), (43, 25), (40, 31)), ink)
    canvas.circle(45, 34, 1.4, ink)
    canvas.circle(55, 34, 1.4, ink)
    canvas.line(((46, 43), (50, 45), (54, 43)), 1.7, ink)
    for x, colour in ((35, (24, 173, 165)), (65, (240, 93, 88))):
        canvas.circle(x, 86, 12, ink)
        canvas.circle(x, 86, 10, colour)
    canvas.line(((35, 81), (35, 86)), 4, white)
    canvas.line(((30, 85), (30, 88), (32, 91), (38, 91), (40, 88), (40, 85)), 1.5, white)
    canvas.line(((35, 91), (35, 94)), 1.5, white)
    canvas.rect(59, 82, 9, 8, white)
    canvas.polygon(((68, 84), (72, 82), (72, 90), (68, 88)), white)


if __name__ == "__main__":
    raise SystemExit(generate(draw, ROOT / "src/app.ico"))
