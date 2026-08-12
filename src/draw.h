// Anti-aliased vector drawing over an 8 bit image. No Windows dependency, so the analysis
// thread can compose the whole processed frame before it reaches the window.
#pragma once

#include <vector>

#include "image.h"

void draw_line(image_u8& img, float x0, float y0, float x1, float y1, uint8_t colour);
void draw_rect(image_u8& img, const rect_i& r, uint8_t colour);

// Connects the 68 landmark points into the usual facial feature outlines.
void draw_face(image_u8& img, const std::vector<point_i>& parts, uint8_t colour);
