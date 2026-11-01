// Piecewise-affine warping of a decoded picture onto newer face geometry.
//
// Both ends of the low-ban wire hold the same reference picture and the same transmitted
// landmarks, so both can move the face they already have to where the newest geometry puts it
// without spending a byte. That is what lets geometry run faster than pictures: the receiver
// animates the face between picture packets, and the next packet only has to correct what the
// warp got wrong. Everything here is integer arithmetic, so the two ends agree exactly whatever
// compiler or optimisation settings built them.
#pragma once

#include <vector>

#include "image.h"

struct mesh_triangle
{
	int a = 0;
	int b = 0;
	int c = 0;
};

// The 68 landmarks plus ten points above the brows, so the forehead moves with the face.
std::vector<point_i> face_mesh_points(const std::vector<point_i>& landmarks);

// Delaunay triangulation by Bowyer-Watson with exact integer predicates. Repeated points are
// left out, so closed lips do not produce zero-area triangles, and every triangle is returned
// with a positive signed area. Coordinates are clamped to +/-2048.
std::vector<mesh_triangle> delaunay_triangles(const std::vector<point_i>& points);

// Moves the faces in `reference`, drawn at `from`, to `to`. Each face is triangulated at its
// new position and every pixel inside the mesh is sampled from the matching triangle at the
// old position; everything outside every mesh is copied. Faces are paired by index and left
// alone unless both sides have all 68 points. Returns false, with `out` an exact copy of
// `reference`, when no face moved.
bool warp_faces(const image_u8& reference, const face_geometry& from, const face_geometry& to, image_u8& out);

// Integer division of every coordinate, rounding towards negative infinity. Transmitted
// geometry is on a two pixel grid, so halving it for the coded picture is exact.
face_geometry scale_geometry(const face_geometry& geometry, int divisor);
