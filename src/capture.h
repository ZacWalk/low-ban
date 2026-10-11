// Webcam capture through Media Foundation. YUY2 is requested because it is what webcams
// natively produce and it hands us chroma for free, which the face detector uses.
#pragma once

#include <string>

#include "image.h"

// Process-wide Media Foundation lifetime. One instance must outlive every webcam.
class media_foundation
{
public:
	media_foundation();
	~media_foundation();

	media_foundation(const media_foundation&) = delete;
	media_foundation& operator=(const media_foundation&) = delete;

private:
	bool started = false;
};

class webcam
{
public:
	webcam() = default;
	~webcam();

	webcam(const webcam&) = delete;
	webcam& operator=(const webcam&) = delete;

	bool open(int width, int height, std::string& error);
	void close();
	bool is_open() const;

	// Blocks until the next frame is available. Returns false on a dropped or malformed
	// sample; the caller should simply try again. When the camera has stopped for good -
	// taken by another application, unplugged - it also sets `error` to say why, and every
	// later read would fail the same way.
	bool read(video_frame& frame, std::string& error);

private:
	struct impl;
	impl* state = nullptr;
};
