// Webcam capture through Media Foundation, and recording and replay of short clips for the
// offline evaluation. YUY2 is requested because it is what webcams natively produce and it
// hands us chroma for free, which the face detector uses.
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

// Records frames to an MP4 file for `evaluate` to replay. The caller decides which frames to
// write and they are stamped at even intervals, so a clip always plays back at exactly
// `frames_per_second`. The H.264 inside is at a rate high enough to be a faithful source.
class clip_writer
{
public:
	clip_writer() = default;
	~clip_writer();

	clip_writer(const clip_writer&) = delete;
	clip_writer& operator=(const clip_writer&) = delete;

	bool open(const std::wstring& path, int width, int height, int frames_per_second, std::string& error);
	bool write(const video_frame& frame, std::string& error);

	// Finishes the file; one that is never closed cannot be played back.
	bool close(std::string& error);
	bool is_open() const;
	int frames() const;

private:
	struct impl;
	impl* state = nullptr;
};

// Plays a recorded clip back as captured frames, each with its time in seconds.
class clip_reader
{
public:
	clip_reader() = default;
	~clip_reader();

	clip_reader(const clip_reader&) = delete;
	clip_reader& operator=(const clip_reader&) = delete;

	// Frames come out at `width` x `height` whatever size the clip was recorded at.
	bool open(const std::wstring& path, int width, int height, std::string& error);

	// False at the end of the clip; `error` is set only when something went wrong.
	bool read(video_frame& frame, double& seconds, std::string& error);
	void close();

private:
	struct impl;
	impl* state = nullptr;
};
