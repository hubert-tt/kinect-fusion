#include "FrameGetter.hh"

#include <libfreenect/libfreenect.hpp>
#include <mutex>

namespace kf
{

KinectFrameGetter::KinectFrameGetter(freenect_context *_ctx, int _index)
    : Freenect::FreenectDevice(_ctx, _index),
      buffer_video(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes),
      buffer_depth(freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2),
      new_rgb_frame(false), new_depth_frame(false)
{
    setDepthFormat(FREENECT_DEPTH_REGISTERED);
}

// Do not call directly, even in child
void KinectFrameGetter::VideoCallback(void *_rgb, uint32_t timestamp)
{
    std::lock_guard<std::mutex> lock(rgb_mutex);

    uint8_t *rgb = static_cast<uint8_t *>(_rgb);
    copy(rgb, rgb + getVideoBufferSize(), buffer_video.begin());
    new_rgb_frame = true;
}

// Do not call directly, even in child
void KinectFrameGetter::DepthCallback(void *_depth, uint32_t timestamp)
{
    std::lock_guard<std::mutex> lock(depth_mutex);

    uint16_t *depth = static_cast<uint16_t *>(_depth);
    copy(depth, depth + getDepthBufferSize() / 2, buffer_depth.begin());
    new_depth_frame = true;
}

bool KinectFrameGetter::get_rgb(std::vector<uint8_t> &buffer)
{
    std::lock_guard<std::mutex> lock(rgb_mutex);

    if (!new_rgb_frame)
        return false;

    buffer.swap(buffer_video);
    new_rgb_frame = false;

    return true;
}

bool KinectFrameGetter::get_depth(std::vector<uint16_t> &buffer)
{
    std::lock_guard<std::mutex> lock(depth_mutex);

    if (!new_depth_frame)
        return false;

    buffer.swap(buffer_depth);
    new_depth_frame = false;

    return true;
}

} // namespace kf
