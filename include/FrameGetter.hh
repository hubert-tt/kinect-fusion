#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#include <libfreenect/libfreenect.hpp>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace kf // Kinect Fusion abbreviation
{

// Interface
class FrameGetter
{
  public:
    virtual ~FrameGetter() = default;

    virtual bool get_rgb(std::vector<uint8_t> &buffer) = 0;
    virtual bool get_depth(std::vector<uint16_t> &buffer) = 0;
};

class FileFrameGetter : public FrameGetter
{
  public:
    FileFrameGetter(const std::string &rgb_list_path, const std::string &depth_list_path);

    bool get_rgb(std::vector<uint8_t> &buffer) override;

    bool get_depth(std::vector<uint16_t> &buffer) override;

    bool next_frame();

  private:
    bool load_file_list(const std::string &file_path, std::vector<std::string> &file_list);

    std::vector<std::string> rgb_files;
    std::vector<std::string> depth_files;
    size_t current_index;
};

class KinectFrameGetter : public FrameGetter, public Freenect::FreenectDevice
{
  public:
    KinectFrameGetter(freenect_context *_ctx, int _index)
        : Freenect::FreenectDevice(_ctx, _index),
          buffer_video(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes),
          buffer_depth(freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED).bytes / 2),
          new_rgb_frame(false), new_depth_frame(false)
    {
        setDepthFormat(FREENECT_DEPTH_REGISTERED);
    }

    // Do not call directly, even in child
    void VideoCallback(void *_rgb, uint32_t timestamp) override
    {
        std::lock_guard<std::mutex> lock(rgb_mutex);

        uint8_t *rgb = static_cast<uint8_t *>(_rgb);
        copy(rgb, rgb + getVideoBufferSize(), buffer_video.begin());
        new_rgb_frame = true;
    }

    // Do not call directly, even in child
    void DepthCallback(void *_depth, uint32_t timestamp) override
    {
        std::lock_guard<std::mutex> lock(depth_mutex);

        uint16_t *depth = static_cast<uint16_t *>(_depth);
        copy(depth, depth + getDepthBufferSize() / 2, buffer_depth.begin());
        new_depth_frame = true;
    }

    bool get_rgb(std::vector<uint8_t> &buffer) override
    {
        std::lock_guard<std::mutex> lock(rgb_mutex);

        if (!new_rgb_frame)
            return false;

        buffer.swap(buffer_video);
        new_rgb_frame = false;

        return true;
    }

    bool get_depth(std::vector<uint16_t> &buffer) override
    {
        std::lock_guard<std::mutex> lock(depth_mutex);

        if (!new_depth_frame)
            return false;

        buffer.swap(buffer_depth);
        new_depth_frame = false;

        return true;
    }

  private:
    std::mutex rgb_mutex;
    std::mutex depth_mutex;
    std::vector<uint8_t> buffer_video;
    std::vector<uint16_t> buffer_depth;
    bool new_rgb_frame;
    bool new_depth_frame;
};

} // namespace kf
