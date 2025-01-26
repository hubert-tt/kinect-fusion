#include <mutex>
#include <vector>

#include <libfreenect/libfreenect.hpp>

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
    KinectFrameGetter(freenect_context *_ctx, int _index);

    // Do not call directly, even in child
    void VideoCallback(void *_rgb, uint32_t timestamp) override;

    // Do not call directly, even in child
    void DepthCallback(void *_depth, uint32_t timestamp) override;

    bool get_rgb(std::vector<uint8_t> &buffer) override;

    bool get_depth(std::vector<uint16_t> &buffer) override;

  private:
    std::mutex rgb_mutex;
    std::mutex depth_mutex;
    std::vector<uint8_t> buffer_video;
    std::vector<uint16_t> buffer_depth;
    bool new_rgb_frame;
    bool new_depth_frame;
};

} // namespace kf
