#include <array>
#include <limits>
#include <vector>

#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace kf
{

constexpr short max_short = std::numeric_limits<short>::max();
constexpr short min_short = std::numeric_limits<short>::min();
constexpr float normalize_short = 1.0f / max_short;

constexpr int voxel_grid_size = 256;
constexpr float voxel_size = 4.0f;
constexpr float truncation_distance = voxel_size * 10;
static constexpr size_t levels = 3;

struct FrameData
{

    size_t max_width;
    size_t max_height;

    std::array<std::vector<uint16_t>, levels> depth_levels;
    std::array<std::vector<uint16_t>, levels> smoothed_depth_levels;
    std::array<std::vector<uint8_t>, levels> color_levels;

    std::array<cv::Mat, levels> vertex_levels;
    std::array<cv::Mat, levels> normal_levels;

    explicit FrameData(size_t width, size_t height) : max_width(width), max_height(height)
    {
        for (size_t i = 0; i < levels; ++i)
        {
            // Divide by 2^i for each level
            size_t current_width = width / (1 << i);
            size_t current_height = height / (1 << i);

            // Preallocate memory for depth and smoothed depth
            depth_levels[i].resize(current_width * current_height);
            smoothed_depth_levels[i].resize(current_width * current_height);

            // Preallocate memory for color (RGB)
            color_levels[i].resize(current_width * current_height * 3);

            // Preallocate vertex and normal matrices for each level
            vertex_levels[i] = cv::Mat(current_height, current_width, CV_32FC3); // 3 channels for vertex (x, y, z)
            normal_levels[i] = cv::Mat(current_height, current_width, CV_32FC3); // 3 channels for normal (nx, ny, nz)
        }
    }

    // Copy mechanisms disabled as it is too costly with such a big volumes of data
    FrameData(const FrameData &) = delete;
    FrameData &operator=(const FrameData &other) = delete;

    FrameData(FrameData &&data) noexcept
        : depth_levels(std::move(data.depth_levels)), smoothed_depth_levels(std::move(data.smoothed_depth_levels)),
          color_levels(std::move(data.color_levels)), vertex_levels(std::move(data.vertex_levels)),
          normal_levels(std::move(data.normal_levels))
    {
    }

    FrameData &operator=(FrameData &&data) noexcept
    {
        depth_levels = std::move(data.depth_levels);
        smoothed_depth_levels = std::move(data.smoothed_depth_levels);
        color_levels = std::move(data.color_levels);
        vertex_levels = std::move(data.vertex_levels);
        normal_levels = std::move(data.normal_levels);
        return *this;
    }
};

// struct ModelData
// {

//     size_t max_width;
//     size_t max_height;

//     std::array<cv::Mat, levels> vertex_levels;
//     std::array<cv::Mat, levels> normal_levels;
//     std::array<cv::Mat, levels> color_levels;

//     explicit ModelData(size_t width, size_t height) : max_width(width), max_height(height)
//     {
//         for (size_t i = 0; i < levels; ++i)
//         {
//             // Divide by 2^i for each level
//             size_t current_width = width / (1 << i);
//             size_t current_height = height / (1 << i);

//             // Preallocate vertex and normal matrices for each level
//             vertex_levels[i] = cv::Mat(current_height, current_width, CV_32FC3); // 3 channels for vertex (x, y, z)
//             normal_levels[i] = cv::Mat(current_height, current_width, CV_32FC3); // 3 channels for normal (nx, ny,
//             nz) color_levels[i] = cv::Mat(current_height, current_width, CV_8UC3);
//         }
//     }

//     // Copy mechanisms disabled as it is too costly with such a big volumes of data
//     ModelData(const ModelData &) = delete;
//     ModelData &operator=(const ModelData &other) = delete;

//     ModelData(ModelData &&data) noexcept
//         : color_levels(std::move(data.color_levels)), vertex_levels(std::move(data.vertex_levels)),
//           normal_levels(std::move(data.normal_levels))
//     {
//     }

//     ModelData &operator=(ModelData &&data) noexcept
//     {
//         color_levels = std::move(data.color_levels);
//         vertex_levels = std::move(data.vertex_levels);
//         normal_levels = std::move(data.normal_levels);
//         return *this;
//     }
// };

struct CameraIntrinsics
{
    float f;  // Focal length, we use the same one for X and Y
    float cx; // Principal point coordinate X
    float cy; // Principal point coordinate Y
};

struct VolumeData
{
    cv::Mat tsdf_volume;   // TSDF volume (Truncated Signed Distance Function) values
    cv::Mat color_volume;  // Stores the color information for each voxel
    cv::Vec3i volume_size; // Size of the volume in voxels (width, height, depth)
    float voxel_scale;     // Physical size of each voxel in millimeters

    // Constructor
    VolumeData(const cv::Vec3i &_volume_size, const float _voxel_scale)
        : tsdf_volume(_volume_size[1] * _volume_size[2], _volume_size[0], CV_16SC2),
          color_volume(_volume_size[1] * _volume_size[2], _volume_size[0], CV_8UC3), volume_size(_volume_size),
          voxel_scale(_voxel_scale)
    {
        tsdf_volume.setTo(cv::Scalar(0, 0));                                   // Initialize TSDF volume to zero
        color_volume.setTo(cv::Scalar(0.45f * 255, 0.45f * 255, 0.45f * 255)); // Initialize color volume to light gray
    }
};

} // namespace kf
