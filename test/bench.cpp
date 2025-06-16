#include <chrono>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp> // Ensure OpenCV is installed
#include <vector>

constexpr short max_short = std::numeric_limits<short>::max();
constexpr short min_short = std::numeric_limits<short>::min();
constexpr float normalize_short = 1.0f / max_short;

volatile int checksum = 0;

// Define your VolumeData structure
struct VolumeData
{
    cv::Mat tsdf_volume;  // TSDF values stored as cv::Mat of Vec2s (TSDF, weight)
    cv::Mat color_volume; // Color values stored as cv::Mat of Vec3b
    Eigen::Vector3i volume_size;
    float voxel_scale;
};

// Define your CameraIntrinsics structure
struct CameraIntrinsics
{
    float f;  // Focal length
    float cx; // Principal point x
    float cy; // Principal point y
};

// Surface reconstruction function prototype
void surface_reconstruction(VolumeData &volume, const std::vector<uint16_t> &depth, const std::vector<uint8_t> &color,
                            const size_t max_width, const size_t max_height, const CameraIntrinsics &camera,
                            const float truncation_distance, const Eigen::Matrix4f &model_view);

int main()
{
    // Volume dimensions
    const int volume_width = 256;
    const int volume_height = 256;
    const int volume_depth = 256;

    // Image dimensions
    const int image_width = 640;
    const int image_height = 480;

    // Initialize volume data
    VolumeData volume;
    volume.volume_size = Eigen::Vector3i(volume_width, volume_height, volume_depth);
    volume.voxel_scale = 0.01f; // 1 cm per voxel
    volume.tsdf_volume = cv::Mat(volume_height * volume_depth, volume_width, CV_16SC2, cv::Scalar(0, 0));
    volume.color_volume = cv::Mat(volume_height * volume_depth, volume_width, CV_8UC3, cv::Scalar(0, 0, 0));

    // Initialize camera intrinsics
    CameraIntrinsics camera;
    camera.f = 525.0f; // Typical Kinect focal length
    camera.cx = image_width / 2.0f;
    camera.cy = image_height / 2.0f;

    // Initialize depth and color images
    std::vector<uint16_t> depth(image_width * image_height, 1000);   // Constant depth of 1 meter (1000 mm)
    std::vector<uint8_t> color(image_width * image_height * 3, 128); // Grey color (R, G, B all set to 128)

    volume.tsdf_volume.setTo(cv::Vec2s(0, 0));     // Initialize TSDF volume
    volume.color_volume.setTo(cv::Vec3b(0, 0, 0)); // Initialize color volume

    // Initialize model view matrix
    Eigen::Matrix4f model_view = Eigen::Matrix4f::Identity();
    model_view(0, 3) = 0.5f; // Translate x by 0.5 meters
    model_view(1, 3) = 0.5f; // Translate y by 0.5 meters
    model_view(2, 3) = 1.0f; // Translate z by 1 meter

    // Truncation distance
    const float truncation_distance = 0.05f; // 5 cm

    for (size_t i = 0; i < 10; i++)
    {
        // Benchmarking
        std::cout << "Starting benchmark..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Call the surface reconstruction method
        surface_reconstruction(volume, depth, color, image_width, image_height, camera, truncation_distance,
                               model_view);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // Output results
        std::cout << "Benchmark completed in " << elapsed_time << " ms." << std::endl;

        std::cout << "Checksum: " << checksum << std::endl;
    }

    return 0;
}

void surface_reconstruction(VolumeData &volume, const std::vector<uint16_t> &depth, const std::vector<uint8_t> &color,
                            const size_t max_width, const size_t max_height, const CameraIntrinsics &camera,
                            const float truncation_distance, const Eigen::Matrix4f &model_view)
{
    // Extract rotation and translation from the model view matrix
    Eigen::Matrix3f rotation = model_view.block<3, 3>(0, 0);
    Eigen::Vector3f translation = model_view.block<3, 1>(0, 3);

    const int volume_width = volume.volume_size[0];
    const int volume_height = volume.volume_size[1];
    const int volume_depth = volume.volume_size[2];
    const float voxel_scale = volume.voxel_scale;
    const int max_weight = 128;

    // Iterate over the 3D voxel grid
    // #pragma omp parallel for collapse(2)
    for (int z = 0; z < volume_depth; ++z)
    {
        for (int y = 0; y < volume_height; ++y)
        {
            for (int x = 0; x < volume_width; ++x)
            {
                // int y = yz / volume_depth; // Decompose yz into y and z
                // int z = yz % volume_depth;
                // Compute voxel position in world coordinates
                Eigen::Vector3f position((x + 0.5f) * voxel_scale, (y + 0.5f) * voxel_scale, (z + 0.5f) * voxel_scale);

                // Transform to camera coordinates
                Eigen::Vector3f camera_pos = rotation * position + translation;

                if (camera_pos.z() <= 0)
                {
                    continue; // Skip voxels behind the camera
                }

                // Project to 2D image plane
                int u = static_cast<int>(std::round(camera_pos.x() / camera_pos.z() * camera.f + camera.cx));
                int v = static_cast<int>(std::round(camera_pos.y() / camera_pos.z() * camera.f + camera.cy));

                // Check if the projection falls within the image bounds
                if (u < 0 || u >= static_cast<int>(max_width) || v < 0 || v >= static_cast<int>(max_height))
                {
                    continue;
                }

                // Retrieve the depth value from the depth image
                float d = static_cast<float>(depth[v * max_width + u]); // mm
                if (d <= 0)
                {
                    continue;
                }

                // Compute SDF
                float sdf = d - camera_pos.norm();

                if (sdf >= -truncation_distance)
                {
                    float new_tsdf = std::min(1.0f, sdf / truncation_distance);
                    // #pragma omp critical
                    {
                        // Retrieve the current TSDF value and weight
                        auto &voxel = volume.tsdf_volume.at<cv::Vec2s>(z * volume_height + y, x);
                        float current_tsdf = voxel[0] * normalize_short; // voxel[0] is short
                        int current_weight = voxel[1];

                        // Update TSDF using weighted average
                        const int add_weight = 1;
                        const float weight_inv = 1.0f / (current_weight + add_weight);
                        float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) * weight_inv;
                        int new_weight = std::min(current_weight + add_weight, max_weight);
                        short new_value =
                            std::clamp(static_cast<short>(updated_tsdf * max_short), min_short, max_short);

                        voxel[0] = static_cast<short>(new_value);
                        voxel[1] = static_cast<short>(new_weight);

                        checksum += voxel[0];
                        checksum += voxel[1];

                        // Update color if within a valid SDF range
                        if (sdf <= truncation_distance / 2 && sdf >= -truncation_distance / 2)
                        {
                            auto &model_color = volume.color_volume.at<cv::Vec3b>(z * volume_height + y, x);

                            const int color_index = (v * max_width + u) * 3;

                            model_color[0] = static_cast<uint8_t>(
                                (current_weight * model_color[0] + add_weight * color[color_index]) * weight_inv);
                            model_color[1] = static_cast<uint8_t>(
                                (current_weight * model_color[1] + add_weight * color[color_index + 1]) * weight_inv);
                            model_color[2] = static_cast<uint8_t>(
                                (current_weight * model_color[2] + add_weight * color[color_index + 2]) * weight_inv);

                            checksum += model_color[0];
                            checksum += model_color[1];
                            checksum += model_color[2];
                        }
                    }
                }
            }
        }
    }
}
