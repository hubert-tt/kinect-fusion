#include <chrono>
#include <iostream>
#include <omp.h> // Required for OpenMP
#include <opencv2/opencv.hpp>

struct CameraParameters
{
    float focal_x, focal_y;
    float principal_x, principal_y;
};

// Single-threaded version
void compute_vertex_map_cpu_single(const cv::Mat &depth_map, cv::Mat &vertex_map, float depth_cutoff,
                                   const CameraParameters &cam_params)
{
    CV_Assert(depth_map.type() == CV_32F);

    int rows = depth_map.rows;
    int cols = depth_map.cols;

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float depth_value = depth_map.at<float>(y, x);
            if (depth_value > depth_cutoff)
                depth_value = 0.0f;

            float X = (x - cam_params.principal_x) * depth_value / cam_params.focal_x;
            float Y = (y - cam_params.principal_y) * depth_value / cam_params.focal_y;
            float Z = depth_value;

            vertex_map.at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
        }
    }
}

// OpenMP-parallelized version
void compute_vertex_map_cpu_parallel(const cv::Mat &depth_map, cv::Mat &vertex_map, float depth_cutoff,
                                     const CameraParameters &cam_params)
{
    CV_Assert(depth_map.type() == CV_32F);

    int rows = depth_map.rows;
    int cols = depth_map.cols;

#pragma omp parallel for collapse(2) // Parallelize rows and columns
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float depth_value = depth_map.at<float>(y, x);
            if (depth_value > depth_cutoff)
                depth_value = 0.0f;

            float X = (x - cam_params.principal_x) * depth_value / cam_params.focal_x;
            float Y = (y - cam_params.principal_y) * depth_value / cam_params.focal_y;
            float Z = depth_value;

            vertex_map.at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
        }
    }
}

int main()
{
    // Define camera parameters
    CameraParameters cam_params = {525.0f, 525.0f, 319.5f, 239.5f}; // Typical Kinect parameters

    // Generate a random depth map (e.g., 480p resolution)
    cv::Mat depth_map(5 * 480, 5 * 640, CV_32F);
    cv::randu(depth_map, 0.5f, 5.0f); // Random depth values between 0.5 and 5.0 meters

    // Define depth cutoff
    float depth_cutoff = 4.0f;

    // Prepare vertex maps
    int rows = depth_map.rows;
    int cols = depth_map.cols;

    cv::Mat vertex_map_single = cv::Mat(rows, cols, CV_32FC3);
    cv::Mat vertex_map_parallel = cv::Mat(rows, cols, CV_32FC3);

    // ---------------- Single-threaded Benchmark ----------------
    auto start_single = std::chrono::high_resolution_clock::now();
    compute_vertex_map_cpu_single(depth_map, vertex_map_single, depth_cutoff, cam_params);
    auto end_single = std::chrono::high_resolution_clock::now();

    double time_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();
    std::cout << "Single-threaded version time: " << time_single << " ms" << std::endl;

    // ---------------- OpenMP-parallelized Benchmark ----------------
    auto start_parallel = std::chrono::high_resolution_clock::now();
    compute_vertex_map_cpu_parallel(depth_map, vertex_map_parallel, depth_cutoff, cam_params);
    auto end_parallel = std::chrono::high_resolution_clock::now();

    double time_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count();
    std::cout << "OpenMP-parallelized version time: " << time_parallel << " ms" << std::endl;

    // ---------------- Verify Results ----------------
    cv::Mat diff;
    cv::absdiff(vertex_map_single, vertex_map_parallel, diff); // Absolute difference
    std::vector<cv::Mat> diff_channels(3);
    cv::split(diff, diff_channels); // Split into channels

    // Compute the per-pixel norm of the difference
    cv::Mat norm_diff = diff_channels[0].mul(diff_channels[0]) + diff_channels[1].mul(diff_channels[1]) +
                        diff_channels[2].mul(diff_channels[2]);
    cv::sqrt(norm_diff, norm_diff); // Take the square root

    // Check if any pixel's norm exceeds the tolerance
    bool is_identical = cv::countNonZero(norm_diff > 1e-6) == 0;

    if (is_identical)
    {
        std::cout << "Results are identical!" << std::endl;
    }
    else
    {
        std::cout << "Results differ!" << std::endl;
    }

    return 0;
}
