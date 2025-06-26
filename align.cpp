#include <cassert>
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "FrameGetter.hh"

using namespace std;

namespace kf
{

struct Voxel
{
    float x_ = 0.0f;
    float y_ = 0.0f;
    float z_ = 0.0f;
    uint8_t color_[3] = {0, 0, 0};
    uint16_t count_ = 0;

    void add_point(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b)
    {
        count_++;
        const float count_inv = 1.0f / count_;
        x_ += (x - x_) * count_inv;
        y_ += (y - y_) * count_inv;
        z_ += (z - z_) * count_inv;
        color_[0] += (r - color_[0]) * count_inv;
        color_[1] += (g - color_[1]) * count_inv;
        color_[2] += (b - color_[2]) * count_inv;
    }
};

std::ostream &operator<<(std::ostream &os, const Voxel &v)
{
    os << "XYZ: (" << v.x_ << ", " << v.y_ << ", " << v.z_ << "), "
       << "RGB: (" << static_cast<int>(v.color_[0]) << ", " << static_cast<int>(v.color_[1]) << ", "
       << static_cast<int>(v.color_[2]) << ")";
    return os;
}

class VoxelGrid
{
  public:
    VoxelGrid(int size = 256, float voxel_size = 0.005f, uint16_t noise_cutoff = 10)
        : size_(size), half_size_(size / 2), voxel_size_(voxel_size), noise_cutoff_(noise_cutoff)
    {
        // pre-allocate for speed
        grid_.resize(size_, std::vector<std::vector<Voxel>>(size_, std::vector<Voxel>(size_)));
    }

    void add_point_to_voxel(const cv::Vec3f &pos, const cv::Vec3b &color)
    {
        // map from world coordinates (meters) to voxel grid indices
        int x = static_cast<int>(pos[0] / voxel_size_) + half_size_;
        int y = static_cast<int>(pos[1] / voxel_size_) + half_size_;
        int z = static_cast<int>(pos[2] / voxel_size_) + half_size_;

        // check if in bounds then add
        if (x >= 0 && x < size_ && y >= 0 && y < size_ && z >= 0 && z < size_)
        {
            std::lock_guard<std::mutex> guard(lock_);
            grid_[x][y][z].add_point(pos[0], pos[1], pos[2], color[2], color[1], color[0]);
        }
    }

    Voxel get_voxel(int x, int y, int z) const
    {
        if (x >= 0 && x < size_ && y >= 0 && y < size_ && z >= 0 && z < size_)
        {
            return grid_[x][y][z];
        }
        throw std::out_of_range("Voxel coordinates out of bounds");
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr to_point_cloud() const
    {
        int white_threshold = 210;

        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        cloud->points.reserve(size_ * size_ * size_);

        for (int x = 0; x < size_; ++x)
        {
            for (int y = 0; y < size_; ++y)
            {
                for (int z = 0; z < size_; ++z)
                {
                    const Voxel &v = grid_[x][y][z];

                    // filter out vertexes with low number of occurences, possible noise
                    if (v.count_ <= noise_cutoff_)
                        continue;

                    // skip points close to white
                    if (v.color_[0] >= white_threshold && v.color_[1] >= white_threshold &&
                        v.color_[2] >= white_threshold)
                    {
                        continue;
                    }

                    pcl::PointXYZRGB p;
                    p.x = v.x_;
                    p.y = v.y_;
                    p.z = v.z_;
                    p.r = v.color_[0];
                    p.g = v.color_[1];
                    p.b = v.color_[2];
                    cloud->points.push_back(p);
                }
            }
        }

        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = true;
        return cloud;
    }

  private:
    int size_; // X, Y and Z
    int half_size_;
    float voxel_size_;      // in meters (e.g. 0.002 m = 2 mm)
    uint16_t noise_cutoff_; // maximal number of occurences at voxel to be considered as noise
    std::vector<std::vector<std::vector<Voxel>>> grid_;
    std::mutex lock_;
};

class TrajectoryRecorder
{
  public:
    TrajectoryRecorder()
    {
        using namespace std::chrono;

        // Start timestamp
        auto now = system_clock::now();
        auto duration = now.time_since_epoch();
        auto secs = duration_cast<seconds>(duration).count();
        auto usecs = duration_cast<microseconds>(duration).count() % 1000000;
        start_timestamp = secs + usecs / 1e6;

        std::ostringstream filename_ss;
        filename_ss << "trajectory_" << std::fixed << std::setprecision(6) << start_timestamp << ".traj";
        filename = filename_ss.str();

        trajectory_txt.open(filename, std::ios::out);
        trajectory_txt << "# trajectory\n";
        trajectory_txt << "# file: '" << filename << "'\n";
        trajectory_txt << "# timestamp tx ty tz qx qy qz qw\n";
    }

    ~TrajectoryRecorder()
    {
        trajectory_txt.close();
    }

    void save_pose(const cv::Mat &pose_4x4, double timestamp)
    {
        assert(pose_4x4.rows == 4 && pose_4x4.cols == 4 && pose_4x4.type() == CV_64FC1);

        Eigen::Matrix4d eigen_mat;
        cv::cv2eigen(pose_4x4, eigen_mat);

        Eigen::Matrix3d rotation = eigen_mat.block<3, 3>(0, 0);
        Eigen::Quaterniond q(rotation);
        Eigen::Vector3d t = eigen_mat.block<3, 1>(0, 3);

        trajectory_txt << std::fixed << std::setprecision(6) << timestamp << " " << t.x() << " " << t.y() << " "
                       << t.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

  private:
    std::string filename;
    std::ofstream trajectory_txt;
    double start_timestamp;
};

} // namespace kf

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <rgb_list.txt> <depth_list.txt>" << std::endl;
        return 1;
    }

    kf::TrajectoryRecorder recorder;
    kf::FileFrameGetter getter(argv[1], argv[2]);
    // fx = 517.3f; fy = 516.5f; cx = 318.6f; cy = 255.3f;
    // cv::Mat K = (cv::Mat_<float>(3, 3) << 517.3f, 0.0f, 318.6f, 0.0f, 516.5f, 255.3f, 0.0f, 0.0f, 1.0f);
    cv::Mat K = (cv::Mat_<float>(3, 3) << 525.0f, 0.0f, 319.5f, 0.0f, 525.0f, 239.5f, 0.0f, 0.0f, 1.0f);

    // const float MIN_DEPTH = 0.f;        // in meters
    // const float MAX_DEPTH = 3.0f;       // in meters
    // const float MAX_DEPTH_DIFF = 0.07f; // in meters
    // const float MAX_POINTS_PART = 0.07f;

    std::vector<int> iterCounts(3);
    iterCounts[0] = 10;
    iterCounts[1] = 5;
    iterCounts[2] = 4;
    // iterCounts[3] = 10;

    std::vector<float> minGradMagnitudes(3);
    minGradMagnitudes[0] = 12;
    minGradMagnitudes[1] = 5;
    minGradMagnitudes[2] = 3;
    // minGradMagnitudes[3] = 1;

    auto odom = cv::rgbd::RgbdICPOdometry::create(K);
    // auto odom = cv::rgbd::FastICPOdometry::create(K);

    odom->setCameraMatrix(K);

    // odom->setMaxRotation(30);
    // odom->setMaxTranslation(0.05);

    const int size = 110;
    const float voxel_size = 0.01f; // in meters
    const uint16_t noise_cutoff = 10;
    kf::VoxelGrid grid(size, voxel_size, noise_cutoff);

    std::vector<uint8_t> rgb_buf;
    std::vector<uint16_t> depth_buf;

    const double depth_sigma = 0.03;
    const double space_sigma = 4.5;

    cv::Mat rgb_prev, depth_prev;
    double timestamp_to_save_traj;
    if (!getter.get_rgb(rgb_buf) || !getter.get_depth(depth_buf) || !getter.get_timestamp(timestamp_to_save_traj))
    {
        std::cerr << "Failed to load first frame" << std::endl;
        return 1;
    }

    rgb_prev = cv::Mat(480, 640, CV_8UC3, rgb_buf.data()).clone();
    depth_prev = cv::Mat(480, 640, CV_16UC1, depth_buf.data()).clone();

    // Convert RGB to grayscale (CV_8UC1)
    cv::Mat rgb_gray_prev;
    cv::cvtColor(rgb_prev, rgb_gray_prev, cv::COLOR_BGR2GRAY);

    // Convert depth to CV_32FC1 in meters
    cv::Mat depth_float_prev;
    depth_prev.convertTo(depth_float_prev, CV_32FC1, 1.0f / 1000.0f);
    cv::Mat invalid_depth_mask = depth_float_prev == 0.f;
    depth_float_prev.setTo(-5 * depth_sigma, invalid_depth_mask);
    cv::Mat depth_filtered_prev;
    cv::bilateralFilter(depth_float_prev, depth_filtered_prev, -1, depth_sigma, space_sigma);
    depth_filtered_prev.setTo(std::numeric_limits<float>::quiet_NaN(), invalid_depth_mask);

    const double offset = size * voxel_size * 0.5;
    cv::Mat Rt_total = cv::Mat::eye(4, 4, CV_64FC1);
    Rt_total.at<double>(0, 3) = -offset * 0.2f;
    Rt_total.at<double>(1, 3) = -offset * 0.3f; // up / down
    Rt_total.at<double>(2, 3) = offset;

    cv::Mat Rt_to_save = cv::Mat::eye(4, 4, CV_64FC1);
    recorder.save_pose(Rt_to_save, timestamp_to_save_traj);

    int counter = 0;
    std::vector<cv::Mat> clouds;

    while (getter.next_frame())
    {
        if (!getter.get_rgb(rgb_buf) || !getter.get_depth(depth_buf) || !getter.get_timestamp(timestamp_to_save_traj))
        {
            break;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        counter++;

        cv::Mat rgb_curr = cv::Mat(480, 640, CV_8UC3, rgb_buf.data()).clone();
        cv::Mat depth_curr = cv::Mat(480, 640, CV_16UC1, depth_buf.data()).clone();

        cv::Mat rgb_gray_curr;
        cv::cvtColor(rgb_curr, rgb_gray_curr, cv::COLOR_BGR2GRAY);

        // TODO: cleaning, filtering, cut off too deep things
        cv::Mat depth_float_curr;
        depth_curr.convertTo(depth_float_curr, CV_32FC1, 1.0f / 1000.0f);
        cv::Mat invalid_depth_mask = depth_float_curr == 0.f;
        depth_float_curr.setTo(-5 * depth_sigma, invalid_depth_mask);
        cv::Mat depth_filtered_curr;
        cv::bilateralFilter(depth_float_curr, depth_filtered_curr, -1, depth_sigma, space_sigma);
        depth_filtered_curr.setTo(std::numeric_limits<float>::quiet_NaN(), invalid_depth_mask);

        cv::Mat Rt = cv::Mat::eye(4, 4, CV_64FC1);
        bool success = odom->compute(rgb_gray_prev, depth_filtered_prev, cv::Mat(), rgb_gray_curr, depth_filtered_curr,
                                     cv::Mat(), Rt);
        if (!success)
        {
            std::cerr << "Odometry failed, skipping frame\n";
            continue;
        }

        Rt_total = Rt * Rt_total;

        Rt_to_save = Rt * Rt_to_save;

        recorder.save_pose(Rt_to_save, timestamp_to_save_traj);

        // std::cout << "Rt_total = " << std::endl << " " << Rt_total << std::endl;

        cv::Mat cloud;
        cv::rgbd::depthTo3d(depth_filtered_curr, K, cloud);

        cv::Mat Rt_total_inv;
        cv::invert(Rt_total, Rt_total_inv);

        if ((counter % 3) == 0)
        {
#pragma omp parallel for collapse(2)
            for (int y = 0; y < cloud.rows; ++y)
            {
                for (int x = 0; x < cloud.cols; ++x)
                {
                    cv::Vec3f p = cloud.at<cv::Vec3f>(y, x);
                    if (!cv::checkRange(p))
                        continue;

                    cv::Mat pt = (cv::Mat_<double>(4, 1) << p[0], p[1], p[2], 1.0f);
                    cv::Mat p_trans = Rt_total_inv * pt;

                    cv::Vec3d p_final(p_trans.at<double>(0), p_trans.at<double>(1), p_trans.at<double>(2));
                    cv::Vec3b color = rgb_curr.at<cv::Vec3b>(y, x);
                    grid.add_point_to_voxel(cv::Vec3f(p_final[0], p_final[1], p_final[2]), color);
                }
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "Processed in " << elapsed_time << " ms." << std::endl;

        depth_filtered_prev = depth_filtered_curr;
        rgb_gray_prev = rgb_gray_curr;
    }

    auto pcl_cloud = grid.to_point_cloud();
    pcl::io::savePCDFile("output_test.pcd", *pcl_cloud);

    std::cout << "Saved " << pcl_cloud->points.size() << " points to output_test.pcd\n";
    return 0;
}
