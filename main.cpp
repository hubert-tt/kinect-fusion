#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <eigen3/Eigen/Eigen>
#include <libfreenect/libfreenect.hpp>
#include <omp.h>
#include <opencv2/opencv.hpp>

#include "FrameGetter.hh"
#include "Types.hh"

volatile int checksum = 0;

namespace kf // Kinect Fusion abbreviation
{

void surface_measurement(FrameData &frame_data, const std::vector<uint16_t> &depth, const std::vector<uint8_t> &color,
                         const CameraIntrinsics &camera, const uint16_t depth_cutoff)
{
    const size_t width = frame_data.max_width;
    const size_t height = frame_data.max_height;

    // Put frames on the bottom of the pyramid
    std::memcpy(frame_data.depth_levels[0].data(), depth.data(), depth.size() * sizeof(uint16_t));
    std::memcpy(frame_data.color_levels[0].data(), color.data(), color.size() * sizeof(uint8_t));

    // Create other levels of pyramid
    for (size_t level = 1; level < frame_data.levels; ++level)
    {
        cv::pyrDown(frame_data.depth_levels[level - 1], frame_data.depth_levels[level]);
    }

    // Filter the levels (used medianBlur instead of bilateralFilter as it can work in place and support u16)
#pragma omp parallel for
    for (size_t level = 0; level < frame_data.levels; ++level)
    {
        cv::medianBlur(frame_data.depth_levels[level], frame_data.smoothed_depth_levels[level], 5);
    }

    for (size_t level = 0; level < frame_data.levels; ++level)
    {
#pragma omp parallel for collapse(2) // Parallelize rows and columns
        for (size_t y = 0; y < height / (1 << level); ++y)
        {
            for (size_t x = 0; x < width / (1 << level); ++x)
            {
                float depth_value = frame_data.smoothed_depth_levels[level][y * width / (1 << level) + x];
                if (depth_value > depth_cutoff)
                    depth_value = 0.0f;

                float X = (x - camera.cx) * depth_value / camera.f;
                float Y = (y - camera.cy) * depth_value / camera.f;
                float Z = depth_value;

                frame_data.vertex_levels[level].at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
            }
        }

#pragma omp parallel for collapse(2) // Parallelize rows and columns
        for (size_t y = 1; y < (height / (1 << level)) - 1; ++y)
        {
            for (size_t x = 1; x < (width / (1 << level)) - 1; ++x)
            {
                // Get the 3D vertices of the neighboring points
                cv::Vec3f left = frame_data.vertex_levels[level].at<cv::Vec3f>(y, x - 1);
                cv::Vec3f right = frame_data.vertex_levels[level].at<cv::Vec3f>(y, x + 1);
                cv::Vec3f upper = frame_data.vertex_levels[level].at<cv::Vec3f>(y - 1, x);
                cv::Vec3f lower = frame_data.vertex_levels[level].at<cv::Vec3f>(y + 1, x);

                // Compute the horizontal and vertical gradients (vectors)
                cv::Vec3f hor = left - right;
                cv::Vec3f ver = upper - lower;

                // Calculate the cross product to find the normal vector
                cv::Vec3f normal = hor.cross(ver);

                // Normalize the normal vector
                float norm = cv::norm(normal);
                if (norm > 0)
                {
                    normal /= norm;
                }
                else
                {
                    normal = cv::Vec3f(0.f, 0.f, 0.f); // If normal can't be computed, set to zero
                }

                // Flip the normal if the z-component is positive (optional for certain orientations)
                if (normal[2] > 0)
                {
                    normal = -normal;
                }

                // Store the computed normal in the normal map
                frame_data.normal_levels[level].at<cv::Vec3f>(y, x) = normal;
            }
        }
    }
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
#pragma omp parallel for collapse(2)
    for (int yz = 0; yz < volume_height * volume_depth; ++yz)
    {
        for (int x = 0; x < volume_width; ++x)
        {
            int y = yz / volume_depth; // Decompose yz into y and z
            int z = yz % volume_depth;
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
#pragma omp critical
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
                    short new_value = std::clamp(static_cast<short>(updated_tsdf * max_short), min_short, max_short);

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
                    }
                }
            }
        }
    }
}

} // namespace kf

#define USE_FILE

#ifndef USE_FILE
Freenect::Freenect freenect;
kf::KinectFrameGetter *device;
#else
kf::FileFrameGetter *device;
#endif

int window(0);                // Glut window identifier
int mx = -1, my = -1;         // Prevous mouse coordinates
float anglex = 0, angley = 0; // Panning angles
float zoom = 1;               // Zoom factor
bool color = true;            // Flag to indicate to use of color in the cloud
uint8_t print_counter = 0;

Eigen::Matrix4f current_pose;

void DrawGLScene()
{
    static std::vector<uint8_t> rgb(640 * 480 * 3);
    static std::vector<uint16_t> depth(640 * 480);
    static std::vector<uint16_t> smooth_depth(640 * 480);
    static kf::FrameData frame_data(640, 480);
    static kf::CameraIntrinsics camera = {.f = 595.f, .cx = (640 - 1) / 2.f, .cy = (480 - 1) / 2.f};
    static kf::VolumeData volume(cv::Vec3i({kf::voxel_grid_size, kf::voxel_grid_size, kf::voxel_grid_size}),
                                 kf::voxel_size); // 512^3 scene with 2.0mm per voxel

    if (device->get_rgb(rgb) == true && device->get_depth(depth) == true)
    {
        kf::surface_measurement(frame_data, depth, rgb, camera, 5500);
        auto start_parallel = std::chrono::high_resolution_clock::now();
        kf::surface_reconstruction(volume, frame_data.depth_levels[0], frame_data.color_levels[0], frame_data.max_width,
                                   frame_data.max_height, camera, kf::truncation_distance, current_pose);
        auto &voxel = volume.tsdf_volume.at<cv::Vec2s>(100, 100);
        std::cout << voxel[0] << " " << voxel[1] << std::endl;
        auto end_parallel = std::chrono::high_resolution_clock::now();

        double time_parallel =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count();
        std::cout << "OpenMP-parallelized version time: " << time_parallel << " ms" << std::endl;

        std::cout << "Checksum: " << checksum << std::endl;

        if (print_counter < 55) // do it for few first frames as first tend to be broken (RGB)
        {
            print_counter++;
        }
        if (print_counter == 54)
        {
            // Open a file to write output
            std::ofstream outfile("output.txt");

            // Check if the file is open
            if (!outfile.is_open())
            {
                std::cerr << "Error opening file!" << std::endl;
                throw;
            }

            // Writing rgb vector in initialization list format to file
            outfile << "rgb = { ";
            for (size_t i = 0; i < rgb.size(); ++i)
            {
                outfile << (int)rgb[i]; // Casting to int to print as numbers
                if (i != rgb.size() - 1)
                    outfile << ", ";
            }
            outfile << " }" << std::endl;

            // Writing depth vector in initialization list format to file
            outfile << "depth = { ";
            for (size_t i = 0; i < depth.size(); ++i)
            {
                outfile << (int)depth[i];
                if (i != depth.size() - 1)
                    outfile << ", ";
            }
            outfile << " }" << std::endl;

            // Close the file
            outfile.close();
        }

#ifdef USE_FILE
        device->next_frame();
#endif
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPointSize(1.0f);

    glBegin(GL_POINTS);

    if (!color)
        glColor3ub(255, 255, 255);
    for (int i = 0; i < 480 * 640; ++i)
    {
        cv::Vec3f vertex = frame_data.vertex_levels[0].at<cv::Vec3f>(i / 640, i % 640);
        if (color)
            glColor3ub(rgb[3 * i + 0],  // R
                       rgb[3 * i + 1],  // G
                       rgb[3 * i + 2]); // B

        // Convert from image plane coordinates to world coordinates
        glVertex3f(vertex[0], vertex[1], vertex[2]);
    }

    glEnd();

    // Draw the world coordinate frame
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3ub(255, 0, 0); // X-axis
    glVertex3f(0, 0, 0);
    glVertex3f(50, 0, 0);
    glColor3ub(0, 255, 0); // Y-axis
    glVertex3f(0, 0, 0);
    glVertex3f(0, 50, 0);
    glColor3ub(0, 0, 255); // Z-axis
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 50);
    glEnd();

    // Place the camera
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(zoom, zoom, 1);
    gluLookAt(-7 * anglex, -7 * angley, -1000.0, 0.0, 0.0, 2000.0, 0.0, -1.0, 0.0);

    glutSwapBuffers();
}

void keyPressed(unsigned char key, int x, int y);

void mouseMoved(int x, int y);

void mouseButtonPressed(int button, int state, int x, int y);

void resizeGLScene(int width, int height);

void idleGLScene();

void printInfo();

int main(int argc, char **argv)
{

    current_pose.setIdentity();
    current_pose(0, 3) = 512 / 2 * 2.0f;
    current_pose(1, 3) = 512 / 2 * 2.0f;
    current_pose(2, 3) = 512 / 2 * 2.0f - 200.0f;

#ifndef USE_FILE
    device = &freenect.createDevice<kf::KinectFrameGetter>(0);
    device->setDepthFormat(FREENECT_DEPTH_REGISTERED);
    device->startVideo();
    device->startDepth();
#else
    device = new kf::FileFrameGetter("../data/rgbd_dataset_freiburg1_xyz/rgb.txt",
                                     "../data/rgbd_dataset_freiburg1_xyz/depth.txt");
#endif

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(0, 0);

    window = glutCreateWindow("LibFreenect");
    glClearColor(0.45f, 0.45f, 0.45f, 0.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.0f);

    glMatrixMode(GL_PROJECTION);
    gluPerspective(50.0, 1.0, 900.0, 11000.0);

    glutDisplayFunc(&DrawGLScene);
    glutIdleFunc(&idleGLScene);
    glutReshapeFunc(&resizeGLScene);
    glutKeyboardFunc(&keyPressed);
    glutMotionFunc(&mouseMoved);
    glutMouseFunc(&mouseButtonPressed);

    printInfo();

    glutMainLoop();

    return 0;
}

void keyPressed(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'C':
    case 'c':
        color = !color;
        break;

    case 'Q':
    case 'q':
    case 0x1B: // ESC
        glutDestroyWindow(window);
#ifndef USE_FILE
        device->stopDepth();
        device->stopVideo();
#endif
        exit(0);
    }
}

void mouseMoved(int x, int y)
{
    if (mx >= 0 && my >= 0)
    {
        anglex += x - mx;
        angley += y - my;
    }

    mx = x;
    my = y;
}

void mouseButtonPressed(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        switch (button)
        {
        case GLUT_LEFT_BUTTON:
            mx = x;
            my = y;
            break;

        case 3:
            zoom *= 1.2f;
            break;

        case 4:
            zoom /= 1.2f;
            break;
        }
    }
    else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
    {
        mx = -1;
        my = -1;
    }
}

void resizeGLScene(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50.0, (float)width / height, 900.0, 11000.0);

    glMatrixMode(GL_MODELVIEW);
}

void idleGLScene()
{
    glutPostRedisplay();
}

void printInfo()
{
    std::cout << "\nAvailable Controls:" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "Rotate       :   Mouse Left Button" << std::endl;
    std::cout << "Zoom         :   Mouse Wheel" << std::endl;
    std::cout << "Toggle Color :   C" << std::endl;
    std::cout << "Quit         :   Q or Esc\n" << std::endl;
}
