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

#include <libfreenect/libfreenect.hpp>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace kf // Kinect Fusion abbreviation
{

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
    FileFrameGetter(const std::string &file_path)
    {
        // Open the file to load the data
        std::ifstream infile(file_path);
        if (!infile.is_open())
        {
            std::cerr << "Error opening file: " << file_path << std::endl;
            return;
        }

        // Read the entire content into a string
        std::stringstream buffer;
        buffer << infile.rdbuf();
        std::string content = buffer.str();

        // Extract rgb data (after "rgb = { ... }")
        size_t rgb_pos = content.find("rgb = { ");
        size_t depth_pos = content.find("depth = { ");

        if (rgb_pos != std::string::npos && depth_pos != std::string::npos)
        {
            rgb_pos += 8;   // Move past "rgb = { "
            depth_pos += 9; // Move past "depth = { "

            // Extract the rgb data between the braces
            size_t rgb_end = content.find(" }", rgb_pos);
            std::string rgb_data = content.substr(rgb_pos, rgb_end - rgb_pos);

            // Extract the depth data between the braces
            size_t depth_end = content.find(" }", depth_pos);
            std::string depth_data = content.substr(depth_pos, depth_end - depth_pos);

            // Parse the numbers from the rgb and depth strings
            parse_data(rgb_data, rgb);
            parse_data(depth_data, depth);
        }

        infile.close();
    }

    bool get_rgb(std::vector<uint8_t> &buffer) override
    {
        if (rgb.empty())
            return false;
        buffer = rgb;
        return true;
    }

    bool get_depth(std::vector<uint16_t> &buffer) override
    {
        if (depth.empty())
            return false;
        buffer = depth;
        return true;
    }

  private:
    // Utility function to parse the data from the string
    template <typename T> void parse_data(const std::string &data, std::vector<T> &buffer)
    {
        std::stringstream ss(data);
        int value;
        while (ss >> value)
        {
            buffer.push_back(static_cast<T>(value));
            if (ss.peek() == ',')
            {
                ss.ignore(); // Skip the comma
            }
        }
    }

    std::vector<uint8_t> rgb;
    std::vector<uint16_t> depth;
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

struct FrameData
{
    static constexpr size_t levels = 3;

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

} // namespace kf

// #define USE_FILE

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

void DrawGLScene()
{
    static std::vector<uint8_t> rgb(640 * 480 * 3);
    static std::vector<uint16_t> depth(640 * 480);
    static std::vector<uint16_t> smooth_depth(640 * 480);
    static kf::FrameData frame_data(640, 480);
    static kf::CameraIntrinsics camera = {.f = 595.f, .cx = (640 - 1) / 2.f, .cy = (480 - 1) / 2.f};

    if (device->get_depth(depth) == true && device->get_rgb(rgb) == true)
    {
        kf::surface_measurement(frame_data, depth, rgb, camera, 2500);

#ifndef USE_FILE
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

int main(int argc, char **argv)
{
#ifndef USE_FILE
    device = &freenect.createDevice<kf::KinectFrameGetter>(0);
    device->setDepthFormat(FREENECT_DEPTH_REGISTERED);
    device->startVideo();
    device->startDepth();
#else
    device = new kf::FileFrameGetter("output.txt");
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
