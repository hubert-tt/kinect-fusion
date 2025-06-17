#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
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

namespace kf
{
class Recorder
{
  public:
    Recorder(const std::string &base_dir)
    {
        using namespace std::filesystem;
        output_dir = base_dir;
        rgb_dir = output_dir / "rgb";
        depth_dir = output_dir / "depth";

        // Create directories
        create_directories(rgb_dir);
        create_directories(depth_dir);

        // Open list files
        rgb_txt.open(output_dir / "rgb.txt", std::ios::out);
        depth_txt.open(output_dir / "depth.txt", std::ios::out);

        // Write headers
        rgb_txt << "# color images\n# timestamp filename\n";
        depth_txt << "# depth maps\n# timestamp filename\n";
    }

    ~Recorder()
    {
        rgb_txt.close();
        depth_txt.close();
    }

    void save_frame(const std::vector<uint8_t> &rgb, const std::vector<uint16_t> &depth, int width, int height)
    {
        using namespace std::chrono;

        auto now = system_clock::now();
        auto duration = now.time_since_epoch();
        auto secs = duration_cast<seconds>(duration).count();
        auto usecs = duration_cast<microseconds>(duration).count() % 1000000;

        double timestamp = secs + usecs / 1e6;

        std::ostringstream filename_ss;
        filename_ss << std::fixed << std::setprecision(6) << timestamp;
        std::string ts_str = filename_ss.str();

        // Convert RGB to cv::Mat
        cv::Mat rgb_img(height, width, CV_8UC3, const_cast<uint8_t *>(rgb.data()));
        cv::Mat bgr_img;
        cv::cvtColor(rgb_img, bgr_img, cv::COLOR_RGB2BGR); // OpenCV uses BGR

        // Scale input depth
        cv::Mat depth_scaled(height, width, CV_16UC1);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                uint16_t d = depth[y * width + x];
                // skalowanie (np. 1m=1000 -> 1m=5000 to mnożenie przez 5)
                uint32_t scaled = static_cast<uint32_t>(d) * 5;
                if (scaled > std::numeric_limits<uint16_t>::max())
                {
                    scaled = std::numeric_limits<uint16_t>::max();
                }
                depth_scaled.at<uint16_t>(y, x) = static_cast<uint16_t>(scaled);
            }
        }

        // Save PNG files
        std::string rgb_path = (rgb_dir / (ts_str + ".png")).string();
        std::string depth_path = (depth_dir / (ts_str + ".png")).string();
        cv::imwrite(rgb_path, bgr_img);
        cv::imwrite(depth_path, depth_scaled);

        // Add to .txt logs
        rgb_txt << ts_str << " rgb/" << ts_str << ".png" << std::endl;
        depth_txt << ts_str << " depth/" << ts_str << ".png" << std::endl;
    }

  private:
    std::filesystem::path output_dir;
    std::filesystem::path rgb_dir;
    std::filesystem::path depth_dir;
    std::ofstream rgb_txt;
    std::ofstream depth_txt;
};
} // namespace kf

Freenect::Freenect freenect;
kf::KinectFrameGetter *device;
kf::Recorder recorder("recorded_files");
kf::Recorder recorder_images("images");

int window(0);                // Glut window identifier
int mx = -1, my = -1;         // Prevous mouse coordinates
float anglex = 0, angley = 0; // Panning angles
float zoom = 1;               // Zoom factor
bool color = true;            // Flag to indicate to use of color in the cloud
bool snapshot = true;
// uint8_t print_counter = 0;

void DrawGLScene()
{
    static std::vector<uint8_t> rgb(640 * 480 * 3);
    static std::vector<uint16_t> depth(640 * 480);
    static kf::CameraIntrinsics camera = {.f = 595.f, .cx = (640 - 1) / 2.f, .cy = (480 - 1) / 2.f};
    bool stop = false;

    if ((device->get_rgb(rgb) == true && device->get_depth(depth) == true) || stop == true)
    {
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPointSize(1.0f);

    glBegin(GL_POINTS);

    recorder.save_frame(rgb, depth, 640, 480);

    if (snapshot)
    {
        recorder_images.save_frame(rgb, depth, 640, 480);
        snapshot = false;
    }

    if (!color)
        glColor3ub(255, 255, 255);
    for (int y = 0; y < 480; ++y)
    {
        for (int x = 0; x < 640; ++x)
        {
            int i = y * 640 + x;
            uint16_t depth_value = depth[i];
            if (depth_value == 0)
                continue;

            float Z = static_cast<float>(depth_value);
            float X = (x - camera.cx) * Z / camera.f;
            float Y = (y - camera.cy) * Z / camera.f;

            if (color)
                glColor3ub(rgb[3 * i + 0],  // R
                           rgb[3 * i + 1],  // G
                           rgb[3 * i + 2]); // B

            glVertex3f(X, Y, Z);
        }
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

    device = &freenect.createDevice<kf::KinectFrameGetter>(0);
    device->setDepthFormat(FREENECT_DEPTH_REGISTERED);
    device->startVideo();
    device->startDepth();

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

void keyPressed(unsigned char key, int qualityx, int y)
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

    case 'r':
    case 'R':
        snapshot = true;
        break;
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
    std::cout << "Quit         :   Q or Esc" << std::endl;
    std::cout << "Snapshot     :   R\n" << std::endl;
}
