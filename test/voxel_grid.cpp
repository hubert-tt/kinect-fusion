#include <iostream>
#include <opencv2/core.hpp> // dla cv::Vec3f i cv::Vec3b
#include <vector>

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
    VoxelGrid(int size, float voxel_size) : size_(size), voxel_size_(voxel_size)
    {
        grid.resize(size_, std::vector<std::vector<Voxel>>(size_, std::vector<Voxel>(size_)));
    }

    void add_point_to_voxel(const cv::Vec3f &pos, const cv::Vec3b &color)
    {
        // map from meters to voxel coordinates
        int x = static_cast<int>(pos[0] / voxel_size_);
        int y = static_cast<int>(pos[1] / voxel_size_);
        int z = static_cast<int>(pos[2] / voxel_size_);

        // check if in bounds then add
        if (x >= 0 && x < size_ && y >= 0 && y < size_ && z >= 0 && z < size_)
        {
            unsigned char r = color[0], g = color[1], b = color[2];
            grid[x][y][z].add_point(pos[0], pos[1], pos[2], r, g, b);
        }
    }

    Voxel get_voxel(int x, int y, int z) const
    {
        if (x >= 0 && x < size_ && y >= 0 && y < size_ && z >= 0 && z < size_)
        {
            return grid[x][y][z];
        }
        throw std::out_of_range("Voxel coordinates out of bounds");
    }

  private:
    int size_;
    float voxel_size_; // in meters (e.g. 0.002 m = 2 mm)
    std::vector<std::vector<std::vector<Voxel>>> grid;
};

int main()
{
    const int size = 512;
    const float voxel_size = 0.002f; // Voxel ma 2x2x2 mm, czyli 0.002 metra
    VoxelGrid grid(size, voxel_size);

    // Przykład: Dodanie punktu do voxela
    cv::Vec3f pos1(0.005f, 0.005f, 0.005f); // Pozycja w metrach
    cv::Vec3b color1(255, 0, 0);            // Kolor czerwony
    grid.add_point_to_voxel(pos1, color1);

    cv::Vec3f pos2(0.005f, 0.005f, 0.0045f); // Pozycja w metrach
    cv::Vec3b color2(0, 255, 0);             // Kolor zielony
    grid.add_point_to_voxel(pos2, color2);

    // Pobieranie i wyświetlanie danych z voxela (np. (x=2, y=2, z=2))
    auto voxel = grid.get_voxel(2, 2, 2);
    std::cout << voxel << std::endl;
    std::cout << sizeof(pos1) << std::endl;
    std::cout << sizeof(color2) << std::endl;
    std::cout << sizeof(voxel) << std::endl;

    return 0;
}
