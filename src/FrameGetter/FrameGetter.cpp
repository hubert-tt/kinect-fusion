#include "FrameGetter.hh"

namespace kf
{
FileFrameGetter::FileFrameGetter(const std::string &rgb_list_path, const std::string &depth_list_path)
{
    // Load RGB file list
    if (!load_file_list(rgb_list_path, rgb_files))
    {
        std::cerr << "Error loading RGB file list from: " << rgb_list_path << std::endl;
    }

    // Load depth file list
    if (!load_file_list(depth_list_path, depth_files))
    {
        std::cerr << "Error loading depth file list from: " << depth_list_path << std::endl;
    }

    // Ensure both lists have the same number of entries
    if (rgb_files.size() != depth_files.size())
    {
        std::cerr << "Mismatch between RGB and depth file list sizes." << std::endl;
    }

    current_index = 0;
}

bool FileFrameGetter::get_rgb(std::vector<uint8_t> &buffer)
{
    if (current_index >= rgb_files.size())
        return false;

    // Load the current RGB image
    cv::Mat image = cv::imread(rgb_files[current_index], cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error loading RGB image: " << rgb_files[current_index] << std::endl;
        return false;
    }

    // Convert the image to a byte buffer
    buffer.assign(image.data, image.data + image.total() * image.elemSize());
    return true;
}

bool FileFrameGetter::get_depth(std::vector<uint16_t> &buffer)

{
    if (current_index >= depth_files.size())
        return false;

    // Load the current depth image
    cv::Mat image = cv::imread(depth_files[current_index], cv::IMREAD_ANYDEPTH);
    if (image.empty())
    {
        std::cerr << "Error loading depth image or incorrect format: " << depth_files[current_index] << std::endl;
        return false;
    }

    cv::Mat scaled = image / 5; // this data is scaled by 5000, ergo 1m = 5000. we want mm

    // Convert the image to a 16-bit buffer
    buffer.assign(reinterpret_cast<uint16_t *>(scaled.data),
                  reinterpret_cast<uint16_t *>(scaled.data) + scaled.total());
    return true;
}

bool FileFrameGetter::next_frame()
{
    if (current_index + 1 < rgb_files.size() && current_index + 1 < depth_files.size())
    {
        ++current_index;
        return true;
    }
    return false;
}

bool FileFrameGetter::load_file_list(const std::string &file_path, std::vector<std::string> &file_list)
{
    std::ifstream infile(file_path);
    if (!infile.is_open())
    {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return false;
    }

    size_t pos = file_path.find_last_of('/');
    std::string base_path;

    // If '/' is found, remove everything after it
    if (pos != std::string::npos)
    {
        base_path = file_path.substr(0, pos + 1);
    }

    std::string line;
    while (std::getline(infile, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string timestamp, filename;
        if (iss >> timestamp >> filename)
        {
            file_list.push_back(base_path + filename);
        }
    }

    infile.close();
    return true;
}

} // namespace kf
