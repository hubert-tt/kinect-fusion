#include <iostream>
#include <string>
#include <unistd.h> // for getopt

#include <pcl/common/common.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/surface/poisson.h>

using namespace pcl;
using namespace std;

int main(int argc, char **argv)
{
    string input_file, output_file, method;

    int opt;
    while ((opt = getopt(argc, argv, "i:o:m:")) != -1)
    {
        switch (opt)
        {
        case 'i':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'm':
            method = optarg;
            break;
        default:
            cerr << "Usage: " << argv[0] << " -i input.ply -o output.ply -m [grid|poisson]" << endl;
            return -1;
        }
    }

    if (input_file.empty() || output_file.empty() || method.empty())
    {
        cerr << "Missing required arguments." << endl;
        cerr << "Usage: " << argv[0] << " -i input.ply -o output.ply -m [grid|poisson]" << endl;
        return -1;
    }

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud_filtered(new PointCloud<PointXYZRGB>);

    if (input_file.size() >= 4 && input_file.substr(input_file.size() - 4) == ".ply")
    {
        if (io::loadPLYFile<PointXYZRGB>(input_file, *cloud) == -1)
        {
            cerr << "Failed to load PLY file: " << input_file << endl;
            return -1;
        }
    }
    else if (input_file.size() >= 4 && input_file.substr(input_file.size() - 4) == ".pcd")
    {
        if (io::loadPCDFile<PointXYZRGB>(input_file, *cloud) == -1)
        {
            cerr << "Failed to load PCD file: " << input_file << endl;
            return -1;
        }
    }
    else
    {
        cerr << "Unsupported file format: " << input_file << endl;
        return -1;
    }

    cout << "loaded" << endl;

    StatisticalOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(0.5);
    sor.filter(*cloud_filtered);

    cout << "begin normal estimation" << endl;
    NormalEstimationOMP<PointXYZRGB, Normal> ne;
    ne.setNumberOfThreads(10);
    ne.setInputCloud(cloud_filtered);
    ne.setRadiusSearch(0.01);
    Eigen::Vector4f centroid;
    compute3DCentroid(*cloud_filtered, centroid);
    ne.setViewPoint(centroid[0], centroid[1], centroid[2]);

    PointCloud<Normal>::Ptr cloud_normals(new PointCloud<Normal>());
    ne.compute(*cloud_normals);
    cout << "normal estimation complete" << endl;

    cout << "reverse normals' direction" << endl;
    for (size_t i = 0; i < cloud_normals->size(); ++i)
    {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
    }

    cout << "combine points and normals" << endl;
    PointCloud<PointXYZRGBNormal>::Ptr cloud_smoothed_normals(new PointCloud<PointXYZRGBNormal>());
    concatenateFields(*cloud_filtered, *cloud_normals, *cloud_smoothed_normals);

    PolygonMesh mesh;

    if (method == "grid")
    {
        cout << "begin grid projection reconstruction" << endl;
        search::KdTree<PointXYZRGBNormal>::Ptr tree(new search::KdTree<PointXYZRGBNormal>);
        tree->setInputCloud(cloud_smoothed_normals);

        GridProjection<PointXYZRGBNormal> gp;
        gp.setInputCloud(cloud_smoothed_normals);
        gp.setSearchMethod(tree);
        gp.setResolution(0.005);
        gp.setPaddingSize(3);
        gp.setNearestNeighborNum(100);
        gp.setMaxBinarySearchLevel(10);
        gp.reconstruct(mesh);
    }
    else if (method == "poisson")
    {
        cout << "begin poisson reconstruction" << endl;
        Poisson<PointXYZRGBNormal> poisson;
        poisson.setDepth(11);
        poisson.setInputCloud(cloud_smoothed_normals);
        poisson.reconstruct(mesh);
    }
    else
    {
        cerr << "Unknown reconstruction method: " << method << ". Use 'grid' or 'poisson'." << endl;
        return -1;
    }

    cout << mesh.polygons.size() << " triangles created" << endl;

    if (io::savePLYFile(output_file, mesh) != 0)
    {
        cerr << "Failed to save output file: " << output_file << endl;
        return -1;
    }

    cout << "Saved to " << output_file << endl;
    return 0;
}
