# KinectFusion

## Description
This was an implementation of the KinectFusion algorithm, designed to run on a CPU only.

The project started during one of my university classes as a foundation for my master's thesis. During the research phase, I noticed that nearly all implementations of KinectFusion (or at least most of them) use GPU acceleration, particularly CUDA. Personally I don't have nVidia GPU, I work on laptops with integrated graphics card, so I thought it would be a fun challenge to implement this algorithm using only a CPU. Since the algorithm was originally published in 2011, I strongly believe that modern computers are now powerful enough, and with further code optimization, this approach should work!

During the development original approach to implement KinectFusion was rejected. It is now completetly different approach, just inspired in some parts by KinectFusion

## Dependencies
The project relies on the following libraries:

- **OpenCV**: For image processing and computer vision tasks.
- **libfreenect**: To interface with Kinect hardware.
- **Eigen**: For linear algebra and mathematical operations.
- **OpenGL / GLUT**: For rendering and visualization of 3D data.
- **OpenMP**: For parallelizing CPU computations.
- **Point Cloud Library**: For point cloud operations.

## How to run it
Make sure that you have all dependencies up and running and your Kinect device is properly connected and detected. Then the build prodecure is pretty standard, just like for most of CMake-based projects:

1. **Clone the repository**:

```bash
git clone git@github.com:hubert-tt/kinect-fusion.git (or git clone https://github.com/hubert-tt/kinect-fusion.git)
cd kinect-fusion
```

2. **Build the project**:

```bash
mkdir build
cd build
cmake ..
make
```

3. **Run the applications**:
```bash
./record # to create a recording
or
./kinect_fusion <rgb_list.txt> <depth_list.txt> # to perform first step of pipeline
or
./recon -i input.[pcl/ply] -o output.ply -m [grid|poisson]
```