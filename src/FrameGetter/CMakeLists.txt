set(SOURCES
    FileFrameGetter.cpp
    KinectFrameGetter.cpp
)

add_library(FrameGetter ${SOURCES})

target_link_libraries(FrameGetter freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${OpenCV_LIBS} ${OpenMP_CXX_FLAGS})