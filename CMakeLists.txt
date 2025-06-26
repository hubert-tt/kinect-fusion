cmake_minimum_required(VERSION 3.10)

project(KinectFusion)

set(THREADS_USE_PTHREADS_WIN32 true)
find_package(Threads)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenGL COMPONENTS OpenGL)
find_package(GLUT)

find_package(PCL 1.3 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -Wall
        -Wextra
        # -pedantic
        -Werror
        -Wno-unused-parameter
        -Wno-deprecated-enum-enum-conversion
        -O3
        # -pedantic-errors
        -ggdb
    )
endif()

if (Threads_FOUND AND OpenGL_OpenGL_FOUND AND GLUT_FOUND)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
        set(OPENGL_TARGET "$<IF:$<TARGET_EXISTS:OpenGL::OpenGL>,OpenGL::OpenGL,OpenGL::GL>")
    else()
        set(OPENGL_TARGET "$<IF:$<BOOL:${OPENGL_opengl_LIBRARY}>,OpenGL::OpenGL,OpenGL::GL>")
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/include)

    add_subdirectory(src EXCLUDE_FROM_ALL)

    include_directories(. ${THREADS_PTHREADS_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

    if (OPENGL_GLU_FOUND)
        add_executable(align align.cpp)
        target_compile_features(align PUBLIC cxx_std_20)
        target_compile_options(align PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(align freenect ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${OpenCV_LIBS} ${OpenMP_CXX_FLAGS} ${PCL_LIBRARIES} FrameGetter)
    
        add_executable(record record.cpp)
        target_compile_features(record PUBLIC cxx_std_20)
        target_compile_options(record PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(record freenect "${OPENGL_TARGET}" OpenGL::GLU GLUT::GLUT ${CMAKE_THREAD_LIBS_INIT} ${MATH_LIB} ${OpenCV_LIBS} ${OpenMP_CXX_FLAGS} FrameGetter)

        add_executable(recon recon.cpp)
        target_link_libraries(recon ${PCL_LIBRARIES})
    endif()
endif()
