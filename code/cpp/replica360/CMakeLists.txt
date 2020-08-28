cmake_minimum_required(VERSION 3.6.0)

project(Replica)

# replica SDK header, lib and shader dir
set(replica360_ROOT_DIR "/mnt/sda1/workspace_linux/replica360/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${replica360_ROOT_DIR}/cmake/")
set(THIRD_PARTY "${replica360_ROOT_DIR}/3rdparty/")
set(PTEX_SHADER_DIR "${replica360_ROOT_DIR}/ReplicaSDK/shaders/" CACHE PATH "Where the shaders live")

# glfw 
find_package(GLEW REQUIRED)

# replica header files
set(Replica360_INCLUDE_DIRS "${replica360_ROOT_DIR}/ReplicaSDK/include/")
set(Replica360_LIBRARIES "${replica360_ROOT_DIR}/build/ReplicaSDK/libptex.so")

# Eigen
set(Eigen_INCLUDE_DIRS "${THIRD_PARTY}/eigen")

# pangolin
set(Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/build/src/include")
list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/include")
set(Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/src/libpangolin.so")

# dl 
find_package(dl REQUIRED)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# OpenCV
find_package(OpenCV REQUIRED ) 

# # openEXR
# #find_package(OpenEXR)
# # the `pkg_check_modules` function is created with this call
# find_package(PkgConfig REQUIRED) 
# pkg_check_modules (OpenEXR REQUIRED OpenEXR) 

# Debug & Release configuration
message("CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'")
set(CMAKE_BUILD_TYPE Release)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -DSHADER_DIR=${PTEX_SHADER_DIR}")
ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse2 -msse3 -Wall -std=c++11 -fopenmp -DSHADER_DIR=${PTEX_SHADER_DIR}")
ELSE()
    message("Some other build type.")
ENDIF()


######  renader cubemap ########
# function: 
add_executable(render_cubemap 
                render_cubemap.cpp 
                turbo_colormap.hpp
                "${replica360_ROOT_DIR}/ReplicaSDK/src/MirrorSurface.cpp")

target_link_libraries(render_cubemap
                    ${OpenCV_LIBS}
                    ${OpenEXR_LIBRARIES}
                    ${Replica360_LIBRARIES}
                    ${Pangolin_LIBRARIES}
                    ${GLEW_LIBRARIES}
                    GL
                    stdc++fs
)

target_include_directories(render_cubemap PRIVATE
                    ${dl_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${OpenCV_INCLUDE_DIRS} 
                    ${OpenEXR_INCLUDE_DIRS}
                    ${Eigen_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${Replica360_INCLUDE_DIRS}
)

