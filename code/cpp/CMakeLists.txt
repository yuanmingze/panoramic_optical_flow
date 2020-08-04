cmake_minimum_required(VERSION 3.6.0)

project(pano_optical_flow)

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

# set(THIRD_PARTY "${CMAKE_CURRENT_LIST_DIR}/3rdparty/")

# set(Eigen_INCLUDE_DIRS "${THIRD_PARTY}/eigen")
# set(Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/build/src/include")
# list(APPEND Pangolin_INCLUDE_DIRS "${THIRD_PARTY}/Pangolin/include")
# set(Pangolin_LIBRARIES "${THIRD_PARTY}/Pangolin/build/src/libpangolin.so")

# find_package(dl REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED) 
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

message("CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'")

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -DSHADER_DIR=${PTEX_SHADER_DIR}")
ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse2 -msse3 -Wall -std=c++11 -fopenmp -DSHADER_DIR=${PTEX_SHADER_DIR}")
ELSE()
    message("Some other build type.")
ENDIF()


# include_directories(${Pangolin_INCLUDE_DIRS})
# include_directories(${EIGEN3_INCLUDE_DIR})
# include_directories(${dl_INCLUDE_DIRS})

# opencv

# # openEXR
# #find_package(OpenEXR)
# # the `pkg_check_modules` function is created with this call
# find_package(PkgConfig REQUIRED) 
# pkg_check_modules (OpenEXR REQUIRED OpenEXR) 
# ${OpenEXR_LIBRARIES}
# ${OpenEXR_INCLUDE_DIRS}

######     Warp with optical flow ########
# function: 
add_executable(replica_otical_flow_post_process data_format_conversion.cpp)

target_link_libraries(replica_otical_flow_post_process
                    ${OpenCV_LIBS}
                    Eigen3::Eigen
                      stdc++fs
)

target_include_directories(replica_otical_flow_post_process PRIVATE
                    ${OpenCV_INCLUDE_DIRS} 
                    ${Eigen_INCLUDE_DIRS}
)

# ######     ReplicaVideoRendererMP     ######
# # function: 
# add_executable(ReplicaVideoRendererMP src/render_video_mp.cpp src/MirrorSurface.cpp)

# target_include_directories(ReplicaVideoRendererMP PRIVATE
#                     "/home/mingze/software/renderdoc_1.9/include/"
# )

# target_link_libraries(ReplicaVideoRendererMP
#                       ${Pangolin_LIBRARIES}
#                       ${dl_LIBRARIES}
#                       GL
#                       GLEW
#                       ptex
#                       stdc++fs
#                       "/home/mingze/software/renderdoc_1.9/lib/librenderdoc.so"
# )

# ######     ReplicaRendererDatasetMP     ######
# # function: 
# add_executable(ReplicaRendererDatasetMP src/render_dataset_mp.cpp src/MirrorSurface.cpp)

# target_link_libraries(ReplicaRendererDatasetMP
#                       ${Pangolin_LIBRARIES}
#                       ${dl_LIBRARIES}
#                       GL
#                       GLEW
#                       ptex
#                       stdc++fs
# )
