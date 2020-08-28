// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#define cimg_display 0

#include "turbo_colormap.hpp"

#include <EGL.h>
#include <PTexLib.h>
#include <string>
#include <pangolin/image/image_convert.h>
#include <Eigen/Geometry>
#include "MirrorRenderer.h"
// #include "CImg.h"
#include <chrono>
#include <random>
#include <iterator>
#include <iostream>
#include <fstream>
#include <map>
#include <DepthMeshLib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <string.h>

#define LOG(ERROR) std::cout

using namespace std::chrono;

// output the depth mpa as one channel numpy array
void output_depht_map_bin(const char *filename, const void *ptr, const int width, const int height)
{
  FILE *pFile;
  pFile = fopen(filename, "wb");
  //Some calculations to fill a[]
  fwrite(ptr, 1, width * height * sizeof(float), pFile);
  fclose(pFile);
}


// write the depth map to a .dpt file (Sintel format).
void output_depth_map_dpt(const char *filename, const void *ptr, const int width, const int height)
{
  if (filename == nullptr)
  {
    std::cout << "Error in " << __FUNCTION__ << ": empty filename.";
    return;
  }

  const char *dot = strrchr(filename, '.');
  if (dot == nullptr)
  {
    std::cout << "Error in " << __FUNCTION__ << ": extension required in filename " << filename << ".";
    return;
  }

  if (strcmp(dot, ".dpt") != 0)
  {
    std::cout << "Error in " << __FUNCTION__ << ": filename " << filename << " should have extension '.dpt'.";
    return;
  }

  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
  {
    std::cout << "Error in " << __FUNCTION__ << ": could not open " << filename;
    return;
  }

  // write the header
  fprintf(stream, "PIEH");
  if ((int)fwrite(&width, sizeof(int), 1, stream) != 1 || (int)fwrite(&height, sizeof(int), 1, stream) != 1)
  {
    std::cout << "Error in writeSintelDptFile(" << filename << "): problem writing header.";
    return;
  }

  // write the depth data
  fwrite(ptr, 1, width * height * sizeof(float), stream);

  fclose(stream);
}

// Writes a depth map to a .dpt file (Sintel format).
void writeSintelDptFile(const char* filename, cv::Mat1f img)
{
	if (filename == nullptr)
	{
		LOG(ERROR) << "Error in writeSintelDptFile: empty filename.";
		return;
	}

	const char* dot = strrchr(filename, '.');
	if (dot == nullptr)
	{
		LOG(ERROR) << "Error in writeSintelDptFile: extension required in filename " << filename << ".";
		return;
	}

	if (strcmp(dot, ".dpt") != 0)
	{
		LOG(ERROR) << "Error in writeSintelDptFile: filename " << filename << " should have extension '.dpt'.";
		return;
	}

	int width = img.cols, height = img.rows, nBands = img.channels();

	if (nBands != 1)
	{
		LOG(ERROR) << "Error in writeSintelDptFile(" << filename << "): image must have 1 channel.";
		return;
	}

	FILE* stream = fopen(filename, "wb");
	if (stream == 0)
	{
		LOG(ERROR) << "Error in writeSintelDptFile: could not open " << filename;
		return;
	}

	// write the header
	fprintf(stream, "PIEH");
	if ((int)fwrite(&width, sizeof(int), 1, stream) != 1
	    || (int)fwrite(&height, sizeof(int), 1, stream) != 1)
	{
		LOG(ERROR) << "Error in writeSintelDptFile(" << filename << "): problem writing header.";
		return;
	}

	// write the rows
	for (int y = 0; y < height; y++)
	{
		if ((int)fwrite(img.row(y).data, sizeof(float), width, stream) != width)
		{
			LOG(ERROR) << "Error in writeSintelDptFile(" << filename << "): problem writing data.";
			return;
		}
	}

	fclose(stream);
}


void visual_depth_map(const cv::Mat &input, cv::Mat &color_map)
{
  double min, max;
  cv::minMaxLoc(input, &min, &max);

  cv::Mat input_norm;
  // cv::normalize(input, input_norm, 255,0);
  input_norm = (input - min) * (255.0 / (max - min));
  input_norm.convertTo(color_map, CV_8UC1); //, scale, 0);

  // cv::Mat color_map_turbo(256,3,CV_8UC3);
  // memcpy(color_map_turbo.data, static_cast<unsigned char *>(turbo_srgb_bytes), 256 * 3);
  // cv::applyColorMap(input_norm, color_map, color_map_turbo);
  cv::applyColorMap(color_map, color_map, cv::COLORMAP_JET);
}

int main(int argc, char *argv[])
{

  auto model_start = high_resolution_clock::now();

  ASSERT(argc == 9, "Usage: ./Path/to/ReplicaViewer mesh.ply textures glass.sur[glass.sur/n] cameraPositions.txt[file.txt/n] spherical[y/n] outputDir width height");

  bool noSurfaceFile = std::string(argv[3]).compare(std::string("n")) == 0 || !pangolin::FileExists(std::string(argv[3]));
  bool noTxtFile = std::string(argv[4]).compare(std::string("n")) == 0 || !pangolin::FileExists(std::string(argv[4]));
  bool spherical = std::string(argv[5]).compare(std::string("y")) == 0;
  int width = std::stoi(std::string(argv[7]));
  int height = std::stoi(std::string(argv[8]));

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  const std::string outputDir(argv[6]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));
  ASSERT(pangolin::FileExists(outputDir));

  //get scene name
  std::string scene;
  const size_t last_slash_idx = meshFile.rfind("/");
  const size_t second2last_slash_idx = meshFile.substr(0, last_slash_idx).rfind("/");
  if (std::string::npos != last_slash_idx)
  {
    scene = meshFile.substr(second2last_slash_idx + 1, last_slash_idx - second2last_slash_idx - 1);
    std::cout << "Generating from scene " << scene << std::endl;
  }

  std::string surfaceFile;
  if (!noSurfaceFile)
  {
    surfaceFile = std::string(argv[3]);
    ASSERT(pangolin::FileExists(surfaceFile));
  }

  std::string navPositions;
  bool navCam = !noTxtFile;
  if (!noTxtFile)
  {
    navPositions = std::string(argv[4]);
    ASSERT(pangolin::FileExists(navPositions));
  }

  // load txt file for data generation
  // FORMAT:
  // idx, camera_position_x, camera_position_y, camera_position_z, rotation_x, rotation_y, rotation_z
  std::vector<std::vector<float>> cameraPos;
  if (navCam)
  {
    std::fstream in(navPositions);
    std::string line;
    int i = 0;
    while (std::getline(in, line))
    {
      float value;
      std::stringstream ss(line);
      cameraPos.push_back(std::vector<float>());

      while (ss >> value)
      {
        cameraPos[i].push_back(value);
      }
      ++i;
    }
  }

  // Setup EGL
  EGLCtx egl;
  egl.PrintInformation();

  //Don't draw backfaces
  GLenum frontFace = GL_CW;
  if (spherical)
  {
    glFrontFace(frontFace);
  }
  else
  {
    frontFace = GL_CCW;
    glFrontFace(frontFace);
  }

  // Setup a framebuffer
  // texture for RGB, depth image rendering
  pangolin::GlTexture render_rgb(width, height, GL_RGB8);
  pangolin::GlTexture render_depth(width, height, GL_R32F);
  pangolin::GlRenderBuffer render_buffer(width, height);

  pangolin::GlFramebuffer frameBuffer_rgb(render_rgb, render_buffer);
  pangolin::GlFramebuffer frameBuffer_depth(render_depth, render_buffer);

  // Setup a camera
  std::vector<float> initCam = {0, 0.5, -0.6230950951576233}; //default
  if (navCam)
  {
    initCam = cameraPos[0];
    std::cout << "First camera position:" << initCam[1] << " " << initCam[2] << " " << initCam[3];
  }

  // set first view/image
  Eigen::Matrix3f model_mat;
  model_mat = Eigen::AngleAxisf(cameraPos[0][6] / 180.0 * M_PI, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(cameraPos[0][5] / 180.0 * M_PI, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(cameraPos[0][4] / 180.0 * M_PI, Eigen::Vector3f::UnitX());

  Eigen::Vector3f eye_point(cameraPos[0][1], cameraPos[0][2], cameraPos[0][3]);
  Eigen::Vector3f target_point = eye_point + model_mat * Eigen::Vector3f(1, 0, 0);
  Eigen::Vector3f up = model_mat * Eigen::Vector3f::UnitZ();
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_BottomLeft(
          width,
          height,
          width / 2.0f,
          width / 2.0f,
          (width - 1.0f) / 2.0f,
          (height - 1.0f) / 2.0f,
          0.1f,
          100.0f),
      pangolin::ModelViewLookAtRDF(eye_point[0], eye_point[1], eye_point[2],
                                   target_point[0], target_point[1], target_point[2],
                                   up[0], up[1], up[2]));

  // Start at some origin
  Eigen::Matrix4d T_camera_world = s_cam.GetModelViewMatrix();

  // For cubemap dataset: rotation matrix of 90 degree for each face of the cubemap
  // t -> t -> t -> u -> d
  Eigen::Transform<double, 3, Eigen::Affine> t(Eigen::AngleAxis<double>(0.5 * M_PI, Eigen::Vector3d::UnitY()));
  Eigen::Transform<double, 3, Eigen::Affine> u(Eigen::AngleAxis<double>(0.5 * M_PI, Eigen::Vector3d::UnitX()));
  Eigen::Transform<double, 3, Eigen::Affine> d(Eigen::AngleAxis<double>(M_PI, Eigen::Vector3d::UnitX()));
  Eigen::Matrix4d R_side = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d R_up = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d R_down = Eigen::Matrix4d::Identity();
  R_side = t.matrix();
  R_up = u.matrix();
  R_down = d.matrix();

  // load mirrors
  std::vector<MirrorSurface> mirrors;
  if (!noSurfaceFile && surfaceFile.length() > 1)
  {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);

    for (size_t i = 0; i < json.size(); i++)
    {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirrors" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // load mesh and textures
  PTexMesh ptexMesh(meshFile, atlasFolder, spherical);
  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float_t> depth_map(width, height);

  size_t numSpots = 20; //default
  if (navCam)
  {
    numSpots = cameraPos.size();
  }
  srand(2019); //random seed

  // 0:front, 1:right, 2:back, 3:left, 4:up, 5:down
  std::map<int, char> image_orientation_map = {{0, 'F'}, {1, 'R'}, {2, 'B'}, {3, 'L'}, {4, 'U'}, {5, 'D'}};
  // rendering the dataset (double equirect pair + interpolation + extrapolation + forward extrapolation)
  for (size_t j = 0; j < numSpots; j++)
  {
    //cubemap dataset
    for (size_t i = 0; i < 6; ++i)
    {
      // 0) Render RGB image
      frameBuffer_rgb.Bind();
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
      glEnable(GL_CULL_FACE);
      ptexMesh.SetExposure(0.01);
      ptexMesh.Render(s_cam, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f));
      render_rgb.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);
      glDisable(GL_CULL_FACE);
      glPopAttrib(); //GL_VIEWPORT_BIT
      frameBuffer_rgb.Unbind();
      // Download and save
      char cubemapFilename[1000];
      snprintf(cubemapFilename, 1000, "%s/%s_%04zu_%c.jpg", outputDir.c_str(), scene.c_str(), j, image_orientation_map[i]);
      // generate RGB images
      pangolin::SaveImage(
          image.UnsafeReinterpret<uint8_t>(),
          pangolin::PixelFormatFromString("RGB24"),
          std::string(cubemapFilename), 100.0);

      // 1) Render Depth map
      frameBuffer_depth.Bind();
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
      glEnable(GL_CULL_FACE);
      ptexMesh.SetExposure(0.01);
      ptexMesh.RenderDepth(s_cam, 1, Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f));
      render_depth.Download(depth_map.ptr, GL_RED, GL_FLOAT);
      glDisable(GL_CULL_FACE);
      glPopAttrib(); //GL_VIEWPORT_BIT
      frameBuffer_depth.Unbind();
      // Download and save
      //char cubemapFilename[1000];
      snprintf(cubemapFilename, 1000, "%s/%s_%04zu_%c_depth.bin", outputDir.c_str(), scene.c_str(), j, image_orientation_map[i]);
      // output depth map to bin file
      output_depht_map_bin(cubemapFilename, depth_map.ptr, width, height);
      //
      cv::Mat depth_map_mat(height, width, CV_32FC1);
      memcpy(depth_map_mat.data, depth_map.ptr, width * height * sizeof(float));
      cv::Mat depth_map_mat_vis;
      visual_depth_map(depth_map_mat, depth_map_mat_vis);
      snprintf(cubemapFilename, 1000, "%s/%s_%04zu_%c_depth_vis.jpg", outputDir.c_str(), scene.c_str(), j, image_orientation_map[i]);
      cv::imwrite(cubemapFilename, depth_map_mat_vis);

      // 2) Update the camera orientation
      if (i < 3)
      {
        //turn to the side
        Eigen::Matrix4d curr_spot_cam_to_world = s_cam.GetModelViewMatrix();
        T_camera_world = R_side.inverse() * curr_spot_cam_to_world;
        s_cam.GetModelViewMatrix() = T_camera_world;
      }
      else if (i == 3)
      {
        //look upward by 90 degree
        Eigen::Matrix4d curr_spot_cam_to_world = s_cam.GetModelViewMatrix();
        T_camera_world = R_up.inverse() * curr_spot_cam_to_world;
        s_cam.GetModelViewMatrix() = T_camera_world;
      }
      else if (i == 4)
      {
        //look downward by 180 degree
        Eigen::Matrix4d curr_spot_cam_to_world = s_cam.GetModelViewMatrix();
        T_camera_world = R_down.inverse() * curr_spot_cam_to_world;
        s_cam.GetModelViewMatrix() = T_camera_world;
      }
    } // end i

    // update camera pose
    if (navCam)
    {
      if (j + 1 < numSpots)
      {
        Eigen::Matrix3f model_mat;
        model_mat = Eigen::AngleAxisf(cameraPos[j + 1][6] / 180.0 * M_PI, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(cameraPos[j + 1][5] / 180.0 * M_PI, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(cameraPos[j + 1][4] / 180.0 * M_PI, Eigen::Vector3f::UnitX());

        Eigen::Vector3f eye_point(cameraPos[j + 1][1], cameraPos[j + 1][2], cameraPos[j + 1][3]);
        Eigen::Vector3f target_point = eye_point + model_mat * Eigen::Vector3f(1, 0, 0);
        Eigen::Vector3f up = model_mat * Eigen::Vector3f::UnitZ();

        s_cam.SetModelViewMatrix(
            pangolin::ModelViewLookAtRDF(
                eye_point[0], eye_point[1], eye_point[2],
                target_point[0], target_point[1], target_point[2],
                up[0], up[1], up[2]));
      }
    }
    else
    {
      continue;
    }
    std::cout << "\r Spot " << j + 1 << "/" << numSpots << std::endl;

  } // end j

  auto model_stop = high_resolution_clock::now();
  auto model_duration = duration_cast<microseconds>(model_stop - model_start);
  std::cout << "Time taken rendering the model " << navPositions.substr(0, navPositions.length() - 9).c_str() << ": " << model_duration.count() << " microseconds" << std::endl;

  return 0;
}
