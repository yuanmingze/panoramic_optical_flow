#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Eigen>

#include <string>
#include <vector>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include <type_traits>
#include <algorithm>

#include <experimental/filesystem>
// #include <filesystem>
// namespace fs = std::filesystem;
namespace fs = std::experimental::filesystem;

#define TAR 202021.250

// 
double thresh_x_max = 50;
double thresh_x_min = -50;
double thresh_y_max = 50;
double thresh_y_min = -50;


inline bool isFlowCorrect(cv::Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static cv::Vec3b computeColor(float fx, float fy)
{
    static bool first = true;
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static cv::Vec3i colorWheel[NCOLS];
    if (first)
    {
        int k = 0;
        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 255 * i / RY, 0);
        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 - 255 * i / YG, 255, 0);
        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255, 255 * i / GC);
        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = cv::Vec3i(0, 255 - 255 * i / CB, 255);
        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255 * i / BM, 0, 255);
        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = cv::Vec3i(255, 0, 255 - 255 * i / MR);
        first = false;
    }
    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;
    // const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const float fk = (a + 1.0f) / 2.0f * (NCOLS); // TODO double check
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;
    cv::Vec3b pix;
    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range
        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }
    return pix;
}

static void drawOpticalFlow(const cv::Mat_<float> &flowx, const cv::Mat_<float> &flowy, cv::Mat &dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));
    // determine motion range:
    float maxrad = maxmotion;
    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                cv::Point2f u(flowx(y, x), flowy(y, x));
                if (!isFlowCorrect(u))
                    continue;
                maxrad = cv::max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }
    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            cv::Point2f u(flowx(y, x), flowy(y, x));
            if (isFlowCorrect(u))
                dst.at<cv::Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}


template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
           // unless the result is subnormal
           || std::fabs(x - y) < std::numeric_limits<T>::min();
}

Eigen::Vector3f cartesian2spherical(Eigen::Vector3f point)
{
    float radius = point.norm();

    // Special case: zero radius
    if (radius < 1e-10)
        return Eigen::Vector3f(0, 0, 0);

    float polar = acosf(-point.y() / radius);
    float azimuth = atan2f(-point.x(), -point.z());

    azimuth = fmod(fmod(azimuth, 2 * M_PI) + 2 * M_PI, 2 * M_PI); // find positive remainder, i.e. in range [0, ]

    return Eigen::Vector3f(radius, azimuth, polar);
}

Eigen::Vector3f depthmap2spherical(const cv::Mat &depth_map, const int azimuth_index, const int polar_index)
{
    // Compute spherical coordinates.
    // for openvslam coordinate
    // double azimuth = (azimuth_index + 0.5f) / depth_map.cols * 2 * M_PI;
    // double polar = double(polar_index) / (depth_map.rows - 1.f) * M_PI; // stretch top/bottom     to caps

    // replia coordinate system
    double azimuth = (-double(azimuth_index) / double(depth_map.cols) + 0.5) * 2 * M_PI;
    double polar = (-double(polar_index) / double(depth_map.rows) + 0.5) * M_PI;

    // Convert from spherical coordinates to Cartesian coordinates (x, y, z).
    //   - azimuth: 0 is -z, pi / 2 is -x, pi is +z, 3pi / 4 is +x
    //   - polar: 0 is -y, pi / 2 is equatorial plane, pi is +y
    double radius = depth_map.at<float>(polar_index, azimuth_index);

    return Eigen::Vector3f(radius, azimuth, polar);
}

Eigen::Vector3f spherical2cartesian(const Eigen::Vector3f &point)
{
    float radius = point.x();
    float azimuth = point.y();
    float polar = point.z();

    // // for openvslam coordinate
    // float x = -radius * sinf(polar) * sinf(azimuth);
    // float y = -radius * cosf(polar);
    // float z = -radius * sinf(polar) * cosf(azimuth);

    // replia coordinate system
    float x = radius * cosf(polar) * cosf(azimuth);
    float y = radius * cosf(polar) * sinf(azimuth);
    float z = radius * sinf(polar);

    return Eigen::Vector3f(x, y, z);
}

void writeSphereMesh(const cv::Mat &depth_map, const std::string &filename)
{
    // Open the mesh file.
    std::ofstream objFile;
    objFile.open(filename, std::fstream::out);

    // TODO: error handling ... overwrite existing file or not?

    // Write vertices for depth map: "v x y z"
    for (int i = 0; i < depth_map.rows; i++)
    {
        for (int j = 0; j < depth_map.cols; j++)
        {
            // Compute spherical coordinates.
            Eigen::Vector3f spherical = depthmap2spherical(depth_map, j, i);

            // Convert from spherical coordinates to Cartesian coordinates (x, y, z).
            Eigen::Vector3f vertex = spherical2cartesian(spherical);

            // Write vertex coordinates.
            objFile << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << '\n';
        }
    }

    // Write triangle faces: "f i1 i2 i3"
    for (int i = 0; i < depth_map.rows - 1; i++) // skip last row, as each row outputs triang    les connecting to t row
    {
        for (int j = 0; j < depth_map.cols; j++)
        {
            // Triangle vertices of quad: (t_op, b_ottom) x (l_eft, r_ight)
            unsigned int index_tl = depth_map.cols * i + j + 1;
            unsigned int index_tr = depth_map.cols * i + (j + 1) % depth_map.cols + 1;
            unsigned int index_bl = depth_map.cols * (i + 1) + j + 1;
            unsigned int index_br = depth_map.cols * (i + 1) + (j + 1) % depth_map.cols + 1;

            objFile << "f " << index_tl << " " << index_tr << " " << index_bl << '\n'; // top-left angle
            objFile << "f " << index_tr << " " << index_br << " " << index_bl << '\n'; // bottom-right angle
        }
    }

    objFile.close();
}

void comput_occlusion(const cv::Mat &src_rgb,
                      const cv::Mat &of_forward_wo_occ,
                      const cv::Mat &src_primitive_id,
                      const cv::Mat &tar_primitive_id,
                      cv::Mat &of_forward_w_occ,
                      cv::Mat &occ)
{
    std::cout << "comput_occlusion use primtive id" << std::endl;
    occ = cv::Mat::zeros(src_rgb.size(), CV_8UC3);
    of_forward_w_occ = of_forward_wo_occ.clone();

    for (int y = 0; y < src_rgb.rows; ++y)
    {
        for (int x = 0; x < src_rgb.cols; ++x)
        {
            bool occlusive = false;
            cv::Point2f f = of_forward_wo_occ.at<cv::Point2f>(y, x);
            int x_new = x + f.x > 0 ? int(x + f.x + 0.5) : int(x + f.x - 0.5);
            int y_new = y + f.y > 0 ? int(y + f.y + 0.5) : int(y + f.y - 0.5);

            // case 1: out image range
            if (x_new < 0 || x_new >= src_rgb.cols ||
                y_new < 0 || y_new >= src_rgb.rows)
            {
                occlusive = true;
            }
            else
            {
                // case 2: the primtive id is not equal
                int src_primtive_id = (int)src_primitive_id.at<float>(y, x);
                if ((x + f.x) != x_new)
                {
                    int x_new_l = int(x + f.x);
                    int x_new_r = int(x + f.x) + 1;
                    int y_new_t = int(y + f.y);
                    int y_new_b = int(y + f.y) + 1;

                    if (x_new_l < 0 || x_new_r > src_rgb.cols || y_new_t < 0 || y_new_b > src_rgb.rows)
                    {
                        continue;
                    }
                    int tar_primtive_id_tl = (int)tar_primitive_id.at<float>(y_new_t, x_new_l);
                    int tar_primtive_id_tr = (int)tar_primitive_id.at<float>(y_new_t, x_new_r);
                    int tar_primtive_id_bl = (int)tar_primitive_id.at<float>(y_new_b, x_new_l);
                    int tar_primtive_id_br = (int)tar_primitive_id.at<float>(y_new_b, x_new_r);
                    // if(!almost_equal(src_primtive_id, tar_primtive_id, 3))
                    if (src_primtive_id != tar_primtive_id_tl &&
                        src_primtive_id != tar_primtive_id_tr &&
                        src_primtive_id != tar_primtive_id_bl &&
                        src_primtive_id != tar_primtive_id_br)
                    {
                        occlusive = true;
                    }
                }
                else
                {
                    int tar_primtive_id = (int)tar_primitive_id.at<float>(y_new, x_new);
                    if (src_primtive_id != tar_primtive_id)
                    {
                        occlusive = true;
                    }
                }
            }

            if (occlusive)
            {
                occ.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                of_forward_w_occ.at<cv::Point2f>(y, x) = cv::Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
            }
        }
    }
}

//
void comput_occlusion(const cv::Mat &src_rgb,
                      const cv::Mat &src_depth,
                      const cv::Mat &target_rgb,
                      const cv::Mat &target_depth,
                      const cv::Mat &src_of_wo_occ,
                      const cv::Mat &src_of_z,
                      cv::Mat &src_of_occ,
                      cv::Mat &occ)
{
    std::cout << "comput_occlusion" << std::endl;
    occ = cv::Mat::zeros(src_rgb.size(), CV_8UC3);
    src_of_occ = src_of_wo_occ.clone();

    for (int y = 0; y < src_rgb.rows; ++y)
    {
        for (int x = 0; x < src_rgb.cols; ++x)
        {
            bool occlusive = false;
            cv::Point2f f = src_of_wo_occ.at<cv::Point2f>(y, x);
            int x_new = x + f.x > 0 ? int(x + f.x + 0.5) : int(x + f.x - 0.5);
            int y_new = y + f.y > 0 ? int(y + f.y + 0.5) : int(y + f.y - 0.5);

            // case 1: out image range
            if (x_new < 0 || x_new >= src_rgb.cols ||
                y_new < 0 || y_new >= src_rgb.rows)
            {
                occlusive = true;
            }
            else
            {
                // case 2: cover by other pixel
                float src_depth_pixel = src_depth.at<float>(y, x);
                float src_of_z_pixel = src_of_z.at<float>(y, x);
                float tar_depth_pixel = target_depth.at<float>(y_new, x_new);
                //if (!almost_equal(src_depth_pixel + src_of_z_pixel, tar_depth_pixel, 1000))
                //if((src_depth_pixel + src_of_z_pixel + 0.05) > tar_depth_pixel)
                if (!(std::abs(src_depth_pixel + src_of_z_pixel - tar_depth_pixel) < 0.4))
                {
                    occlusive = true;
                }

                // // case 3: the color is very different
                // cv::Vec3b src_rgb_pixel = src_rgb.at<cv::Vec3b>(y, x);
                // cv::Vec3b tar_rgb_pixel = target_rgb.at<cv::Vec3b>(y_new, x_new);
                // if (std::abs(src_rgb_pixel[0] - tar_rgb_pixel[0]) > 30 ||
                //     std::abs(src_rgb_pixel[1] - tar_rgb_pixel[1]) > 30 ||
                //     std::abs(src_rgb_pixel[2] - tar_rgb_pixel[2]) > 30)
                // {
                //     occlusive = true;
                // }
            }

            if (occlusive)
            {
                occ.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                src_of_occ.at<cv::Point2f>(y, x) = cv::Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
            }
        }
    }
}

void load_depth_png(const std::string &file_path, cv::Mat &depth)
{
    cv::Mat depth_raw = cv::imread(file_path);
    depth_raw.convertTo(depth_raw, CV_32FC3); // or CV_32F works (too)

    cv::Mat depth_rgb[3];
    cv::split(depth_raw, depth_rgb);
    depth = cv::Mat::zeros(depth_raw.rows, depth_raw.cols, CV_32FC1);
    cv::Mat b = depth_rgb[0];
    cv::Mat g = depth_rgb[1];
    cv::Mat r = depth_rgb[2];
    depth = r + g * 256.0 + b * 256.0 * 256.0;
    depth = depth / 65536.0;
}

void load_depth_bin(const std::string &file_path, cv::Mat &depth, const int width, const int height)
{
    // load binary data from disk
    FILE *pFile;
    size_t lSize;
    char *buffer;
    size_t result;

    pFile = fopen(file_path.c_str(), "rb");
    if (pFile == NULL)
    {
        fputs("File error", stderr);
        exit(1);
    }

    // obtain file size:
    fseek(pFile, 0, SEEK_END);
    lSize = ftell(pFile);
    rewind(pFile);

    // allocate memory to contain the whole file:
    buffer = (char *)malloc(sizeof(float) * width * height);
    if (buffer == NULL)
    {
        fputs("Memory error", stderr);
        exit(2);
    }

    // copy the file into the buffer:
    result = fread(buffer, 1, lSize, pFile);
    if (result != lSize)
    {
        fputs("Reading error", stderr);
        exit(3);
    }

    /* the whole file is now loaded in the memory buffer. */
    // std::vector<cv::Mat> of_list;
    depth = cv::Mat::zeros(height, width, CV_32FC1);
    memcpy(depth.data, buffer, sizeof(float) * width * height);

    // terminate
    fclose(pFile);
    free(buffer);
}

void load_of_png(const std::string &file_path, cv::Mat &of)
{
    cv::Mat of_raw = cv::imread(file_path);
    of_raw.convertTo(of_raw, CV_32FC3); // or CV_32F works (too)

    cv::Mat of_x_rgb[3];
    cv::split(of_raw, of_x_rgb);
    of = cv::Mat::zeros(of_raw.rows, of_raw.cols, CV_32FC1);
    cv::Mat b = of_x_rgb[0];
    cv::Mat g = of_x_rgb[1];
    cv::Mat r = of_x_rgb[2];
    of = r + g * 256.0 + b * 256.0 * 256.0;
    of = of / 1024.0 - 4096.0;
}

void load_of_bin(const std::string &file_path,
                 std::vector<cv::Mat> &of_list,
                 cv::Mat &primitive_id,
                 int width, int height)
{
    FILE *pFile;
    size_t lSize;
    char *buffer;
    size_t result;

    pFile = fopen(file_path.c_str(), "rb");
    if (pFile == NULL)
    {
        fputs("File error", stderr);
        exit(1);
    }

    // obtain file size:
    fseek(pFile, 0, SEEK_END);
    lSize = ftell(pFile);
    rewind(pFile);

    // allocate memory to contain the whole file:
    buffer = (char *)malloc(sizeof(float) * width * height * 4);
    if (buffer == NULL)
    {
        fputs("Memory error", stderr);
        exit(2);
    }

    // copy the file into the buffer:
    result = fread(buffer, 1, lSize, pFile);
    if (result != lSize)
    {
        fputs("Reading error", stderr);
        exit(3);
    }

    /* the whole file is now loaded in the memory buffer. */
    // std::vector<cv::Mat> of_list;
    cv::Mat of = cv::Mat::zeros(height, width, CV_32FC4);
    memcpy(of.data, buffer, sizeof(float) * width * height * 4);
    cv::split(of, of_list);
    primitive_id = of_list[2];
    of_list.pop_back();
    of_list.pop_back();

    // terminate
    fclose(pFile);
    free(buffer);
}

void show_image(const cv::Mat &data, const std::string info = "")
{
    std::cout << "show visualed " << info << std::endl;
    cv::namedWindow(info, cv::WINDOW_AUTOSIZE);
    if (data.dims == 2)
    {
        cv::Mat sobelx;
        Sobel(data, sobelx, CV_32F, 1, 0);
        double minVal, maxVal;
        minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
        cv::Mat draw;
        sobelx.convertTo(draw, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        imshow(info, draw);
        cv::waitKey(); // Wait for a keystroke in the window
    }
    else if (data.dims == 3)
    {
        imshow(info, data);
        cv::waitKey(); // Wait for a keystroke in the window
    }
}

void warp_image_backward(const cv::Mat &src_img, const cv::Mat &of, cv::Mat &src_img_warpped)
{
    cv::Mat map_x(of.size(), CV_32FC1);
    cv::Mat map_y(of.size(), CV_32FC1);
    for (int y = 0; y < src_img.rows; ++y)
    {
        for (int x = 0; x < src_img.cols; ++x)
        {
            cv::Point2f f = of.at<cv::Point2f>(y, x);
            map_x.at<float>(y, x) = (float)(x + f.x);
            map_y.at<float>(y, x) = (float)(y + f.y);
        }
    }
    cv::remap(src_img, src_img_warpped, map_x, map_y, cv::INTER_LINEAR);
}

void warp_image_forward(const cv::Mat &src_img, const cv::Mat &of, cv::Mat &src_img_warpped)
{
    src_img_warpped = cv::Mat::zeros(src_img.size(), CV_8UC3);

    for (int y = 0; y < src_img.rows; ++y)
    {
        for (int x = 0; x < src_img.cols; ++x)
        {
            cv::Point2f f = of.at<cv::Point2f>(y, x);
            int x_new = x + f.x > 0 ? int(x + f.x + 0.5) : int(x + f.x - 0.5);
            int y_new = y + f.y > 0 ? int(y + f.y + 0.5) : int(y + f.y - 0.5);
            if (x_new < 0 || x_new >= src_img.cols)
            {
                continue;
            }
            if (y_new < 0 || y_new >= src_img.rows)
            {
                continue;
            }
            // if (src_img_warpped.at<cv::Vec3b>(y_new, x_new)[0] != 0 ||
            //     src_img_warpped.at<cv::Vec3b>(y_new, x_new)[1] != 0 ||
            //     src_img_warpped.at<cv::Vec3b>(y_new, x_new)[2] != 0)
            // {
            //     continue;
            // }
            src_img_warpped.at<cv::Vec3b>(y_new, x_new) = src_img.at<cv::Vec3b>(y, x);
        }
    }
}

void test_occlusion()
{
    // 00--> 01 --> 02
    std::string data_root = "/mnt/sda1/workspace_linux/replica360/data/test_00/";

    // 1) read data

    // rgb image
    std::string src_img_path = data_root + "hotel_0_0001_rgb.jpg";
    std::string tar_img_path = data_root + "hotel_0_0002_rgb.jpg";
    cv::Mat src_rgb = cv::imread(src_img_path);
    cv::Mat tar_rgb = cv::imread(tar_img_path);

    // depth image
    std::string src_depth_path = data_root + "hotel_0_0001_depth.png";
    std::string tar_depth_path = data_root + "hotel_0_0002_depth.png";
    cv::Mat src_depth;
    load_depth_png(src_depth_path, src_depth);
    cv::Mat tar_depth;
    load_depth_png(tar_depth_path, tar_depth);

    // optical flow
    // std::string of_x_path = data_root + "hotel_0_0001_opticalflow_x.png";
    // std::string of_y_path = data_root + "hotel_0_0001_opticalflow_y.png";
    // std::string of_z_path = data_root + "hotel_0_0001_opticalflow_z.png";
    // cv::Mat of_x  = cv::Mat::zeros(src_rgb.size(), CV_32FC1);
    // load_of(of_x_path, of_x);
    // cv::Mat of_y = cv::Mat::zeros(src_rgb.size(), CV_32FC1);
    // load_of(of_y_path, of_y);
    // cv::Mat of_z = cv::Mat::zeros(src_rgb.size(), CV_32FC1);
    // load_of(of_z_path, of_z);

    std::vector<cv::Mat> of_list_forward;
    std::vector<cv::Mat> of_list_backward;
    // of_list.push_back(of_x);
    // of_list.push_back(of_y);
    std::string of_path_forward = data_root + "hotel_0_0001_opticalflow_forward.bin";
    std::string of_path_backward = data_root + "hotel_0_0002_opticalflow_forward.bin";

    cv::Mat src_primitive_id;
    cv::Mat tar_primitive_id;

    load_of_bin(of_path_forward, of_list_forward, src_primitive_id, src_rgb.cols, src_rgb.rows);
    load_of_bin(of_path_backward, of_list_backward, tar_primitive_id, src_rgb.cols, src_rgb.rows);

    cv::Mat of_forward_wo_occ;
    cv::Mat of_backward_wo_occ;
    cv::merge(of_list_forward, of_forward_wo_occ);
    cv::merge(of_list_backward, of_backward_wo_occ);

    // output visualizaion
    cv::Mat of_visual;
    //FlowIOOpenCVWrapper::flowToColor(of_forward_wo_occ);
    cv::Mat planes[2];
    cv::split(of_forward_wo_occ, planes);
    drawOpticalFlow(planes[0], planes[1], of_visual);

    cv::imwrite(data_root + "hotel_0_0001_opticalflow_forward_wo_occ_visual.jpg", of_visual);
    //of_visual = FlowIOOpenCVWrapper::flowToColor(of_backward_wo_occ);
    cv::split(of_backward_wo_occ, planes);
    drawOpticalFlow(planes[0], planes[1], of_visual);
    cv::imwrite(data_root + "hotel_0_0001_opticalflow_backward_wo_occ_visual.jpg", of_visual);

    // comput of without occ
    cv::Mat of_forward_occ;
    cv::Mat occ;
    // comput_occlusion(src_rgb, src_depth, tar_rgb, tar_depth,of_wo_occ, of_z, of_occ, occ);
    // show_image(tar_primitive_id);
    comput_occlusion(src_rgb, of_forward_wo_occ, src_primitive_id, tar_primitive_id, of_forward_occ, occ);
    cv::imwrite(data_root + "hotel_0_0001_opticalflow_occ.jpg", occ);
    // cv::Mat of_occ_visual = FlowIOOpenCVWrapper::flowToColor(of_forward_occ);
    cv::Mat of_occ_visual ;
    cv::split(of_forward_occ, planes);
    drawOpticalFlow(planes[0], planes[1], of_occ_visual);
    cv::imwrite(data_root + "hotel_0_0001_opticalflow_occ_visual.jpg", of_occ_visual);

    // 2) warp image with of_wo_occ
    cv::Mat warp_src_rgb_wo_occ;
    warp_image_forward(src_rgb, of_forward_wo_occ, warp_src_rgb_wo_occ);
    cv::imwrite(data_root + "hotel_0_0001_rgb_warpped_wo_occ.jpg", warp_src_rgb_wo_occ);

    // warp image with of_occ
    cv::Mat warped_src_rgb_occ;
    warp_image_forward(src_rgb, of_forward_occ, warped_src_rgb_occ);
    cv::imwrite(data_root + "hotel_0_0001_rgb_warpped_occ.jpg", warped_src_rgb_occ);
}

void write_image_jpeg(const std::string &path, const cv::Mat &data)
{
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(20);
    try
    {
        cv::imwrite(path, data, compression_params);
    }
    catch (std::runtime_error &ex)
    {
        fprintf(stderr, "Exception converting image to JPEG format: %s\n", ex.what());
    }
}

void flowrite(cv::String addr, cv::Mat &flo)
{

    cv::Size imgsize = flo.size();
    int high = imgsize.height;
    int width = imgsize.width;
    std::ofstream fout(addr, std::ios::binary);
    char *data = flo.ptr<char>(0);
    if (!fout)
    {
        return;
    }
    else
    {
        fout << "PIEH";
        fout.write((char *)&width, sizeof(int));
        fout.write((char *)&high, sizeof(int));
        fout.write(data, high * width * 2 * sizeof(float));
    }
    fout.close();
}

void floread(cv::String addr, cv::Mat &flo)
{
    std::ifstream fin(addr, std::ios::binary);
    char buffer[sizeof(float)];
    fin.read(buffer, sizeof(float));
    float tar = ((float *)buffer)[0];
    if (tar != TAR)
    {
        fin.close();
        return;
    }
    fin.read(buffer, sizeof(int));
    int high = ((int *)buffer)[0];
    fin.read(buffer, sizeof(int));
    int width = ((int *)buffer)[0];
    flo = cv::Mat(cv::Size(high, width), CV_32FC2);
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < high; j++)
        {
            if (!fin.eof())
            {
                float *data = flo.ptr<float>(i, j);
                fin.read(buffer, sizeof(float));
                data[0] = ((float *)buffer)[0];
                fin.read(buffer, sizeof(float));
                data[1] = ((float *)buffer)[0];
            }
        }
    }
    fin.close();
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "specify the option root_dir." << std::endl;
        std::cout << "optio: root_dir center_view[y/n]" << std::endl;
        return -1;
    }

    // root directory
    std::string root_dir(argv[1]);
    std::string depth_convert_enable(argv[2]);
    std::string of_convert_enable(argv[3]);
    if (depth_convert_enable != "y" && depth_convert_enable != "n")
    {
        std::cout << "please specify the depth_convert_enable [y/n]" << std::endl;
    }
    if (of_convert_enable != "y" && of_convert_enable != "n")
    {
        std::cout << "please specify the of_convert_enable [y/n]" << std::endl;
    }
    std::cout << "data convertion root folder is :" <<  root_dir << std::endl;

    // get image resolution
    std::string first_rgb_image = root_dir + "/0000_rgb.jpg";
    cv::Mat first_rgb_image_mat = cv::imread(first_rgb_image);
    int width = first_rgb_image_mat.cols;
    int height = first_rgb_image_mat.rows;
    std::cout << "the image width is :" << width << "  Height is :" << height << std::endl;

    // set of visual threshold
    thresh_x_max = width * 0.75;
    thresh_x_min = -width * 0.75;
    thresh_y_max = height * 0.75;
    thresh_y_min = -height * 0.75;
    std::cout << "the optical flow threshold x is :" << thresh_x_min << " to " << thresh_x_max
              << "\t threshold y is :" << thresh_y_min << " to " << thresh_y_max << std::endl;

    std::vector<fs::directory_entry> file_list;
    fs::path p(root_dir);
    fs::directory_iterator start(p);
    fs::directory_iterator end;
    std::transform(start, end, std::back_inserter(file_list),
                   [](const fs::directory_entry &entry) -> fs::directory_entry { return entry; });

#pragma omp parallel num_threads(24)
    {
#pragma omp for schedule(static, 8)
        // data format convertion
        // for (auto &p : fs::directory_iterator(root_dir))
        for (size_t i = 0; i < file_list.size(); i++)
        {
            fs::path path = file_list[i].path();
            //printf("Processing %d \n", i);
            if (path.extension() != ".flo" && path.extension() != ".bin")
            {
                continue;
            }
            else if (path.string().find("_opticalflow_") != std::string::npos && of_convert_enable == "y")
            {
                std::cout << "optical flow " << path << std::endl;
                // load optical flow from binary
                // cv::Mat depth_diff;
                // load_of_bin(path, of_list, depth_diff, width, height);
                cv::Mat of;
                floread(path.string().c_str(), of);
                fs::path of_file_path(path);

                // cv::merge(of_list, of);
                std::vector<cv::Mat> of_list;
                cv::split(of, of_list);


                // output *.flo
                // of_file_path.replace_extension(".flo");
                // FlowIOOpenCVWrapper::write(of_file_path, of);
                // flowrite(of_file_path.string().c_str(), of);

                // remove max and min value
                double max_x, min_x;
                double max_y, min_y;
                cv::minMaxLoc(of_list[0], &min_x, &max_x);
                cv::minMaxLoc(of_list[1], &min_y, &max_y);
                std::cout << "max_x:" << max_x << "\t min_x:" << min_x
                          << "\t max_y:" << max_y << "\t min_y:" << min_y << std::endl;
                if (max_x > thresh_x_max || min_x < thresh_x_min || max_y > thresh_y_max || min_y < thresh_y_min)
                {
                    threshold(of, of, 50, 0, cv::THRESH_TRUNC);
                    threshold(of, of, -50, 0, cv::THRESH_TOZERO);
                }

                // output visualizaion *.png images
                fs::path of_visual_path(path);
                of_visual_path.replace_extension(".jpg");
                // cv::Mat of_visual = FlowIOOpenCVWrapper::flowToColor(of);
                cv::Mat of_visual;
                cv::Mat planes[2];
                cv::split(of, planes);
                drawOpticalFlow(planes[0], planes[1], of_visual);//, planes[0].rows * 0.2);
                write_image_jpeg(of_visual_path.string(), of_visual);
            }
            else if (path.string().find("_depth") != std::string::npos && depth_convert_enable == "y")
            {
                std::cout << "depth map" << path << std::endl;
                cv::Mat depth_map;
                load_depth_bin(path.string(), depth_map, width, height);

                // convert the depht to obj
                fs::path obj_path(path);
                obj_path.replace_extension(".obj");
                writeSphereMesh(depth_map, obj_path.string());

                // output the colormap
                cv::Mat img_color;
                cv::Mat depth_color;
                cv::Mat depht_map_color;
                double min;
                double max;
                cv::minMaxIdx(depth_map, &min, &max);
                depth_map.convertTo(depht_map_color, CV_8UC1, 255 / (max - min), -min);
                applyColorMap(depht_map_color, depth_color, cv::COLORMAP_JET);
                fs::path img_path(path);
                img_path.replace_extension(".jpg");
                write_image_jpeg(img_path.string(), depth_color);
            }
        }
    }

} //main