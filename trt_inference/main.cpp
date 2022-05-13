#include <chrono>
#include <fstream>
#include <iostream>

#include "common.h"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ATen/ATen.h>
#include <torch/script.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

// Change these according to your dataset
#define DEVICE 0 // GPU id
#define WIDTH 2048
#define HEIGHT 1024
#define LABEL_DIVISOR 1000
#define MAX_DEPTH 80.

int main(int argc, char** argv) {
  // Class colors from cityscapes. Change according to your dataset
  cv::Mat class_colors = cv::Mat::zeros(256, 1, CV_8UC3);
  class_colors.at<cv::Vec3b>(0, 0) = {204, 209, 72};
  class_colors.at<cv::Vec3b>(0, 1) = {128, 64, 128};
  class_colors.at<cv::Vec3b>(0, 2) = {232, 35, 244};
  class_colors.at<cv::Vec3b>(0, 3) = {70, 70, 70};
  class_colors.at<cv::Vec3b>(0, 4) = {156, 102, 102};
  class_colors.at<cv::Vec3b>(0, 5) = {153, 153, 190};
  class_colors.at<cv::Vec3b>(0, 6) = {153, 153, 153};
  class_colors.at<cv::Vec3b>(0, 7) = {30, 170, 250};
  class_colors.at<cv::Vec3b>(0, 8) = {0, 220, 220};
  class_colors.at<cv::Vec3b>(0, 9) = {35, 142, 107};
  class_colors.at<cv::Vec3b>(0, 10) = {152, 251, 152};
  class_colors.at<cv::Vec3b>(0, 11) = {180, 130, 70};
  class_colors.at<cv::Vec3b>(0, 12) = {60, 20, 220};
  class_colors.at<cv::Vec3b>(0, 13) = {0, 0, 255};
  class_colors.at<cv::Vec3b>(0, 14) = {142, 0, 0};
  class_colors.at<cv::Vec3b>(0, 15) = {70, 0, 0};
  class_colors.at<cv::Vec3b>(0, 16) = {100, 60, 0};
  class_colors.at<cv::Vec3b>(0, 17) = {100, 80, 0};
  class_colors.at<cv::Vec3b>(0, 18) = {230, 0, 0};
  class_colors.at<cv::Vec3b>(0, 19) = {32, 11, 119};

  cudaSetDevice(DEVICE);

  if (argc < 5) {
    std::cerr
        << "usage: " << argv[0]
        << " model.plan postprocessing.pt image.png camera_calibration.json";
    return -1;
  }

  // deserialize the .plan file
  std::ifstream file(argv[1], std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << argv[1] << " error!" << std::endl;
    return -1;
  }
  char* trtModelStream = nullptr;
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  trtModelStream = new char[size];
  file.read(trtModelStream, size);
  file.close();

  // Create inference engine and context
  TRTUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(gLogger)};
  CHECK_FOR_NULLPTR(runtime, "failed to create infer runtime");

  TRTUniquePtr<nvinfer1::ICudaEngine> engine{
      runtime->deserializeCudaEngine(trtModelStream, size)};
  CHECK_FOR_NULLPTR(engine, "failed to create trt engine");

  TRTUniquePtr<nvinfer1::IExecutionContext> context{
      engine->createExecutionContext()};
  CHECK_FOR_NULLPTR(context, "failed to create execution context");

  delete[] trtModelStream;

  // Get binding indices
  void* buffers[5];
  const int input_idx = engine->getBindingIndex("input_image");
  const int output_semantic_idx = engine->getBindingIndex("semantic");
  const int output_center_idx = engine->getBindingIndex("center");
  const int output_offset_idx = engine->getBindingIndex("offset");
  const int output_depth_idx = engine->getBindingIndex("depth");

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc(
      (void**)&buffers[input_idx], 1 * 3 * HEIGHT * WIDTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      (void**)&buffers[output_semantic_idx],
      1 * 1 * HEIGHT * WIDTH * sizeof(int)));
  CUDA_CHECK(cudaMalloc(
      (void**)&buffers[output_center_idx],
      1 * 1 * HEIGHT * WIDTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      (void**)&buffers[output_offset_idx],
      1 * 2 * HEIGHT * WIDTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      (void**)&buffers[output_depth_idx],
      1 * 1 * HEIGHT * WIDTH * sizeof(float)));

  // Load postprocessing torchscript module
  c10::InferenceMode guard(true);
  torch::jit::script::Module postprocessing_module;
  try {
    postprocessing_module = torch::jit::load(argv[2]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading postprocessing model" << std::endl;
    return -1;
  }

  // Create inputs for postprocessing module
  std::vector<torch::jit::IValue> postprocessing_inputs;
  postprocessing_inputs.push_back(torch::from_blob(
      buffers[output_semantic_idx],
      {1, 1, HEIGHT, WIDTH},
      torch::dtype(torch::kInt32)
          .device(torch::kCUDA, DEVICE)
          .requires_grad(false)));
  postprocessing_inputs.push_back(torch::from_blob(
      buffers[output_center_idx],
      {1, 1, HEIGHT, WIDTH},
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, DEVICE)
          .requires_grad(false)));
  postprocessing_inputs.push_back(torch::from_blob(
      buffers[output_offset_idx],
      {1, 2, HEIGHT, WIDTH},
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, DEVICE)
          .requires_grad(false)));
  postprocessing_inputs.push_back(torch::from_blob(
      buffers[output_depth_idx],
      {1, 1, HEIGHT, WIDTH},
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, DEVICE)
          .requires_grad(false)));

  // Create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Read and preprocess image
  cv::Mat img = cv::imread(argv[3]);
  cv::resize(img, img, cv::Size(WIDTH, HEIGHT), cv::INTER_LINEAR);
  cv::imwrite("img.png", img);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  img.convertTo(img, CV_32FC3, 1. / 255.);
  if (!img.isContinuous()) {
    img = img.clone();
  }

  // Read camera info
  std::ifstream camera_info_file(argv[4]);
  std::string str;
  float camera_height = 0.;
  float fx = 0.;
  float fy = 0.;
  float cx = 0.;
  float cy = 0.;
  while (std::getline(camera_info_file, str)) {
    // We assume a calibration file in Cityscapes camera format! Please adjust,
    // if a different format is used
    if (str.find("z") != std::string::npos) {
      camera_height = stof(str.substr(13, std::string::npos));
    } else if (str.find("fx") != std::string::npos) {
      fx = stof(str.substr(14, std::string::npos));
    } else if (str.find("fy") != std::string::npos) {
      fy = stof(str.substr(14, std::string::npos));
    } else if (str.find("u0") != std::string::npos) {
      cx = stof(str.substr(14, std::string::npos));
    } else if (str.find("v0") != std::string::npos) {
      cy = stof(str.substr(14, std::string::npos));
    }
  }
  if (camera_height == 0. || fx == 0. || fy == 0. || cx == 0. || cy == 0.) {
    std::cerr << "invalid camera_calibration" << std::endl;
    return -1;
  }

  postprocessing_inputs.push_back(torch::tensor(
      {{1. / fx, 0., -1. * cx / fx},
       {0., 1. / fy, -1. * cy / fy},
       {0., 0., 1.}},
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, DEVICE)
          .requires_grad(false)));
  postprocessing_inputs.push_back(torch::tensor(
      {camera_height},
      torch::dtype(torch::kFloat32)
          .device(torch::kCUDA, DEVICE)
          .requires_grad(false)));

  cv::Mat panoptic_prediction(HEIGHT, WIDTH, CV_32SC1);
  cv::Mat depth_prediction(HEIGHT, WIDTH, CV_32FC1);
  cv::Mat xyz_points(HEIGHT, WIDTH, CV_32FC4);

  // Warmup model
  for (size_t i = 0; i < 10; ++i) {
    context->enqueueV2(buffers, stream, nullptr);
    postprocessing_module.forward(postprocessing_inputs);
  }

  // Run inference and test speed
  auto start = std::chrono::system_clock::now();
  // Copy input image to GPU buffer
  CUDA_CHECK(cudaMemcpyAsync(
      buffers[input_idx],
      img.data,
      1 * 3 * HEIGHT * WIDTH * sizeof(float),
      cudaMemcpyHostToDevice,
      stream));
  context->enqueueV2(buffers, stream, nullptr);
  auto output = postprocessing_module.forward(postprocessing_inputs)
                    .toTuple()
                    ->elements();
  torch::Tensor panoptic_output = output.at(0).toTensor();
  torch::Tensor depth_output = output.at(1).toTensor();
  torch::Tensor xyz_points_output = output.at(2).toTensor();
  cudaStreamSynchronize(stream);
  auto end = std::chrono::system_clock::now();
  std::cout << "inference time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                   .count()
            << " ms" << std::endl;

  CUDA_CHECK(cudaMemcpyAsync(
      (void*)panoptic_prediction.data,
      panoptic_output.data_ptr(),
      1 * HEIGHT * WIDTH * sizeof(int),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      (void*)depth_prediction.data,
      depth_output.data_ptr(),
      1 * HEIGHT * WIDTH * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      (void*)xyz_points.data,
      xyz_points_output.data_ptr(),
      4 * HEIGHT * WIDTH * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream));

  // Convert predictions to color images and write them to files
  // Semantic part
  cv::Mat semantic_color_image;
  cv::Mat semantic_prediction = panoptic_prediction.clone();
  // Instance ids to semantic class ids
  std::for_each(
      semantic_prediction.begin<int>(),
      semantic_prediction.end<int>(),
      [](int& pixel) {
        if (pixel > LABEL_DIVISOR) {
          pixel = pixel / LABEL_DIVISOR;
        }
      });
  semantic_prediction.convertTo(semantic_prediction, CV_8UC1);
  cv::applyColorMap(semantic_prediction, semantic_color_image, class_colors);
  cv::imwrite("semantic.png", semantic_color_image);

  // Instance part
  cv::Mat instance_color_image;
  // Panoptic ids to instance ids. Set semantic ids to zero
  std::for_each(
      panoptic_prediction.begin<int>(),
      panoptic_prediction.end<int>(),
      [](int& pixel) {
        if (pixel > LABEL_DIVISOR) {
          pixel = pixel % LABEL_DIVISOR;
        } else {
          pixel = 0;
        }
      });
  panoptic_prediction.convertTo(panoptic_prediction, CV_8UC1);
  cv::normalize(
      panoptic_prediction, panoptic_prediction, 0., 255., cv::NORM_MINMAX);
  cv::applyColorMap(
      panoptic_prediction, instance_color_image, cv::COLORMAP_VIRIDIS);
  cv::imwrite("instance.png", instance_color_image);

  // Combine semantic and instance color image
  cv::Mat panoptic_color_image;
  cv::Mat mask;
  cv::inRange(
      instance_color_image, cv::Scalar(84, 1, 68), cv::Scalar(84, 1, 68), mask);
  instance_color_image.setTo(cv::Scalar(0, 0, 0), mask);
  cv::bitwise_or(
      semantic_color_image, instance_color_image, panoptic_color_image);
  cv::imwrite("panoptic.png", panoptic_color_image);

  // Convert depth prediction to color image and write it to a file
  cv::threshold(
      depth_prediction, depth_prediction, MAX_DEPTH, 255., cv::THRESH_TRUNC);
  cv::normalize(depth_prediction, depth_prediction, 0., 1., cv::NORM_MINMAX);
  depth_prediction = 1. - depth_prediction;
  depth_prediction.convertTo(depth_prediction, CV_8UC1, 255.);
  cv::applyColorMap(depth_prediction, depth_prediction, cv::COLORMAP_PLASMA);
  cv::imwrite("depth.png", depth_prediction);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(buffers[input_idx]));
  CUDA_CHECK(cudaFree(buffers[output_semantic_idx]));
  CUDA_CHECK(cudaFree(buffers[output_center_idx]));
  CUDA_CHECK(cudaFree(buffers[output_offset_idx]));
  CUDA_CHECK(cudaFree(buffers[output_depth_idx]));

  return 0;
}
