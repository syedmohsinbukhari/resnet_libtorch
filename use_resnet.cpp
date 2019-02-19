#include <torch/script.h>               // One-stop header.
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: resnet_libtorch <path-to-resnet-script-module> <path-to-input-image>\n";
        return -1;
    }

    // Configuration
    int input_image_size = 224;
    int batch_size = 8;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
    assert(module != nullptr);

    // Read input image
    cv::Mat origin_image = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    // Preprocess image (resize, put on GPU)
    cv::Mat resized_image;
    cv::cvtColor(origin_image, resized_image, cv::COLOR_RGB2BGR);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);
    auto img_var = torch::autograd::make_variable(img_tensor, false);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var.to(at::kCUDA));

    // Execute the model and turn its output into a tensor.
    at::Tensor output;
    auto duration = duration_cast<milliseconds>(std::chrono::high_resolution_clock::now()-std::chrono::high_resolution_clock::now());
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        output = module->forward(inputs).toTensor().to(at::kCPU);
        auto end = std::chrono::high_resolution_clock::now();
        duration = duration_cast<milliseconds>(end - start);
    }
    
    at::Tensor max_ind = at::argmax(output);
    std::cout << "class_id: " << max_ind.item<int>() << std::endl;
    std::cout << "Time take for forward pass: " << duration.count() << " ms" << std::endl;

    std::cout << "Done\n";
}
