#include <iostream>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int main(int argc,char**argv)
{
    //load a pytorch model
    cout<<argv[1]<<endl;
    //加载模型n
    torch::jit::script::Module encoder = torch::jit::load("/home/guoben/Project/monodepth2_cpp/model/encoder.cpt");
    encoder.to(at::kCUDA);
    torch::jit::script::Module decoder = torch::jit::load("/home/guoben/Project/monodepth2_cpp/model/decoder.cpt");
    decoder.to(at::kCUDA);

    //读取图片
    cv::Mat src=cv::imread(argv[1]);
    imshow("img",src);
    cv::waitKey(0);
    
    cv::Mat input_mat;
    int w=1024;
    int h=320;

    cv::resize(src, input_mat, cv::Size(w, h));
    input_mat.convertTo(input_mat, CV_32FC3, 1. / 255.);
    torch::Tensor tensor_image = torch::from_blob(input_mat.data, {1, input_mat.rows, input_mat.cols, 3},torch::kF32);

    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = tensor_image.to(at::kCUDA);

    std::vector<torch::IValue> batch;
    batch.push_back(tensor_image);
    auto result_encoder = encoder.forward(batch);
    //    cout<<*result_encoder.type()<<endl;
    batch.clear();
    batch.push_back(result_encoder);
    auto result_decoder = decoder.forward(batch);
    auto tensor_result = result_decoder.toTensor().to(at::kCPU);

    tensor_result = tensor_result.permute({0, 3, 2, 1});
//        cout<<tensor_result.sizes()<<endl;
    cv::Mat disp = cv::Mat(h, w, CV_32FC1, tensor_result.data_ptr());
    cv::resize(disp, disp, cv::Size(src.cols, src.rows));
    disp *= 512;
    cout<<disp.at<float>(0,0)<<endl;
    disp.convertTo(disp, CV_8UC1);

    cv::cvtColor(disp, disp, CV_GRAY2BGR);
    cv::imshow("src", disp);
    cv::waitKey(0);
    cout<<disp.at<float>(0,0)<<endl;

    return 0;
}
