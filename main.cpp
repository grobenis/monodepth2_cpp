#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int main(int argc,char**argv)
{
    //load a pytorch model
    //加载模型
    torch::jit::script::Module encoder = torch::jit::load("/home/guoben/Project/monodepth2.cpp/model/encoder.cpt");
    encoder.to(at::kCUDA);
    torch::jit::script::Module decoder = torch::jit::load("/home/guoben/Project/monodepth2.cpp/model/decoder.cpt");
    decoder.to(at::kCUDA);

    //读取图片
    cv::Mat src=cv::imread("/home/guoben/Project/monodepth2.cpp/test_image.jpg");
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
    disp.convertTo(disp, CV_8UC1);
    cv::cvtColor(disp, disp, CV_GRAY2BGR);
    
    cv::imshow("src", disp);
    cv::waitKey(0);

    // src.push_back(disp);
//        vector<cv::Mat> channels={disp,disp,disp};
//        cv::merge(channels,disp);
    cv::imshow("src", src);
    cv::waitKey(0);
    // cv::resize(src, src, cv::Size(), 0.5, 0.5);
//        cv::imshow("result",disp);
    // cv::imshow("src", src);
    // cv::waitKey(0);


//    cout<<tensor_image.sizes()<<endl;
////    tensor_image = tensor_image.unsqueeze(0);
//    tensor_image = tensor_image.to(at::kCUDA);
//    //[0,1]
//
//    std::vector<torch::IValue> batch;//inputs
//    batch.push_back(tensor_image);

//    //网络的前向计算
//    auto outputs= decoder.forward(batch);// inference
//
////    batch.clear();
////    batch.push_back(outputs);
//    auto  tensor_output = outputs.toTensor();
//
////    tensor_output = tensor_output.permute({0,2,3,1});//change the dimentions of tensor
//    // tensor_output=tensor_output.squeeze(0).detach().permute({1,2,0});
//    // tensor_output=tensor_output.mul(255).clamp(0,255).to(torch::kU8);
//    tensor_output=tensor_output.to(at::kCPU);
//
//    cv::Mat disp=cv::Mat(h,w,CV_32FC1,tensor_output.data_ptr());
//
//    double minVal; double maxVal;
//    cv::minMaxLoc(disp, &minVal, &maxVal);
//    disp /= maxVal;
//
//    cv::resize(disp,disp,cv::Size(src.cols,src.rows));
//    disp*=255;
//
//    disp.convertTo(disp,CV_8UC1);
//    cv::cvtColor(disp,disp,cv::COLOR_GRAY2BGR);
//    cout<<disp<<endl;
//    src.push_back(disp);
//    //cv::resize(src,src,cv::Size(),0.5,0.5);
//
//    cv::imwrite("src.jpg",disp);

    cout<<"Done"<<endl;
    return 0;
}
