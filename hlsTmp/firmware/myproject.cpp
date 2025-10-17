#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t y_local[N_INPUT_1_1], input2_t y_profile[N_INPUT_1_2],
    result_t layer7_out[N_LAYER_7]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_profile complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=y_local,y_profile,layer7_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight4_t, 812>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 58>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight7_t, 58>(w7, "w7.txt");
        nnet::load_weights_from_txt<bias7_t, 1>(b7, "b7.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer3_t layer3_out[OUT_CONCAT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::concatenate1d<input_t, input2_t, layer3_t, config3>(y_local, y_profile, layer3_out); // concatenate

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // dense1

    layer6_t layer6_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::relu<layer4_t, layer6_t, relu_config6>(layer4_out, layer6_out); // relu1

    nnet::dense<layer6_t, result_t, config7>(layer6_out, layer7_out, w7, b7); // output_dense

}
