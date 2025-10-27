#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t x_profile[N_INPUT_1_1], input2_t z_global[N_INPUT_1_2], input3_t y_profile[N_INPUT_1_3], input4_t y_local[N_INPUT_1_4],
    result_t layer25_out[N_LAYER_23]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_profile complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=z_global complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_profile complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_profile,z_global,y_profile,y_local,layer25_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight7_t, 704>(w7, "w7.txt");
        nnet::load_weights_from_txt<bias7_t, 32>(b7, "b7.txt");
        nnet::load_weights_from_txt<weight9_t, 448>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 32>(b9, "b9.txt");
        nnet::load_weights_from_txt<weight14_t, 8192>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 128>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight17_t, 8192>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 64>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight20_t, 2048>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 32>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight23_t, 32>(w23, "w23.txt");
        nnet::load_weights_from_txt<bias23_t, 1>(b23, "b23.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer5_t layer5_out[OUT_CONCAT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::concatenate1d<input_t, input2_t, layer5_t, config5>(x_profile, z_global, layer5_out); // xz_concat

    layer6_t layer6_out[OUT_CONCAT_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::concatenate1d<input3_t, input4_t, layer6_t, config6>(y_profile, y_local, layer6_out); // yl_concat

    layer7_t layer7_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::dense<layer5_t, layer7_t, config7>(layer5_out, layer7_out, w7, b7); // xz_dense1

    layer9_t layer9_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::dense<layer6_t, layer9_t, config9>(layer6_out, layer9_out, w9, b9); // yl_dense1

    layer11_t layer11_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::relu<layer7_t, layer11_t, relu_config11>(layer7_out, layer11_out); // xz_relu1

    layer12_t layer12_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::relu<layer9_t, layer12_t, relu_config12>(layer9_out, layer12_out); // yl_relu1

    layer13_t layer13_out[OUT_CONCAT_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::concatenate1d<layer11_t, layer12_t, layer13_t, config13>(layer11_out, layer12_out, layer13_out); // merged_features

    layer14_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // merged_dense1

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::relu<layer14_t, layer16_t, relu_config16>(layer14_out, layer16_out); // merged_relu1

    layer17_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17); // merged_dense2

    layer19_t layer19_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::relu<layer17_t, layer19_t, relu_config19>(layer17_out, layer19_out); // merged_relu2

    layer20_t layer20_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::dense<layer19_t, layer20_t, config20>(layer19_out, layer20_out, w20, b20); // merged_dense3

    layer22_t layer22_out[N_LAYER_20];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::relu<layer20_t, layer22_t, relu_config22>(layer20_out, layer22_out); // merged_relu3

    layer23_t layer23_out[N_LAYER_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::dense<layer22_t, layer23_t, config23>(layer22_out, layer23_out, w23, b23); // output_dense

    nnet::hard_tanh<layer23_t, result_t, hard_tanh_config25>(layer23_out, layer25_out); // output

}
