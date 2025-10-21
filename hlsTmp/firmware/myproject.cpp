#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t y_size[N_INPUT_1_1], input2_t y_local[N_INPUT_1_2],
    result_t layer21_out[N_LAYER_19]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=y_size complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=y_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=y_size,y_local,layer21_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight4_t, 34>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 17>(b4, "b4.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale22_t, 17>(s22, "s22.txt");
        nnet::load_weights_from_txt<bias22_t, 17>(b22, "b22.txt");
        nnet::load_weights_from_txt<weight7_t, 340>(w7, "w7.txt");
        nnet::load_weights_from_txt<bias7_t, 20>(b7, "b7.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale23_t, 20>(s23, "s23.txt");
        nnet::load_weights_from_txt<bias23_t, 20>(b23, "b23.txt");
        nnet::load_weights_from_txt<weight10_t, 180>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 9>(b10, "b10.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale24_t, 9>(s24, "s24.txt");
        nnet::load_weights_from_txt<bias24_t, 9>(b24, "b24.txt");
        nnet::load_weights_from_txt<weight13_t, 144>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 16>(b13, "b13.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale25_t, 16>(s25, "s25.txt");
        nnet::load_weights_from_txt<bias25_t, 16>(b25, "b25.txt");
        nnet::load_weights_from_txt<weight16_t, 128>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 8>(b16, "b16.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale26_t, 8>(s26, "s26.txt");
        nnet::load_weights_from_txt<bias26_t, 8>(b26, "b26.txt");
        nnet::load_weights_from_txt<weight19_t, 8>(w19, "w19.txt");
        nnet::load_weights_from_txt<bias19_t, 1>(b19, "b19.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale27_t, 1>(s27, "s27.txt");
        nnet::load_weights_from_txt<bias27_t, 1>(b27, "b27.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer3_t layer3_out[OUT_CONCAT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::concatenate1d<input_t, input2_t, layer3_t, config3>(y_size, y_local, layer3_out); // concatenate_1

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // dense1

    layer22_t layer22_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::normalize<layer4_t, layer22_t, config22>(layer4_out, layer22_out, s22, b22); // dense1_alpha

    layer6_t layer6_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::relu<layer22_t, layer6_t, relu_config6>(layer22_out, layer6_out); // q_relu1

    layer7_t layer7_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::dense<layer6_t, layer7_t, config7>(layer6_out, layer7_out, w7, b7); // dense2

    layer23_t layer23_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::normalize<layer7_t, layer23_t, config23>(layer7_out, layer23_out, s23, b23); // dense2_alpha

    layer9_t layer9_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<layer23_t, layer9_t, relu_config9>(layer23_out, layer9_out); // q_relu2

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // dense3

    layer24_t layer24_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::normalize<layer10_t, layer24_t, config24>(layer10_out, layer24_out, s24, b24); // dense3_alpha

    layer12_t layer12_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::relu<layer24_t, layer12_t, relu_config12>(layer24_out, layer12_out); // q_relu3

    layer13_t layer13_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer12_t, layer13_t, config13>(layer12_out, layer13_out, w13, b13); // dense4

    layer25_t layer25_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::normalize<layer13_t, layer25_t, config25>(layer13_out, layer25_out, s25, b25); // dense4_alpha

    layer15_t layer15_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::relu<layer25_t, layer15_t, relu_config15>(layer25_out, layer15_out); // q_relu4

    layer16_t layer16_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::dense<layer15_t, layer16_t, config16>(layer15_out, layer16_out, w16, b16); // dense5

    layer26_t layer26_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0
    nnet::normalize<layer16_t, layer26_t, config26>(layer16_out, layer26_out, s26, b26); // dense5_alpha

    layer18_t layer18_out[N_LAYER_16];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::relu<layer26_t, layer18_t, relu_config18>(layer26_out, layer18_out); // q_relu5

    layer19_t layer19_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::dense<layer18_t, layer19_t, config19>(layer18_out, layer19_out, w19, b19); // output_dense

    layer27_t layer27_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::normalize<layer19_t, layer27_t, config27>(layer19_out, layer27_out, s27, b27); // output_dense_alpha

    nnet::hard_tanh<layer27_t, result_t, hard_tanh_config21>(layer27_out, layer21_out); // output

}
