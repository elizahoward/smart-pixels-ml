#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 21
#define N_INPUT_1_2 1
#define N_INPUT_1_3 13
#define N_INPUT_1_4 1
#define OUT_CONCAT_5 22
#define OUT_CONCAT_6 14
#define N_LAYER_7 32
#define N_LAYER_9 32
#define N_LAYER_7 32
#define N_LAYER_9 32
#define OUT_CONCAT_13 64
#define N_LAYER_14 128
#define N_LAYER_14 128
#define N_LAYER_17 64
#define N_LAYER_17 64
#define N_LAYER_20 32
#define N_LAYER_20 32
#define N_LAYER_23 1
#define N_LAYER_23 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> input2_t;
typedef ap_fixed<16,6> input3_t;
typedef ap_fixed<16,6> input4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<4,1> weight7_t;
typedef ap_fixed<4,1> bias7_t;
typedef ap_uint<1> layer7_index;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<4,1> weight9_t;
typedef ap_fixed<4,1> bias9_t;
typedef ap_uint<1> layer9_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer11_t;
typedef ap_fixed<18,8> xz_relu1_table_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer12_t;
typedef ap_fixed<18,8> yl_relu1_table_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<4,1> weight14_t;
typedef ap_fixed<4,1> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer16_t;
typedef ap_fixed<18,8> merged_relu1_table_t;
typedef ap_fixed<16,6> layer17_t;
typedef ap_fixed<4,1> weight17_t;
typedef ap_fixed<4,1> bias17_t;
typedef ap_uint<1> layer17_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer19_t;
typedef ap_fixed<18,8> merged_relu2_table_t;
typedef ap_fixed<16,6> layer20_t;
typedef ap_fixed<4,1> weight20_t;
typedef ap_fixed<4,1> bias20_t;
typedef ap_uint<1> layer20_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer22_t;
typedef ap_fixed<18,8> merged_relu3_table_t;
typedef ap_fixed<16,6> layer23_t;
typedef ap_fixed<4,1> weight23_t;
typedef ap_fixed<4,1> bias23_t;
typedef ap_uint<1> layer23_index;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> result_t;
typedef ap_ufixed<2,0> slope25_t;
typedef ap_ufixed<2,0> shift25_t;
typedef ap_fixed<18,8> output_table_t;

#endif
