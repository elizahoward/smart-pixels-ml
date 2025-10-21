#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1
#define N_INPUT_1_2 1
#define OUT_CONCAT_3 2
#define N_LAYER_4 17
#define N_LAYER_4 17
#define N_LAYER_4 17
#define N_LAYER_7 20
#define N_LAYER_7 20
#define N_LAYER_7 20
#define N_LAYER_10 9
#define N_LAYER_10 9
#define N_LAYER_10 9
#define N_LAYER_13 16
#define N_LAYER_13 16
#define N_LAYER_13 16
#define N_LAYER_16 8
#define N_LAYER_16 8
#define N_LAYER_16 8
#define N_LAYER_19 1
#define N_LAYER_19 1
#define N_LAYER_19 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> input2_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<2,1> weight4_t;
typedef ap_fixed<2,1> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<16,6> layer22_t;
typedef struct exponent_scale22_t {ap_uint<1> sign;ap_int<3> weight; } exponent_scale22_t;
typedef ap_fixed<2,1> bias22_t;
typedef ap_ufixed<2,0,AP_RND_CONV,AP_SAT> layer6_t;
typedef ap_fixed<18,8> q_relu1_table_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<2,1> weight7_t;
typedef ap_fixed<2,1> bias7_t;
typedef ap_uint<1> layer7_index;
typedef ap_fixed<16,6> layer23_t;
typedef struct exponent_scale23_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale23_t;
typedef ap_fixed<2,1> bias23_t;
typedef ap_ufixed<2,0,AP_RND_CONV,AP_SAT> layer9_t;
typedef ap_fixed<18,8> q_relu2_table_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<2,1> weight10_t;
typedef ap_fixed<2,1> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> layer24_t;
typedef struct exponent_scale24_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale24_t;
typedef ap_fixed<2,1> bias24_t;
typedef ap_ufixed<2,0,AP_RND_CONV,AP_SAT> layer12_t;
typedef ap_fixed<18,8> q_relu3_table_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<2,1> weight13_t;
typedef ap_fixed<2,1> bias13_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<16,6> layer25_t;
typedef struct exponent_scale25_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale25_t;
typedef ap_fixed<2,1> bias25_t;
typedef ap_ufixed<2,0,AP_RND_CONV,AP_SAT> layer15_t;
typedef ap_fixed<18,8> q_relu4_table_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<2,1> weight16_t;
typedef ap_fixed<2,1> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<16,6> layer26_t;
typedef struct exponent_scale26_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale26_t;
typedef ap_fixed<2,1> bias26_t;
typedef ap_ufixed<2,0,AP_RND_CONV,AP_SAT> layer18_t;
typedef ap_fixed<18,8> q_relu5_table_t;
typedef ap_fixed<16,6> layer19_t;
typedef ap_fixed<2,1> weight19_t;
typedef ap_fixed<2,1> bias19_t;
typedef ap_uint<1> layer19_index;
typedef ap_fixed<16,6> layer27_t;
typedef struct exponent_scale27_t {ap_uint<1> sign;ap_int<2> weight; } exponent_scale27_t;
typedef ap_fixed<2,1> bias27_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> result_t;
typedef ap_ufixed<2,0> slope21_t;
typedef ap_ufixed<2,0> shift21_t;
typedef ap_fixed<18,8> output_table_t;

#endif
