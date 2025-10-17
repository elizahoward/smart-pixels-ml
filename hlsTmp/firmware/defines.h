#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1
#define N_INPUT_1_2 13
#define OUT_CONCAT_3 14
#define N_LAYER_4 58
#define N_LAYER_4 58
#define N_LAYER_7 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> input2_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<4,1> weight4_t;
typedef ap_fixed<4,1> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> layer6_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<4,1> weight7_t;
typedef ap_fixed<4,1> bias7_t;
typedef ap_uint<1> layer7_index;

#endif
