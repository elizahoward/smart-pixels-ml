#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t x_profile[N_INPUT_1_1], input2_t z_global[N_INPUT_1_2], input3_t y_profile[N_INPUT_1_3], input4_t y_local[N_INPUT_1_4],
    result_t layer25_out[N_LAYER_23]
);

#endif
