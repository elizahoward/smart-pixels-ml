#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t y_size[N_INPUT_1_1], input2_t y_local[N_INPUT_1_2],
    result_t layer21_out[N_LAYER_19]
);

#endif
