//Numpy array shape [8, 1]
//Min -0.500000000000
//Max 0.500000000000
//Number of zeros 3

#ifndef W19_H_
#define W19_H_

#ifndef __SYNTHESIS__
weight19_t w19[8];
#else
weight19_t w19[8] = {0.5, -0.5, 0.0, 0.0, -0.5, 0.5, 0.5, 0.0};
#endif

#endif
