//Numpy array shape [8]
//Min 0.000000000000
//Max 0.000000000000
//Number of zeros 8

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[8];
#else
bias16_t b16[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
#endif

#endif
