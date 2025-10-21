//Numpy array shape [1]
//Min 0.500000000000
//Max 1.000000000000
//Number of zeros 0

#ifndef S24_H_
#define S24_H_

#ifndef __SYNTHESIS__
exponent_scale24_t s24[9];
#else
exponent_scale24_t s24[9] = {{1.0, 0}, {1.0, 0}, {1.0, 0}, {1.0, 0}, {1.0, -1}, {1.0, 0}, {1.0, 0}, {1.0, 0}, {1.0, 0}};
#endif

#endif
