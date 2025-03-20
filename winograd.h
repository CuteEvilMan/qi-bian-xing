#pragma once

void winograd_convolution(float *__restrict__ image,
                          const int irows,
                          const int icols,
                          const int C,
                          float *__restrict__ filter,
                          const int K,
                          const int batch,
                          float *__restrict__ out);