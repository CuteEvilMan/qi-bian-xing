#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cmath>

#include "utils.h"
#include "winograd.h"

#define LOOP_NUM 3

double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1.e-6;
}

void naive_conv(float *image,
                float *filter,
                float *out,
                const int batch_size,
                const int input_channel,
                const int height,
                const int width,
                const int output_channel) {
  typedef float(*image_tensor_t)[input_channel][height][width];
  typedef float(*filter_tensor_t)[input_channel][FLT_H][FLT_W];
  typedef float(*out_tensor_t)[output_channel][height - 2][width - 2];
  image_tensor_t image_tensor = (image_tensor_t)image;
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  out_tensor_t out_tensor = (out_tensor_t)out;
#pragma omp parallel for collapse(4)
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t oc = 0; oc < output_channel; ++oc) {
      for (int64_t oh = 0; oh < height - 2; ++oh) {
        for (int64_t ow = 0; ow < width - 2; ++ow) {
          out_tensor[batch][oc][oh][ow] = 0;
          for (int64_t ic = 0; ic < input_channel; ++ic) {
            for (int64_t kh = 0; kh < 3; ++kh) {
              for (int64_t kw = 0; kw < 3; ++kw) {
                out_tensor[batch][oc][oh][ow] += image_tensor[batch][ic][oh + kh][ow + kw] *
                                                 filter_tensor[oc][ic][kh][kw];
              }
            }
          }
        }
      }
    }
  }
}

void test_on_one_layer(const int layer_idx,
                       const int validation_mode,
                       const int image_height,
                       const int image_width,
                       const int input_channels,
                       const int output_channels,
                       const int batch,
                       long *total_flops,
                       double *total_time) {
  const int output_height = image_height - 2;
  const int output_width = image_width - 2;

  float *image, *filter, *out;
  image = (float *)malloc(sizeof(float) * batch * input_channels * image_height * image_width);
  assert(image != NULL);
  filter = (float *)malloc(sizeof(float) * output_channels * input_channels * FLT_H * FLT_W);
  assert(filter != NULL);
  out = (float *)malloc(sizeof(float) * batch * output_channels * output_height * output_width);
  assert(out != NULL);

#pragma omp parallel for
  for (long i = 0; i < (long)batch * input_channels * image_height * image_width; i++)
    image[i] = (float)(i % 10 + 1);

#pragma omp parallel for
  for (long i = 0; i < output_channels * input_channels * FLT_H * FLT_W; i++)
    filter[i] = (float)(i / (FLT_H * FLT_W) + 1);

  if (validation_mode) {  // Verify mode. Check the result
    float *out_ref = (float *)malloc(sizeof(float) * batch * output_channels * output_height * output_width);
    assert(out_ref != NULL);
    winograd_convolution(
        image, image_height, image_width, input_channels, filter, output_channels, batch, out);
    naive_conv(image, filter, out_ref, batch, input_channels, image_height, image_width, output_channels);
    printf("Layer %-2d: (Channel Height Weight Filter Batch) = ", layer_idx);
    printf(
        "(%-3d %-3d %-3d %-3d %-3d) : ", input_channels, image_height, image_width, output_channels, batch);
    long n;
    for (n = 0; n < (long)batch * output_height * output_width * output_channels; n++)
      if (fabs((out[n] - out_ref[n]) / out_ref[n]) > 1e-2 || isnan(out[n]) || isinf(out[n])) {
        printf("Validation Failed !");
        printf("winogradConv[%ld] = %f || directConv[%ld] = %f \n", n, out[n], n, out_ref[n]);
        break;
      }
    if (n == (long)batch * output_height * output_width * output_channels) printf("Validation Passed !\n");
    free(out_ref);
  } else {
    double start_time = timestamp();
    for (int i = 0; i < LOOP_NUM; i++) {
      winograd_convolution(
          image, image_height, image_width, input_channels, filter, output_channels, batch, out);
    }
    double end_time = timestamp();
    double elapse_time_all = end_time - start_time;
    double elapse_time = elapse_time_all / LOOP_NUM;
    *total_time += elapse_time;
    long nflops = (long)batch * output_channels * input_channels * (image_height - 2) * (image_width - 2) *
                  FLT_H * FLT_W * 2;
    double gflops = (double)nflops * 1.0e-9 / elapse_time;
    *total_flops += nflops;
    printf("Layer %-2d:  Elapse time %lf ms. ( %7.2lf GFlops) \n", layer_idx, elapse_time * 1000, gflops);
  }

  free(image);
  free(filter);
  free(out);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s layer.conf [validation=0/1] \n", argv[0]);
    printf("Please provided layer configs. Aborting\n");
    exit(-1);
  }
  FILE *input = fopen(argv[1], "r");
  if (!input) {
    printf("File open failed. Aborting...\n");
    exit(-1);
  }
  int validation_mode = 0;  // 0 : benchmark mode, 1: validation mode
  if (argc > 2) validation_mode = atoi(argv[2]);

  int layer_num;
  fscanf(input, "%d", &layer_num);
  if (layer_num <= 0) {
    printf("Invalid layer num %d. Aborting\n", layer_num);
    fclose(input);
    exit(1);
  }
  int *IC_arr = (int *)malloc(sizeof(int) * layer_num);     // Channel
  int *H_arr = (int *)malloc(sizeof(int) * layer_num);      // Image Height
  int *W_arr = (int *)malloc(sizeof(int) * layer_num);      // Image Width
  int *OC_arr = (int *)malloc(sizeof(int) * layer_num);     // Filters
  int *Batch_arr = (int *)malloc(sizeof(int) * layer_num);  // Batch

  for (int l = 0; l < layer_num; ++l) {
    fscanf(input, "%d%d%d%d%d", &IC_arr[l], &H_arr[l], &W_arr[l], &OC_arr[l], &Batch_arr[l]);
  }
  fclose(input);

  double total_time = 0.0;
  long total_flops = 0;
  for (int l = 0; l < layer_num; l++) {
    test_on_one_layer(l,
                      validation_mode,
                      H_arr[l],
                      W_arr[l],
                      IC_arr[l],
                      OC_arr[l],
                      Batch_arr[l],
                      &total_flops,
                      &total_time);
  }

  if (!validation_mode) {
    printf("Total elapse time:");
    printf(" %lf. ( %7.2lf GFlops) \n", total_time, (double)total_flops * 1.0e-9 / total_time);
  }

  if (IC_arr) free(IC_arr);
  if (H_arr) free(H_arr);
  if (W_arr) free(W_arr);
  if (OC_arr) free(OC_arr);
  if (Batch_arr) free(Batch_arr);
  return 0;
}
