#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "utils.h"

#include <pthread.h>


void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  


 // #pragma omp parallel for

  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
    //#pragma omp parallel
    //#pragma omp parallel for
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
   /*   float z0, z1, z2, z3, z4, z5, z6;
      z6 = packed_image_tensor[0][w][idx];

      z0 = 4.0f * z6;
//z0=4.0f*packed_image_tensor[0][w][idx]-5.0f*packed_image_tensor[2][w][idx]+packed_image_tensor[4][w][idx]


z0 = 4.0f * packed_image_tensor[0][w][idx] 
     -5.0f * packed_image_tensor[2][w][idx] 
     + packed_image_tensor[4][w][idx];
z1 = -4.0f * packed_image_tensor[1][w][idx] 
   -4.0f * packed_image_tensor[2][w][idx] 
     + packed_image_tensor[3][w][idx] 
     + packed_image_tensor[4][w][idx];
z2 = 4.0f * packed_image_tensor[1][w][idx] 
     -4.0f * packed_image_tensor[2][w][idx] 
     - packed_image_tensor[3][w][idx] 
     + packed_image_tensor[4][w][idx];
     z3 = -2.0f * packed_image_tensor[1][w][idx] 
     - packed_image_tensor[2][w][idx] 
     + 2.0f * packed_image_tensor[3][w][idx] 
     + packed_image_tensor[4][w][idx];
     z4 = 2.0f * packed_image_tensor[1][w][idx] 
     - packed_image_tensor[2][w][idx] 
     -2.0f * packed_image_tensor[3][w][idx] 
     + packed_image_tensor[4][w][idx];
     z5 = 4.0f * packed_image_tensor[1][w][idx] 
     -5.0f * packed_image_tensor[3][w][idx] 
     + packed_image_tensor[5][w][idx];













 
      z6 = packed_image_tensor[1][w][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = packed_image_tensor[2][w][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packed_image_tensor[3][w][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packed_image_tensor[4][w][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packed_image_tensor[5][w][idx];

      z5 += z6;

      V_tensor[0][w][idx] = z0;
      V_tensor[1][w][idx] = z1;
      V_tensor[2][w][idx] = z2;
      V_tensor[3][w][idx] = z3;
      V_tensor[4][w][idx] = z4;
      V_tensor[5][w][idx] = z5;*/
// 预加载高频访问的通道数据到寄存器
const float ch0 = packed_image_tensor[0][w][idx];
const float ch1 = packed_image_tensor[1][w][idx];
const float ch2 = packed_image_tensor[2][w][idx];
const float ch3 = packed_image_tensor[3][w][idx];
const float ch4 = packed_image_tensor[4][w][idx];
const float ch5 = packed_image_tensor[5][w][idx];

// 提前计算公共差值
const float diff1_2 = ch1 - ch2;
const float diff1_3 = ch1 - ch3;
const float temp = ch4 - ch2;
// 展开V_tensor计算，最大化指令级并行
V_tensor[0][w][idx] = 4.0f * ch0 - 5.0f * ch2 + ch4;
V_tensor[5][w][idx] = 4.0f * ch1 - 5.0f * ch3 + ch5;

const float common_term = (-4.0f * diff1_2) + ch3 + ch4;
V_tensor[1][w][idx] = common_term;
V_tensor[2][w][idx] = (4.0f * diff1_2) - ch3 + ch4;

const float temp_common = temp;
V_tensor[3][w][idx] = (-2.0f * diff1_3) + temp_common;
V_tensor[4][w][idx] = (2.0f * diff1_3) + temp_common;


    }  
    // #pragma omp parallel for collapse(1) schedule(static) num_threads(8)
//#pragma omp parallel for
    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    /* float z0, z1, z2, z3, z4, z5, z6;
      z6 = V_tensor[h][0][idx];

      z0 = 4.0f * z6;

      z6 = V_tensor[h][1][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = V_tensor[h][2][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V_tensor[h][3][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V_tensor[h][4][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V_tensor[h][5][idx];

      z5 += z6;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
    */


  // 优化步骤1：预加载所有输入数据到寄存器
const float v0 = V_tensor[h][0][idx];
const float v1 = V_tensor[h][1][idx];
const float v2 = V_tensor[h][2][idx];
const float v3 = V_tensor[h][3][idx];
const float v4 = V_tensor[h][4][idx];
const float v5 = V_tensor[h][5][idx];

// 优化步骤2：代数重组提升ILP
const float v0_term = 4.0f * v0 - 5.0f * v2 + v4;
const float v1_term = -4.0f * v1 - 4.0f * v2 + v3 + v4;
const float v5_term = 4.0f * v1 - 5.0f * v3 + v5;

// 优化步骤3：合并公共计算路径
const float v2_4 = v2 + v4;  // 公共子表达式
const float v1_4 = v1 + v4;
const float v3_scale = 2.0f * v3;

V_tensor[h][0][idx] = v0_term;
V_tensor[h][1][idx] = -4.0f*(v1 + v2) + v3 + v4; // 展开表达式
V_tensor[h][2][idx] = 4.0f*(v1 - v2) - v3 + v4;
V_tensor[h][3][idx] = -2.0f*(v1 + 0.5f*v2) + v3_scale + v4;
V_tensor[h][4][idx] = 2.0f*(v1 - v2) - v3_scale + v4;
V_tensor[h][5][idx] = v5_term;



    }

  }

}

void filter_transform(float *__restrict__ packed_filter,
                      float *__restrict__ U,
                      const filter_shape_t fs,
                      const U_shape_t us,
                      const int64_t collapsed_dim_size) 
{

  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  #pragma omp parallel for collapse(2) schedule(dynamic, 16)
 // #pragma omp parallel for
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
   // #pragma omp parallel for
    for (int64_t w = 0; w < fs.w; ++w) {
     /* float z0, z1, z2, z3, z4, z5, z6;
      z6 = packed_filter_tensor[0][w][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packed_filter_tensor[1][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packed_filter_tensor[2][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[0][w][idx] = z0;
      U_tensor[1][w][idx] = z1;
      U_tensor[2][w][idx] = z2;
      U_tensor[3][w][idx] = z3;
      U_tensor[4][w][idx] = z4;
      U_tensor[5][w][idx] = z5;*/

     // 优化步骤1：预加载所有输入数据到寄存器
const float f0 = packed_filter_tensor[0][w][idx];
const float f1 = packed_filter_tensor[1][w][idx];
const float f2 = packed_filter_tensor[2][w][idx];

// 优化步骤2：预计算倒数常量（编译器可能自动完成，显式写出更友好）
constexpr float inv4 = 1.0f/4.0f;    // 0.25f
constexpr float inv6 = 1.0f/6.0f;    // ~0.166667f 
constexpr float inv12 = 1.0f/12.0f;  // ~0.083333f
constexpr float inv24 = 1.0f/24.0f;  // ~0.041667f

// 优化步骤3：计算流重组（提升ILP）
const float f0_inv4 = f0 * inv4;          // U0
const float f1_inv6 = f1 * inv6;          // tmp1
const float f02_sum = (f0 + f2) * -inv6;  // tmp2

// 优化步骤4：并行计算独立路径
U_tensor[0][w][idx] = f0_inv4;  // 独立计算路径1
U_tensor[5][w][idx] = f2;       // 独立计算路径2

// 优化步骤5：合并关联计算
const float u3_core = f0*inv24 + f1*inv12;  // 提前计算公共部分
U_tensor[3][w][idx] = u3_core + f2*inv6;    // 完成U3计算
U_tensor[4][w][idx] = U_tensor[3][w][idx] - f1_inv6; // 直接使用U3结果

// 优化步骤6：SIMD友好型表达式
U_tensor[1][w][idx] = f02_sum - f1_inv6;  // 原始tmp2 - tmp1
U_tensor[2][w][idx] = f02_sum + f1_inv6;  // 原始tmp2 + tmp1

   // #pragma omp parallel for

    for (int64_t h = 0; h < us.h; ++h) {
    /*  float z0, z1, z2, z3, z4, z5, z6;
      z6 = U_tensor[h][0][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U_tensor[h][1][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U_tensor[h][2][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[h][0][idx] = z0;
      U_tensor[h][1][idx] = z1;
      U_tensor[h][2][idx] = z2;
      U_tensor[h][3][idx] = z3;
      U_tensor[h][4][idx] = z4;
      U_tensor[h][5][idx] = z5;*/
   
        // 优化步骤1：预加载所有输入数据到寄存器
const float u0 = U_tensor[h][0][idx];
const float u1 = U_tensor[h][1][idx];
const float u2 = U_tensor[h][2][idx];

// 优化步骤2：预计算倒数常量（使用更高效的表示）
constexpr float inv4   = 0.25f;     // 1/4
constexpr float inv6   = 0.16666667f; // 1/6
constexpr float inv12  = 0.08333333f; // 1/12
constexpr float inv24  = 0.04166667f; // 1/24

// 优化步骤3：代数重组与公共子表达式消除
const float u0_inv24 = u0 * inv24;
const float u1_inv12 = u1 * inv12;
const float u2_inv6 = u2 * inv6;

U_tensor[h][0][idx] = u0 * inv4;              // z0
U_tensor[h][1][idx] = -inv6 * (u0 + u1 + u2); // z1
U_tensor[h][2][idx] = inv6 * (u1 - u0 - u2);  // z2
U_tensor[h][3][idx] = u0_inv24 + u1_inv12 + u2_inv6; // z3
U_tensor[h][4][idx] = u0_inv24 - u1_inv12 + u2_inv6; // z4
U_tensor[h][5][idx] = u2;                      // z5

    
    }
  }
}
}
void output_transform(float *__restrict__ M,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size)
                       {
  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  
  //#pragma omp parallel for
  for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
   // #pragma omp parallel for
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      /*float z0, z1, z2, z3, z4;
      z4 = M_tensor[0][w][idx];
      z0 = z4;

      z4 = M_tensor[1][w][idx];
      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = M_tensor[2][w][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M_tensor[3][w][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M_tensor[4][w][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M_tensor[5][w][idx];
      z3 += z4;

      Y_tensor[0][w][idx] = z0;
      Y_tensor[1][w][idx] = z1;
      Y_tensor[2][w][idx] = z2;
      Y_tensor[3][w][idx] = z3;*/
      // 优化步骤1：预加载所有输入数据到寄存器
const float m0 = M_tensor[0][w][idx];
const float m1 = M_tensor[1][w][idx];
const float m2 = M_tensor[2][w][idx];
const float m3 = M_tensor[3][w][idx];
const float m4 = M_tensor[4][w][idx];
const float m5 = M_tensor[5][w][idx];

// 优化步骤2：代数重组与公共子表达式消除
const float m3_2x = 2.0f * m3;
const float m4_2x = 2.0f * m4;
const float m3_4x = 4.0f * m3;
const float m4_4x = 4.0f * m4;

// 优化步骤3：并行计算各输出通道
Y_tensor[0][w][idx] = m0 + m1 + m2 + m3 + m4;          // z0
Y_tensor[1][w][idx] = m1 - m2 + m3_2x - m4_2x;         // z1
Y_tensor[2][w][idx] = m1 + m2 + m3_4x + m4_4x;         // z2
Y_tensor[3][w][idx] = m1 - m2 + 8.0f*(m3 - m4) + m5;   // z3

    }

    //#pragma omp parallel for
    for (int64_t h = 0; h < ti.tile_out_h; ++h) {
      float y0 = Y_tensor[h][0][idx];
      float y1 = Y_tensor[h][1][idx];
      float y2 = Y_tensor[h][2][idx];
      float y3 = Y_tensor[h][3][idx];
      float y4 = Y_tensor[h][4][idx];
      float y5 = Y_tensor[h][5][idx];
      
      // 预计算公共子表达式
      float sum1_2 = y1 + y2;
      float diff1_2 = y1 - y2;
      float sum3_4 = y3 + y4;
      float diff3_4 = y3 - y4;
      
      // 最终计算结果
      Y_tensor[h][0][idx] = y0 + sum1_2 + sum3_4;
      Y_tensor[h][1][idx] = diff1_2 + 2.0f * diff3_4;
      Y_tensor[h][2][idx] = sum1_2 + 4.0f * sum3_4;
      Y_tensor[h][3][idx] = diff1_2 + 8.0f * diff3_4 + y5;
      
    }
  }
}

void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

  #pragma omp parallel for collapse(4)

  for (int64_t h = 0; h < fs.h; ++h)
    for (int64_t w = 0; w < fs.w; ++w)
      for (int64_t oc = 0; oc < fs.oc; oc++)
        for (int64_t ic = 0; ic < fs.ic; ic++)
          packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
}

void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
  typedef float(*packedImage_tensor_t)[ti.tile_in_w][ti.num_tiles][is.ic];
  typedef float(*image_tensor_t)[is.ic][is.h][is.w];
  packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  image_tensor_t image_tensor = (image_tensor_t)image;

#pragma omp parallel for collapse(4)

  for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
    for (int64_t ic = 0; ic < is.ic; ic++) {
      for (int64_t h = 0; h < ti.tile_in_h; ++h) {
        for (int64_t w = 0; w < ti.tile_in_w; ++w) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if ( ( hh << 2 ) + h < is.h && (ww << 2 ) + w < is.w)
            packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][(( hh << 2 ) + h)][(ww << 2 )+w];
          else
            packed_image_tensor[h][w][tile][ic] = 0;
        }
      }
    }
  }
}

void output_unpacking_store(float *__restrict__ Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) {
  typedef float(*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  #pragma omp parallel for collapse(4)

  for (int64_t h = 0; h < ti.tile_out_h; ++h) {
    for (int64_t w = 0; w < ti.tile_out_w; ++w) {
      for (int64_t oc = 0; oc < os.oc; oc++) {
        for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if ((hh << 2) + h < os.h && (ww << 2) + w < os.w)
            out_tensor[batch][oc][((hh << 2) + h)][((ww << 2) + w)] = Y_tensor[h][w][oc][tile];
        }
      }
    }
  }
}

void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
  typedef float(*A_tensor_t)[K];
  typedef float(*B_tensor_t)[K];
  typedef float(*C_tensor_t)[M];
  A_tensor_t A_tensor = (A_tensor_t)A;
  B_tensor_t B_tensor = (B_tensor_t)B;
  C_tensor_t C_tensor = (C_tensor_t)C;

 // #pragma omp parallel for collapse(2)
 // #pragma omp parallel

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      C_tensor[n][m] = 0;
        //#pragma omp parallel for
      for (int64_t k = 0; k < K; ++k) {
        C_tensor[n][m] += A_tensor[m][k] * B_tensor[n][k];
      }
    }
  }
}

void winograd_convolution(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
  /* new vars of shape */
  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  const out_shape_t os = get_output_shape(is, fs);
  const tiling_info_t ti = get_tiling_info(is, os);
  const U_shape_t us = get_U_shape(fs, ti);
  const V_shape_t vs = get_V_shape(is, ti);

  float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
  float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
  float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
  float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
  float *Y = (float *)malloc(sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

  //#pragma omp parallel

  filter_packing(filter, packed_filter, fs);
  filter_transform(packed_filter, U, fs, us, us.oc * us.ic);

 // #pragma omp parallel

  image_packing(image, packed_image, is, ti);
  image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

 // #pragma omp parallel for collapse(2)

  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
      typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
      typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
      U_tensor_t U_tensor = (U_tensor_t)U;
      V_tensor_t V_tensor = (V_tensor_t)V;
      M_tensor_t M_tensor = (M_tensor_t)M;

      sgemm(vs.num_tiles,
            us.oc,
            us.ic,
            (float *)(V_tensor[h][w]),
            (float *)(U_tensor[h][w]),
            (float *)(M_tensor[h][w]));
    }
  }

  output_transform(M, Y, ti, us.oc * vs.num_tiles);
  output_unpacking_store(Y, out, os, ti);

  free(packed_filter);
  free(packed_image);
  free(U);
  free(V);
  free(M);
  free(Y);
}