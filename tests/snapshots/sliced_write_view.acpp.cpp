// === base name ===
kernel_939857c66e

// === header ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_939857c66e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_939857c66e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_939857c66e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×13(32×13) {0..32}×{0..13} strided
        // m1 32×13(32×13) {0..32}×{0..13} strided
        // m2 13×13(13×13) {0..13}×{0..13} strided
        // m3 32×13(32×13) {0..32}×{0..13} strided
        // m4 13×13(13×13) {0..13}×{0..13} strided
        // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{0..1})[-1, 1]
        // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 32);
                #pragma unroll
                for (int32_t v10_i1 = 10; v10_i1 < 13; ++v10_i1) {
                  float v18_data = glb_m1[(v15_lead + (v10_i1 * 32))];
                  r0[(v9_i0 + (v10_i1 - 10))] = v18_data;
                }
              }
              float r1[1]{};
              // r1 = load{g>r}(glb_m2);
              bool v26_g = v8_lead < 13;
              if ((v8_lead >= 10) && v26_g) {
                #pragma unroll
                for (int32_t v28_i1 = 8; v28_i1 < 9; ++v28_i1) {
                  float v36_data = glb_m2[(v8_lead + (v28_i1 * 13))];
                  r1[(v28_i1 - 8)] = v36_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[1]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 1)] [(10, 13)]
              float ir2[1]{};
              float v44_data = r0[0];
              float v45_data = r1[0];
              float v48_data = ir2[0];
              ir2[0] = (v48_data + (v44_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 10))));
              float v53_data = r0[1];
              float v57_data = ir2[0];
              ir2[0] = (v57_data + (v53_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 11))));
              float v62_data = r0[2];
              float v66_data = ir2[0];
              ir2[0] = (v66_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 12))));
              #pragma unroll
              for (int32_t v71_n0 = 0; v71_n0 < 1; ++v71_n0) {
                #pragma unroll
                for (int32_t v72_n1 = 0; v72_n1 < 1; ++v72_n1) {
                  int32_t v73_a = v71_n0 + v72_n1;
                  float v74_data = ir2[v73_a];
                  r2[v73_a] = v74_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v79_i0 = 0; v79_i0 < 1; ++v79_i0) {
                int32_t v87_lead = v8_lead + (v79_i0 * 32);
                #pragma unroll
                for (int32_t v80_i1 = 0; v80_i1 < 1; ++v80_i1) {
                  float v82_data = r2[(v79_i0 + v80_i1)];
                  glb_m0[(v87_lead + ((v80_i1 + 8) * 32))] = v82_data;
                }
              }
              float r3[13]{};
              // r3 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v95_i0 = 0; v95_i0 < 1; ++v95_i0) {
                int32_t v101_lead = v8_lead + (v95_i0 * 32);
                #pragma unroll
                for (int32_t v96_i1 = 0; v96_i1 < 13; ++v96_i1) {
                  float v104_data = glb_m0[(v101_lead + (v96_i1 * 32))];
                  r3[(v95_i0 + v96_i1)] = v104_data;
                }
              }
              float r4[13]{};
              // r4 = load{g>r}(glb_m4);
              if (v26_g) {
                #pragma unroll
                for (int32_t v111_i1 = 0; v111_i1 < 13; ++v111_i1) {
                  float v119_data = glb_m4[(v8_lead + (v111_i1 * 13))];
                  r4[v111_i1] = v119_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m0););
              // wait(r4 = load{g>r}(glb_m4););
              float r5[13]{};
              // r5 = +(r3 * r4) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir5[13]{};
              float v126_data = r3[0];
              float v127_data = r4[0];
              float v130_data = ir5[0];
              ir5[0] = (v130_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 0))));
              float v133_data = r4[1];
              float v136_data = ir5[1];
              ir5[1] = (v136_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 0))));
              float v139_data = r4[2];
              float v142_data = ir5[2];
              ir5[2] = (v142_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 0))));
              float v145_data = r4[3];
              float v148_data = ir5[3];
              ir5[3] = (v148_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 0))));
              float v151_data = r4[4];
              float v154_data = ir5[4];
              ir5[4] = (v154_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 0))));
              float v157_data = r4[5];
              float v160_data = ir5[5];
              ir5[5] = (v160_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 0))));
              float v163_data = r4[6];
              float v166_data = ir5[6];
              ir5[6] = (v166_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 0))));
              float v169_data = r4[7];
              float v172_data = ir5[7];
              ir5[7] = (v172_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 0))));
              float v175_data = r4[8];
              float v178_data = ir5[8];
              ir5[8] = (v178_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 0))));
              float v181_data = r4[9];
              float v184_data = ir5[9];
              ir5[9] = (v184_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 0))));
              float v187_data = r4[10];
              float v190_data = ir5[10];
              ir5[10] = (v190_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 0))));
              float v193_data = r4[11];
              float v196_data = ir5[11];
              ir5[11] = (v196_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 0))));
              float v199_data = r4[12];
              float v202_data = ir5[12];
              ir5[12] = (v202_data + (v126_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 0))));
              float v207_data = r3[1];
              float v211_data = ir5[0];
              ir5[0] = (v211_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
              float v217_data = ir5[1];
              ir5[1] = (v217_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 1))));
              float v223_data = ir5[2];
              ir5[2] = (v223_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 1))));
              float v229_data = ir5[3];
              ir5[3] = (v229_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 1))));
              float v235_data = ir5[4];
              ir5[4] = (v235_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 1))));
              float v241_data = ir5[5];
              ir5[5] = (v241_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 1))));
              float v247_data = ir5[6];
              ir5[6] = (v247_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 1))));
              float v253_data = ir5[7];
              ir5[7] = (v253_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 1))));
              float v259_data = ir5[8];
              ir5[8] = (v259_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 1))));
              float v265_data = ir5[9];
              ir5[9] = (v265_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 1))));
              float v271_data = ir5[10];
              ir5[10] = (v271_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 1))));
              float v277_data = ir5[11];
              ir5[11] = (v277_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 1))));
              float v283_data = ir5[12];
              ir5[12] = (v283_data + (v207_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 1))));
              float v288_data = r3[2];
              float v292_data = ir5[0];
              ir5[0] = (v292_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 2))));
              float v298_data = ir5[1];
              ir5[1] = (v298_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 2))));
              float v304_data = ir5[2];
              ir5[2] = (v304_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 2))));
              float v310_data = ir5[3];
              ir5[3] = (v310_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 2))));
              float v316_data = ir5[4];
              ir5[4] = (v316_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 2))));
              float v322_data = ir5[5];
              ir5[5] = (v322_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 2))));
              float v328_data = ir5[6];
              ir5[6] = (v328_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 2))));
              float v334_data = ir5[7];
              ir5[7] = (v334_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
              float v340_data = ir5[8];
              ir5[8] = (v340_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
              float v346_data = ir5[9];
              ir5[9] = (v346_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 2))));
              float v352_data = ir5[10];
              ir5[10] = (v352_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 2))));
              float v358_data = ir5[11];
              ir5[11] = (v358_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 2))));
              float v364_data = ir5[12];
              ir5[12] = (v364_data + (v288_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 2))));
              float v369_data = r3[3];
              float v373_data = ir5[0];
              ir5[0] = (v373_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 3))));
              float v379_data = ir5[1];
              ir5[1] = (v379_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 3))));
              float v385_data = ir5[2];
              ir5[2] = (v385_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 3))));
              float v391_data = ir5[3];
              ir5[3] = (v391_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 3))));
              float v397_data = ir5[4];
              ir5[4] = (v397_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 3))));
              float v403_data = ir5[5];
              ir5[5] = (v403_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 3))));
              float v409_data = ir5[6];
              ir5[6] = (v409_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 3))));
              float v415_data = ir5[7];
              ir5[7] = (v415_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 3))));
              float v421_data = ir5[8];
              ir5[8] = (v421_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 3))));
              float v427_data = ir5[9];
              ir5[9] = (v427_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 3))));
              float v433_data = ir5[10];
              ir5[10] = (v433_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 3))));
              float v439_data = ir5[11];
              ir5[11] = (v439_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 3))));
              float v445_data = ir5[12];
              ir5[12] = (v445_data + (v369_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 3))));
              float v450_data = r3[4];
              float v454_data = ir5[0];
              ir5[0] = (v454_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 4))));
              float v460_data = ir5[1];
              ir5[1] = (v460_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 4))));
              float v466_data = ir5[2];
              ir5[2] = (v466_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 4))));
              float v472_data = ir5[3];
              ir5[3] = (v472_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 4))));
              float v478_data = ir5[4];
              ir5[4] = (v478_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 4))));
              float v484_data = ir5[5];
              ir5[5] = (v484_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 4))));
              float v490_data = ir5[6];
              ir5[6] = (v490_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 4))));
              float v496_data = ir5[7];
              ir5[7] = (v496_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 4))));
              float v502_data = ir5[8];
              ir5[8] = (v502_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 4))));
              float v508_data = ir5[9];
              ir5[9] = (v508_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 4))));
              float v514_data = ir5[10];
              ir5[10] = (v514_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 4))));
              float v520_data = ir5[11];
              ir5[11] = (v520_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 4))));
              float v526_data = ir5[12];
              ir5[12] = (v526_data + (v450_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 4))));
              float v531_data = r3[5];
              float v535_data = ir5[0];
              ir5[0] = (v535_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 5))));
              float v541_data = ir5[1];
              ir5[1] = (v541_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 5))));
              float v547_data = ir5[2];
              ir5[2] = (v547_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 5))));
              float v553_data = ir5[3];
              ir5[3] = (v553_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 5))));
              float v559_data = ir5[4];
              ir5[4] = (v559_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 5))));
              float v565_data = ir5[5];
              ir5[5] = (v565_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 5))));
              float v571_data = ir5[6];
              ir5[6] = (v571_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 5))));
              float v577_data = ir5[7];
              ir5[7] = (v577_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 5))));
              float v583_data = ir5[8];
              ir5[8] = (v583_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 5))));
              float v589_data = ir5[9];
              ir5[9] = (v589_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 5))));
              float v595_data = ir5[10];
              ir5[10] = (v595_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 5))));
              float v601_data = ir5[11];
              ir5[11] = (v601_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 5))));
              float v607_data = ir5[12];
              ir5[12] = (v607_data + (v531_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 5))));
              float v612_data = r3[6];
              float v616_data = ir5[0];
              ir5[0] = (v616_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 6))));
              float v622_data = ir5[1];
              ir5[1] = (v622_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 6))));
              float v628_data = ir5[2];
              ir5[2] = (v628_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 6))));
              float v634_data = ir5[3];
              ir5[3] = (v634_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 6))));
              float v640_data = ir5[4];
              ir5[4] = (v640_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 6))));
              float v646_data = ir5[5];
              ir5[5] = (v646_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 6))));
              float v652_data = ir5[6];
              ir5[6] = (v652_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 6))));
              float v658_data = ir5[7];
              ir5[7] = (v658_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 6))));
              float v664_data = ir5[8];
              ir5[8] = (v664_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 6))));
              float v670_data = ir5[9];
              ir5[9] = (v670_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 6))));
              float v676_data = ir5[10];
              ir5[10] = (v676_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 6))));
              float v682_data = ir5[11];
              ir5[11] = (v682_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 6))));
              float v688_data = ir5[12];
              ir5[12] = (v688_data + (v612_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 6))));
              float v693_data = r3[7];
              float v697_data = ir5[0];
              ir5[0] = (v697_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 7))));
              float v703_data = ir5[1];
              ir5[1] = (v703_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 7))));
              float v709_data = ir5[2];
              ir5[2] = (v709_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 7))));
              float v715_data = ir5[3];
              ir5[3] = (v715_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 7))));
              float v721_data = ir5[4];
              ir5[4] = (v721_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 7))));
              float v727_data = ir5[5];
              ir5[5] = (v727_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 7))));
              float v733_data = ir5[6];
              ir5[6] = (v733_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 7))));
              float v739_data = ir5[7];
              ir5[7] = (v739_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 7))));
              float v745_data = ir5[8];
              ir5[8] = (v745_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 7))));
              float v751_data = ir5[9];
              ir5[9] = (v751_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 7))));
              float v757_data = ir5[10];
              ir5[10] = (v757_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 7))));
              float v763_data = ir5[11];
              ir5[11] = (v763_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 7))));
              float v769_data = ir5[12];
              ir5[12] = (v769_data + (v693_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 7))));
              float v774_data = r3[8];
              float v778_data = ir5[0];
              ir5[0] = (v778_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 8))));
              float v784_data = ir5[1];
              ir5[1] = (v784_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 8))));
              float v790_data = ir5[2];
              ir5[2] = (v790_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 8))));
              float v796_data = ir5[3];
              ir5[3] = (v796_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 8))));
              float v802_data = ir5[4];
              ir5[4] = (v802_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 8))));
              float v808_data = ir5[5];
              ir5[5] = (v808_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 8))));
              float v814_data = ir5[6];
              ir5[6] = (v814_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 8))));
              float v820_data = ir5[7];
              ir5[7] = (v820_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 8))));
              float v826_data = ir5[8];
              ir5[8] = (v826_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 8))));
              float v832_data = ir5[9];
              ir5[9] = (v832_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 8))));
              float v838_data = ir5[10];
              ir5[10] = (v838_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 8))));
              float v844_data = ir5[11];
              ir5[11] = (v844_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 8))));
              float v850_data = ir5[12];
              ir5[12] = (v850_data + (v774_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 8))));
              float v855_data = r3[9];
              float v859_data = ir5[0];
              ir5[0] = (v859_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 9))));
              float v865_data = ir5[1];
              ir5[1] = (v865_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 9))));
              float v871_data = ir5[2];
              ir5[2] = (v871_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 9))));
              float v877_data = ir5[3];
              ir5[3] = (v877_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 9))));
              float v883_data = ir5[4];
              ir5[4] = (v883_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 9))));
              float v889_data = ir5[5];
              ir5[5] = (v889_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 9))));
              float v895_data = ir5[6];
              ir5[6] = (v895_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 9))));
              float v901_data = ir5[7];
              ir5[7] = (v901_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 9))));
              float v907_data = ir5[8];
              ir5[8] = (v907_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 9))));
              float v913_data = ir5[9];
              ir5[9] = (v913_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 9))));
              float v919_data = ir5[10];
              ir5[10] = (v919_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 9))));
              float v925_data = ir5[11];
              ir5[11] = (v925_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 9))));
              float v931_data = ir5[12];
              ir5[12] = (v931_data + (v855_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 9))));
              float v936_data = r3[10];
              float v940_data = ir5[0];
              ir5[0] = (v940_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 10))));
              float v946_data = ir5[1];
              ir5[1] = (v946_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 10))));
              float v952_data = ir5[2];
              ir5[2] = (v952_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 10))));
              float v958_data = ir5[3];
              ir5[3] = (v958_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 10))));
              float v964_data = ir5[4];
              ir5[4] = (v964_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 10))));
              float v970_data = ir5[5];
              ir5[5] = (v970_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 10))));
              float v976_data = ir5[6];
              ir5[6] = (v976_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 10))));
              float v982_data = ir5[7];
              ir5[7] = (v982_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 10))));
              float v988_data = ir5[8];
              ir5[8] = (v988_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 10))));
              float v994_data = ir5[9];
              ir5[9] = (v994_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 10))));
              float v1000_data = ir5[10];
              ir5[10] = (v1000_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 10))));
              float v1006_data = ir5[11];
              ir5[11] = (v1006_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 10))));
              float v1012_data = ir5[12];
              ir5[12] = (v1012_data + (v936_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 10))));
              float v1017_data = r3[11];
              float v1021_data = ir5[0];
              ir5[0] = (v1021_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 11))));
              float v1027_data = ir5[1];
              ir5[1] = (v1027_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 11))));
              float v1033_data = ir5[2];
              ir5[2] = (v1033_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 11))));
              float v1039_data = ir5[3];
              ir5[3] = (v1039_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 11))));
              float v1045_data = ir5[4];
              ir5[4] = (v1045_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 11))));
              float v1051_data = ir5[5];
              ir5[5] = (v1051_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 11))));
              float v1057_data = ir5[6];
              ir5[6] = (v1057_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 11))));
              float v1063_data = ir5[7];
              ir5[7] = (v1063_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 11))));
              float v1069_data = ir5[8];
              ir5[8] = (v1069_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 11))));
              float v1075_data = ir5[9];
              ir5[9] = (v1075_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 11))));
              float v1081_data = ir5[10];
              ir5[10] = (v1081_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 11))));
              float v1087_data = ir5[11];
              ir5[11] = (v1087_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 11))));
              float v1093_data = ir5[12];
              ir5[12] = (v1093_data + (v1017_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 11))));
              float v1098_data = r3[12];
              float v1102_data = ir5[0];
              ir5[0] = (v1102_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 12))));
              float v1108_data = ir5[1];
              ir5[1] = (v1108_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 12))));
              float v1114_data = ir5[2];
              ir5[2] = (v1114_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 12))));
              float v1120_data = ir5[3];
              ir5[3] = (v1120_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 12))));
              float v1126_data = ir5[4];
              ir5[4] = (v1126_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 12))));
              float v1132_data = ir5[5];
              ir5[5] = (v1132_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 12))));
              float v1138_data = ir5[6];
              ir5[6] = (v1138_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 12))));
              float v1144_data = ir5[7];
              ir5[7] = (v1144_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 12))));
              float v1150_data = ir5[8];
              ir5[8] = (v1150_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 12))));
              float v1156_data = ir5[9];
              ir5[9] = (v1156_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 12))));
              float v1162_data = ir5[10];
              ir5[10] = (v1162_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 12))));
              float v1168_data = ir5[11];
              ir5[11] = (v1168_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v193_data, 12))));
              float v1174_data = ir5[12];
              ir5[12] = (v1174_data + (v1098_data * (sycl::group_broadcast(item.get_sub_group(), v199_data, 12))));
              #pragma unroll
              for (int32_t v1179_n0 = 0; v1179_n0 < 1; ++v1179_n0) {
                #pragma unroll
                for (int32_t v1180_n1 = 0; v1180_n1 < 13; ++v1180_n1) {
                  int32_t v1181_a = v1179_n0 + v1180_n1;
                  float v1182_data = ir5[v1181_a];
                  r5[v1181_a] = v1182_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1187_i0 = 0; v1187_i0 < 1; ++v1187_i0) {
                int32_t v1195_lead = v8_lead + (v1187_i0 * 32);
                #pragma unroll
                for (int32_t v1188_i1 = 0; v1188_i1 < 13; ++v1188_i1) {
                  float v1190_data = r5[(v1187_i0 + v1188_i1)];
                  glb_m3[(v1195_lead + (v1188_i1 * 32))] = v1190_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

