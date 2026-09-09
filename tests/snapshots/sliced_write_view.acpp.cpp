// === base name ===
kernel_852cbda0cadb0ca3

// === header ===
void launcher_kernel_852cbda0cadb0ca3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_852cbda0cadb0ca3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_852cbda0cadb0ca3(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_852cbda0cadb0ca3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
                int32_t v19_lead = v12_lead + (v13_i0 * 32);
                #pragma unroll
                for (int32_t v14_i1 = 10; v14_i1 < 13; ++v14_i1) {
                  float v22_data = glb_m1[(v19_lead + (v14_i1 * 32))];
                  r0[(v13_i0 + (v14_i1 - 10))] = v22_data;
                }
              }
              float r1[1]{};
              // r1 = load{g>r}(glb_m2);
              if ((v12_lead >= 10) && (v12_lead < 13)) {
                #pragma unroll
                for (int32_t v32_i1 = 8; v32_i1 < 9; ++v32_i1) {
                  float v40_data = glb_m2[(v12_lead + (v32_i1 * 13))];
                  r1[(v32_i1 - 8)] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[1]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 1)] [(10, 13)]
              float ir2[1]{};
              float v48_data = r0[0];
              float v49_data = r1[0];
              float v52_data = ir2[0];
              ir2[0] = (v52_data + (v48_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              float v57_data = r0[1];
              float v61_data = ir2[0];
              ir2[0] = (v61_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              float v66_data = r0[2];
              float v70_data = ir2[0];
              ir2[0] = (v70_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              #pragma unroll
              for (int32_t v75_n0 = 0; v75_n0 < 1; ++v75_n0) {
                #pragma unroll
                for (int32_t v76_n1 = 0; v76_n1 < 1; ++v76_n1) {
                  int32_t v77_a = v75_n0 + v76_n1;
                  float v78_data = ir2[v77_a];
                  r2[v77_a] = v78_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v83_i0 = 0; v83_i0 < 1; ++v83_i0) {
                int32_t v91_lead = v12_lead + (v83_i0 * 32);
                #pragma unroll
                for (int32_t v84_i1 = 0; v84_i1 < 1; ++v84_i1) {
                  float v86_data = r2[(v83_i0 + v84_i1)];
                  glb_m0[(v91_lead + ((v84_i1 + 8) * 32))] = v86_data;
                }
              }
              float r3[13]{};
              // r3 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v99_i0 = 0; v99_i0 < 1; ++v99_i0) {
                int32_t v105_lead = v12_lead + (v99_i0 * 32);
                #pragma unroll
                for (int32_t v100_i1 = 0; v100_i1 < 13; ++v100_i1) {
                  float v108_data = glb_m0[(v105_lead + (v100_i1 * 32))];
                  r3[(v99_i0 + v100_i1)] = v108_data;
                }
              }
              float r4[13]{};
              // r4 = load{g>r}(glb_m4);
              if (v12_lead < 13) {
                #pragma unroll
                for (int32_t v115_i1 = 0; v115_i1 < 13; ++v115_i1) {
                  float v123_data = glb_m4[(v12_lead + (v115_i1 * 13))];
                  r4[v115_i1] = v123_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m0););
              // wait(r4 = load{g>r}(glb_m4););
              float r5[13]{};
              // r5 = +(r3 * r4) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir5[13]{};
              float v130_data = r3[0];
              float v131_data = r4[0];
              float v134_data = ir5[0];
              ir5[0] = (v134_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 0))));
              float v137_data = r4[1];
              float v140_data = ir5[1];
              ir5[1] = (v140_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 0))));
              float v143_data = r4[2];
              float v146_data = ir5[2];
              ir5[2] = (v146_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 0))));
              float v149_data = r4[3];
              float v152_data = ir5[3];
              ir5[3] = (v152_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 0))));
              float v155_data = r4[4];
              float v158_data = ir5[4];
              ir5[4] = (v158_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 0))));
              float v161_data = r4[5];
              float v164_data = ir5[5];
              ir5[5] = (v164_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 0))));
              float v167_data = r4[6];
              float v170_data = ir5[6];
              ir5[6] = (v170_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 0))));
              float v173_data = r4[7];
              float v176_data = ir5[7];
              ir5[7] = (v176_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 0))));
              float v179_data = r4[8];
              float v182_data = ir5[8];
              ir5[8] = (v182_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 0))));
              float v185_data = r4[9];
              float v188_data = ir5[9];
              ir5[9] = (v188_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 0))));
              float v191_data = r4[10];
              float v194_data = ir5[10];
              ir5[10] = (v194_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 0))));
              float v197_data = r4[11];
              float v200_data = ir5[11];
              ir5[11] = (v200_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 0))));
              float v203_data = r4[12];
              float v206_data = ir5[12];
              ir5[12] = (v206_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 0))));
              float v211_data = r3[1];
              float v215_data = ir5[0];
              ir5[0] = (v215_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
              float v221_data = ir5[1];
              ir5[1] = (v221_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 1))));
              float v227_data = ir5[2];
              ir5[2] = (v227_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 1))));
              float v233_data = ir5[3];
              ir5[3] = (v233_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 1))));
              float v239_data = ir5[4];
              ir5[4] = (v239_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 1))));
              float v245_data = ir5[5];
              ir5[5] = (v245_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 1))));
              float v251_data = ir5[6];
              ir5[6] = (v251_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 1))));
              float v257_data = ir5[7];
              ir5[7] = (v257_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 1))));
              float v263_data = ir5[8];
              ir5[8] = (v263_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 1))));
              float v269_data = ir5[9];
              ir5[9] = (v269_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 1))));
              float v275_data = ir5[10];
              ir5[10] = (v275_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 1))));
              float v281_data = ir5[11];
              ir5[11] = (v281_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 1))));
              float v287_data = ir5[12];
              ir5[12] = (v287_data + (v211_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 1))));
              float v292_data = r3[2];
              float v296_data = ir5[0];
              ir5[0] = (v296_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 2))));
              float v302_data = ir5[1];
              ir5[1] = (v302_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 2))));
              float v308_data = ir5[2];
              ir5[2] = (v308_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 2))));
              float v314_data = ir5[3];
              ir5[3] = (v314_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 2))));
              float v320_data = ir5[4];
              ir5[4] = (v320_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 2))));
              float v326_data = ir5[5];
              ir5[5] = (v326_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 2))));
              float v332_data = ir5[6];
              ir5[6] = (v332_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
              float v338_data = ir5[7];
              ir5[7] = (v338_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 2))));
              float v344_data = ir5[8];
              ir5[8] = (v344_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 2))));
              float v350_data = ir5[9];
              ir5[9] = (v350_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 2))));
              float v356_data = ir5[10];
              ir5[10] = (v356_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 2))));
              float v362_data = ir5[11];
              ir5[11] = (v362_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 2))));
              float v368_data = ir5[12];
              ir5[12] = (v368_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 2))));
              float v373_data = r3[3];
              float v377_data = ir5[0];
              ir5[0] = (v377_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 3))));
              float v383_data = ir5[1];
              ir5[1] = (v383_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 3))));
              float v389_data = ir5[2];
              ir5[2] = (v389_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 3))));
              float v395_data = ir5[3];
              ir5[3] = (v395_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 3))));
              float v401_data = ir5[4];
              ir5[4] = (v401_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 3))));
              float v407_data = ir5[5];
              ir5[5] = (v407_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 3))));
              float v413_data = ir5[6];
              ir5[6] = (v413_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 3))));
              float v419_data = ir5[7];
              ir5[7] = (v419_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 3))));
              float v425_data = ir5[8];
              ir5[8] = (v425_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 3))));
              float v431_data = ir5[9];
              ir5[9] = (v431_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 3))));
              float v437_data = ir5[10];
              ir5[10] = (v437_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 3))));
              float v443_data = ir5[11];
              ir5[11] = (v443_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 3))));
              float v449_data = ir5[12];
              ir5[12] = (v449_data + (v373_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 3))));
              float v454_data = r3[4];
              float v458_data = ir5[0];
              ir5[0] = (v458_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 4))));
              float v464_data = ir5[1];
              ir5[1] = (v464_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 4))));
              float v470_data = ir5[2];
              ir5[2] = (v470_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 4))));
              float v476_data = ir5[3];
              ir5[3] = (v476_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 4))));
              float v482_data = ir5[4];
              ir5[4] = (v482_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 4))));
              float v488_data = ir5[5];
              ir5[5] = (v488_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 4))));
              float v494_data = ir5[6];
              ir5[6] = (v494_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 4))));
              float v500_data = ir5[7];
              ir5[7] = (v500_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 4))));
              float v506_data = ir5[8];
              ir5[8] = (v506_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 4))));
              float v512_data = ir5[9];
              ir5[9] = (v512_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 4))));
              float v518_data = ir5[10];
              ir5[10] = (v518_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 4))));
              float v524_data = ir5[11];
              ir5[11] = (v524_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 4))));
              float v530_data = ir5[12];
              ir5[12] = (v530_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 4))));
              float v535_data = r3[5];
              float v539_data = ir5[0];
              ir5[0] = (v539_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 5))));
              float v545_data = ir5[1];
              ir5[1] = (v545_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 5))));
              float v551_data = ir5[2];
              ir5[2] = (v551_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 5))));
              float v557_data = ir5[3];
              ir5[3] = (v557_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 5))));
              float v563_data = ir5[4];
              ir5[4] = (v563_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 5))));
              float v569_data = ir5[5];
              ir5[5] = (v569_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 5))));
              float v575_data = ir5[6];
              ir5[6] = (v575_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 5))));
              float v581_data = ir5[7];
              ir5[7] = (v581_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 5))));
              float v587_data = ir5[8];
              ir5[8] = (v587_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 5))));
              float v593_data = ir5[9];
              ir5[9] = (v593_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 5))));
              float v599_data = ir5[10];
              ir5[10] = (v599_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 5))));
              float v605_data = ir5[11];
              ir5[11] = (v605_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 5))));
              float v611_data = ir5[12];
              ir5[12] = (v611_data + (v535_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 5))));
              float v616_data = r3[6];
              float v620_data = ir5[0];
              ir5[0] = (v620_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 6))));
              float v626_data = ir5[1];
              ir5[1] = (v626_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 6))));
              float v632_data = ir5[2];
              ir5[2] = (v632_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 6))));
              float v638_data = ir5[3];
              ir5[3] = (v638_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 6))));
              float v644_data = ir5[4];
              ir5[4] = (v644_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 6))));
              float v650_data = ir5[5];
              ir5[5] = (v650_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 6))));
              float v656_data = ir5[6];
              ir5[6] = (v656_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 6))));
              float v662_data = ir5[7];
              ir5[7] = (v662_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 6))));
              float v668_data = ir5[8];
              ir5[8] = (v668_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 6))));
              float v674_data = ir5[9];
              ir5[9] = (v674_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 6))));
              float v680_data = ir5[10];
              ir5[10] = (v680_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 6))));
              float v686_data = ir5[11];
              ir5[11] = (v686_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 6))));
              float v692_data = ir5[12];
              ir5[12] = (v692_data + (v616_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 6))));
              float v697_data = r3[7];
              float v701_data = ir5[0];
              ir5[0] = (v701_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 7))));
              float v707_data = ir5[1];
              ir5[1] = (v707_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 7))));
              float v713_data = ir5[2];
              ir5[2] = (v713_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 7))));
              float v719_data = ir5[3];
              ir5[3] = (v719_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 7))));
              float v725_data = ir5[4];
              ir5[4] = (v725_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 7))));
              float v731_data = ir5[5];
              ir5[5] = (v731_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 7))));
              float v737_data = ir5[6];
              ir5[6] = (v737_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 7))));
              float v743_data = ir5[7];
              ir5[7] = (v743_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 7))));
              float v749_data = ir5[8];
              ir5[8] = (v749_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 7))));
              float v755_data = ir5[9];
              ir5[9] = (v755_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 7))));
              float v761_data = ir5[10];
              ir5[10] = (v761_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 7))));
              float v767_data = ir5[11];
              ir5[11] = (v767_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 7))));
              float v773_data = ir5[12];
              ir5[12] = (v773_data + (v697_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 7))));
              float v778_data = r3[8];
              float v782_data = ir5[0];
              ir5[0] = (v782_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 8))));
              float v788_data = ir5[1];
              ir5[1] = (v788_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 8))));
              float v794_data = ir5[2];
              ir5[2] = (v794_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 8))));
              float v800_data = ir5[3];
              ir5[3] = (v800_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 8))));
              float v806_data = ir5[4];
              ir5[4] = (v806_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 8))));
              float v812_data = ir5[5];
              ir5[5] = (v812_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 8))));
              float v818_data = ir5[6];
              ir5[6] = (v818_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 8))));
              float v824_data = ir5[7];
              ir5[7] = (v824_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 8))));
              float v830_data = ir5[8];
              ir5[8] = (v830_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 8))));
              float v836_data = ir5[9];
              ir5[9] = (v836_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 8))));
              float v842_data = ir5[10];
              ir5[10] = (v842_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 8))));
              float v848_data = ir5[11];
              ir5[11] = (v848_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 8))));
              float v854_data = ir5[12];
              ir5[12] = (v854_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 8))));
              float v859_data = r3[9];
              float v863_data = ir5[0];
              ir5[0] = (v863_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 9))));
              float v869_data = ir5[1];
              ir5[1] = (v869_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 9))));
              float v875_data = ir5[2];
              ir5[2] = (v875_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 9))));
              float v881_data = ir5[3];
              ir5[3] = (v881_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 9))));
              float v887_data = ir5[4];
              ir5[4] = (v887_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 9))));
              float v893_data = ir5[5];
              ir5[5] = (v893_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 9))));
              float v899_data = ir5[6];
              ir5[6] = (v899_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 9))));
              float v905_data = ir5[7];
              ir5[7] = (v905_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 9))));
              float v911_data = ir5[8];
              ir5[8] = (v911_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 9))));
              float v917_data = ir5[9];
              ir5[9] = (v917_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 9))));
              float v923_data = ir5[10];
              ir5[10] = (v923_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 9))));
              float v929_data = ir5[11];
              ir5[11] = (v929_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 9))));
              float v935_data = ir5[12];
              ir5[12] = (v935_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 9))));
              float v940_data = r3[10];
              float v944_data = ir5[0];
              ir5[0] = (v944_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 10))));
              float v950_data = ir5[1];
              ir5[1] = (v950_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 10))));
              float v956_data = ir5[2];
              ir5[2] = (v956_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 10))));
              float v962_data = ir5[3];
              ir5[3] = (v962_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 10))));
              float v968_data = ir5[4];
              ir5[4] = (v968_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 10))));
              float v974_data = ir5[5];
              ir5[5] = (v974_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 10))));
              float v980_data = ir5[6];
              ir5[6] = (v980_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 10))));
              float v986_data = ir5[7];
              ir5[7] = (v986_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 10))));
              float v992_data = ir5[8];
              ir5[8] = (v992_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 10))));
              float v998_data = ir5[9];
              ir5[9] = (v998_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 10))));
              float v1004_data = ir5[10];
              ir5[10] = (v1004_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 10))));
              float v1010_data = ir5[11];
              ir5[11] = (v1010_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 10))));
              float v1016_data = ir5[12];
              ir5[12] = (v1016_data + (v940_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 10))));
              float v1021_data = r3[11];
              float v1025_data = ir5[0];
              ir5[0] = (v1025_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 11))));
              float v1031_data = ir5[1];
              ir5[1] = (v1031_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 11))));
              float v1037_data = ir5[2];
              ir5[2] = (v1037_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 11))));
              float v1043_data = ir5[3];
              ir5[3] = (v1043_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 11))));
              float v1049_data = ir5[4];
              ir5[4] = (v1049_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 11))));
              float v1055_data = ir5[5];
              ir5[5] = (v1055_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 11))));
              float v1061_data = ir5[6];
              ir5[6] = (v1061_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 11))));
              float v1067_data = ir5[7];
              ir5[7] = (v1067_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 11))));
              float v1073_data = ir5[8];
              ir5[8] = (v1073_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 11))));
              float v1079_data = ir5[9];
              ir5[9] = (v1079_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 11))));
              float v1085_data = ir5[10];
              ir5[10] = (v1085_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 11))));
              float v1091_data = ir5[11];
              ir5[11] = (v1091_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 11))));
              float v1097_data = ir5[12];
              ir5[12] = (v1097_data + (v1021_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 11))));
              float v1102_data = r3[12];
              float v1106_data = ir5[0];
              ir5[0] = (v1106_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 12))));
              float v1112_data = ir5[1];
              ir5[1] = (v1112_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 12))));
              float v1118_data = ir5[2];
              ir5[2] = (v1118_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 12))));
              float v1124_data = ir5[3];
              ir5[3] = (v1124_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 12))));
              float v1130_data = ir5[4];
              ir5[4] = (v1130_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 12))));
              float v1136_data = ir5[5];
              ir5[5] = (v1136_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 12))));
              float v1142_data = ir5[6];
              ir5[6] = (v1142_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 12))));
              float v1148_data = ir5[7];
              ir5[7] = (v1148_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 12))));
              float v1154_data = ir5[8];
              ir5[8] = (v1154_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 12))));
              float v1160_data = ir5[9];
              ir5[9] = (v1160_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 12))));
              float v1166_data = ir5[10];
              ir5[10] = (v1166_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 12))));
              float v1172_data = ir5[11];
              ir5[11] = (v1172_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 12))));
              float v1178_data = ir5[12];
              ir5[12] = (v1178_data + (v1102_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 12))));
              #pragma unroll
              for (int32_t v1183_n0 = 0; v1183_n0 < 1; ++v1183_n0) {
                #pragma unroll
                for (int32_t v1184_n1 = 0; v1184_n1 < 13; ++v1184_n1) {
                  int32_t v1185_a = v1183_n0 + v1184_n1;
                  float v1186_data = ir5[v1185_a];
                  r5[v1185_a] = v1186_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1191_i0 = 0; v1191_i0 < 1; ++v1191_i0) {
                int32_t v1199_lead = v12_lead + (v1191_i0 * 32);
                #pragma unroll
                for (int32_t v1192_i1 = 0; v1192_i1 < 13; ++v1192_i1) {
                  float v1194_data = r5[(v1191_i0 + v1192_i1)];
                  glb_m3[(v1199_lead + (v1192_i1 * 32))] = v1194_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

