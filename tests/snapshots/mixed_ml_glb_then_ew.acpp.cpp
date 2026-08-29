// === base name ===
kernel_8c9d1a8467

// === header ===
void launcher_kernel_8c9d1a8467(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8c9d1a8467(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8c9d1a8467(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8c9d1a8467(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(M)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v9_lead = item.get_local_id(0) % 16;
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v11_i1 = 0; v11_i1 < 8; ++v11_i1) {
                  float v19_data = glb_m1[(v9_lead + (v11_i1 * 8))];
                  r0[v11_i1] = v19_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              float v22_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v22_lin;
              float v23_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v23_lin;
              float v24_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v24_lin;
              float v25_lin = glb_m2[48 + item.get_local_id(0) * 1];
              r1[3] = v25_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir2[8]{};
              if (v9_lead < 8) {
                float v32_data = r0[0];
                float v33_data = r1[0];
                float v36_data = ir2[0];
                ir2[0] = (v36_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v33_data, 0))));
                float v39_data = r1[1];
                float v42_data = ir2[1];
                ir2[1] = (v42_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v39_data, 0))));
                float v45_data = r1[2];
                float v48_data = ir2[2];
                ir2[2] = (v48_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 0))));
                float v51_data = r1[3];
                float v54_data = ir2[3];
                ir2[3] = (v54_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 0))));
                float v57_data = r1[4];
                float v60_data = ir2[4];
                ir2[4] = (v60_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 0))));
                float v63_data = r1[5];
                float v66_data = ir2[5];
                ir2[5] = (v66_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
                float v69_data = r1[6];
                float v72_data = ir2[6];
                ir2[6] = (v72_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
                float v75_data = r1[7];
                float v78_data = ir2[7];
                ir2[7] = (v78_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
              }
              if (v9_lead < 8) {
                float v84_data = r0[1];
                float v85_data = r1[0];
                float v88_data = ir2[0];
                ir2[0] = (v88_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 1))));
                float v91_data = r1[1];
                float v94_data = ir2[1];
                ir2[1] = (v94_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 1))));
                float v97_data = r1[2];
                float v100_data = ir2[2];
                ir2[2] = (v100_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 1))));
                float v103_data = r1[3];
                float v106_data = ir2[3];
                ir2[3] = (v106_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 1))));
                float v109_data = r1[4];
                float v112_data = ir2[4];
                ir2[4] = (v112_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 1))));
                float v115_data = r1[5];
                float v118_data = ir2[5];
                ir2[5] = (v118_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 1))));
                float v121_data = r1[6];
                float v124_data = ir2[6];
                ir2[6] = (v124_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 1))));
                float v127_data = r1[7];
                float v130_data = ir2[7];
                ir2[7] = (v130_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
              }
              if (v9_lead < 8) {
                float v136_data = r0[2];
                float v137_data = r1[0];
                float v140_data = ir2[0];
                ir2[0] = (v140_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 2))));
                float v143_data = r1[1];
                float v146_data = ir2[1];
                ir2[1] = (v146_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 2))));
                float v149_data = r1[2];
                float v152_data = ir2[2];
                ir2[2] = (v152_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 2))));
                float v155_data = r1[3];
                float v158_data = ir2[3];
                ir2[3] = (v158_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 2))));
                float v161_data = r1[4];
                float v164_data = ir2[4];
                ir2[4] = (v164_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 2))));
                float v167_data = r1[5];
                float v170_data = ir2[5];
                ir2[5] = (v170_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
                float v173_data = r1[6];
                float v176_data = ir2[6];
                ir2[6] = (v176_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 2))));
                float v179_data = r1[7];
                float v182_data = ir2[7];
                ir2[7] = (v182_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 2))));
              }
              if (v9_lead < 8) {
                float v188_data = r0[3];
                float v189_data = r1[0];
                float v192_data = ir2[0];
                ir2[0] = (v192_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 3))));
                float v195_data = r1[1];
                float v198_data = ir2[1];
                ir2[1] = (v198_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 3))));
                float v201_data = r1[2];
                float v204_data = ir2[2];
                ir2[2] = (v204_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 3))));
                float v207_data = r1[3];
                float v210_data = ir2[3];
                ir2[3] = (v210_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 3))));
                float v213_data = r1[4];
                float v216_data = ir2[4];
                ir2[4] = (v216_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v213_data, 3))));
                float v219_data = r1[5];
                float v222_data = ir2[5];
                ir2[5] = (v222_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v219_data, 3))));
                float v225_data = r1[6];
                float v228_data = ir2[6];
                ir2[6] = (v228_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 3))));
                float v231_data = r1[7];
                float v234_data = ir2[7];
                ir2[7] = (v234_data + (v188_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 3))));
              }
              if (v9_lead < 8) {
                float v240_data = r0[4];
                float v241_data = r1[0];
                float v244_data = ir2[0];
                ir2[0] = (v244_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 4))));
                float v247_data = r1[1];
                float v250_data = ir2[1];
                ir2[1] = (v250_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v247_data, 4))));
                float v253_data = r1[2];
                float v256_data = ir2[2];
                ir2[2] = (v256_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v253_data, 4))));
                float v259_data = r1[3];
                float v262_data = ir2[3];
                ir2[3] = (v262_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v259_data, 4))));
                float v265_data = r1[4];
                float v268_data = ir2[4];
                ir2[4] = (v268_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v265_data, 4))));
                float v271_data = r1[5];
                float v274_data = ir2[5];
                ir2[5] = (v274_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v271_data, 4))));
                float v277_data = r1[6];
                float v280_data = ir2[6];
                ir2[6] = (v280_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v277_data, 4))));
                float v283_data = r1[7];
                float v286_data = ir2[7];
                ir2[7] = (v286_data + (v240_data * (sycl::group_broadcast(item.get_sub_group(), v283_data, 4))));
              }
              if (v9_lead < 8) {
                float v292_data = r0[5];
                float v293_data = r1[0];
                float v296_data = ir2[0];
                ir2[0] = (v296_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v293_data, 5))));
                float v299_data = r1[1];
                float v302_data = ir2[1];
                ir2[1] = (v302_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v299_data, 5))));
                float v305_data = r1[2];
                float v308_data = ir2[2];
                ir2[2] = (v308_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 5))));
                float v311_data = r1[3];
                float v314_data = ir2[3];
                ir2[3] = (v314_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 5))));
                float v317_data = r1[4];
                float v320_data = ir2[4];
                ir2[4] = (v320_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 5))));
                float v323_data = r1[5];
                float v326_data = ir2[5];
                ir2[5] = (v326_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 5))));
                float v329_data = r1[6];
                float v332_data = ir2[6];
                ir2[6] = (v332_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v329_data, 5))));
                float v335_data = r1[7];
                float v338_data = ir2[7];
                ir2[7] = (v338_data + (v292_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 5))));
              }
              if (v9_lead < 8) {
                float v344_data = r0[6];
                float v345_data = r1[0];
                float v348_data = ir2[0];
                ir2[0] = (v348_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 6))));
                float v351_data = r1[1];
                float v354_data = ir2[1];
                ir2[1] = (v354_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 6))));
                float v357_data = r1[2];
                float v360_data = ir2[2];
                ir2[2] = (v360_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 6))));
                float v363_data = r1[3];
                float v366_data = ir2[3];
                ir2[3] = (v366_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 6))));
                float v369_data = r1[4];
                float v372_data = ir2[4];
                ir2[4] = (v372_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 6))));
                float v375_data = r1[5];
                float v378_data = ir2[5];
                ir2[5] = (v378_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v375_data, 6))));
                float v381_data = r1[6];
                float v384_data = ir2[6];
                ir2[6] = (v384_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v381_data, 6))));
                float v387_data = r1[7];
                float v390_data = ir2[7];
                ir2[7] = (v390_data + (v344_data * (sycl::group_broadcast(item.get_sub_group(), v387_data, 6))));
              }
              if (v9_lead < 8) {
                float v396_data = r0[7];
                float v397_data = r1[0];
                float v400_data = ir2[0];
                ir2[0] = (v400_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 7))));
                float v403_data = r1[1];
                float v406_data = ir2[1];
                ir2[1] = (v406_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 7))));
                float v409_data = r1[2];
                float v412_data = ir2[2];
                ir2[2] = (v412_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v409_data, 7))));
                float v415_data = r1[3];
                float v418_data = ir2[3];
                ir2[3] = (v418_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v415_data, 7))));
                float v421_data = r1[4];
                float v424_data = ir2[4];
                ir2[4] = (v424_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v421_data, 7))));
                float v427_data = r1[5];
                float v430_data = ir2[5];
                ir2[5] = (v430_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 7))));
                float v433_data = r1[6];
                float v436_data = ir2[6];
                ir2[6] = (v436_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 7))));
                float v439_data = r1[7];
                float v442_data = ir2[7];
                ir2[7] = (v442_data + (v396_data * (sycl::group_broadcast(item.get_sub_group(), v439_data, 7))));
              }
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v448_n1 = 0; v448_n1 < 8; ++v448_n1) {
                  float v450_data = ir2[v448_n1];
                  r2[v448_n1] = v450_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v456_i1 = 0; v456_i1 < 8; ++v456_i1) {
                  float v458_data = r2[v456_i1];
                  glb_m0[(v9_lead + (v456_i1 * 8))] = v458_data;
                }
              }
              // glb_m3 = abs(glb_m0)
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v470_k1 = 0; v470_k1 < 8; ++v470_k1) {
                  int32_t v476_a = v470_k1 * 8;
                  float v478_data = glb_m0[(v9_lead + v476_a)];
                  glb_m3[(v9_lead + v476_a)] = (sycl::fabs(v478_data));
                }
              }
            }
          }
        }
      });
    }
  });
}

