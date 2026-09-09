// === base name ===
kernel_59733a5fa15371c6

// === header ===
void launcher_kernel_59733a5fa15371c6(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_59733a5fa15371c6(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_59733a5fa15371c6(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_59733a5fa15371c6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×12(12×12) {0..12}×{0..12} strided
        // m2 12×8(12×8) {0..12}×{0..8} strided
        // m3 12×12(12×12) {0..12}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 12×12(12×12) {0..12}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m7 12×12(12×12) {0..12}×{0..12} strided
        // m8 12×8(12×8) {0..12}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v18_lead = item.get_local_id(0) % 16;
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
                  float v28_data = glb_m1[(v18_lead + (v20_i1 * 12))];
                  r0[v20_i1] = v28_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
                  float v43_data = glb_m2[(v18_lead + (v35_i1 * 12))];
                  r1[v35_i1] = v43_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v50_i1 = 0; v50_i1 < 12; ++v50_i1) {
                  float v58_data = glb_m3[(v18_lead + (v50_i1 * 12))];
                  r3[v50_i1] = v58_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              if (v18_lead < 12) {
                float v66_data = r0[0];
                float v67_data = r1[0];
                float v70_data = ir2[0];
                ir2[0] = (v70_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[1];
                float v76_data = ir2[1];
                ir2[1] = (v76_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[2];
                float v82_data = ir2[2];
                ir2[2] = (v82_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
                float v85_data = r1[3];
                float v88_data = ir2[3];
                ir2[3] = (v88_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
                float v91_data = r1[4];
                float v94_data = ir2[4];
                ir2[4] = (v94_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
                float v97_data = r1[5];
                float v100_data = ir2[5];
                ir2[5] = (v100_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 0))));
                float v103_data = r1[6];
                float v106_data = ir2[6];
                ir2[6] = (v106_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 0))));
                float v109_data = r1[7];
                float v112_data = ir2[7];
                ir2[7] = (v112_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 0))));
              }
              if (v18_lead < 12) {
                float v118_data = r0[1];
                float v119_data = r1[0];
                float v122_data = ir2[0];
                ir2[0] = (v122_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[1];
                float v128_data = ir2[1];
                ir2[1] = (v128_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
                float v131_data = r1[2];
                float v134_data = ir2[2];
                ir2[2] = (v134_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
                float v137_data = r1[3];
                float v140_data = ir2[3];
                ir2[3] = (v140_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 1))));
                float v143_data = r1[4];
                float v146_data = ir2[4];
                ir2[4] = (v146_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 1))));
                float v149_data = r1[5];
                float v152_data = ir2[5];
                ir2[5] = (v152_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 1))));
                float v155_data = r1[6];
                float v158_data = ir2[6];
                ir2[6] = (v158_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 1))));
                float v161_data = r1[7];
                float v164_data = ir2[7];
                ir2[7] = (v164_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 1))));
              }
              if (v18_lead < 12) {
                float v170_data = r0[2];
                float v171_data = r1[0];
                float v174_data = ir2[0];
                ir2[0] = (v174_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
                float v177_data = r1[1];
                float v180_data = ir2[1];
                ir2[1] = (v180_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
                float v183_data = r1[2];
                float v186_data = ir2[2];
                ir2[2] = (v186_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 2))));
                float v189_data = r1[3];
                float v192_data = ir2[3];
                ir2[3] = (v192_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 2))));
                float v195_data = r1[4];
                float v198_data = ir2[4];
                ir2[4] = (v198_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 2))));
                float v201_data = r1[5];
                float v204_data = ir2[5];
                ir2[5] = (v204_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 2))));
                float v207_data = r1[6];
                float v210_data = ir2[6];
                ir2[6] = (v210_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 2))));
                float v213_data = r1[7];
                float v216_data = ir2[7];
                ir2[7] = (v216_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v213_data, 2))));
              }
              if (v18_lead < 12) {
                float v222_data = r0[3];
                float v223_data = r1[0];
                float v226_data = ir2[0];
                ir2[0] = (v226_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 3))));
                float v229_data = r1[1];
                float v232_data = ir2[1];
                ir2[1] = (v232_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 3))));
                float v235_data = r1[2];
                float v238_data = ir2[2];
                ir2[2] = (v238_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 3))));
                float v241_data = r1[3];
                float v244_data = ir2[3];
                ir2[3] = (v244_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 3))));
                float v247_data = r1[4];
                float v250_data = ir2[4];
                ir2[4] = (v250_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v247_data, 3))));
                float v253_data = r1[5];
                float v256_data = ir2[5];
                ir2[5] = (v256_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v253_data, 3))));
                float v259_data = r1[6];
                float v262_data = ir2[6];
                ir2[6] = (v262_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v259_data, 3))));
                float v265_data = r1[7];
                float v268_data = ir2[7];
                ir2[7] = (v268_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v265_data, 3))));
              }
              if (v18_lead < 12) {
                float v274_data = r0[4];
                float v275_data = r1[0];
                float v278_data = ir2[0];
                ir2[0] = (v278_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
                float v281_data = r1[1];
                float v284_data = ir2[1];
                ir2[1] = (v284_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 4))));
                float v287_data = r1[2];
                float v290_data = ir2[2];
                ir2[2] = (v290_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v287_data, 4))));
                float v293_data = r1[3];
                float v296_data = ir2[3];
                ir2[3] = (v296_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v293_data, 4))));
                float v299_data = r1[4];
                float v302_data = ir2[4];
                ir2[4] = (v302_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v299_data, 4))));
                float v305_data = r1[5];
                float v308_data = ir2[5];
                ir2[5] = (v308_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 4))));
                float v311_data = r1[6];
                float v314_data = ir2[6];
                ir2[6] = (v314_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 4))));
                float v317_data = r1[7];
                float v320_data = ir2[7];
                ir2[7] = (v320_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 4))));
              }
              if (v18_lead < 12) {
                float v326_data = r0[5];
                float v327_data = r1[0];
                float v330_data = ir2[0];
                ir2[0] = (v330_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 5))));
                float v333_data = r1[1];
                float v336_data = ir2[1];
                ir2[1] = (v336_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 5))));
                float v339_data = r1[2];
                float v342_data = ir2[2];
                ir2[2] = (v342_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 5))));
                float v345_data = r1[3];
                float v348_data = ir2[3];
                ir2[3] = (v348_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 5))));
                float v351_data = r1[4];
                float v354_data = ir2[4];
                ir2[4] = (v354_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 5))));
                float v357_data = r1[5];
                float v360_data = ir2[5];
                ir2[5] = (v360_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 5))));
                float v363_data = r1[6];
                float v366_data = ir2[6];
                ir2[6] = (v366_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 5))));
                float v369_data = r1[7];
                float v372_data = ir2[7];
                ir2[7] = (v372_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 5))));
              }
              if (v18_lead < 12) {
                float v378_data = r0[6];
                float v379_data = r1[0];
                float v382_data = ir2[0];
                ir2[0] = (v382_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 6))));
                float v385_data = r1[1];
                float v388_data = ir2[1];
                ir2[1] = (v388_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 6))));
                float v391_data = r1[2];
                float v394_data = ir2[2];
                ir2[2] = (v394_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 6))));
                float v397_data = r1[3];
                float v400_data = ir2[3];
                ir2[3] = (v400_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 6))));
                float v403_data = r1[4];
                float v406_data = ir2[4];
                ir2[4] = (v406_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 6))));
                float v409_data = r1[5];
                float v412_data = ir2[5];
                ir2[5] = (v412_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v409_data, 6))));
                float v415_data = r1[6];
                float v418_data = ir2[6];
                ir2[6] = (v418_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v415_data, 6))));
                float v421_data = r1[7];
                float v424_data = ir2[7];
                ir2[7] = (v424_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v421_data, 6))));
              }
              if (v18_lead < 12) {
                float v430_data = r0[7];
                float v431_data = r1[0];
                float v434_data = ir2[0];
                ir2[0] = (v434_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 7))));
                float v437_data = r1[1];
                float v440_data = ir2[1];
                ir2[1] = (v440_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 7))));
                float v443_data = r1[2];
                float v446_data = ir2[2];
                ir2[2] = (v446_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 7))));
                float v449_data = r1[3];
                float v452_data = ir2[3];
                ir2[3] = (v452_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 7))));
                float v455_data = r1[4];
                float v458_data = ir2[4];
                ir2[4] = (v458_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v455_data, 7))));
                float v461_data = r1[5];
                float v464_data = ir2[5];
                ir2[5] = (v464_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v461_data, 7))));
                float v467_data = r1[6];
                float v470_data = ir2[6];
                ir2[6] = (v470_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v467_data, 7))));
                float v473_data = r1[7];
                float v476_data = ir2[7];
                ir2[7] = (v476_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v473_data, 7))));
              }
              if (v18_lead < 12) {
                float v482_data = r0[8];
                float v483_data = r1[0];
                float v486_data = ir2[0];
                ir2[0] = (v486_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 8))));
                float v489_data = r1[1];
                float v492_data = ir2[1];
                ir2[1] = (v492_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 8))));
                float v495_data = r1[2];
                float v498_data = ir2[2];
                ir2[2] = (v498_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v495_data, 8))));
                float v501_data = r1[3];
                float v504_data = ir2[3];
                ir2[3] = (v504_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v501_data, 8))));
                float v507_data = r1[4];
                float v510_data = ir2[4];
                ir2[4] = (v510_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v507_data, 8))));
                float v513_data = r1[5];
                float v516_data = ir2[5];
                ir2[5] = (v516_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v513_data, 8))));
                float v519_data = r1[6];
                float v522_data = ir2[6];
                ir2[6] = (v522_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v519_data, 8))));
                float v525_data = r1[7];
                float v528_data = ir2[7];
                ir2[7] = (v528_data + (v482_data * (sycl::group_broadcast(item.get_sub_group(), v525_data, 8))));
              }
              if (v18_lead < 12) {
                float v534_data = r0[9];
                float v535_data = r1[0];
                float v538_data = ir2[0];
                ir2[0] = (v538_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v535_data, 9))));
                float v541_data = r1[1];
                float v544_data = ir2[1];
                ir2[1] = (v544_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v541_data, 9))));
                float v547_data = r1[2];
                float v550_data = ir2[2];
                ir2[2] = (v550_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v547_data, 9))));
                float v553_data = r1[3];
                float v556_data = ir2[3];
                ir2[3] = (v556_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v553_data, 9))));
                float v559_data = r1[4];
                float v562_data = ir2[4];
                ir2[4] = (v562_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v559_data, 9))));
                float v565_data = r1[5];
                float v568_data = ir2[5];
                ir2[5] = (v568_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v565_data, 9))));
                float v571_data = r1[6];
                float v574_data = ir2[6];
                ir2[6] = (v574_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v571_data, 9))));
                float v577_data = r1[7];
                float v580_data = ir2[7];
                ir2[7] = (v580_data + (v534_data * (sycl::group_broadcast(item.get_sub_group(), v577_data, 9))));
              }
              if (v18_lead < 12) {
                float v586_data = r0[10];
                float v587_data = r1[0];
                float v590_data = ir2[0];
                ir2[0] = (v590_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v587_data, 10))));
                float v593_data = r1[1];
                float v596_data = ir2[1];
                ir2[1] = (v596_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v593_data, 10))));
                float v599_data = r1[2];
                float v602_data = ir2[2];
                ir2[2] = (v602_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 10))));
                float v605_data = r1[3];
                float v608_data = ir2[3];
                ir2[3] = (v608_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v605_data, 10))));
                float v611_data = r1[4];
                float v614_data = ir2[4];
                ir2[4] = (v614_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v611_data, 10))));
                float v617_data = r1[5];
                float v620_data = ir2[5];
                ir2[5] = (v620_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v617_data, 10))));
                float v623_data = r1[6];
                float v626_data = ir2[6];
                ir2[6] = (v626_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v623_data, 10))));
                float v629_data = r1[7];
                float v632_data = ir2[7];
                ir2[7] = (v632_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v629_data, 10))));
              }
              if (v18_lead < 12) {
                float v638_data = r0[11];
                float v639_data = r1[0];
                float v642_data = ir2[0];
                ir2[0] = (v642_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v639_data, 11))));
                float v645_data = r1[1];
                float v648_data = ir2[1];
                ir2[1] = (v648_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v645_data, 11))));
                float v651_data = r1[2];
                float v654_data = ir2[2];
                ir2[2] = (v654_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v651_data, 11))));
                float v657_data = r1[3];
                float v660_data = ir2[3];
                ir2[3] = (v660_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v657_data, 11))));
                float v663_data = r1[4];
                float v666_data = ir2[4];
                ir2[4] = (v666_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v663_data, 11))));
                float v669_data = r1[5];
                float v672_data = ir2[5];
                ir2[5] = (v672_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v669_data, 11))));
                float v675_data = r1[6];
                float v678_data = ir2[6];
                ir2[6] = (v678_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v675_data, 11))));
                float v681_data = r1[7];
                float v684_data = ir2[7];
                ir2[7] = (v684_data + (v638_data * (sycl::group_broadcast(item.get_sub_group(), v681_data, 11))));
              }
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v690_n1 = 0; v690_n1 < 8; ++v690_n1) {
                  float v692_data = ir2[v690_n1];
                  r2[v690_n1] = v692_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v699_i1 = 0; v699_i1 < 8; ++v699_i1) {
                  float v707_data = glb_m4[(v18_lead + (v699_i1 * 12))];
                  r4[v699_i1] = v707_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v714_i1 = 0; v714_i1 < 12; ++v714_i1) {
                  float v722_data = glb_m5[(v18_lead + (v714_i1 * 12))];
                  r6[v714_i1] = v722_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              if (v18_lead < 12) {
                float v730_data = r3[0];
                float v731_data = r4[0];
                float v734_data = ir5[0];
                ir5[0] = (v734_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 0))));
                float v737_data = r4[1];
                float v740_data = ir5[1];
                ir5[1] = (v740_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v737_data, 0))));
                float v743_data = r4[2];
                float v746_data = ir5[2];
                ir5[2] = (v746_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v743_data, 0))));
                float v749_data = r4[3];
                float v752_data = ir5[3];
                ir5[3] = (v752_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v749_data, 0))));
                float v755_data = r4[4];
                float v758_data = ir5[4];
                ir5[4] = (v758_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v755_data, 0))));
                float v761_data = r4[5];
                float v764_data = ir5[5];
                ir5[5] = (v764_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v761_data, 0))));
                float v767_data = r4[6];
                float v770_data = ir5[6];
                ir5[6] = (v770_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v767_data, 0))));
                float v773_data = r4[7];
                float v776_data = ir5[7];
                ir5[7] = (v776_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v773_data, 0))));
              }
              if (v18_lead < 12) {
                float v782_data = r3[1];
                float v783_data = r4[0];
                float v786_data = ir5[0];
                ir5[0] = (v786_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v783_data, 1))));
                float v789_data = r4[1];
                float v792_data = ir5[1];
                ir5[1] = (v792_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v789_data, 1))));
                float v795_data = r4[2];
                float v798_data = ir5[2];
                ir5[2] = (v798_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v795_data, 1))));
                float v801_data = r4[3];
                float v804_data = ir5[3];
                ir5[3] = (v804_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v801_data, 1))));
                float v807_data = r4[4];
                float v810_data = ir5[4];
                ir5[4] = (v810_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v807_data, 1))));
                float v813_data = r4[5];
                float v816_data = ir5[5];
                ir5[5] = (v816_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v813_data, 1))));
                float v819_data = r4[6];
                float v822_data = ir5[6];
                ir5[6] = (v822_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v819_data, 1))));
                float v825_data = r4[7];
                float v828_data = ir5[7];
                ir5[7] = (v828_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v825_data, 1))));
              }
              if (v18_lead < 12) {
                float v834_data = r3[2];
                float v835_data = r4[0];
                float v838_data = ir5[0];
                ir5[0] = (v838_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v835_data, 2))));
                float v841_data = r4[1];
                float v844_data = ir5[1];
                ir5[1] = (v844_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v841_data, 2))));
                float v847_data = r4[2];
                float v850_data = ir5[2];
                ir5[2] = (v850_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v847_data, 2))));
                float v853_data = r4[3];
                float v856_data = ir5[3];
                ir5[3] = (v856_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v853_data, 2))));
                float v859_data = r4[4];
                float v862_data = ir5[4];
                ir5[4] = (v862_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v859_data, 2))));
                float v865_data = r4[5];
                float v868_data = ir5[5];
                ir5[5] = (v868_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v865_data, 2))));
                float v871_data = r4[6];
                float v874_data = ir5[6];
                ir5[6] = (v874_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v871_data, 2))));
                float v877_data = r4[7];
                float v880_data = ir5[7];
                ir5[7] = (v880_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v877_data, 2))));
              }
              if (v18_lead < 12) {
                float v886_data = r3[3];
                float v887_data = r4[0];
                float v890_data = ir5[0];
                ir5[0] = (v890_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v887_data, 3))));
                float v893_data = r4[1];
                float v896_data = ir5[1];
                ir5[1] = (v896_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v893_data, 3))));
                float v899_data = r4[2];
                float v902_data = ir5[2];
                ir5[2] = (v902_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v899_data, 3))));
                float v905_data = r4[3];
                float v908_data = ir5[3];
                ir5[3] = (v908_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v905_data, 3))));
                float v911_data = r4[4];
                float v914_data = ir5[4];
                ir5[4] = (v914_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v911_data, 3))));
                float v917_data = r4[5];
                float v920_data = ir5[5];
                ir5[5] = (v920_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v917_data, 3))));
                float v923_data = r4[6];
                float v926_data = ir5[6];
                ir5[6] = (v926_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v923_data, 3))));
                float v929_data = r4[7];
                float v932_data = ir5[7];
                ir5[7] = (v932_data + (v886_data * (sycl::group_broadcast(item.get_sub_group(), v929_data, 3))));
              }
              if (v18_lead < 12) {
                float v938_data = r3[4];
                float v939_data = r4[0];
                float v942_data = ir5[0];
                ir5[0] = (v942_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v939_data, 4))));
                float v945_data = r4[1];
                float v948_data = ir5[1];
                ir5[1] = (v948_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v945_data, 4))));
                float v951_data = r4[2];
                float v954_data = ir5[2];
                ir5[2] = (v954_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v951_data, 4))));
                float v957_data = r4[3];
                float v960_data = ir5[3];
                ir5[3] = (v960_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v957_data, 4))));
                float v963_data = r4[4];
                float v966_data = ir5[4];
                ir5[4] = (v966_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v963_data, 4))));
                float v969_data = r4[5];
                float v972_data = ir5[5];
                ir5[5] = (v972_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v969_data, 4))));
                float v975_data = r4[6];
                float v978_data = ir5[6];
                ir5[6] = (v978_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v975_data, 4))));
                float v981_data = r4[7];
                float v984_data = ir5[7];
                ir5[7] = (v984_data + (v938_data * (sycl::group_broadcast(item.get_sub_group(), v981_data, 4))));
              }
              if (v18_lead < 12) {
                float v990_data = r3[5];
                float v991_data = r4[0];
                float v994_data = ir5[0];
                ir5[0] = (v994_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v991_data, 5))));
                float v997_data = r4[1];
                float v1000_data = ir5[1];
                ir5[1] = (v1000_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v997_data, 5))));
                float v1003_data = r4[2];
                float v1006_data = ir5[2];
                ir5[2] = (v1006_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v1003_data, 5))));
                float v1009_data = r4[3];
                float v1012_data = ir5[3];
                ir5[3] = (v1012_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v1009_data, 5))));
                float v1015_data = r4[4];
                float v1018_data = ir5[4];
                ir5[4] = (v1018_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v1015_data, 5))));
                float v1021_data = r4[5];
                float v1024_data = ir5[5];
                ir5[5] = (v1024_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v1021_data, 5))));
                float v1027_data = r4[6];
                float v1030_data = ir5[6];
                ir5[6] = (v1030_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v1027_data, 5))));
                float v1033_data = r4[7];
                float v1036_data = ir5[7];
                ir5[7] = (v1036_data + (v990_data * (sycl::group_broadcast(item.get_sub_group(), v1033_data, 5))));
              }
              if (v18_lead < 12) {
                float v1042_data = r3[6];
                float v1043_data = r4[0];
                float v1046_data = ir5[0];
                ir5[0] = (v1046_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1043_data, 6))));
                float v1049_data = r4[1];
                float v1052_data = ir5[1];
                ir5[1] = (v1052_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1049_data, 6))));
                float v1055_data = r4[2];
                float v1058_data = ir5[2];
                ir5[2] = (v1058_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1055_data, 6))));
                float v1061_data = r4[3];
                float v1064_data = ir5[3];
                ir5[3] = (v1064_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1061_data, 6))));
                float v1067_data = r4[4];
                float v1070_data = ir5[4];
                ir5[4] = (v1070_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1067_data, 6))));
                float v1073_data = r4[5];
                float v1076_data = ir5[5];
                ir5[5] = (v1076_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1073_data, 6))));
                float v1079_data = r4[6];
                float v1082_data = ir5[6];
                ir5[6] = (v1082_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1079_data, 6))));
                float v1085_data = r4[7];
                float v1088_data = ir5[7];
                ir5[7] = (v1088_data + (v1042_data * (sycl::group_broadcast(item.get_sub_group(), v1085_data, 6))));
              }
              if (v18_lead < 12) {
                float v1094_data = r3[7];
                float v1095_data = r4[0];
                float v1098_data = ir5[0];
                ir5[0] = (v1098_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1095_data, 7))));
                float v1101_data = r4[1];
                float v1104_data = ir5[1];
                ir5[1] = (v1104_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1101_data, 7))));
                float v1107_data = r4[2];
                float v1110_data = ir5[2];
                ir5[2] = (v1110_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1107_data, 7))));
                float v1113_data = r4[3];
                float v1116_data = ir5[3];
                ir5[3] = (v1116_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1113_data, 7))));
                float v1119_data = r4[4];
                float v1122_data = ir5[4];
                ir5[4] = (v1122_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1119_data, 7))));
                float v1125_data = r4[5];
                float v1128_data = ir5[5];
                ir5[5] = (v1128_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1125_data, 7))));
                float v1131_data = r4[6];
                float v1134_data = ir5[6];
                ir5[6] = (v1134_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1131_data, 7))));
                float v1137_data = r4[7];
                float v1140_data = ir5[7];
                ir5[7] = (v1140_data + (v1094_data * (sycl::group_broadcast(item.get_sub_group(), v1137_data, 7))));
              }
              if (v18_lead < 12) {
                float v1146_data = r3[8];
                float v1147_data = r4[0];
                float v1150_data = ir5[0];
                ir5[0] = (v1150_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1147_data, 8))));
                float v1153_data = r4[1];
                float v1156_data = ir5[1];
                ir5[1] = (v1156_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 8))));
                float v1159_data = r4[2];
                float v1162_data = ir5[2];
                ir5[2] = (v1162_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 8))));
                float v1165_data = r4[3];
                float v1168_data = ir5[3];
                ir5[3] = (v1168_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 8))));
                float v1171_data = r4[4];
                float v1174_data = ir5[4];
                ir5[4] = (v1174_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 8))));
                float v1177_data = r4[5];
                float v1180_data = ir5[5];
                ir5[5] = (v1180_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 8))));
                float v1183_data = r4[6];
                float v1186_data = ir5[6];
                ir5[6] = (v1186_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 8))));
                float v1189_data = r4[7];
                float v1192_data = ir5[7];
                ir5[7] = (v1192_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 8))));
              }
              if (v18_lead < 12) {
                float v1198_data = r3[9];
                float v1199_data = r4[0];
                float v1202_data = ir5[0];
                ir5[0] = (v1202_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1199_data, 9))));
                float v1205_data = r4[1];
                float v1208_data = ir5[1];
                ir5[1] = (v1208_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1205_data, 9))));
                float v1211_data = r4[2];
                float v1214_data = ir5[2];
                ir5[2] = (v1214_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1211_data, 9))));
                float v1217_data = r4[3];
                float v1220_data = ir5[3];
                ir5[3] = (v1220_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1217_data, 9))));
                float v1223_data = r4[4];
                float v1226_data = ir5[4];
                ir5[4] = (v1226_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1223_data, 9))));
                float v1229_data = r4[5];
                float v1232_data = ir5[5];
                ir5[5] = (v1232_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1229_data, 9))));
                float v1235_data = r4[6];
                float v1238_data = ir5[6];
                ir5[6] = (v1238_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1235_data, 9))));
                float v1241_data = r4[7];
                float v1244_data = ir5[7];
                ir5[7] = (v1244_data + (v1198_data * (sycl::group_broadcast(item.get_sub_group(), v1241_data, 9))));
              }
              if (v18_lead < 12) {
                float v1250_data = r3[10];
                float v1251_data = r4[0];
                float v1254_data = ir5[0];
                ir5[0] = (v1254_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1251_data, 10))));
                float v1257_data = r4[1];
                float v1260_data = ir5[1];
                ir5[1] = (v1260_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1257_data, 10))));
                float v1263_data = r4[2];
                float v1266_data = ir5[2];
                ir5[2] = (v1266_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1263_data, 10))));
                float v1269_data = r4[3];
                float v1272_data = ir5[3];
                ir5[3] = (v1272_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1269_data, 10))));
                float v1275_data = r4[4];
                float v1278_data = ir5[4];
                ir5[4] = (v1278_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1275_data, 10))));
                float v1281_data = r4[5];
                float v1284_data = ir5[5];
                ir5[5] = (v1284_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1281_data, 10))));
                float v1287_data = r4[6];
                float v1290_data = ir5[6];
                ir5[6] = (v1290_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1287_data, 10))));
                float v1293_data = r4[7];
                float v1296_data = ir5[7];
                ir5[7] = (v1296_data + (v1250_data * (sycl::group_broadcast(item.get_sub_group(), v1293_data, 10))));
              }
              if (v18_lead < 12) {
                float v1302_data = r3[11];
                float v1303_data = r4[0];
                float v1306_data = ir5[0];
                ir5[0] = (v1306_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1303_data, 11))));
                float v1309_data = r4[1];
                float v1312_data = ir5[1];
                ir5[1] = (v1312_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1309_data, 11))));
                float v1315_data = r4[2];
                float v1318_data = ir5[2];
                ir5[2] = (v1318_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 11))));
                float v1321_data = r4[3];
                float v1324_data = ir5[3];
                ir5[3] = (v1324_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 11))));
                float v1327_data = r4[4];
                float v1330_data = ir5[4];
                ir5[4] = (v1330_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 11))));
                float v1333_data = r4[5];
                float v1336_data = ir5[5];
                ir5[5] = (v1336_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 11))));
                float v1339_data = r4[6];
                float v1342_data = ir5[6];
                ir5[6] = (v1342_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 11))));
                float v1345_data = r4[7];
                float v1348_data = ir5[7];
                ir5[7] = (v1348_data + (v1302_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 11))));
              }
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v1354_n1 = 0; v1354_n1 < 8; ++v1354_n1) {
                  float v1356_data = ir5[v1354_n1];
                  float v1358_data = r2[v1354_n1];
                  r5[v1354_n1] = (v1358_data + v1356_data);
                }
              }
              float r7[8]{};
              // r7 = load{g>r}(glb_m6);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v1366_i1 = 0; v1366_i1 < 8; ++v1366_i1) {
                  float v1374_data = glb_m6[(v18_lead + (v1366_i1 * 12))];
                  r7[v1366_i1] = v1374_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m5););
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v1381_i1 = 0; v1381_i1 < 12; ++v1381_i1) {
                  float v1389_data = glb_m7[(v18_lead + (v1381_i1 * 12))];
                  r9[v1381_i1] = v1389_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m6););
              float r8[8]{};
              // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              if (v18_lead < 12) {
                float v1397_data = r6[0];
                float v1398_data = r7[0];
                float v1401_data = ir8[0];
                ir8[0] = (v1401_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1398_data, 0))));
                float v1404_data = r7[1];
                float v1407_data = ir8[1];
                ir8[1] = (v1407_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1404_data, 0))));
                float v1410_data = r7[2];
                float v1413_data = ir8[2];
                ir8[2] = (v1413_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1410_data, 0))));
                float v1416_data = r7[3];
                float v1419_data = ir8[3];
                ir8[3] = (v1419_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1416_data, 0))));
                float v1422_data = r7[4];
                float v1425_data = ir8[4];
                ir8[4] = (v1425_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1422_data, 0))));
                float v1428_data = r7[5];
                float v1431_data = ir8[5];
                ir8[5] = (v1431_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1428_data, 0))));
                float v1434_data = r7[6];
                float v1437_data = ir8[6];
                ir8[6] = (v1437_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1434_data, 0))));
                float v1440_data = r7[7];
                float v1443_data = ir8[7];
                ir8[7] = (v1443_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v1440_data, 0))));
              }
              if (v18_lead < 12) {
                float v1449_data = r6[1];
                float v1450_data = r7[0];
                float v1453_data = ir8[0];
                ir8[0] = (v1453_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1450_data, 1))));
                float v1456_data = r7[1];
                float v1459_data = ir8[1];
                ir8[1] = (v1459_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1456_data, 1))));
                float v1462_data = r7[2];
                float v1465_data = ir8[2];
                ir8[2] = (v1465_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1462_data, 1))));
                float v1468_data = r7[3];
                float v1471_data = ir8[3];
                ir8[3] = (v1471_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1468_data, 1))));
                float v1474_data = r7[4];
                float v1477_data = ir8[4];
                ir8[4] = (v1477_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1474_data, 1))));
                float v1480_data = r7[5];
                float v1483_data = ir8[5];
                ir8[5] = (v1483_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1480_data, 1))));
                float v1486_data = r7[6];
                float v1489_data = ir8[6];
                ir8[6] = (v1489_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1486_data, 1))));
                float v1492_data = r7[7];
                float v1495_data = ir8[7];
                ir8[7] = (v1495_data + (v1449_data * (sycl::group_broadcast(item.get_sub_group(), v1492_data, 1))));
              }
              if (v18_lead < 12) {
                float v1501_data = r6[2];
                float v1502_data = r7[0];
                float v1505_data = ir8[0];
                ir8[0] = (v1505_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1502_data, 2))));
                float v1508_data = r7[1];
                float v1511_data = ir8[1];
                ir8[1] = (v1511_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1508_data, 2))));
                float v1514_data = r7[2];
                float v1517_data = ir8[2];
                ir8[2] = (v1517_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1514_data, 2))));
                float v1520_data = r7[3];
                float v1523_data = ir8[3];
                ir8[3] = (v1523_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1520_data, 2))));
                float v1526_data = r7[4];
                float v1529_data = ir8[4];
                ir8[4] = (v1529_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1526_data, 2))));
                float v1532_data = r7[5];
                float v1535_data = ir8[5];
                ir8[5] = (v1535_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1532_data, 2))));
                float v1538_data = r7[6];
                float v1541_data = ir8[6];
                ir8[6] = (v1541_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1538_data, 2))));
                float v1544_data = r7[7];
                float v1547_data = ir8[7];
                ir8[7] = (v1547_data + (v1501_data * (sycl::group_broadcast(item.get_sub_group(), v1544_data, 2))));
              }
              if (v18_lead < 12) {
                float v1553_data = r6[3];
                float v1554_data = r7[0];
                float v1557_data = ir8[0];
                ir8[0] = (v1557_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1554_data, 3))));
                float v1560_data = r7[1];
                float v1563_data = ir8[1];
                ir8[1] = (v1563_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1560_data, 3))));
                float v1566_data = r7[2];
                float v1569_data = ir8[2];
                ir8[2] = (v1569_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1566_data, 3))));
                float v1572_data = r7[3];
                float v1575_data = ir8[3];
                ir8[3] = (v1575_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1572_data, 3))));
                float v1578_data = r7[4];
                float v1581_data = ir8[4];
                ir8[4] = (v1581_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1578_data, 3))));
                float v1584_data = r7[5];
                float v1587_data = ir8[5];
                ir8[5] = (v1587_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1584_data, 3))));
                float v1590_data = r7[6];
                float v1593_data = ir8[6];
                ir8[6] = (v1593_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1590_data, 3))));
                float v1596_data = r7[7];
                float v1599_data = ir8[7];
                ir8[7] = (v1599_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1596_data, 3))));
              }
              if (v18_lead < 12) {
                float v1605_data = r6[4];
                float v1606_data = r7[0];
                float v1609_data = ir8[0];
                ir8[0] = (v1609_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1606_data, 4))));
                float v1612_data = r7[1];
                float v1615_data = ir8[1];
                ir8[1] = (v1615_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1612_data, 4))));
                float v1618_data = r7[2];
                float v1621_data = ir8[2];
                ir8[2] = (v1621_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1618_data, 4))));
                float v1624_data = r7[3];
                float v1627_data = ir8[3];
                ir8[3] = (v1627_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1624_data, 4))));
                float v1630_data = r7[4];
                float v1633_data = ir8[4];
                ir8[4] = (v1633_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1630_data, 4))));
                float v1636_data = r7[5];
                float v1639_data = ir8[5];
                ir8[5] = (v1639_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1636_data, 4))));
                float v1642_data = r7[6];
                float v1645_data = ir8[6];
                ir8[6] = (v1645_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 4))));
                float v1648_data = r7[7];
                float v1651_data = ir8[7];
                ir8[7] = (v1651_data + (v1605_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 4))));
              }
              if (v18_lead < 12) {
                float v1657_data = r6[5];
                float v1658_data = r7[0];
                float v1661_data = ir8[0];
                ir8[0] = (v1661_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1658_data, 5))));
                float v1664_data = r7[1];
                float v1667_data = ir8[1];
                ir8[1] = (v1667_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1664_data, 5))));
                float v1670_data = r7[2];
                float v1673_data = ir8[2];
                ir8[2] = (v1673_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1670_data, 5))));
                float v1676_data = r7[3];
                float v1679_data = ir8[3];
                ir8[3] = (v1679_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1676_data, 5))));
                float v1682_data = r7[4];
                float v1685_data = ir8[4];
                ir8[4] = (v1685_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1682_data, 5))));
                float v1688_data = r7[5];
                float v1691_data = ir8[5];
                ir8[5] = (v1691_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1688_data, 5))));
                float v1694_data = r7[6];
                float v1697_data = ir8[6];
                ir8[6] = (v1697_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1694_data, 5))));
                float v1700_data = r7[7];
                float v1703_data = ir8[7];
                ir8[7] = (v1703_data + (v1657_data * (sycl::group_broadcast(item.get_sub_group(), v1700_data, 5))));
              }
              if (v18_lead < 12) {
                float v1709_data = r6[6];
                float v1710_data = r7[0];
                float v1713_data = ir8[0];
                ir8[0] = (v1713_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1710_data, 6))));
                float v1716_data = r7[1];
                float v1719_data = ir8[1];
                ir8[1] = (v1719_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1716_data, 6))));
                float v1722_data = r7[2];
                float v1725_data = ir8[2];
                ir8[2] = (v1725_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1722_data, 6))));
                float v1728_data = r7[3];
                float v1731_data = ir8[3];
                ir8[3] = (v1731_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1728_data, 6))));
                float v1734_data = r7[4];
                float v1737_data = ir8[4];
                ir8[4] = (v1737_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1734_data, 6))));
                float v1740_data = r7[5];
                float v1743_data = ir8[5];
                ir8[5] = (v1743_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1740_data, 6))));
                float v1746_data = r7[6];
                float v1749_data = ir8[6];
                ir8[6] = (v1749_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 6))));
                float v1752_data = r7[7];
                float v1755_data = ir8[7];
                ir8[7] = (v1755_data + (v1709_data * (sycl::group_broadcast(item.get_sub_group(), v1752_data, 6))));
              }
              if (v18_lead < 12) {
                float v1761_data = r6[7];
                float v1762_data = r7[0];
                float v1765_data = ir8[0];
                ir8[0] = (v1765_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1762_data, 7))));
                float v1768_data = r7[1];
                float v1771_data = ir8[1];
                ir8[1] = (v1771_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1768_data, 7))));
                float v1774_data = r7[2];
                float v1777_data = ir8[2];
                ir8[2] = (v1777_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1774_data, 7))));
                float v1780_data = r7[3];
                float v1783_data = ir8[3];
                ir8[3] = (v1783_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1780_data, 7))));
                float v1786_data = r7[4];
                float v1789_data = ir8[4];
                ir8[4] = (v1789_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1786_data, 7))));
                float v1792_data = r7[5];
                float v1795_data = ir8[5];
                ir8[5] = (v1795_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1792_data, 7))));
                float v1798_data = r7[6];
                float v1801_data = ir8[6];
                ir8[6] = (v1801_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1798_data, 7))));
                float v1804_data = r7[7];
                float v1807_data = ir8[7];
                ir8[7] = (v1807_data + (v1761_data * (sycl::group_broadcast(item.get_sub_group(), v1804_data, 7))));
              }
              if (v18_lead < 12) {
                float v1813_data = r6[8];
                float v1814_data = r7[0];
                float v1817_data = ir8[0];
                ir8[0] = (v1817_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1814_data, 8))));
                float v1820_data = r7[1];
                float v1823_data = ir8[1];
                ir8[1] = (v1823_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1820_data, 8))));
                float v1826_data = r7[2];
                float v1829_data = ir8[2];
                ir8[2] = (v1829_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1826_data, 8))));
                float v1832_data = r7[3];
                float v1835_data = ir8[3];
                ir8[3] = (v1835_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1832_data, 8))));
                float v1838_data = r7[4];
                float v1841_data = ir8[4];
                ir8[4] = (v1841_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1838_data, 8))));
                float v1844_data = r7[5];
                float v1847_data = ir8[5];
                ir8[5] = (v1847_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1844_data, 8))));
                float v1850_data = r7[6];
                float v1853_data = ir8[6];
                ir8[6] = (v1853_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1850_data, 8))));
                float v1856_data = r7[7];
                float v1859_data = ir8[7];
                ir8[7] = (v1859_data + (v1813_data * (sycl::group_broadcast(item.get_sub_group(), v1856_data, 8))));
              }
              if (v18_lead < 12) {
                float v1865_data = r6[9];
                float v1866_data = r7[0];
                float v1869_data = ir8[0];
                ir8[0] = (v1869_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1866_data, 9))));
                float v1872_data = r7[1];
                float v1875_data = ir8[1];
                ir8[1] = (v1875_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1872_data, 9))));
                float v1878_data = r7[2];
                float v1881_data = ir8[2];
                ir8[2] = (v1881_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1878_data, 9))));
                float v1884_data = r7[3];
                float v1887_data = ir8[3];
                ir8[3] = (v1887_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1884_data, 9))));
                float v1890_data = r7[4];
                float v1893_data = ir8[4];
                ir8[4] = (v1893_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1890_data, 9))));
                float v1896_data = r7[5];
                float v1899_data = ir8[5];
                ir8[5] = (v1899_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1896_data, 9))));
                float v1902_data = r7[6];
                float v1905_data = ir8[6];
                ir8[6] = (v1905_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1902_data, 9))));
                float v1908_data = r7[7];
                float v1911_data = ir8[7];
                ir8[7] = (v1911_data + (v1865_data * (sycl::group_broadcast(item.get_sub_group(), v1908_data, 9))));
              }
              if (v18_lead < 12) {
                float v1917_data = r6[10];
                float v1918_data = r7[0];
                float v1921_data = ir8[0];
                ir8[0] = (v1921_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1918_data, 10))));
                float v1924_data = r7[1];
                float v1927_data = ir8[1];
                ir8[1] = (v1927_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1924_data, 10))));
                float v1930_data = r7[2];
                float v1933_data = ir8[2];
                ir8[2] = (v1933_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1930_data, 10))));
                float v1936_data = r7[3];
                float v1939_data = ir8[3];
                ir8[3] = (v1939_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1936_data, 10))));
                float v1942_data = r7[4];
                float v1945_data = ir8[4];
                ir8[4] = (v1945_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 10))));
                float v1948_data = r7[5];
                float v1951_data = ir8[5];
                ir8[5] = (v1951_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 10))));
                float v1954_data = r7[6];
                float v1957_data = ir8[6];
                ir8[6] = (v1957_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1954_data, 10))));
                float v1960_data = r7[7];
                float v1963_data = ir8[7];
                ir8[7] = (v1963_data + (v1917_data * (sycl::group_broadcast(item.get_sub_group(), v1960_data, 10))));
              }
              if (v18_lead < 12) {
                float v1969_data = r6[11];
                float v1970_data = r7[0];
                float v1973_data = ir8[0];
                ir8[0] = (v1973_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v1970_data, 11))));
                float v1976_data = r7[1];
                float v1979_data = ir8[1];
                ir8[1] = (v1979_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v1976_data, 11))));
                float v1982_data = r7[2];
                float v1985_data = ir8[2];
                ir8[2] = (v1985_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v1982_data, 11))));
                float v1988_data = r7[3];
                float v1991_data = ir8[3];
                ir8[3] = (v1991_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v1988_data, 11))));
                float v1994_data = r7[4];
                float v1997_data = ir8[4];
                ir8[4] = (v1997_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v1994_data, 11))));
                float v2000_data = r7[5];
                float v2003_data = ir8[5];
                ir8[5] = (v2003_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v2000_data, 11))));
                float v2006_data = r7[6];
                float v2009_data = ir8[6];
                ir8[6] = (v2009_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v2006_data, 11))));
                float v2012_data = r7[7];
                float v2015_data = ir8[7];
                ir8[7] = (v2015_data + (v1969_data * (sycl::group_broadcast(item.get_sub_group(), v2012_data, 11))));
              }
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v2021_n1 = 0; v2021_n1 < 8; ++v2021_n1) {
                  float v2023_data = ir8[v2021_n1];
                  float v2025_data = r5[v2021_n1];
                  r8[v2021_n1] = (v2025_data + v2023_data);
                }
              }
              float r10[8]{};
              // r10 = load{g>r}(glb_m8);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v2033_i1 = 0; v2033_i1 < 8; ++v2033_i1) {
                  float v2041_data = glb_m8[(v18_lead + (v2033_i1 * 12))];
                  r10[v2033_i1] = v2041_data;
                }
              }
              // wait(r9 = load{g>r}(glb_m7););
              // wait(r10 = load{g>r}(glb_m8););
              float r11[8]{};
              // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir11[8]{};
              if (v18_lead < 12) {
                float v2049_data = r9[0];
                float v2050_data = r10[0];
                float v2053_data = ir11[0];
                ir11[0] = (v2053_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2050_data, 0))));
                float v2056_data = r10[1];
                float v2059_data = ir11[1];
                ir11[1] = (v2059_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2056_data, 0))));
                float v2062_data = r10[2];
                float v2065_data = ir11[2];
                ir11[2] = (v2065_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2062_data, 0))));
                float v2068_data = r10[3];
                float v2071_data = ir11[3];
                ir11[3] = (v2071_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2068_data, 0))));
                float v2074_data = r10[4];
                float v2077_data = ir11[4];
                ir11[4] = (v2077_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2074_data, 0))));
                float v2080_data = r10[5];
                float v2083_data = ir11[5];
                ir11[5] = (v2083_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2080_data, 0))));
                float v2086_data = r10[6];
                float v2089_data = ir11[6];
                ir11[6] = (v2089_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2086_data, 0))));
                float v2092_data = r10[7];
                float v2095_data = ir11[7];
                ir11[7] = (v2095_data + (v2049_data * (sycl::group_broadcast(item.get_sub_group(), v2092_data, 0))));
              }
              if (v18_lead < 12) {
                float v2101_data = r9[1];
                float v2102_data = r10[0];
                float v2105_data = ir11[0];
                ir11[0] = (v2105_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2102_data, 1))));
                float v2108_data = r10[1];
                float v2111_data = ir11[1];
                ir11[1] = (v2111_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2108_data, 1))));
                float v2114_data = r10[2];
                float v2117_data = ir11[2];
                ir11[2] = (v2117_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2114_data, 1))));
                float v2120_data = r10[3];
                float v2123_data = ir11[3];
                ir11[3] = (v2123_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2120_data, 1))));
                float v2126_data = r10[4];
                float v2129_data = ir11[4];
                ir11[4] = (v2129_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2126_data, 1))));
                float v2132_data = r10[5];
                float v2135_data = ir11[5];
                ir11[5] = (v2135_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2132_data, 1))));
                float v2138_data = r10[6];
                float v2141_data = ir11[6];
                ir11[6] = (v2141_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2138_data, 1))));
                float v2144_data = r10[7];
                float v2147_data = ir11[7];
                ir11[7] = (v2147_data + (v2101_data * (sycl::group_broadcast(item.get_sub_group(), v2144_data, 1))));
              }
              if (v18_lead < 12) {
                float v2153_data = r9[2];
                float v2154_data = r10[0];
                float v2157_data = ir11[0];
                ir11[0] = (v2157_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2154_data, 2))));
                float v2160_data = r10[1];
                float v2163_data = ir11[1];
                ir11[1] = (v2163_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2160_data, 2))));
                float v2166_data = r10[2];
                float v2169_data = ir11[2];
                ir11[2] = (v2169_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2166_data, 2))));
                float v2172_data = r10[3];
                float v2175_data = ir11[3];
                ir11[3] = (v2175_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2172_data, 2))));
                float v2178_data = r10[4];
                float v2181_data = ir11[4];
                ir11[4] = (v2181_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2178_data, 2))));
                float v2184_data = r10[5];
                float v2187_data = ir11[5];
                ir11[5] = (v2187_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2184_data, 2))));
                float v2190_data = r10[6];
                float v2193_data = ir11[6];
                ir11[6] = (v2193_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2190_data, 2))));
                float v2196_data = r10[7];
                float v2199_data = ir11[7];
                ir11[7] = (v2199_data + (v2153_data * (sycl::group_broadcast(item.get_sub_group(), v2196_data, 2))));
              }
              if (v18_lead < 12) {
                float v2205_data = r9[3];
                float v2206_data = r10[0];
                float v2209_data = ir11[0];
                ir11[0] = (v2209_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2206_data, 3))));
                float v2212_data = r10[1];
                float v2215_data = ir11[1];
                ir11[1] = (v2215_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2212_data, 3))));
                float v2218_data = r10[2];
                float v2221_data = ir11[2];
                ir11[2] = (v2221_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2218_data, 3))));
                float v2224_data = r10[3];
                float v2227_data = ir11[3];
                ir11[3] = (v2227_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2224_data, 3))));
                float v2230_data = r10[4];
                float v2233_data = ir11[4];
                ir11[4] = (v2233_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2230_data, 3))));
                float v2236_data = r10[5];
                float v2239_data = ir11[5];
                ir11[5] = (v2239_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2236_data, 3))));
                float v2242_data = r10[6];
                float v2245_data = ir11[6];
                ir11[6] = (v2245_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2242_data, 3))));
                float v2248_data = r10[7];
                float v2251_data = ir11[7];
                ir11[7] = (v2251_data + (v2205_data * (sycl::group_broadcast(item.get_sub_group(), v2248_data, 3))));
              }
              if (v18_lead < 12) {
                float v2257_data = r9[4];
                float v2258_data = r10[0];
                float v2261_data = ir11[0];
                ir11[0] = (v2261_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2258_data, 4))));
                float v2264_data = r10[1];
                float v2267_data = ir11[1];
                ir11[1] = (v2267_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2264_data, 4))));
                float v2270_data = r10[2];
                float v2273_data = ir11[2];
                ir11[2] = (v2273_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2270_data, 4))));
                float v2276_data = r10[3];
                float v2279_data = ir11[3];
                ir11[3] = (v2279_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2276_data, 4))));
                float v2282_data = r10[4];
                float v2285_data = ir11[4];
                ir11[4] = (v2285_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2282_data, 4))));
                float v2288_data = r10[5];
                float v2291_data = ir11[5];
                ir11[5] = (v2291_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2288_data, 4))));
                float v2294_data = r10[6];
                float v2297_data = ir11[6];
                ir11[6] = (v2297_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2294_data, 4))));
                float v2300_data = r10[7];
                float v2303_data = ir11[7];
                ir11[7] = (v2303_data + (v2257_data * (sycl::group_broadcast(item.get_sub_group(), v2300_data, 4))));
              }
              if (v18_lead < 12) {
                float v2309_data = r9[5];
                float v2310_data = r10[0];
                float v2313_data = ir11[0];
                ir11[0] = (v2313_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2310_data, 5))));
                float v2316_data = r10[1];
                float v2319_data = ir11[1];
                ir11[1] = (v2319_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2316_data, 5))));
                float v2322_data = r10[2];
                float v2325_data = ir11[2];
                ir11[2] = (v2325_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2322_data, 5))));
                float v2328_data = r10[3];
                float v2331_data = ir11[3];
                ir11[3] = (v2331_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2328_data, 5))));
                float v2334_data = r10[4];
                float v2337_data = ir11[4];
                ir11[4] = (v2337_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2334_data, 5))));
                float v2340_data = r10[5];
                float v2343_data = ir11[5];
                ir11[5] = (v2343_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2340_data, 5))));
                float v2346_data = r10[6];
                float v2349_data = ir11[6];
                ir11[6] = (v2349_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2346_data, 5))));
                float v2352_data = r10[7];
                float v2355_data = ir11[7];
                ir11[7] = (v2355_data + (v2309_data * (sycl::group_broadcast(item.get_sub_group(), v2352_data, 5))));
              }
              if (v18_lead < 12) {
                float v2361_data = r9[6];
                float v2362_data = r10[0];
                float v2365_data = ir11[0];
                ir11[0] = (v2365_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2362_data, 6))));
                float v2368_data = r10[1];
                float v2371_data = ir11[1];
                ir11[1] = (v2371_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2368_data, 6))));
                float v2374_data = r10[2];
                float v2377_data = ir11[2];
                ir11[2] = (v2377_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2374_data, 6))));
                float v2380_data = r10[3];
                float v2383_data = ir11[3];
                ir11[3] = (v2383_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2380_data, 6))));
                float v2386_data = r10[4];
                float v2389_data = ir11[4];
                ir11[4] = (v2389_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2386_data, 6))));
                float v2392_data = r10[5];
                float v2395_data = ir11[5];
                ir11[5] = (v2395_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2392_data, 6))));
                float v2398_data = r10[6];
                float v2401_data = ir11[6];
                ir11[6] = (v2401_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2398_data, 6))));
                float v2404_data = r10[7];
                float v2407_data = ir11[7];
                ir11[7] = (v2407_data + (v2361_data * (sycl::group_broadcast(item.get_sub_group(), v2404_data, 6))));
              }
              if (v18_lead < 12) {
                float v2413_data = r9[7];
                float v2414_data = r10[0];
                float v2417_data = ir11[0];
                ir11[0] = (v2417_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2414_data, 7))));
                float v2420_data = r10[1];
                float v2423_data = ir11[1];
                ir11[1] = (v2423_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2420_data, 7))));
                float v2426_data = r10[2];
                float v2429_data = ir11[2];
                ir11[2] = (v2429_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2426_data, 7))));
                float v2432_data = r10[3];
                float v2435_data = ir11[3];
                ir11[3] = (v2435_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2432_data, 7))));
                float v2438_data = r10[4];
                float v2441_data = ir11[4];
                ir11[4] = (v2441_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2438_data, 7))));
                float v2444_data = r10[5];
                float v2447_data = ir11[5];
                ir11[5] = (v2447_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2444_data, 7))));
                float v2450_data = r10[6];
                float v2453_data = ir11[6];
                ir11[6] = (v2453_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2450_data, 7))));
                float v2456_data = r10[7];
                float v2459_data = ir11[7];
                ir11[7] = (v2459_data + (v2413_data * (sycl::group_broadcast(item.get_sub_group(), v2456_data, 7))));
              }
              if (v18_lead < 12) {
                float v2465_data = r9[8];
                float v2466_data = r10[0];
                float v2469_data = ir11[0];
                ir11[0] = (v2469_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2466_data, 8))));
                float v2472_data = r10[1];
                float v2475_data = ir11[1];
                ir11[1] = (v2475_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2472_data, 8))));
                float v2478_data = r10[2];
                float v2481_data = ir11[2];
                ir11[2] = (v2481_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2478_data, 8))));
                float v2484_data = r10[3];
                float v2487_data = ir11[3];
                ir11[3] = (v2487_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2484_data, 8))));
                float v2490_data = r10[4];
                float v2493_data = ir11[4];
                ir11[4] = (v2493_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2490_data, 8))));
                float v2496_data = r10[5];
                float v2499_data = ir11[5];
                ir11[5] = (v2499_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2496_data, 8))));
                float v2502_data = r10[6];
                float v2505_data = ir11[6];
                ir11[6] = (v2505_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2502_data, 8))));
                float v2508_data = r10[7];
                float v2511_data = ir11[7];
                ir11[7] = (v2511_data + (v2465_data * (sycl::group_broadcast(item.get_sub_group(), v2508_data, 8))));
              }
              if (v18_lead < 12) {
                float v2517_data = r9[9];
                float v2518_data = r10[0];
                float v2521_data = ir11[0];
                ir11[0] = (v2521_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2518_data, 9))));
                float v2524_data = r10[1];
                float v2527_data = ir11[1];
                ir11[1] = (v2527_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2524_data, 9))));
                float v2530_data = r10[2];
                float v2533_data = ir11[2];
                ir11[2] = (v2533_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2530_data, 9))));
                float v2536_data = r10[3];
                float v2539_data = ir11[3];
                ir11[3] = (v2539_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2536_data, 9))));
                float v2542_data = r10[4];
                float v2545_data = ir11[4];
                ir11[4] = (v2545_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2542_data, 9))));
                float v2548_data = r10[5];
                float v2551_data = ir11[5];
                ir11[5] = (v2551_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2548_data, 9))));
                float v2554_data = r10[6];
                float v2557_data = ir11[6];
                ir11[6] = (v2557_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2554_data, 9))));
                float v2560_data = r10[7];
                float v2563_data = ir11[7];
                ir11[7] = (v2563_data + (v2517_data * (sycl::group_broadcast(item.get_sub_group(), v2560_data, 9))));
              }
              if (v18_lead < 12) {
                float v2569_data = r9[10];
                float v2570_data = r10[0];
                float v2573_data = ir11[0];
                ir11[0] = (v2573_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2570_data, 10))));
                float v2576_data = r10[1];
                float v2579_data = ir11[1];
                ir11[1] = (v2579_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2576_data, 10))));
                float v2582_data = r10[2];
                float v2585_data = ir11[2];
                ir11[2] = (v2585_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2582_data, 10))));
                float v2588_data = r10[3];
                float v2591_data = ir11[3];
                ir11[3] = (v2591_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2588_data, 10))));
                float v2594_data = r10[4];
                float v2597_data = ir11[4];
                ir11[4] = (v2597_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2594_data, 10))));
                float v2600_data = r10[5];
                float v2603_data = ir11[5];
                ir11[5] = (v2603_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2600_data, 10))));
                float v2606_data = r10[6];
                float v2609_data = ir11[6];
                ir11[6] = (v2609_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2606_data, 10))));
                float v2612_data = r10[7];
                float v2615_data = ir11[7];
                ir11[7] = (v2615_data + (v2569_data * (sycl::group_broadcast(item.get_sub_group(), v2612_data, 10))));
              }
              if (v18_lead < 12) {
                float v2621_data = r9[11];
                float v2622_data = r10[0];
                float v2625_data = ir11[0];
                ir11[0] = (v2625_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2622_data, 11))));
                float v2628_data = r10[1];
                float v2631_data = ir11[1];
                ir11[1] = (v2631_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2628_data, 11))));
                float v2634_data = r10[2];
                float v2637_data = ir11[2];
                ir11[2] = (v2637_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2634_data, 11))));
                float v2640_data = r10[3];
                float v2643_data = ir11[3];
                ir11[3] = (v2643_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2640_data, 11))));
                float v2646_data = r10[4];
                float v2649_data = ir11[4];
                ir11[4] = (v2649_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2646_data, 11))));
                float v2652_data = r10[5];
                float v2655_data = ir11[5];
                ir11[5] = (v2655_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2652_data, 11))));
                float v2658_data = r10[6];
                float v2661_data = ir11[6];
                ir11[6] = (v2661_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2658_data, 11))));
                float v2664_data = r10[7];
                float v2667_data = ir11[7];
                ir11[7] = (v2667_data + (v2621_data * (sycl::group_broadcast(item.get_sub_group(), v2664_data, 11))));
              }
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v2673_n1 = 0; v2673_n1 < 8; ++v2673_n1) {
                  float v2675_data = ir11[v2673_n1];
                  float v2677_data = r8[v2673_n1];
                  r11[v2673_n1] = (v2677_data + v2675_data);
                }
              }
              // glb_m0 = store{r>g}(r11);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v2684_i1 = 0; v2684_i1 < 8; ++v2684_i1) {
                  float v2686_data = r11[v2684_i1];
                  glb_m0[(v18_lead + (v2684_i1 * 12))] = v2686_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

