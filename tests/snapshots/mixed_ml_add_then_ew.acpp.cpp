// === base name ===
kernel_dfdcb467baf2f1ee

// === header ===
void launcher_kernel_dfdcb467baf2f1ee(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_dfdcb467baf2f1ee(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_dfdcb467baf2f1ee(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_dfdcb467baf2f1ee(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m4 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v3_batchId0 * 64 + 0 + m4_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v19_lead = item.get_local_id(0) % 16;
              if (v19_lead < 8) {
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
                  float v29_data = glb_m0[(v19_lead + (v21_i1 * 8))];
                  r0[v21_i1] = v29_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v19_lead < 8) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
                  float v44_data = glb_m1[(v19_lead + (v36_i1 * 8))];
                  r1[v36_i1] = v44_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = load{g>r}(glb_m2);
              if (v19_lead < 8) {
                #pragma unroll
                for (int32_t v51_i1 = 0; v51_i1 < 8; ++v51_i1) {
                  float v59_data = glb_m2[(v19_lead + (v51_i1 * 8))];
                  r3[v51_i1] = v59_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v19_lead < 8) {
                float v66_data = r0[0];
                float v67_data = r1[0];
                float v70_data = r2[0];
                r2[0] = (v70_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 0))));
                float v73_data = r1[1];
                float v76_data = r2[1];
                r2[1] = (v76_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 0))));
                float v79_data = r1[2];
                float v82_data = r2[2];
                r2[2] = (v82_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 0))));
                float v85_data = r1[3];
                float v88_data = r2[3];
                r2[3] = (v88_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 0))));
                float v91_data = r1[4];
                float v94_data = r2[4];
                r2[4] = (v94_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 0))));
                float v97_data = r1[5];
                float v100_data = r2[5];
                r2[5] = (v100_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 0))));
                float v103_data = r1[6];
                float v106_data = r2[6];
                r2[6] = (v106_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 0))));
                float v109_data = r1[7];
                float v112_data = r2[7];
                r2[7] = (v112_data + (v66_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 0))));
              }
              if (v19_lead < 8) {
                float v118_data = r0[1];
                float v119_data = r1[0];
                float v122_data = r2[0];
                r2[0] = (v122_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
                float v125_data = r1[1];
                float v128_data = r2[1];
                r2[1] = (v128_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
                float v131_data = r1[2];
                float v134_data = r2[2];
                r2[2] = (v134_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
                float v137_data = r1[3];
                float v140_data = r2[3];
                r2[3] = (v140_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 1))));
                float v143_data = r1[4];
                float v146_data = r2[4];
                r2[4] = (v146_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 1))));
                float v149_data = r1[5];
                float v152_data = r2[5];
                r2[5] = (v152_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 1))));
                float v155_data = r1[6];
                float v158_data = r2[6];
                r2[6] = (v158_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 1))));
                float v161_data = r1[7];
                float v164_data = r2[7];
                r2[7] = (v164_data + (v118_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 1))));
              }
              if (v19_lead < 8) {
                float v170_data = r0[2];
                float v171_data = r1[0];
                float v174_data = r2[0];
                r2[0] = (v174_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v171_data, 2))));
                float v177_data = r1[1];
                float v180_data = r2[1];
                r2[1] = (v180_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 2))));
                float v183_data = r1[2];
                float v186_data = r2[2];
                r2[2] = (v186_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 2))));
                float v189_data = r1[3];
                float v192_data = r2[3];
                r2[3] = (v192_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 2))));
                float v195_data = r1[4];
                float v198_data = r2[4];
                r2[4] = (v198_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 2))));
                float v201_data = r1[5];
                float v204_data = r2[5];
                r2[5] = (v204_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 2))));
                float v207_data = r1[6];
                float v210_data = r2[6];
                r2[6] = (v210_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 2))));
                float v213_data = r1[7];
                float v216_data = r2[7];
                r2[7] = (v216_data + (v170_data * (sycl::group_broadcast(item.get_sub_group(), v213_data, 2))));
              }
              if (v19_lead < 8) {
                float v222_data = r0[3];
                float v223_data = r1[0];
                float v226_data = r2[0];
                r2[0] = (v226_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 3))));
                float v229_data = r1[1];
                float v232_data = r2[1];
                r2[1] = (v232_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 3))));
                float v235_data = r1[2];
                float v238_data = r2[2];
                r2[2] = (v238_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 3))));
                float v241_data = r1[3];
                float v244_data = r2[3];
                r2[3] = (v244_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 3))));
                float v247_data = r1[4];
                float v250_data = r2[4];
                r2[4] = (v250_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v247_data, 3))));
                float v253_data = r1[5];
                float v256_data = r2[5];
                r2[5] = (v256_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v253_data, 3))));
                float v259_data = r1[6];
                float v262_data = r2[6];
                r2[6] = (v262_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v259_data, 3))));
                float v265_data = r1[7];
                float v268_data = r2[7];
                r2[7] = (v268_data + (v222_data * (sycl::group_broadcast(item.get_sub_group(), v265_data, 3))));
              }
              if (v19_lead < 8) {
                float v274_data = r0[4];
                float v275_data = r1[0];
                float v278_data = r2[0];
                r2[0] = (v278_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 4))));
                float v281_data = r1[1];
                float v284_data = r2[1];
                r2[1] = (v284_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 4))));
                float v287_data = r1[2];
                float v290_data = r2[2];
                r2[2] = (v290_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v287_data, 4))));
                float v293_data = r1[3];
                float v296_data = r2[3];
                r2[3] = (v296_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v293_data, 4))));
                float v299_data = r1[4];
                float v302_data = r2[4];
                r2[4] = (v302_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v299_data, 4))));
                float v305_data = r1[5];
                float v308_data = r2[5];
                r2[5] = (v308_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 4))));
                float v311_data = r1[6];
                float v314_data = r2[6];
                r2[6] = (v314_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 4))));
                float v317_data = r1[7];
                float v320_data = r2[7];
                r2[7] = (v320_data + (v274_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 4))));
              }
              if (v19_lead < 8) {
                float v326_data = r0[5];
                float v327_data = r1[0];
                float v330_data = r2[0];
                r2[0] = (v330_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 5))));
                float v333_data = r1[1];
                float v336_data = r2[1];
                r2[1] = (v336_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 5))));
                float v339_data = r1[2];
                float v342_data = r2[2];
                r2[2] = (v342_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 5))));
                float v345_data = r1[3];
                float v348_data = r2[3];
                r2[3] = (v348_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 5))));
                float v351_data = r1[4];
                float v354_data = r2[4];
                r2[4] = (v354_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 5))));
                float v357_data = r1[5];
                float v360_data = r2[5];
                r2[5] = (v360_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 5))));
                float v363_data = r1[6];
                float v366_data = r2[6];
                r2[6] = (v366_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 5))));
                float v369_data = r1[7];
                float v372_data = r2[7];
                r2[7] = (v372_data + (v326_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 5))));
              }
              if (v19_lead < 8) {
                float v378_data = r0[6];
                float v379_data = r1[0];
                float v382_data = r2[0];
                r2[0] = (v382_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 6))));
                float v385_data = r1[1];
                float v388_data = r2[1];
                r2[1] = (v388_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 6))));
                float v391_data = r1[2];
                float v394_data = r2[2];
                r2[2] = (v394_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 6))));
                float v397_data = r1[3];
                float v400_data = r2[3];
                r2[3] = (v400_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 6))));
                float v403_data = r1[4];
                float v406_data = r2[4];
                r2[4] = (v406_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 6))));
                float v409_data = r1[5];
                float v412_data = r2[5];
                r2[5] = (v412_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v409_data, 6))));
                float v415_data = r1[6];
                float v418_data = r2[6];
                r2[6] = (v418_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v415_data, 6))));
                float v421_data = r1[7];
                float v424_data = r2[7];
                r2[7] = (v424_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v421_data, 6))));
              }
              if (v19_lead < 8) {
                float v430_data = r0[7];
                float v431_data = r1[0];
                float v434_data = r2[0];
                r2[0] = (v434_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 7))));
                float v437_data = r1[1];
                float v440_data = r2[1];
                r2[1] = (v440_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 7))));
                float v443_data = r1[2];
                float v446_data = r2[2];
                r2[2] = (v446_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 7))));
                float v449_data = r1[3];
                float v452_data = r2[3];
                r2[3] = (v452_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 7))));
                float v455_data = r1[4];
                float v458_data = r2[4];
                r2[4] = (v458_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v455_data, 7))));
                float v461_data = r1[5];
                float v464_data = r2[5];
                r2[5] = (v464_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v461_data, 7))));
                float v467_data = r1[6];
                float v470_data = r2[6];
                r2[6] = (v470_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v467_data, 7))));
                float v473_data = r1[7];
                float v476_data = r2[7];
                r2[7] = (v476_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v473_data, 7))));
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m3);
              if (v19_lead < 8) {
                #pragma unroll
                for (int32_t v483_i1 = 0; v483_i1 < 8; ++v483_i1) {
                  float v491_data = glb_m3[(v19_lead + (v483_i1 * 8))];
                  r4[v483_i1] = v491_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              // wait(r4 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir5[8]{};
              if (v19_lead < 8) {
                float v499_data = r3[0];
                float v500_data = r4[0];
                float v503_data = ir5[0];
                ir5[0] = (v503_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v500_data, 0))));
                float v506_data = r4[1];
                float v509_data = ir5[1];
                ir5[1] = (v509_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v506_data, 0))));
                float v512_data = r4[2];
                float v515_data = ir5[2];
                ir5[2] = (v515_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v512_data, 0))));
                float v518_data = r4[3];
                float v521_data = ir5[3];
                ir5[3] = (v521_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v518_data, 0))));
                float v524_data = r4[4];
                float v527_data = ir5[4];
                ir5[4] = (v527_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v524_data, 0))));
                float v530_data = r4[5];
                float v533_data = ir5[5];
                ir5[5] = (v533_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v530_data, 0))));
                float v536_data = r4[6];
                float v539_data = ir5[6];
                ir5[6] = (v539_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v536_data, 0))));
                float v542_data = r4[7];
                float v545_data = ir5[7];
                ir5[7] = (v545_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 0))));
              }
              if (v19_lead < 8) {
                float v551_data = r3[1];
                float v552_data = r4[0];
                float v555_data = ir5[0];
                ir5[0] = (v555_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v552_data, 1))));
                float v558_data = r4[1];
                float v561_data = ir5[1];
                ir5[1] = (v561_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v558_data, 1))));
                float v564_data = r4[2];
                float v567_data = ir5[2];
                ir5[2] = (v567_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v564_data, 1))));
                float v570_data = r4[3];
                float v573_data = ir5[3];
                ir5[3] = (v573_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v570_data, 1))));
                float v576_data = r4[4];
                float v579_data = ir5[4];
                ir5[4] = (v579_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v576_data, 1))));
                float v582_data = r4[5];
                float v585_data = ir5[5];
                ir5[5] = (v585_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v582_data, 1))));
                float v588_data = r4[6];
                float v591_data = ir5[6];
                ir5[6] = (v591_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v588_data, 1))));
                float v594_data = r4[7];
                float v597_data = ir5[7];
                ir5[7] = (v597_data + (v551_data * (sycl::group_broadcast(item.get_sub_group(), v594_data, 1))));
              }
              if (v19_lead < 8) {
                float v603_data = r3[2];
                float v604_data = r4[0];
                float v607_data = ir5[0];
                ir5[0] = (v607_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v604_data, 2))));
                float v610_data = r4[1];
                float v613_data = ir5[1];
                ir5[1] = (v613_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v610_data, 2))));
                float v616_data = r4[2];
                float v619_data = ir5[2];
                ir5[2] = (v619_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v616_data, 2))));
                float v622_data = r4[3];
                float v625_data = ir5[3];
                ir5[3] = (v625_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v622_data, 2))));
                float v628_data = r4[4];
                float v631_data = ir5[4];
                ir5[4] = (v631_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v628_data, 2))));
                float v634_data = r4[5];
                float v637_data = ir5[5];
                ir5[5] = (v637_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v634_data, 2))));
                float v640_data = r4[6];
                float v643_data = ir5[6];
                ir5[6] = (v643_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v640_data, 2))));
                float v646_data = r4[7];
                float v649_data = ir5[7];
                ir5[7] = (v649_data + (v603_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 2))));
              }
              if (v19_lead < 8) {
                float v655_data = r3[3];
                float v656_data = r4[0];
                float v659_data = ir5[0];
                ir5[0] = (v659_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v656_data, 3))));
                float v662_data = r4[1];
                float v665_data = ir5[1];
                ir5[1] = (v665_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v662_data, 3))));
                float v668_data = r4[2];
                float v671_data = ir5[2];
                ir5[2] = (v671_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v668_data, 3))));
                float v674_data = r4[3];
                float v677_data = ir5[3];
                ir5[3] = (v677_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v674_data, 3))));
                float v680_data = r4[4];
                float v683_data = ir5[4];
                ir5[4] = (v683_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v680_data, 3))));
                float v686_data = r4[5];
                float v689_data = ir5[5];
                ir5[5] = (v689_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v686_data, 3))));
                float v692_data = r4[6];
                float v695_data = ir5[6];
                ir5[6] = (v695_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v692_data, 3))));
                float v698_data = r4[7];
                float v701_data = ir5[7];
                ir5[7] = (v701_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v698_data, 3))));
              }
              if (v19_lead < 8) {
                float v707_data = r3[4];
                float v708_data = r4[0];
                float v711_data = ir5[0];
                ir5[0] = (v711_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v708_data, 4))));
                float v714_data = r4[1];
                float v717_data = ir5[1];
                ir5[1] = (v717_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v714_data, 4))));
                float v720_data = r4[2];
                float v723_data = ir5[2];
                ir5[2] = (v723_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v720_data, 4))));
                float v726_data = r4[3];
                float v729_data = ir5[3];
                ir5[3] = (v729_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v726_data, 4))));
                float v732_data = r4[4];
                float v735_data = ir5[4];
                ir5[4] = (v735_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v732_data, 4))));
                float v738_data = r4[5];
                float v741_data = ir5[5];
                ir5[5] = (v741_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v738_data, 4))));
                float v744_data = r4[6];
                float v747_data = ir5[6];
                ir5[6] = (v747_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v744_data, 4))));
                float v750_data = r4[7];
                float v753_data = ir5[7];
                ir5[7] = (v753_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v750_data, 4))));
              }
              if (v19_lead < 8) {
                float v759_data = r3[5];
                float v760_data = r4[0];
                float v763_data = ir5[0];
                ir5[0] = (v763_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v760_data, 5))));
                float v766_data = r4[1];
                float v769_data = ir5[1];
                ir5[1] = (v769_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v766_data, 5))));
                float v772_data = r4[2];
                float v775_data = ir5[2];
                ir5[2] = (v775_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v772_data, 5))));
                float v778_data = r4[3];
                float v781_data = ir5[3];
                ir5[3] = (v781_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v778_data, 5))));
                float v784_data = r4[4];
                float v787_data = ir5[4];
                ir5[4] = (v787_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v784_data, 5))));
                float v790_data = r4[5];
                float v793_data = ir5[5];
                ir5[5] = (v793_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v790_data, 5))));
                float v796_data = r4[6];
                float v799_data = ir5[6];
                ir5[6] = (v799_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v796_data, 5))));
                float v802_data = r4[7];
                float v805_data = ir5[7];
                ir5[7] = (v805_data + (v759_data * (sycl::group_broadcast(item.get_sub_group(), v802_data, 5))));
              }
              if (v19_lead < 8) {
                float v811_data = r3[6];
                float v812_data = r4[0];
                float v815_data = ir5[0];
                ir5[0] = (v815_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v812_data, 6))));
                float v818_data = r4[1];
                float v821_data = ir5[1];
                ir5[1] = (v821_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v818_data, 6))));
                float v824_data = r4[2];
                float v827_data = ir5[2];
                ir5[2] = (v827_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v824_data, 6))));
                float v830_data = r4[3];
                float v833_data = ir5[3];
                ir5[3] = (v833_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v830_data, 6))));
                float v836_data = r4[4];
                float v839_data = ir5[4];
                ir5[4] = (v839_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v836_data, 6))));
                float v842_data = r4[5];
                float v845_data = ir5[5];
                ir5[5] = (v845_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 6))));
                float v848_data = r4[6];
                float v851_data = ir5[6];
                ir5[6] = (v851_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 6))));
                float v854_data = r4[7];
                float v857_data = ir5[7];
                ir5[7] = (v857_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v854_data, 6))));
              }
              if (v19_lead < 8) {
                float v863_data = r3[7];
                float v864_data = r4[0];
                float v867_data = ir5[0];
                ir5[0] = (v867_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v864_data, 7))));
                float v870_data = r4[1];
                float v873_data = ir5[1];
                ir5[1] = (v873_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v870_data, 7))));
                float v876_data = r4[2];
                float v879_data = ir5[2];
                ir5[2] = (v879_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v876_data, 7))));
                float v882_data = r4[3];
                float v885_data = ir5[3];
                ir5[3] = (v885_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v882_data, 7))));
                float v888_data = r4[4];
                float v891_data = ir5[4];
                ir5[4] = (v891_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v888_data, 7))));
                float v894_data = r4[5];
                float v897_data = ir5[5];
                ir5[5] = (v897_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v894_data, 7))));
                float v900_data = r4[6];
                float v903_data = ir5[6];
                ir5[6] = (v903_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v900_data, 7))));
                float v906_data = r4[7];
                float v909_data = ir5[7];
                ir5[7] = (v909_data + (v863_data * (sycl::group_broadcast(item.get_sub_group(), v906_data, 7))));
              }
              if (v19_lead < 8) {
                #pragma unroll
                for (int32_t v915_n1 = 0; v915_n1 < 8; ++v915_n1) {
                  float v917_data = ir5[v915_n1];
                  float v919_data = r2[v915_n1];
                  r5[v915_n1] = (v919_data + v917_data);
                }
              }
              // s0 = store{r>s}(localShrMem0, r5);
              if (v19_lead < 8) {
                #pragma unroll
                for (int32_t v926_i1 = 0; v926_i1 < 8; ++v926_i1) {
                  float v928_data = r5[v926_i1];
                  int32_t v935_a = v19_lead + (v926_i1 * 8);
                  s0[(v935_a ^ ((v935_a >> 5) & 31))] = v928_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m4 = abs(s0)
              if (v19_lead < 8) {
                #pragma unroll
                for (int32_t v943_k1 = 0; v943_k1 < 8; ++v943_k1) {
                  int32_t v949_a = v943_k1 * 8;
                  int32_t v950_a = v19_lead + v949_a;
                  float v954_data = s0[(v950_a ^ ((v950_a >> 5) & 31))];
                  glb_m4[(v19_lead + v949_a)] = (sycl::fabs(v954_data));
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

