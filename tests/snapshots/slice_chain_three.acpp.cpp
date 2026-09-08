// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08703cce1d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08703cce1d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
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
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v9_lead = item.get_local_id(0) % 16;
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v11_i1 = 0; v11_i1 < 6; ++v11_i1) {
                  float v19_data = glb_m0[(v9_lead + (v11_i1 * 12))];
                  r0[v11_i1] = v19_data;
                }
              }
              float r1[6]{};
              // r1 = load{g>r}(glb_m1);
              if (v9_lead < 6) {
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 6; ++v26_i1) {
                  float v34_data = glb_m1[(v9_lead + (v26_i1 * 6))];
                  r1[v26_i1] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v41_i1 = 0; v41_i1 < 12; ++v41_i1) {
                  float v49_data = glb_m3[(v9_lead + (v41_i1 * 12))];
                  r3[v41_i1] = v49_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              if (v9_lead < 12) {
                float v56_data = r0[0];
                float v57_data = r1[0];
                float v60_data = r2[0];
                r2[0] = (v60_data + (v56_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 0))));
                float v63_data = r1[1];
                float v66_data = r2[1];
                r2[1] = (v66_data + (v56_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
                float v69_data = r1[2];
                float v72_data = r2[2];
                r2[2] = (v72_data + (v56_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
                float v75_data = r1[3];
                float v78_data = r2[3];
                r2[3] = (v78_data + (v56_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
                float v81_data = r1[4];
                float v84_data = r2[4];
                r2[4] = (v84_data + (v56_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
                float v87_data = r1[5];
                float v90_data = r2[5];
                r2[5] = (v90_data + (v56_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
              }
              if (v9_lead < 12) {
                float v96_data = r0[1];
                float v97_data = r1[0];
                float v100_data = r2[0];
                r2[0] = (v100_data + (v96_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 1))));
                float v103_data = r1[1];
                float v106_data = r2[1];
                r2[1] = (v106_data + (v96_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 1))));
                float v109_data = r1[2];
                float v112_data = r2[2];
                r2[2] = (v112_data + (v96_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 1))));
                float v115_data = r1[3];
                float v118_data = r2[3];
                r2[3] = (v118_data + (v96_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 1))));
                float v121_data = r1[4];
                float v124_data = r2[4];
                r2[4] = (v124_data + (v96_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 1))));
                float v127_data = r1[5];
                float v130_data = r2[5];
                r2[5] = (v130_data + (v96_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
              }
              if (v9_lead < 12) {
                float v136_data = r0[2];
                float v137_data = r1[0];
                float v140_data = r2[0];
                r2[0] = (v140_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 2))));
                float v143_data = r1[1];
                float v146_data = r2[1];
                r2[1] = (v146_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 2))));
                float v149_data = r1[2];
                float v152_data = r2[2];
                r2[2] = (v152_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 2))));
                float v155_data = r1[3];
                float v158_data = r2[3];
                r2[3] = (v158_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 2))));
                float v161_data = r1[4];
                float v164_data = r2[4];
                r2[4] = (v164_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 2))));
                float v167_data = r1[5];
                float v170_data = r2[5];
                r2[5] = (v170_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
              }
              if (v9_lead < 12) {
                float v176_data = r0[3];
                float v177_data = r1[0];
                float v180_data = r2[0];
                r2[0] = (v180_data + (v176_data * (sycl::group_broadcast(item.get_sub_group(), v177_data, 3))));
                float v183_data = r1[1];
                float v186_data = r2[1];
                r2[1] = (v186_data + (v176_data * (sycl::group_broadcast(item.get_sub_group(), v183_data, 3))));
                float v189_data = r1[2];
                float v192_data = r2[2];
                r2[2] = (v192_data + (v176_data * (sycl::group_broadcast(item.get_sub_group(), v189_data, 3))));
                float v195_data = r1[3];
                float v198_data = r2[3];
                r2[3] = (v198_data + (v176_data * (sycl::group_broadcast(item.get_sub_group(), v195_data, 3))));
                float v201_data = r1[4];
                float v204_data = r2[4];
                r2[4] = (v204_data + (v176_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 3))));
                float v207_data = r1[5];
                float v210_data = r2[5];
                r2[5] = (v210_data + (v176_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 3))));
              }
              if (v9_lead < 12) {
                float v216_data = r0[4];
                float v217_data = r1[0];
                float v220_data = r2[0];
                r2[0] = (v220_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v217_data, 4))));
                float v223_data = r1[1];
                float v226_data = r2[1];
                r2[1] = (v226_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v223_data, 4))));
                float v229_data = r1[2];
                float v232_data = r2[2];
                r2[2] = (v232_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 4))));
                float v235_data = r1[3];
                float v238_data = r2[3];
                r2[3] = (v238_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 4))));
                float v241_data = r1[4];
                float v244_data = r2[4];
                r2[4] = (v244_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 4))));
                float v247_data = r1[5];
                float v250_data = r2[5];
                r2[5] = (v250_data + (v216_data * (sycl::group_broadcast(item.get_sub_group(), v247_data, 4))));
              }
              if (v9_lead < 12) {
                float v256_data = r0[5];
                float v257_data = r1[0];
                float v260_data = r2[0];
                r2[0] = (v260_data + (v256_data * (sycl::group_broadcast(item.get_sub_group(), v257_data, 5))));
                float v263_data = r1[1];
                float v266_data = r2[1];
                r2[1] = (v266_data + (v256_data * (sycl::group_broadcast(item.get_sub_group(), v263_data, 5))));
                float v269_data = r1[2];
                float v272_data = r2[2];
                r2[2] = (v272_data + (v256_data * (sycl::group_broadcast(item.get_sub_group(), v269_data, 5))));
                float v275_data = r1[3];
                float v278_data = r2[3];
                r2[3] = (v278_data + (v256_data * (sycl::group_broadcast(item.get_sub_group(), v275_data, 5))));
                float v281_data = r1[4];
                float v284_data = r2[4];
                r2[4] = (v284_data + (v256_data * (sycl::group_broadcast(item.get_sub_group(), v281_data, 5))));
                float v287_data = r1[5];
                float v290_data = r2[5];
                r2[5] = (v290_data + (v256_data * (sycl::group_broadcast(item.get_sub_group(), v287_data, 5))));
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r4[6]{};
              // r4 = +(r3 * r2) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir4[6]{};
              if (v9_lead < 12) {
                float v298_data = r3[0];
                float v299_data = r2[0];
                float v302_data = ir4[0];
                ir4[0] = (v302_data + (v298_data * (sycl::group_broadcast(item.get_sub_group(), v299_data, 0))));
                float v305_data = r2[1];
                float v308_data = ir4[1];
                ir4[1] = (v308_data + (v298_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 0))));
                float v311_data = r2[2];
                float v314_data = ir4[2];
                ir4[2] = (v314_data + (v298_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 0))));
                float v317_data = r2[3];
                float v320_data = ir4[3];
                ir4[3] = (v320_data + (v298_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 0))));
                float v323_data = r2[4];
                float v326_data = ir4[4];
                ir4[4] = (v326_data + (v298_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 0))));
                float v329_data = r2[5];
                float v332_data = ir4[5];
                ir4[5] = (v332_data + (v298_data * (sycl::group_broadcast(item.get_sub_group(), v329_data, 0))));
              }
              if (v9_lead < 12) {
                float v338_data = r3[1];
                float v339_data = r2[0];
                float v342_data = ir4[0];
                ir4[0] = (v342_data + (v338_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 1))));
                float v345_data = r2[1];
                float v348_data = ir4[1];
                ir4[1] = (v348_data + (v338_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 1))));
                float v351_data = r2[2];
                float v354_data = ir4[2];
                ir4[2] = (v354_data + (v338_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 1))));
                float v357_data = r2[3];
                float v360_data = ir4[3];
                ir4[3] = (v360_data + (v338_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 1))));
                float v363_data = r2[4];
                float v366_data = ir4[4];
                ir4[4] = (v366_data + (v338_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 1))));
                float v369_data = r2[5];
                float v372_data = ir4[5];
                ir4[5] = (v372_data + (v338_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 1))));
              }
              if (v9_lead < 12) {
                float v378_data = r3[2];
                float v379_data = r2[0];
                float v382_data = ir4[0];
                ir4[0] = (v382_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 2))));
                float v385_data = r2[1];
                float v388_data = ir4[1];
                ir4[1] = (v388_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 2))));
                float v391_data = r2[2];
                float v394_data = ir4[2];
                ir4[2] = (v394_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 2))));
                float v397_data = r2[3];
                float v400_data = ir4[3];
                ir4[3] = (v400_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 2))));
                float v403_data = r2[4];
                float v406_data = ir4[4];
                ir4[4] = (v406_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 2))));
                float v409_data = r2[5];
                float v412_data = ir4[5];
                ir4[5] = (v412_data + (v378_data * (sycl::group_broadcast(item.get_sub_group(), v409_data, 2))));
              }
              if (v9_lead < 12) {
                float v418_data = r3[3];
                float v419_data = r2[0];
                float v422_data = ir4[0];
                ir4[0] = (v422_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 3))));
                float v425_data = r2[1];
                float v428_data = ir4[1];
                ir4[1] = (v428_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 3))));
                float v431_data = r2[2];
                float v434_data = ir4[2];
                ir4[2] = (v434_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 3))));
                float v437_data = r2[3];
                float v440_data = ir4[3];
                ir4[3] = (v440_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v437_data, 3))));
                float v443_data = r2[4];
                float v446_data = ir4[4];
                ir4[4] = (v446_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v443_data, 3))));
                float v449_data = r2[5];
                float v452_data = ir4[5];
                ir4[5] = (v452_data + (v418_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 3))));
              }
              if (v9_lead < 12) {
                float v458_data = r3[4];
                float v459_data = r2[0];
                float v462_data = ir4[0];
                ir4[0] = (v462_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 4))));
                float v465_data = r2[1];
                float v468_data = ir4[1];
                ir4[1] = (v468_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v465_data, 4))));
                float v471_data = r2[2];
                float v474_data = ir4[2];
                ir4[2] = (v474_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 4))));
                float v477_data = r2[3];
                float v480_data = ir4[3];
                ir4[3] = (v480_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 4))));
                float v483_data = r2[4];
                float v486_data = ir4[4];
                ir4[4] = (v486_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 4))));
                float v489_data = r2[5];
                float v492_data = ir4[5];
                ir4[5] = (v492_data + (v458_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 4))));
              }
              if (v9_lead < 12) {
                float v498_data = r3[5];
                float v499_data = r2[0];
                float v502_data = ir4[0];
                ir4[0] = (v502_data + (v498_data * (sycl::group_broadcast(item.get_sub_group(), v499_data, 5))));
                float v505_data = r2[1];
                float v508_data = ir4[1];
                ir4[1] = (v508_data + (v498_data * (sycl::group_broadcast(item.get_sub_group(), v505_data, 5))));
                float v511_data = r2[2];
                float v514_data = ir4[2];
                ir4[2] = (v514_data + (v498_data * (sycl::group_broadcast(item.get_sub_group(), v511_data, 5))));
                float v517_data = r2[3];
                float v520_data = ir4[3];
                ir4[3] = (v520_data + (v498_data * (sycl::group_broadcast(item.get_sub_group(), v517_data, 5))));
                float v523_data = r2[4];
                float v526_data = ir4[4];
                ir4[4] = (v526_data + (v498_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 5))));
                float v529_data = r2[5];
                float v532_data = ir4[5];
                ir4[5] = (v532_data + (v498_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 5))));
              }
              if (v9_lead < 12) {
                float v538_data = r3[6];
                float v539_data = r2[0];
                float v542_data = ir4[0];
                ir4[0] = (v542_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v539_data, 6))));
                float v545_data = r2[1];
                float v548_data = ir4[1];
                ir4[1] = (v548_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v545_data, 6))));
                float v551_data = r2[2];
                float v554_data = ir4[2];
                ir4[2] = (v554_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v551_data, 6))));
                float v557_data = r2[3];
                float v560_data = ir4[3];
                ir4[3] = (v560_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v557_data, 6))));
                float v563_data = r2[4];
                float v566_data = ir4[4];
                ir4[4] = (v566_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v563_data, 6))));
                float v569_data = r2[5];
                float v572_data = ir4[5];
                ir4[5] = (v572_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v569_data, 6))));
              }
              if (v9_lead < 12) {
                float v578_data = r3[7];
                float v579_data = r2[0];
                float v582_data = ir4[0];
                ir4[0] = (v582_data + (v578_data * (sycl::group_broadcast(item.get_sub_group(), v579_data, 7))));
                float v585_data = r2[1];
                float v588_data = ir4[1];
                ir4[1] = (v588_data + (v578_data * (sycl::group_broadcast(item.get_sub_group(), v585_data, 7))));
                float v591_data = r2[2];
                float v594_data = ir4[2];
                ir4[2] = (v594_data + (v578_data * (sycl::group_broadcast(item.get_sub_group(), v591_data, 7))));
                float v597_data = r2[3];
                float v600_data = ir4[3];
                ir4[3] = (v600_data + (v578_data * (sycl::group_broadcast(item.get_sub_group(), v597_data, 7))));
                float v603_data = r2[4];
                float v606_data = ir4[4];
                ir4[4] = (v606_data + (v578_data * (sycl::group_broadcast(item.get_sub_group(), v603_data, 7))));
                float v609_data = r2[5];
                float v612_data = ir4[5];
                ir4[5] = (v612_data + (v578_data * (sycl::group_broadcast(item.get_sub_group(), v609_data, 7))));
              }
              if (v9_lead < 12) {
                float v618_data = r3[8];
                float v619_data = r2[0];
                float v622_data = ir4[0];
                ir4[0] = (v622_data + (v618_data * (sycl::group_broadcast(item.get_sub_group(), v619_data, 8))));
                float v625_data = r2[1];
                float v628_data = ir4[1];
                ir4[1] = (v628_data + (v618_data * (sycl::group_broadcast(item.get_sub_group(), v625_data, 8))));
                float v631_data = r2[2];
                float v634_data = ir4[2];
                ir4[2] = (v634_data + (v618_data * (sycl::group_broadcast(item.get_sub_group(), v631_data, 8))));
                float v637_data = r2[3];
                float v640_data = ir4[3];
                ir4[3] = (v640_data + (v618_data * (sycl::group_broadcast(item.get_sub_group(), v637_data, 8))));
                float v643_data = r2[4];
                float v646_data = ir4[4];
                ir4[4] = (v646_data + (v618_data * (sycl::group_broadcast(item.get_sub_group(), v643_data, 8))));
                float v649_data = r2[5];
                float v652_data = ir4[5];
                ir4[5] = (v652_data + (v618_data * (sycl::group_broadcast(item.get_sub_group(), v649_data, 8))));
              }
              if (v9_lead < 12) {
                float v658_data = r3[9];
                float v659_data = r2[0];
                float v662_data = ir4[0];
                ir4[0] = (v662_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v659_data, 9))));
                float v665_data = r2[1];
                float v668_data = ir4[1];
                ir4[1] = (v668_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v665_data, 9))));
                float v671_data = r2[2];
                float v674_data = ir4[2];
                ir4[2] = (v674_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v671_data, 9))));
                float v677_data = r2[3];
                float v680_data = ir4[3];
                ir4[3] = (v680_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v677_data, 9))));
                float v683_data = r2[4];
                float v686_data = ir4[4];
                ir4[4] = (v686_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v683_data, 9))));
                float v689_data = r2[5];
                float v692_data = ir4[5];
                ir4[5] = (v692_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v689_data, 9))));
              }
              if (v9_lead < 12) {
                float v698_data = r3[10];
                float v699_data = r2[0];
                float v702_data = ir4[0];
                ir4[0] = (v702_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v699_data, 10))));
                float v705_data = r2[1];
                float v708_data = ir4[1];
                ir4[1] = (v708_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v705_data, 10))));
                float v711_data = r2[2];
                float v714_data = ir4[2];
                ir4[2] = (v714_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v711_data, 10))));
                float v717_data = r2[3];
                float v720_data = ir4[3];
                ir4[3] = (v720_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v717_data, 10))));
                float v723_data = r2[4];
                float v726_data = ir4[4];
                ir4[4] = (v726_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v723_data, 10))));
                float v729_data = r2[5];
                float v732_data = ir4[5];
                ir4[5] = (v732_data + (v698_data * (sycl::group_broadcast(item.get_sub_group(), v729_data, 10))));
              }
              if (v9_lead < 12) {
                float v738_data = r3[11];
                float v739_data = r2[0];
                float v742_data = ir4[0];
                ir4[0] = (v742_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v739_data, 11))));
                float v745_data = r2[1];
                float v748_data = ir4[1];
                ir4[1] = (v748_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v745_data, 11))));
                float v751_data = r2[2];
                float v754_data = ir4[2];
                ir4[2] = (v754_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v751_data, 11))));
                float v757_data = r2[3];
                float v760_data = ir4[3];
                ir4[3] = (v760_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v757_data, 11))));
                float v763_data = r2[4];
                float v766_data = ir4[4];
                ir4[4] = (v766_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v763_data, 11))));
                float v769_data = r2[5];
                float v772_data = ir4[5];
                ir4[5] = (v772_data + (v738_data * (sycl::group_broadcast(item.get_sub_group(), v769_data, 11))));
              }
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v778_n1 = 0; v778_n1 < 6; ++v778_n1) {
                  float v780_data = ir4[v778_n1];
                  r4[v778_n1] = v780_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v9_lead < 12) {
                #pragma unroll
                for (int32_t v786_i1 = 0; v786_i1 < 6; ++v786_i1) {
                  float v788_data = r4[v786_i1];
                  glb_m2[(v9_lead + (v786_i1 * 12))] = v788_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

