// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_609dd06e89(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_609dd06e89(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
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
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v10_lead = item.get_local_id(0) % 16;
              if (v10_lead < 8) {
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
                  float v20_data = glb_m0[(v10_lead + (v12_i1 * 8))];
                  r0[v12_i1] = v20_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              float v23_lin = glb_m1[0 + item.get_local_id(0) * 1];
              r1[0] = v23_lin;
              float v24_lin = glb_m1[16 + item.get_local_id(0) * 1];
              r1[1] = v24_lin;
              float v25_lin = glb_m1[32 + item.get_local_id(0) * 1];
              r1[2] = v25_lin;
              float v26_lin = glb_m1[48 + item.get_local_id(0) * 1];
              r1[3] = v26_lin;
              // wait(r0 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = load{g>r}(glb_m2);
              if (v10_lead < 8) {
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
                  float v40_data = glb_m2[(v10_lead + (v32_i1 * 8))];
                  r3[v32_i1] = v40_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v10_lead < 8) {
                float v47_data = r0[0];
                float v48_data = r1[0];
                float v51_data = r2[0];
                r2[0] = (v51_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 0))));
                float v54_data = r1[1];
                float v57_data = r2[1];
                r2[1] = (v57_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v54_data, 0))));
                float v60_data = r1[2];
                float v63_data = r2[2];
                r2[2] = (v63_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v60_data, 0))));
                float v66_data = r1[3];
                float v69_data = r2[3];
                r2[3] = (v69_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 0))));
                float v72_data = r1[4];
                float v75_data = r2[4];
                r2[4] = (v75_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 0))));
                float v78_data = r1[5];
                float v81_data = r2[5];
                r2[5] = (v81_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 0))));
                float v84_data = r1[6];
                float v87_data = r2[6];
                r2[6] = (v87_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 0))));
                float v90_data = r1[7];
                float v93_data = r2[7];
                r2[7] = (v93_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 0))));
              }
              if (v10_lead < 8) {
                float v99_data = r0[1];
                float v100_data = r1[0];
                float v103_data = r2[0];
                r2[0] = (v103_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 1))));
                float v106_data = r1[1];
                float v109_data = r2[1];
                r2[1] = (v109_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 1))));
                float v112_data = r1[2];
                float v115_data = r2[2];
                r2[2] = (v115_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 1))));
                float v118_data = r1[3];
                float v121_data = r2[3];
                r2[3] = (v121_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 1))));
                float v124_data = r1[4];
                float v127_data = r2[4];
                r2[4] = (v127_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 1))));
                float v130_data = r1[5];
                float v133_data = r2[5];
                r2[5] = (v133_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 1))));
                float v136_data = r1[6];
                float v139_data = r2[6];
                r2[6] = (v139_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 1))));
                float v142_data = r1[7];
                float v145_data = r2[7];
                r2[7] = (v145_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 1))));
              }
              if (v10_lead < 8) {
                float v151_data = r0[2];
                float v152_data = r1[0];
                float v155_data = r2[0];
                r2[0] = (v155_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 2))));
                float v158_data = r1[1];
                float v161_data = r2[1];
                r2[1] = (v161_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v158_data, 2))));
                float v164_data = r1[2];
                float v167_data = r2[2];
                r2[2] = (v167_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v164_data, 2))));
                float v170_data = r1[3];
                float v173_data = r2[3];
                r2[3] = (v173_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v170_data, 2))));
                float v176_data = r1[4];
                float v179_data = r2[4];
                r2[4] = (v179_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v176_data, 2))));
                float v182_data = r1[5];
                float v185_data = r2[5];
                r2[5] = (v185_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v182_data, 2))));
                float v188_data = r1[6];
                float v191_data = r2[6];
                r2[6] = (v191_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v188_data, 2))));
                float v194_data = r1[7];
                float v197_data = r2[7];
                r2[7] = (v197_data + (v151_data * (sycl::group_broadcast(item.get_sub_group(), v194_data, 2))));
              }
              if (v10_lead < 8) {
                float v203_data = r0[3];
                float v204_data = r1[0];
                float v207_data = r2[0];
                r2[0] = (v207_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v204_data, 3))));
                float v210_data = r1[1];
                float v213_data = r2[1];
                r2[1] = (v213_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v210_data, 3))));
                float v216_data = r1[2];
                float v219_data = r2[2];
                r2[2] = (v219_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v216_data, 3))));
                float v222_data = r1[3];
                float v225_data = r2[3];
                r2[3] = (v225_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v222_data, 3))));
                float v228_data = r1[4];
                float v231_data = r2[4];
                r2[4] = (v231_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v228_data, 3))));
                float v234_data = r1[5];
                float v237_data = r2[5];
                r2[5] = (v237_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v234_data, 3))));
                float v240_data = r1[6];
                float v243_data = r2[6];
                r2[6] = (v243_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v240_data, 3))));
                float v246_data = r1[7];
                float v249_data = r2[7];
                r2[7] = (v249_data + (v203_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 3))));
              }
              if (v10_lead < 8) {
                float v255_data = r0[4];
                float v256_data = r1[0];
                float v259_data = r2[0];
                r2[0] = (v259_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v256_data, 4))));
                float v262_data = r1[1];
                float v265_data = r2[1];
                r2[1] = (v265_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v262_data, 4))));
                float v268_data = r1[2];
                float v271_data = r2[2];
                r2[2] = (v271_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v268_data, 4))));
                float v274_data = r1[3];
                float v277_data = r2[3];
                r2[3] = (v277_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v274_data, 4))));
                float v280_data = r1[4];
                float v283_data = r2[4];
                r2[4] = (v283_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v280_data, 4))));
                float v286_data = r1[5];
                float v289_data = r2[5];
                r2[5] = (v289_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v286_data, 4))));
                float v292_data = r1[6];
                float v295_data = r2[6];
                r2[6] = (v295_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v292_data, 4))));
                float v298_data = r1[7];
                float v301_data = r2[7];
                r2[7] = (v301_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v298_data, 4))));
              }
              if (v10_lead < 8) {
                float v307_data = r0[5];
                float v308_data = r1[0];
                float v311_data = r2[0];
                r2[0] = (v311_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v308_data, 5))));
                float v314_data = r1[1];
                float v317_data = r2[1];
                r2[1] = (v317_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v314_data, 5))));
                float v320_data = r1[2];
                float v323_data = r2[2];
                r2[2] = (v323_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v320_data, 5))));
                float v326_data = r1[3];
                float v329_data = r2[3];
                r2[3] = (v329_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v326_data, 5))));
                float v332_data = r1[4];
                float v335_data = r2[4];
                r2[4] = (v335_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v332_data, 5))));
                float v338_data = r1[5];
                float v341_data = r2[5];
                r2[5] = (v341_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v338_data, 5))));
                float v344_data = r1[6];
                float v347_data = r2[6];
                r2[6] = (v347_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v344_data, 5))));
                float v350_data = r1[7];
                float v353_data = r2[7];
                r2[7] = (v353_data + (v307_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 5))));
              }
              if (v10_lead < 8) {
                float v359_data = r0[6];
                float v360_data = r1[0];
                float v363_data = r2[0];
                r2[0] = (v363_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v360_data, 6))));
                float v366_data = r1[1];
                float v369_data = r2[1];
                r2[1] = (v369_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v366_data, 6))));
                float v372_data = r1[2];
                float v375_data = r2[2];
                r2[2] = (v375_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v372_data, 6))));
                float v378_data = r1[3];
                float v381_data = r2[3];
                r2[3] = (v381_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v378_data, 6))));
                float v384_data = r1[4];
                float v387_data = r2[4];
                r2[4] = (v387_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 6))));
                float v390_data = r1[5];
                float v393_data = r2[5];
                r2[5] = (v393_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v390_data, 6))));
                float v396_data = r1[6];
                float v399_data = r2[6];
                r2[6] = (v399_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v396_data, 6))));
                float v402_data = r1[7];
                float v405_data = r2[7];
                r2[7] = (v405_data + (v359_data * (sycl::group_broadcast(item.get_sub_group(), v402_data, 6))));
              }
              if (v10_lead < 8) {
                float v411_data = r0[7];
                float v412_data = r1[0];
                float v415_data = r2[0];
                r2[0] = (v415_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v412_data, 7))));
                float v418_data = r1[1];
                float v421_data = r2[1];
                r2[1] = (v421_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v418_data, 7))));
                float v424_data = r1[2];
                float v427_data = r2[2];
                r2[2] = (v427_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v424_data, 7))));
                float v430_data = r1[3];
                float v433_data = r2[3];
                r2[3] = (v433_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v430_data, 7))));
                float v436_data = r1[4];
                float v439_data = r2[4];
                r2[4] = (v439_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v436_data, 7))));
                float v442_data = r1[5];
                float v445_data = r2[5];
                r2[5] = (v445_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 7))));
                float v448_data = r1[6];
                float v451_data = r2[6];
                r2[6] = (v451_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 7))));
                float v454_data = r1[7];
                float v457_data = r2[7];
                r2[7] = (v457_data + (v411_data * (sycl::group_broadcast(item.get_sub_group(), v454_data, 7))));
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m3);
              float v460_lin = glb_m3[0 + item.get_local_id(0) * 1];
              r4[0] = v460_lin;
              float v461_lin = glb_m3[16 + item.get_local_id(0) * 1];
              r4[1] = v461_lin;
              float v462_lin = glb_m3[32 + item.get_local_id(0) * 1];
              r4[2] = v462_lin;
              float v463_lin = glb_m3[48 + item.get_local_id(0) * 1];
              r4[3] = v463_lin;
              // wait(r3 = load{g>r}(glb_m2););
              // wait(r4 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir5[8]{};
              if (v10_lead < 8) {
                float v470_data = r3[0];
                float v471_data = r4[0];
                float v474_data = ir5[0];
                ir5[0] = (v474_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v471_data, 0))));
                float v477_data = r4[1];
                float v480_data = ir5[1];
                ir5[1] = (v480_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 0))));
                float v483_data = r4[2];
                float v486_data = ir5[2];
                ir5[2] = (v486_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 0))));
                float v489_data = r4[3];
                float v492_data = ir5[3];
                ir5[3] = (v492_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 0))));
                float v495_data = r4[4];
                float v498_data = ir5[4];
                ir5[4] = (v498_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v495_data, 0))));
                float v501_data = r4[5];
                float v504_data = ir5[5];
                ir5[5] = (v504_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v501_data, 0))));
                float v507_data = r4[6];
                float v510_data = ir5[6];
                ir5[6] = (v510_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v507_data, 0))));
                float v513_data = r4[7];
                float v516_data = ir5[7];
                ir5[7] = (v516_data + (v470_data * (sycl::group_broadcast(item.get_sub_group(), v513_data, 0))));
              }
              if (v10_lead < 8) {
                float v522_data = r3[1];
                float v523_data = r4[0];
                float v526_data = ir5[0];
                ir5[0] = (v526_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 1))));
                float v529_data = r4[1];
                float v532_data = ir5[1];
                ir5[1] = (v532_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v529_data, 1))));
                float v535_data = r4[2];
                float v538_data = ir5[2];
                ir5[2] = (v538_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v535_data, 1))));
                float v541_data = r4[3];
                float v544_data = ir5[3];
                ir5[3] = (v544_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v541_data, 1))));
                float v547_data = r4[4];
                float v550_data = ir5[4];
                ir5[4] = (v550_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v547_data, 1))));
                float v553_data = r4[5];
                float v556_data = ir5[5];
                ir5[5] = (v556_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v553_data, 1))));
                float v559_data = r4[6];
                float v562_data = ir5[6];
                ir5[6] = (v562_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v559_data, 1))));
                float v565_data = r4[7];
                float v568_data = ir5[7];
                ir5[7] = (v568_data + (v522_data * (sycl::group_broadcast(item.get_sub_group(), v565_data, 1))));
              }
              if (v10_lead < 8) {
                float v574_data = r3[2];
                float v575_data = r4[0];
                float v578_data = ir5[0];
                ir5[0] = (v578_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v575_data, 2))));
                float v581_data = r4[1];
                float v584_data = ir5[1];
                ir5[1] = (v584_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v581_data, 2))));
                float v587_data = r4[2];
                float v590_data = ir5[2];
                ir5[2] = (v590_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v587_data, 2))));
                float v593_data = r4[3];
                float v596_data = ir5[3];
                ir5[3] = (v596_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v593_data, 2))));
                float v599_data = r4[4];
                float v602_data = ir5[4];
                ir5[4] = (v602_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 2))));
                float v605_data = r4[5];
                float v608_data = ir5[5];
                ir5[5] = (v608_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v605_data, 2))));
                float v611_data = r4[6];
                float v614_data = ir5[6];
                ir5[6] = (v614_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v611_data, 2))));
                float v617_data = r4[7];
                float v620_data = ir5[7];
                ir5[7] = (v620_data + (v574_data * (sycl::group_broadcast(item.get_sub_group(), v617_data, 2))));
              }
              if (v10_lead < 8) {
                float v626_data = r3[3];
                float v627_data = r4[0];
                float v630_data = ir5[0];
                ir5[0] = (v630_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 3))));
                float v633_data = r4[1];
                float v636_data = ir5[1];
                ir5[1] = (v636_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v633_data, 3))));
                float v639_data = r4[2];
                float v642_data = ir5[2];
                ir5[2] = (v642_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v639_data, 3))));
                float v645_data = r4[3];
                float v648_data = ir5[3];
                ir5[3] = (v648_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v645_data, 3))));
                float v651_data = r4[4];
                float v654_data = ir5[4];
                ir5[4] = (v654_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v651_data, 3))));
                float v657_data = r4[5];
                float v660_data = ir5[5];
                ir5[5] = (v660_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v657_data, 3))));
                float v663_data = r4[6];
                float v666_data = ir5[6];
                ir5[6] = (v666_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v663_data, 3))));
                float v669_data = r4[7];
                float v672_data = ir5[7];
                ir5[7] = (v672_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v669_data, 3))));
              }
              if (v10_lead < 8) {
                float v678_data = r3[4];
                float v679_data = r4[0];
                float v682_data = ir5[0];
                ir5[0] = (v682_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v679_data, 4))));
                float v685_data = r4[1];
                float v688_data = ir5[1];
                ir5[1] = (v688_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v685_data, 4))));
                float v691_data = r4[2];
                float v694_data = ir5[2];
                ir5[2] = (v694_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v691_data, 4))));
                float v697_data = r4[3];
                float v700_data = ir5[3];
                ir5[3] = (v700_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v697_data, 4))));
                float v703_data = r4[4];
                float v706_data = ir5[4];
                ir5[4] = (v706_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v703_data, 4))));
                float v709_data = r4[5];
                float v712_data = ir5[5];
                ir5[5] = (v712_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v709_data, 4))));
                float v715_data = r4[6];
                float v718_data = ir5[6];
                ir5[6] = (v718_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v715_data, 4))));
                float v721_data = r4[7];
                float v724_data = ir5[7];
                ir5[7] = (v724_data + (v678_data * (sycl::group_broadcast(item.get_sub_group(), v721_data, 4))));
              }
              if (v10_lead < 8) {
                float v730_data = r3[5];
                float v731_data = r4[0];
                float v734_data = ir5[0];
                ir5[0] = (v734_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 5))));
                float v737_data = r4[1];
                float v740_data = ir5[1];
                ir5[1] = (v740_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v737_data, 5))));
                float v743_data = r4[2];
                float v746_data = ir5[2];
                ir5[2] = (v746_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v743_data, 5))));
                float v749_data = r4[3];
                float v752_data = ir5[3];
                ir5[3] = (v752_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v749_data, 5))));
                float v755_data = r4[4];
                float v758_data = ir5[4];
                ir5[4] = (v758_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v755_data, 5))));
                float v761_data = r4[5];
                float v764_data = ir5[5];
                ir5[5] = (v764_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v761_data, 5))));
                float v767_data = r4[6];
                float v770_data = ir5[6];
                ir5[6] = (v770_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v767_data, 5))));
                float v773_data = r4[7];
                float v776_data = ir5[7];
                ir5[7] = (v776_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v773_data, 5))));
              }
              if (v10_lead < 8) {
                float v782_data = r3[6];
                float v783_data = r4[0];
                float v786_data = ir5[0];
                ir5[0] = (v786_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v783_data, 6))));
                float v789_data = r4[1];
                float v792_data = ir5[1];
                ir5[1] = (v792_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v789_data, 6))));
                float v795_data = r4[2];
                float v798_data = ir5[2];
                ir5[2] = (v798_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v795_data, 6))));
                float v801_data = r4[3];
                float v804_data = ir5[3];
                ir5[3] = (v804_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v801_data, 6))));
                float v807_data = r4[4];
                float v810_data = ir5[4];
                ir5[4] = (v810_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v807_data, 6))));
                float v813_data = r4[5];
                float v816_data = ir5[5];
                ir5[5] = (v816_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v813_data, 6))));
                float v819_data = r4[6];
                float v822_data = ir5[6];
                ir5[6] = (v822_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v819_data, 6))));
                float v825_data = r4[7];
                float v828_data = ir5[7];
                ir5[7] = (v828_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v825_data, 6))));
              }
              if (v10_lead < 8) {
                float v834_data = r3[7];
                float v835_data = r4[0];
                float v838_data = ir5[0];
                ir5[0] = (v838_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v835_data, 7))));
                float v841_data = r4[1];
                float v844_data = ir5[1];
                ir5[1] = (v844_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v841_data, 7))));
                float v847_data = r4[2];
                float v850_data = ir5[2];
                ir5[2] = (v850_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v847_data, 7))));
                float v853_data = r4[3];
                float v856_data = ir5[3];
                ir5[3] = (v856_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v853_data, 7))));
                float v859_data = r4[4];
                float v862_data = ir5[4];
                ir5[4] = (v862_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v859_data, 7))));
                float v865_data = r4[5];
                float v868_data = ir5[5];
                ir5[5] = (v868_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v865_data, 7))));
                float v871_data = r4[6];
                float v874_data = ir5[6];
                ir5[6] = (v874_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v871_data, 7))));
                float v877_data = r4[7];
                float v880_data = ir5[7];
                ir5[7] = (v880_data + (v834_data * (sycl::group_broadcast(item.get_sub_group(), v877_data, 7))));
              }
              if (v10_lead < 8) {
                #pragma unroll
                for (int32_t v886_n1 = 0; v886_n1 < 8; ++v886_n1) {
                  float v888_data = ir5[v886_n1];
                  float v890_data = r2[v886_n1];
                  r5[v886_n1] = (v890_data + v888_data);
                }
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r5);
              if (v10_lead < 8) {
                #pragma unroll
                for (int32_t v898_i1 = 0; v898_i1 < 8; ++v898_i1) {
                  float v900_data = r5[v898_i1];
                  int32_t v907_a = v10_lead + (v898_i1 * 8);
                  s0[(v907_a ^ ((v907_a >> 5) & 31))] = v900_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m4 = abs(s0)
              if (v10_lead < 8) {
                #pragma unroll
                for (int32_t v915_k1 = 0; v915_k1 < 8; ++v915_k1) {
                  int32_t v921_a = v915_k1 * 8;
                  int32_t v922_a = v10_lead + v921_a;
                  float v926_data = s0[(v922_a ^ ((v922_a >> 5) & 31))];
                  glb_m4[(v10_lead + v921_a)] = (sycl::fabs(v926_data));
                }
              }
            }
          }
        }
      });
    }
  });
}

