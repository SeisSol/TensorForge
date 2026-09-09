// === base name ===
kernel_065b3f7b414a75a2

// === header ===
void launcher_kernel_065b3f7b414a75a2(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_065b3f7b414a75a2(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_065b3f7b414a75a2(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_065b3f7b414a75a2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v15_lead = item.get_local_id(0) % 16;
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 8; ++v17_i1) {
                  float v25_data = glb_m0[(v15_lead + (v17_i1 * 8))];
                  r0[v17_i1] = v25_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
                  float v40_data = glb_m1[(v15_lead + (v32_i1 * 8))];
                  r1[v32_i1] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = load{g>r}(glb_m2);
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v47_i1 = 0; v47_i1 < 8; ++v47_i1) {
                  float v55_data = glb_m2[(v15_lead + (v47_i1 * 8))];
                  r3[v47_i1] = v55_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v15_lead < 8) {
                float v62_data = r0[0];
                float v63_data = r1[0];
                float v66_data = r2[0];
                r2[0] = (v66_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
                float v69_data = r1[1];
                float v72_data = r2[1];
                r2[1] = (v72_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
                float v75_data = r1[2];
                float v78_data = r2[2];
                r2[2] = (v78_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
                float v81_data = r1[3];
                float v84_data = r2[3];
                r2[3] = (v84_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
                float v87_data = r1[4];
                float v90_data = r2[4];
                r2[4] = (v90_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
                float v93_data = r1[5];
                float v96_data = r2[5];
                r2[5] = (v96_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 0))));
                float v99_data = r1[6];
                float v102_data = r2[6];
                r2[6] = (v102_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 0))));
                float v105_data = r1[7];
                float v108_data = r2[7];
                r2[7] = (v108_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 0))));
              }
              if (v15_lead < 8) {
                float v114_data = r0[1];
                float v115_data = r1[0];
                float v118_data = r2[0];
                r2[0] = (v118_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 1))));
                float v121_data = r1[1];
                float v124_data = r2[1];
                r2[1] = (v124_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 1))));
                float v127_data = r1[2];
                float v130_data = r2[2];
                r2[2] = (v130_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
                float v133_data = r1[3];
                float v136_data = r2[3];
                r2[3] = (v136_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 1))));
                float v139_data = r1[4];
                float v142_data = r2[4];
                r2[4] = (v142_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 1))));
                float v145_data = r1[5];
                float v148_data = r2[5];
                r2[5] = (v148_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 1))));
                float v151_data = r1[6];
                float v154_data = r2[6];
                r2[6] = (v154_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 1))));
                float v157_data = r1[7];
                float v160_data = r2[7];
                r2[7] = (v160_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 1))));
              }
              if (v15_lead < 8) {
                float v166_data = r0[2];
                float v167_data = r1[0];
                float v170_data = r2[0];
                r2[0] = (v170_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
                float v173_data = r1[1];
                float v176_data = r2[1];
                r2[1] = (v176_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 2))));
                float v179_data = r1[2];
                float v182_data = r2[2];
                r2[2] = (v182_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 2))));
                float v185_data = r1[3];
                float v188_data = r2[3];
                r2[3] = (v188_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 2))));
                float v191_data = r1[4];
                float v194_data = r2[4];
                r2[4] = (v194_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 2))));
                float v197_data = r1[5];
                float v200_data = r2[5];
                r2[5] = (v200_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 2))));
                float v203_data = r1[6];
                float v206_data = r2[6];
                r2[6] = (v206_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 2))));
                float v209_data = r1[7];
                float v212_data = r2[7];
                r2[7] = (v212_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 2))));
              }
              if (v15_lead < 8) {
                float v218_data = r0[3];
                float v219_data = r1[0];
                float v222_data = r2[0];
                r2[0] = (v222_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v219_data, 3))));
                float v225_data = r1[1];
                float v228_data = r2[1];
                r2[1] = (v228_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 3))));
                float v231_data = r1[2];
                float v234_data = r2[2];
                r2[2] = (v234_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 3))));
                float v237_data = r1[3];
                float v240_data = r2[3];
                r2[3] = (v240_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v237_data, 3))));
                float v243_data = r1[4];
                float v246_data = r2[4];
                r2[4] = (v246_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v243_data, 3))));
                float v249_data = r1[5];
                float v252_data = r2[5];
                r2[5] = (v252_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v249_data, 3))));
                float v255_data = r1[6];
                float v258_data = r2[6];
                r2[6] = (v258_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v255_data, 3))));
                float v261_data = r1[7];
                float v264_data = r2[7];
                r2[7] = (v264_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v261_data, 3))));
              }
              if (v15_lead < 8) {
                float v270_data = r0[4];
                float v271_data = r1[0];
                float v274_data = r2[0];
                r2[0] = (v274_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v271_data, 4))));
                float v277_data = r1[1];
                float v280_data = r2[1];
                r2[1] = (v280_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v277_data, 4))));
                float v283_data = r1[2];
                float v286_data = r2[2];
                r2[2] = (v286_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v283_data, 4))));
                float v289_data = r1[3];
                float v292_data = r2[3];
                r2[3] = (v292_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v289_data, 4))));
                float v295_data = r1[4];
                float v298_data = r2[4];
                r2[4] = (v298_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v295_data, 4))));
                float v301_data = r1[5];
                float v304_data = r2[5];
                r2[5] = (v304_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v301_data, 4))));
                float v307_data = r1[6];
                float v310_data = r2[6];
                r2[6] = (v310_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 4))));
                float v313_data = r1[7];
                float v316_data = r2[7];
                r2[7] = (v316_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 4))));
              }
              if (v15_lead < 8) {
                float v322_data = r0[5];
                float v323_data = r1[0];
                float v326_data = r2[0];
                r2[0] = (v326_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 5))));
                float v329_data = r1[1];
                float v332_data = r2[1];
                r2[1] = (v332_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v329_data, 5))));
                float v335_data = r1[2];
                float v338_data = r2[2];
                r2[2] = (v338_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 5))));
                float v341_data = r1[3];
                float v344_data = r2[3];
                r2[3] = (v344_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v341_data, 5))));
                float v347_data = r1[4];
                float v350_data = r2[4];
                r2[4] = (v350_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v347_data, 5))));
                float v353_data = r1[5];
                float v356_data = r2[5];
                r2[5] = (v356_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v353_data, 5))));
                float v359_data = r1[6];
                float v362_data = r2[6];
                r2[6] = (v362_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v359_data, 5))));
                float v365_data = r1[7];
                float v368_data = r2[7];
                r2[7] = (v368_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 5))));
              }
              if (v15_lead < 8) {
                float v374_data = r0[6];
                float v375_data = r1[0];
                float v378_data = r2[0];
                r2[0] = (v378_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v375_data, 6))));
                float v381_data = r1[1];
                float v384_data = r2[1];
                r2[1] = (v384_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v381_data, 6))));
                float v387_data = r1[2];
                float v390_data = r2[2];
                r2[2] = (v390_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v387_data, 6))));
                float v393_data = r1[3];
                float v396_data = r2[3];
                r2[3] = (v396_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v393_data, 6))));
                float v399_data = r1[4];
                float v402_data = r2[4];
                r2[4] = (v402_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 6))));
                float v405_data = r1[5];
                float v408_data = r2[5];
                r2[5] = (v408_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v405_data, 6))));
                float v411_data = r1[6];
                float v414_data = r2[6];
                r2[6] = (v414_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v411_data, 6))));
                float v417_data = r1[7];
                float v420_data = r2[7];
                r2[7] = (v420_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v417_data, 6))));
              }
              if (v15_lead < 8) {
                float v426_data = r0[7];
                float v427_data = r1[0];
                float v430_data = r2[0];
                r2[0] = (v430_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 7))));
                float v433_data = r1[1];
                float v436_data = r2[1];
                r2[1] = (v436_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 7))));
                float v439_data = r1[2];
                float v442_data = r2[2];
                r2[2] = (v442_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v439_data, 7))));
                float v445_data = r1[3];
                float v448_data = r2[3];
                r2[3] = (v448_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v445_data, 7))));
                float v451_data = r1[4];
                float v454_data = r2[4];
                r2[4] = (v454_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v451_data, 7))));
                float v457_data = r1[5];
                float v460_data = r2[5];
                r2[5] = (v460_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v457_data, 7))));
                float v463_data = r1[6];
                float v466_data = r2[6];
                r2[6] = (v466_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v463_data, 7))));
                float v469_data = r1[7];
                float v472_data = r2[7];
                r2[7] = (v472_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v469_data, 7))));
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m3);
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v479_i1 = 0; v479_i1 < 8; ++v479_i1) {
                  float v487_data = glb_m3[(v15_lead + (v479_i1 * 8))];
                  r4[v479_i1] = v487_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              // wait(r4 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir5[8]{};
              if (v15_lead < 8) {
                float v495_data = r3[0];
                float v496_data = r4[0];
                float v499_data = ir5[0];
                ir5[0] = (v499_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v496_data, 0))));
                float v502_data = r4[1];
                float v505_data = ir5[1];
                ir5[1] = (v505_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v502_data, 0))));
                float v508_data = r4[2];
                float v511_data = ir5[2];
                ir5[2] = (v511_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v508_data, 0))));
                float v514_data = r4[3];
                float v517_data = ir5[3];
                ir5[3] = (v517_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v514_data, 0))));
                float v520_data = r4[4];
                float v523_data = ir5[4];
                ir5[4] = (v523_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v520_data, 0))));
                float v526_data = r4[5];
                float v529_data = ir5[5];
                ir5[5] = (v529_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v526_data, 0))));
                float v532_data = r4[6];
                float v535_data = ir5[6];
                ir5[6] = (v535_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v532_data, 0))));
                float v538_data = r4[7];
                float v541_data = ir5[7];
                ir5[7] = (v541_data + (v495_data * (sycl::group_broadcast(item.get_sub_group(), v538_data, 0))));
              }
              if (v15_lead < 8) {
                float v547_data = r3[1];
                float v548_data = r4[0];
                float v551_data = ir5[0];
                ir5[0] = (v551_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 1))));
                float v554_data = r4[1];
                float v557_data = ir5[1];
                ir5[1] = (v557_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v554_data, 1))));
                float v560_data = r4[2];
                float v563_data = ir5[2];
                ir5[2] = (v563_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v560_data, 1))));
                float v566_data = r4[3];
                float v569_data = ir5[3];
                ir5[3] = (v569_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v566_data, 1))));
                float v572_data = r4[4];
                float v575_data = ir5[4];
                ir5[4] = (v575_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v572_data, 1))));
                float v578_data = r4[5];
                float v581_data = ir5[5];
                ir5[5] = (v581_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v578_data, 1))));
                float v584_data = r4[6];
                float v587_data = ir5[6];
                ir5[6] = (v587_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v584_data, 1))));
                float v590_data = r4[7];
                float v593_data = ir5[7];
                ir5[7] = (v593_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v590_data, 1))));
              }
              if (v15_lead < 8) {
                float v599_data = r3[2];
                float v600_data = r4[0];
                float v603_data = ir5[0];
                ir5[0] = (v603_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v600_data, 2))));
                float v606_data = r4[1];
                float v609_data = ir5[1];
                ir5[1] = (v609_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v606_data, 2))));
                float v612_data = r4[2];
                float v615_data = ir5[2];
                ir5[2] = (v615_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v612_data, 2))));
                float v618_data = r4[3];
                float v621_data = ir5[3];
                ir5[3] = (v621_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v618_data, 2))));
                float v624_data = r4[4];
                float v627_data = ir5[4];
                ir5[4] = (v627_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v624_data, 2))));
                float v630_data = r4[5];
                float v633_data = ir5[5];
                ir5[5] = (v633_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v630_data, 2))));
                float v636_data = r4[6];
                float v639_data = ir5[6];
                ir5[6] = (v639_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v636_data, 2))));
                float v642_data = r4[7];
                float v645_data = ir5[7];
                ir5[7] = (v645_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 2))));
              }
              if (v15_lead < 8) {
                float v651_data = r3[3];
                float v652_data = r4[0];
                float v655_data = ir5[0];
                ir5[0] = (v655_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v652_data, 3))));
                float v658_data = r4[1];
                float v661_data = ir5[1];
                ir5[1] = (v661_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v658_data, 3))));
                float v664_data = r4[2];
                float v667_data = ir5[2];
                ir5[2] = (v667_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v664_data, 3))));
                float v670_data = r4[3];
                float v673_data = ir5[3];
                ir5[3] = (v673_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v670_data, 3))));
                float v676_data = r4[4];
                float v679_data = ir5[4];
                ir5[4] = (v679_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v676_data, 3))));
                float v682_data = r4[5];
                float v685_data = ir5[5];
                ir5[5] = (v685_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v682_data, 3))));
                float v688_data = r4[6];
                float v691_data = ir5[6];
                ir5[6] = (v691_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v688_data, 3))));
                float v694_data = r4[7];
                float v697_data = ir5[7];
                ir5[7] = (v697_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v694_data, 3))));
              }
              if (v15_lead < 8) {
                float v703_data = r3[4];
                float v704_data = r4[0];
                float v707_data = ir5[0];
                ir5[0] = (v707_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v704_data, 4))));
                float v710_data = r4[1];
                float v713_data = ir5[1];
                ir5[1] = (v713_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v710_data, 4))));
                float v716_data = r4[2];
                float v719_data = ir5[2];
                ir5[2] = (v719_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v716_data, 4))));
                float v722_data = r4[3];
                float v725_data = ir5[3];
                ir5[3] = (v725_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v722_data, 4))));
                float v728_data = r4[4];
                float v731_data = ir5[4];
                ir5[4] = (v731_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v728_data, 4))));
                float v734_data = r4[5];
                float v737_data = ir5[5];
                ir5[5] = (v737_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v734_data, 4))));
                float v740_data = r4[6];
                float v743_data = ir5[6];
                ir5[6] = (v743_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v740_data, 4))));
                float v746_data = r4[7];
                float v749_data = ir5[7];
                ir5[7] = (v749_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 4))));
              }
              if (v15_lead < 8) {
                float v755_data = r3[5];
                float v756_data = r4[0];
                float v759_data = ir5[0];
                ir5[0] = (v759_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v756_data, 5))));
                float v762_data = r4[1];
                float v765_data = ir5[1];
                ir5[1] = (v765_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v762_data, 5))));
                float v768_data = r4[2];
                float v771_data = ir5[2];
                ir5[2] = (v771_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v768_data, 5))));
                float v774_data = r4[3];
                float v777_data = ir5[3];
                ir5[3] = (v777_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v774_data, 5))));
                float v780_data = r4[4];
                float v783_data = ir5[4];
                ir5[4] = (v783_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v780_data, 5))));
                float v786_data = r4[5];
                float v789_data = ir5[5];
                ir5[5] = (v789_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v786_data, 5))));
                float v792_data = r4[6];
                float v795_data = ir5[6];
                ir5[6] = (v795_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v792_data, 5))));
                float v798_data = r4[7];
                float v801_data = ir5[7];
                ir5[7] = (v801_data + (v755_data * (sycl::group_broadcast(item.get_sub_group(), v798_data, 5))));
              }
              if (v15_lead < 8) {
                float v807_data = r3[6];
                float v808_data = r4[0];
                float v811_data = ir5[0];
                ir5[0] = (v811_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v808_data, 6))));
                float v814_data = r4[1];
                float v817_data = ir5[1];
                ir5[1] = (v817_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v814_data, 6))));
                float v820_data = r4[2];
                float v823_data = ir5[2];
                ir5[2] = (v823_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v820_data, 6))));
                float v826_data = r4[3];
                float v829_data = ir5[3];
                ir5[3] = (v829_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v826_data, 6))));
                float v832_data = r4[4];
                float v835_data = ir5[4];
                ir5[4] = (v835_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v832_data, 6))));
                float v838_data = r4[5];
                float v841_data = ir5[5];
                ir5[5] = (v841_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v838_data, 6))));
                float v844_data = r4[6];
                float v847_data = ir5[6];
                ir5[6] = (v847_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v844_data, 6))));
                float v850_data = r4[7];
                float v853_data = ir5[7];
                ir5[7] = (v853_data + (v807_data * (sycl::group_broadcast(item.get_sub_group(), v850_data, 6))));
              }
              if (v15_lead < 8) {
                float v859_data = r3[7];
                float v860_data = r4[0];
                float v863_data = ir5[0];
                ir5[0] = (v863_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v860_data, 7))));
                float v866_data = r4[1];
                float v869_data = ir5[1];
                ir5[1] = (v869_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v866_data, 7))));
                float v872_data = r4[2];
                float v875_data = ir5[2];
                ir5[2] = (v875_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v872_data, 7))));
                float v878_data = r4[3];
                float v881_data = ir5[3];
                ir5[3] = (v881_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v878_data, 7))));
                float v884_data = r4[4];
                float v887_data = ir5[4];
                ir5[4] = (v887_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v884_data, 7))));
                float v890_data = r4[5];
                float v893_data = ir5[5];
                ir5[5] = (v893_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v890_data, 7))));
                float v896_data = r4[6];
                float v899_data = ir5[6];
                ir5[6] = (v899_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v896_data, 7))));
                float v902_data = r4[7];
                float v905_data = ir5[7];
                ir5[7] = (v905_data + (v859_data * (sycl::group_broadcast(item.get_sub_group(), v902_data, 7))));
              }
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v911_n1 = 0; v911_n1 < 8; ++v911_n1) {
                  float v913_data = ir5[v911_n1];
                  float v915_data = r2[v911_n1];
                  r5[v911_n1] = (v915_data + v913_data);
                }
              }
              // s0 = store{r>s}(localShrMem0, r5);
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v922_i1 = 0; v922_i1 < 8; ++v922_i1) {
                  float v924_data = r5[v922_i1];
                  int32_t v931_a = v15_lead + (v922_i1 * 8);
                  s0[(v931_a ^ ((v931_a >> 5) & 31))] = v924_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m4 = abs(s0)
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v939_k1 = 0; v939_k1 < 8; ++v939_k1) {
                  int32_t v945_a = v939_k1 * 8;
                  int32_t v946_a = v15_lead + v945_a;
                  float v950_data = s0[(v946_a ^ ((v946_a >> 5) & 31))];
                  glb_m4[(v15_lead + v945_a)] = (sycl::fabs(v950_data));
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

