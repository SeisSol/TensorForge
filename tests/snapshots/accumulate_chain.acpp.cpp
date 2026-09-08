// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8a03a3cd0d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8a03a3cd0d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
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
              int32_t v14_lead = item.get_local_id(0) % 16;
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
                  float v24_data = glb_m1[(v14_lead + (v16_i1 * 12))];
                  r0[v16_i1] = v24_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
                  float v39_data = glb_m2[(v14_lead + (v31_i1 * 12))];
                  r1[v31_i1] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v46_i1 = 0; v46_i1 < 12; ++v46_i1) {
                  float v54_data = glb_m3[(v14_lead + (v46_i1 * 12))];
                  r3[v46_i1] = v54_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              if (v14_lead < 12) {
                float v62_data = r0[0];
                float v63_data = r1[0];
                float v66_data = ir2[0];
                ir2[0] = (v66_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
                float v69_data = r1[1];
                float v72_data = ir2[1];
                ir2[1] = (v72_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
                float v75_data = r1[2];
                float v78_data = ir2[2];
                ir2[2] = (v78_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
                float v81_data = r1[3];
                float v84_data = ir2[3];
                ir2[3] = (v84_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
                float v87_data = r1[4];
                float v90_data = ir2[4];
                ir2[4] = (v90_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
                float v93_data = r1[5];
                float v96_data = ir2[5];
                ir2[5] = (v96_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 0))));
                float v99_data = r1[6];
                float v102_data = ir2[6];
                ir2[6] = (v102_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v99_data, 0))));
                float v105_data = r1[7];
                float v108_data = ir2[7];
                ir2[7] = (v108_data + (v62_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 0))));
              }
              if (v14_lead < 12) {
                float v114_data = r0[1];
                float v115_data = r1[0];
                float v118_data = ir2[0];
                ir2[0] = (v118_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 1))));
                float v121_data = r1[1];
                float v124_data = ir2[1];
                ir2[1] = (v124_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 1))));
                float v127_data = r1[2];
                float v130_data = ir2[2];
                ir2[2] = (v130_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
                float v133_data = r1[3];
                float v136_data = ir2[3];
                ir2[3] = (v136_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 1))));
                float v139_data = r1[4];
                float v142_data = ir2[4];
                ir2[4] = (v142_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 1))));
                float v145_data = r1[5];
                float v148_data = ir2[5];
                ir2[5] = (v148_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 1))));
                float v151_data = r1[6];
                float v154_data = ir2[6];
                ir2[6] = (v154_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 1))));
                float v157_data = r1[7];
                float v160_data = ir2[7];
                ir2[7] = (v160_data + (v114_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 1))));
              }
              if (v14_lead < 12) {
                float v166_data = r0[2];
                float v167_data = r1[0];
                float v170_data = ir2[0];
                ir2[0] = (v170_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
                float v173_data = r1[1];
                float v176_data = ir2[1];
                ir2[1] = (v176_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 2))));
                float v179_data = r1[2];
                float v182_data = ir2[2];
                ir2[2] = (v182_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 2))));
                float v185_data = r1[3];
                float v188_data = ir2[3];
                ir2[3] = (v188_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 2))));
                float v191_data = r1[4];
                float v194_data = ir2[4];
                ir2[4] = (v194_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 2))));
                float v197_data = r1[5];
                float v200_data = ir2[5];
                ir2[5] = (v200_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 2))));
                float v203_data = r1[6];
                float v206_data = ir2[6];
                ir2[6] = (v206_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 2))));
                float v209_data = r1[7];
                float v212_data = ir2[7];
                ir2[7] = (v212_data + (v166_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 2))));
              }
              if (v14_lead < 12) {
                float v218_data = r0[3];
                float v219_data = r1[0];
                float v222_data = ir2[0];
                ir2[0] = (v222_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v219_data, 3))));
                float v225_data = r1[1];
                float v228_data = ir2[1];
                ir2[1] = (v228_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 3))));
                float v231_data = r1[2];
                float v234_data = ir2[2];
                ir2[2] = (v234_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 3))));
                float v237_data = r1[3];
                float v240_data = ir2[3];
                ir2[3] = (v240_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v237_data, 3))));
                float v243_data = r1[4];
                float v246_data = ir2[4];
                ir2[4] = (v246_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v243_data, 3))));
                float v249_data = r1[5];
                float v252_data = ir2[5];
                ir2[5] = (v252_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v249_data, 3))));
                float v255_data = r1[6];
                float v258_data = ir2[6];
                ir2[6] = (v258_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v255_data, 3))));
                float v261_data = r1[7];
                float v264_data = ir2[7];
                ir2[7] = (v264_data + (v218_data * (sycl::group_broadcast(item.get_sub_group(), v261_data, 3))));
              }
              if (v14_lead < 12) {
                float v270_data = r0[4];
                float v271_data = r1[0];
                float v274_data = ir2[0];
                ir2[0] = (v274_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v271_data, 4))));
                float v277_data = r1[1];
                float v280_data = ir2[1];
                ir2[1] = (v280_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v277_data, 4))));
                float v283_data = r1[2];
                float v286_data = ir2[2];
                ir2[2] = (v286_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v283_data, 4))));
                float v289_data = r1[3];
                float v292_data = ir2[3];
                ir2[3] = (v292_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v289_data, 4))));
                float v295_data = r1[4];
                float v298_data = ir2[4];
                ir2[4] = (v298_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v295_data, 4))));
                float v301_data = r1[5];
                float v304_data = ir2[5];
                ir2[5] = (v304_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v301_data, 4))));
                float v307_data = r1[6];
                float v310_data = ir2[6];
                ir2[6] = (v310_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 4))));
                float v313_data = r1[7];
                float v316_data = ir2[7];
                ir2[7] = (v316_data + (v270_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 4))));
              }
              if (v14_lead < 12) {
                float v322_data = r0[5];
                float v323_data = r1[0];
                float v326_data = ir2[0];
                ir2[0] = (v326_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 5))));
                float v329_data = r1[1];
                float v332_data = ir2[1];
                ir2[1] = (v332_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v329_data, 5))));
                float v335_data = r1[2];
                float v338_data = ir2[2];
                ir2[2] = (v338_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 5))));
                float v341_data = r1[3];
                float v344_data = ir2[3];
                ir2[3] = (v344_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v341_data, 5))));
                float v347_data = r1[4];
                float v350_data = ir2[4];
                ir2[4] = (v350_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v347_data, 5))));
                float v353_data = r1[5];
                float v356_data = ir2[5];
                ir2[5] = (v356_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v353_data, 5))));
                float v359_data = r1[6];
                float v362_data = ir2[6];
                ir2[6] = (v362_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v359_data, 5))));
                float v365_data = r1[7];
                float v368_data = ir2[7];
                ir2[7] = (v368_data + (v322_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 5))));
              }
              if (v14_lead < 12) {
                float v374_data = r0[6];
                float v375_data = r1[0];
                float v378_data = ir2[0];
                ir2[0] = (v378_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v375_data, 6))));
                float v381_data = r1[1];
                float v384_data = ir2[1];
                ir2[1] = (v384_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v381_data, 6))));
                float v387_data = r1[2];
                float v390_data = ir2[2];
                ir2[2] = (v390_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v387_data, 6))));
                float v393_data = r1[3];
                float v396_data = ir2[3];
                ir2[3] = (v396_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v393_data, 6))));
                float v399_data = r1[4];
                float v402_data = ir2[4];
                ir2[4] = (v402_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 6))));
                float v405_data = r1[5];
                float v408_data = ir2[5];
                ir2[5] = (v408_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v405_data, 6))));
                float v411_data = r1[6];
                float v414_data = ir2[6];
                ir2[6] = (v414_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v411_data, 6))));
                float v417_data = r1[7];
                float v420_data = ir2[7];
                ir2[7] = (v420_data + (v374_data * (sycl::group_broadcast(item.get_sub_group(), v417_data, 6))));
              }
              if (v14_lead < 12) {
                float v426_data = r0[7];
                float v427_data = r1[0];
                float v430_data = ir2[0];
                ir2[0] = (v430_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 7))));
                float v433_data = r1[1];
                float v436_data = ir2[1];
                ir2[1] = (v436_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 7))));
                float v439_data = r1[2];
                float v442_data = ir2[2];
                ir2[2] = (v442_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v439_data, 7))));
                float v445_data = r1[3];
                float v448_data = ir2[3];
                ir2[3] = (v448_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v445_data, 7))));
                float v451_data = r1[4];
                float v454_data = ir2[4];
                ir2[4] = (v454_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v451_data, 7))));
                float v457_data = r1[5];
                float v460_data = ir2[5];
                ir2[5] = (v460_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v457_data, 7))));
                float v463_data = r1[6];
                float v466_data = ir2[6];
                ir2[6] = (v466_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v463_data, 7))));
                float v469_data = r1[7];
                float v472_data = ir2[7];
                ir2[7] = (v472_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v469_data, 7))));
              }
              if (v14_lead < 12) {
                float v478_data = r0[8];
                float v479_data = r1[0];
                float v482_data = ir2[0];
                ir2[0] = (v482_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v479_data, 8))));
                float v485_data = r1[1];
                float v488_data = ir2[1];
                ir2[1] = (v488_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v485_data, 8))));
                float v491_data = r1[2];
                float v494_data = ir2[2];
                ir2[2] = (v494_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v491_data, 8))));
                float v497_data = r1[3];
                float v500_data = ir2[3];
                ir2[3] = (v500_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v497_data, 8))));
                float v503_data = r1[4];
                float v506_data = ir2[4];
                ir2[4] = (v506_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v503_data, 8))));
                float v509_data = r1[5];
                float v512_data = ir2[5];
                ir2[5] = (v512_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v509_data, 8))));
                float v515_data = r1[6];
                float v518_data = ir2[6];
                ir2[6] = (v518_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v515_data, 8))));
                float v521_data = r1[7];
                float v524_data = ir2[7];
                ir2[7] = (v524_data + (v478_data * (sycl::group_broadcast(item.get_sub_group(), v521_data, 8))));
              }
              if (v14_lead < 12) {
                float v530_data = r0[9];
                float v531_data = r1[0];
                float v534_data = ir2[0];
                ir2[0] = (v534_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v531_data, 9))));
                float v537_data = r1[1];
                float v540_data = ir2[1];
                ir2[1] = (v540_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v537_data, 9))));
                float v543_data = r1[2];
                float v546_data = ir2[2];
                ir2[2] = (v546_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v543_data, 9))));
                float v549_data = r1[3];
                float v552_data = ir2[3];
                ir2[3] = (v552_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v549_data, 9))));
                float v555_data = r1[4];
                float v558_data = ir2[4];
                ir2[4] = (v558_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v555_data, 9))));
                float v561_data = r1[5];
                float v564_data = ir2[5];
                ir2[5] = (v564_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v561_data, 9))));
                float v567_data = r1[6];
                float v570_data = ir2[6];
                ir2[6] = (v570_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v567_data, 9))));
                float v573_data = r1[7];
                float v576_data = ir2[7];
                ir2[7] = (v576_data + (v530_data * (sycl::group_broadcast(item.get_sub_group(), v573_data, 9))));
              }
              if (v14_lead < 12) {
                float v582_data = r0[10];
                float v583_data = r1[0];
                float v586_data = ir2[0];
                ir2[0] = (v586_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v583_data, 10))));
                float v589_data = r1[1];
                float v592_data = ir2[1];
                ir2[1] = (v592_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v589_data, 10))));
                float v595_data = r1[2];
                float v598_data = ir2[2];
                ir2[2] = (v598_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v595_data, 10))));
                float v601_data = r1[3];
                float v604_data = ir2[3];
                ir2[3] = (v604_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v601_data, 10))));
                float v607_data = r1[4];
                float v610_data = ir2[4];
                ir2[4] = (v610_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v607_data, 10))));
                float v613_data = r1[5];
                float v616_data = ir2[5];
                ir2[5] = (v616_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v613_data, 10))));
                float v619_data = r1[6];
                float v622_data = ir2[6];
                ir2[6] = (v622_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v619_data, 10))));
                float v625_data = r1[7];
                float v628_data = ir2[7];
                ir2[7] = (v628_data + (v582_data * (sycl::group_broadcast(item.get_sub_group(), v625_data, 10))));
              }
              if (v14_lead < 12) {
                float v634_data = r0[11];
                float v635_data = r1[0];
                float v638_data = ir2[0];
                ir2[0] = (v638_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v635_data, 11))));
                float v641_data = r1[1];
                float v644_data = ir2[1];
                ir2[1] = (v644_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v641_data, 11))));
                float v647_data = r1[2];
                float v650_data = ir2[2];
                ir2[2] = (v650_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v647_data, 11))));
                float v653_data = r1[3];
                float v656_data = ir2[3];
                ir2[3] = (v656_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v653_data, 11))));
                float v659_data = r1[4];
                float v662_data = ir2[4];
                ir2[4] = (v662_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v659_data, 11))));
                float v665_data = r1[5];
                float v668_data = ir2[5];
                ir2[5] = (v668_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v665_data, 11))));
                float v671_data = r1[6];
                float v674_data = ir2[6];
                ir2[6] = (v674_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v671_data, 11))));
                float v677_data = r1[7];
                float v680_data = ir2[7];
                ir2[7] = (v680_data + (v634_data * (sycl::group_broadcast(item.get_sub_group(), v677_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v686_n1 = 0; v686_n1 < 8; ++v686_n1) {
                  float v688_data = ir2[v686_n1];
                  r2[v686_n1] = v688_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v695_i1 = 0; v695_i1 < 8; ++v695_i1) {
                  float v703_data = glb_m4[(v14_lead + (v695_i1 * 12))];
                  r4[v695_i1] = v703_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v710_i1 = 0; v710_i1 < 12; ++v710_i1) {
                  float v718_data = glb_m5[(v14_lead + (v710_i1 * 12))];
                  r6[v710_i1] = v718_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              if (v14_lead < 12) {
                float v726_data = r3[0];
                float v727_data = r4[0];
                float v730_data = ir5[0];
                ir5[0] = (v730_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v727_data, 0))));
                float v733_data = r4[1];
                float v736_data = ir5[1];
                ir5[1] = (v736_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v733_data, 0))));
                float v739_data = r4[2];
                float v742_data = ir5[2];
                ir5[2] = (v742_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v739_data, 0))));
                float v745_data = r4[3];
                float v748_data = ir5[3];
                ir5[3] = (v748_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v745_data, 0))));
                float v751_data = r4[4];
                float v754_data = ir5[4];
                ir5[4] = (v754_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v751_data, 0))));
                float v757_data = r4[5];
                float v760_data = ir5[5];
                ir5[5] = (v760_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v757_data, 0))));
                float v763_data = r4[6];
                float v766_data = ir5[6];
                ir5[6] = (v766_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v763_data, 0))));
                float v769_data = r4[7];
                float v772_data = ir5[7];
                ir5[7] = (v772_data + (v726_data * (sycl::group_broadcast(item.get_sub_group(), v769_data, 0))));
              }
              if (v14_lead < 12) {
                float v778_data = r3[1];
                float v779_data = r4[0];
                float v782_data = ir5[0];
                ir5[0] = (v782_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v779_data, 1))));
                float v785_data = r4[1];
                float v788_data = ir5[1];
                ir5[1] = (v788_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v785_data, 1))));
                float v791_data = r4[2];
                float v794_data = ir5[2];
                ir5[2] = (v794_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v791_data, 1))));
                float v797_data = r4[3];
                float v800_data = ir5[3];
                ir5[3] = (v800_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v797_data, 1))));
                float v803_data = r4[4];
                float v806_data = ir5[4];
                ir5[4] = (v806_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v803_data, 1))));
                float v809_data = r4[5];
                float v812_data = ir5[5];
                ir5[5] = (v812_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v809_data, 1))));
                float v815_data = r4[6];
                float v818_data = ir5[6];
                ir5[6] = (v818_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v815_data, 1))));
                float v821_data = r4[7];
                float v824_data = ir5[7];
                ir5[7] = (v824_data + (v778_data * (sycl::group_broadcast(item.get_sub_group(), v821_data, 1))));
              }
              if (v14_lead < 12) {
                float v830_data = r3[2];
                float v831_data = r4[0];
                float v834_data = ir5[0];
                ir5[0] = (v834_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v831_data, 2))));
                float v837_data = r4[1];
                float v840_data = ir5[1];
                ir5[1] = (v840_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v837_data, 2))));
                float v843_data = r4[2];
                float v846_data = ir5[2];
                ir5[2] = (v846_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v843_data, 2))));
                float v849_data = r4[3];
                float v852_data = ir5[3];
                ir5[3] = (v852_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v849_data, 2))));
                float v855_data = r4[4];
                float v858_data = ir5[4];
                ir5[4] = (v858_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v855_data, 2))));
                float v861_data = r4[5];
                float v864_data = ir5[5];
                ir5[5] = (v864_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v861_data, 2))));
                float v867_data = r4[6];
                float v870_data = ir5[6];
                ir5[6] = (v870_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v867_data, 2))));
                float v873_data = r4[7];
                float v876_data = ir5[7];
                ir5[7] = (v876_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v873_data, 2))));
              }
              if (v14_lead < 12) {
                float v882_data = r3[3];
                float v883_data = r4[0];
                float v886_data = ir5[0];
                ir5[0] = (v886_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v883_data, 3))));
                float v889_data = r4[1];
                float v892_data = ir5[1];
                ir5[1] = (v892_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v889_data, 3))));
                float v895_data = r4[2];
                float v898_data = ir5[2];
                ir5[2] = (v898_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v895_data, 3))));
                float v901_data = r4[3];
                float v904_data = ir5[3];
                ir5[3] = (v904_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v901_data, 3))));
                float v907_data = r4[4];
                float v910_data = ir5[4];
                ir5[4] = (v910_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v907_data, 3))));
                float v913_data = r4[5];
                float v916_data = ir5[5];
                ir5[5] = (v916_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v913_data, 3))));
                float v919_data = r4[6];
                float v922_data = ir5[6];
                ir5[6] = (v922_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v919_data, 3))));
                float v925_data = r4[7];
                float v928_data = ir5[7];
                ir5[7] = (v928_data + (v882_data * (sycl::group_broadcast(item.get_sub_group(), v925_data, 3))));
              }
              if (v14_lead < 12) {
                float v934_data = r3[4];
                float v935_data = r4[0];
                float v938_data = ir5[0];
                ir5[0] = (v938_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v935_data, 4))));
                float v941_data = r4[1];
                float v944_data = ir5[1];
                ir5[1] = (v944_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v941_data, 4))));
                float v947_data = r4[2];
                float v950_data = ir5[2];
                ir5[2] = (v950_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v947_data, 4))));
                float v953_data = r4[3];
                float v956_data = ir5[3];
                ir5[3] = (v956_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v953_data, 4))));
                float v959_data = r4[4];
                float v962_data = ir5[4];
                ir5[4] = (v962_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v959_data, 4))));
                float v965_data = r4[5];
                float v968_data = ir5[5];
                ir5[5] = (v968_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v965_data, 4))));
                float v971_data = r4[6];
                float v974_data = ir5[6];
                ir5[6] = (v974_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v971_data, 4))));
                float v977_data = r4[7];
                float v980_data = ir5[7];
                ir5[7] = (v980_data + (v934_data * (sycl::group_broadcast(item.get_sub_group(), v977_data, 4))));
              }
              if (v14_lead < 12) {
                float v986_data = r3[5];
                float v987_data = r4[0];
                float v990_data = ir5[0];
                ir5[0] = (v990_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v987_data, 5))));
                float v993_data = r4[1];
                float v996_data = ir5[1];
                ir5[1] = (v996_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v993_data, 5))));
                float v999_data = r4[2];
                float v1002_data = ir5[2];
                ir5[2] = (v1002_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v999_data, 5))));
                float v1005_data = r4[3];
                float v1008_data = ir5[3];
                ir5[3] = (v1008_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v1005_data, 5))));
                float v1011_data = r4[4];
                float v1014_data = ir5[4];
                ir5[4] = (v1014_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v1011_data, 5))));
                float v1017_data = r4[5];
                float v1020_data = ir5[5];
                ir5[5] = (v1020_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v1017_data, 5))));
                float v1023_data = r4[6];
                float v1026_data = ir5[6];
                ir5[6] = (v1026_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v1023_data, 5))));
                float v1029_data = r4[7];
                float v1032_data = ir5[7];
                ir5[7] = (v1032_data + (v986_data * (sycl::group_broadcast(item.get_sub_group(), v1029_data, 5))));
              }
              if (v14_lead < 12) {
                float v1038_data = r3[6];
                float v1039_data = r4[0];
                float v1042_data = ir5[0];
                ir5[0] = (v1042_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1039_data, 6))));
                float v1045_data = r4[1];
                float v1048_data = ir5[1];
                ir5[1] = (v1048_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1045_data, 6))));
                float v1051_data = r4[2];
                float v1054_data = ir5[2];
                ir5[2] = (v1054_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1051_data, 6))));
                float v1057_data = r4[3];
                float v1060_data = ir5[3];
                ir5[3] = (v1060_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1057_data, 6))));
                float v1063_data = r4[4];
                float v1066_data = ir5[4];
                ir5[4] = (v1066_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1063_data, 6))));
                float v1069_data = r4[5];
                float v1072_data = ir5[5];
                ir5[5] = (v1072_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1069_data, 6))));
                float v1075_data = r4[6];
                float v1078_data = ir5[6];
                ir5[6] = (v1078_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1075_data, 6))));
                float v1081_data = r4[7];
                float v1084_data = ir5[7];
                ir5[7] = (v1084_data + (v1038_data * (sycl::group_broadcast(item.get_sub_group(), v1081_data, 6))));
              }
              if (v14_lead < 12) {
                float v1090_data = r3[7];
                float v1091_data = r4[0];
                float v1094_data = ir5[0];
                ir5[0] = (v1094_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1091_data, 7))));
                float v1097_data = r4[1];
                float v1100_data = ir5[1];
                ir5[1] = (v1100_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1097_data, 7))));
                float v1103_data = r4[2];
                float v1106_data = ir5[2];
                ir5[2] = (v1106_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1103_data, 7))));
                float v1109_data = r4[3];
                float v1112_data = ir5[3];
                ir5[3] = (v1112_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1109_data, 7))));
                float v1115_data = r4[4];
                float v1118_data = ir5[4];
                ir5[4] = (v1118_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1115_data, 7))));
                float v1121_data = r4[5];
                float v1124_data = ir5[5];
                ir5[5] = (v1124_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1121_data, 7))));
                float v1127_data = r4[6];
                float v1130_data = ir5[6];
                ir5[6] = (v1130_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1127_data, 7))));
                float v1133_data = r4[7];
                float v1136_data = ir5[7];
                ir5[7] = (v1136_data + (v1090_data * (sycl::group_broadcast(item.get_sub_group(), v1133_data, 7))));
              }
              if (v14_lead < 12) {
                float v1142_data = r3[8];
                float v1143_data = r4[0];
                float v1146_data = ir5[0];
                ir5[0] = (v1146_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1143_data, 8))));
                float v1149_data = r4[1];
                float v1152_data = ir5[1];
                ir5[1] = (v1152_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 8))));
                float v1155_data = r4[2];
                float v1158_data = ir5[2];
                ir5[2] = (v1158_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 8))));
                float v1161_data = r4[3];
                float v1164_data = ir5[3];
                ir5[3] = (v1164_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 8))));
                float v1167_data = r4[4];
                float v1170_data = ir5[4];
                ir5[4] = (v1170_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 8))));
                float v1173_data = r4[5];
                float v1176_data = ir5[5];
                ir5[5] = (v1176_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 8))));
                float v1179_data = r4[6];
                float v1182_data = ir5[6];
                ir5[6] = (v1182_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 8))));
                float v1185_data = r4[7];
                float v1188_data = ir5[7];
                ir5[7] = (v1188_data + (v1142_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 8))));
              }
              if (v14_lead < 12) {
                float v1194_data = r3[9];
                float v1195_data = r4[0];
                float v1198_data = ir5[0];
                ir5[0] = (v1198_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 9))));
                float v1201_data = r4[1];
                float v1204_data = ir5[1];
                ir5[1] = (v1204_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 9))));
                float v1207_data = r4[2];
                float v1210_data = ir5[2];
                ir5[2] = (v1210_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 9))));
                float v1213_data = r4[3];
                float v1216_data = ir5[3];
                ir5[3] = (v1216_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 9))));
                float v1219_data = r4[4];
                float v1222_data = ir5[4];
                ir5[4] = (v1222_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 9))));
                float v1225_data = r4[5];
                float v1228_data = ir5[5];
                ir5[5] = (v1228_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 9))));
                float v1231_data = r4[6];
                float v1234_data = ir5[6];
                ir5[6] = (v1234_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1231_data, 9))));
                float v1237_data = r4[7];
                float v1240_data = ir5[7];
                ir5[7] = (v1240_data + (v1194_data * (sycl::group_broadcast(item.get_sub_group(), v1237_data, 9))));
              }
              if (v14_lead < 12) {
                float v1246_data = r3[10];
                float v1247_data = r4[0];
                float v1250_data = ir5[0];
                ir5[0] = (v1250_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1247_data, 10))));
                float v1253_data = r4[1];
                float v1256_data = ir5[1];
                ir5[1] = (v1256_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1253_data, 10))));
                float v1259_data = r4[2];
                float v1262_data = ir5[2];
                ir5[2] = (v1262_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1259_data, 10))));
                float v1265_data = r4[3];
                float v1268_data = ir5[3];
                ir5[3] = (v1268_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1265_data, 10))));
                float v1271_data = r4[4];
                float v1274_data = ir5[4];
                ir5[4] = (v1274_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1271_data, 10))));
                float v1277_data = r4[5];
                float v1280_data = ir5[5];
                ir5[5] = (v1280_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1277_data, 10))));
                float v1283_data = r4[6];
                float v1286_data = ir5[6];
                ir5[6] = (v1286_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1283_data, 10))));
                float v1289_data = r4[7];
                float v1292_data = ir5[7];
                ir5[7] = (v1292_data + (v1246_data * (sycl::group_broadcast(item.get_sub_group(), v1289_data, 10))));
              }
              if (v14_lead < 12) {
                float v1298_data = r3[11];
                float v1299_data = r4[0];
                float v1302_data = ir5[0];
                ir5[0] = (v1302_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1299_data, 11))));
                float v1305_data = r4[1];
                float v1308_data = ir5[1];
                ir5[1] = (v1308_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1305_data, 11))));
                float v1311_data = r4[2];
                float v1314_data = ir5[2];
                ir5[2] = (v1314_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1311_data, 11))));
                float v1317_data = r4[3];
                float v1320_data = ir5[3];
                ir5[3] = (v1320_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1317_data, 11))));
                float v1323_data = r4[4];
                float v1326_data = ir5[4];
                ir5[4] = (v1326_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 11))));
                float v1329_data = r4[5];
                float v1332_data = ir5[5];
                ir5[5] = (v1332_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 11))));
                float v1335_data = r4[6];
                float v1338_data = ir5[6];
                ir5[6] = (v1338_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 11))));
                float v1341_data = r4[7];
                float v1344_data = ir5[7];
                ir5[7] = (v1344_data + (v1298_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1350_n1 = 0; v1350_n1 < 8; ++v1350_n1) {
                  float v1352_data = ir5[v1350_n1];
                  float v1354_data = r2[v1350_n1];
                  r5[v1350_n1] = (v1354_data + v1352_data);
                }
              }
              float r7[8]{};
              // r7 = load{g>r}(glb_m6);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1362_i1 = 0; v1362_i1 < 8; ++v1362_i1) {
                  float v1370_data = glb_m6[(v14_lead + (v1362_i1 * 12))];
                  r7[v1362_i1] = v1370_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m5););
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1377_i1 = 0; v1377_i1 < 12; ++v1377_i1) {
                  float v1385_data = glb_m7[(v14_lead + (v1377_i1 * 12))];
                  r9[v1377_i1] = v1385_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m6););
              float r8[8]{};
              // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              if (v14_lead < 12) {
                float v1393_data = r6[0];
                float v1394_data = r7[0];
                float v1397_data = ir8[0];
                ir8[0] = (v1397_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1394_data, 0))));
                float v1400_data = r7[1];
                float v1403_data = ir8[1];
                ir8[1] = (v1403_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1400_data, 0))));
                float v1406_data = r7[2];
                float v1409_data = ir8[2];
                ir8[2] = (v1409_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1406_data, 0))));
                float v1412_data = r7[3];
                float v1415_data = ir8[3];
                ir8[3] = (v1415_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1412_data, 0))));
                float v1418_data = r7[4];
                float v1421_data = ir8[4];
                ir8[4] = (v1421_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1418_data, 0))));
                float v1424_data = r7[5];
                float v1427_data = ir8[5];
                ir8[5] = (v1427_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1424_data, 0))));
                float v1430_data = r7[6];
                float v1433_data = ir8[6];
                ir8[6] = (v1433_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1430_data, 0))));
                float v1436_data = r7[7];
                float v1439_data = ir8[7];
                ir8[7] = (v1439_data + (v1393_data * (sycl::group_broadcast(item.get_sub_group(), v1436_data, 0))));
              }
              if (v14_lead < 12) {
                float v1445_data = r6[1];
                float v1446_data = r7[0];
                float v1449_data = ir8[0];
                ir8[0] = (v1449_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 1))));
                float v1452_data = r7[1];
                float v1455_data = ir8[1];
                ir8[1] = (v1455_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1452_data, 1))));
                float v1458_data = r7[2];
                float v1461_data = ir8[2];
                ir8[2] = (v1461_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1458_data, 1))));
                float v1464_data = r7[3];
                float v1467_data = ir8[3];
                ir8[3] = (v1467_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1464_data, 1))));
                float v1470_data = r7[4];
                float v1473_data = ir8[4];
                ir8[4] = (v1473_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1470_data, 1))));
                float v1476_data = r7[5];
                float v1479_data = ir8[5];
                ir8[5] = (v1479_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1476_data, 1))));
                float v1482_data = r7[6];
                float v1485_data = ir8[6];
                ir8[6] = (v1485_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1482_data, 1))));
                float v1488_data = r7[7];
                float v1491_data = ir8[7];
                ir8[7] = (v1491_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1488_data, 1))));
              }
              if (v14_lead < 12) {
                float v1497_data = r6[2];
                float v1498_data = r7[0];
                float v1501_data = ir8[0];
                ir8[0] = (v1501_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1498_data, 2))));
                float v1504_data = r7[1];
                float v1507_data = ir8[1];
                ir8[1] = (v1507_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1504_data, 2))));
                float v1510_data = r7[2];
                float v1513_data = ir8[2];
                ir8[2] = (v1513_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1510_data, 2))));
                float v1516_data = r7[3];
                float v1519_data = ir8[3];
                ir8[3] = (v1519_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1516_data, 2))));
                float v1522_data = r7[4];
                float v1525_data = ir8[4];
                ir8[4] = (v1525_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1522_data, 2))));
                float v1528_data = r7[5];
                float v1531_data = ir8[5];
                ir8[5] = (v1531_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1528_data, 2))));
                float v1534_data = r7[6];
                float v1537_data = ir8[6];
                ir8[6] = (v1537_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1534_data, 2))));
                float v1540_data = r7[7];
                float v1543_data = ir8[7];
                ir8[7] = (v1543_data + (v1497_data * (sycl::group_broadcast(item.get_sub_group(), v1540_data, 2))));
              }
              if (v14_lead < 12) {
                float v1549_data = r6[3];
                float v1550_data = r7[0];
                float v1553_data = ir8[0];
                ir8[0] = (v1553_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1550_data, 3))));
                float v1556_data = r7[1];
                float v1559_data = ir8[1];
                ir8[1] = (v1559_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1556_data, 3))));
                float v1562_data = r7[2];
                float v1565_data = ir8[2];
                ir8[2] = (v1565_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1562_data, 3))));
                float v1568_data = r7[3];
                float v1571_data = ir8[3];
                ir8[3] = (v1571_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1568_data, 3))));
                float v1574_data = r7[4];
                float v1577_data = ir8[4];
                ir8[4] = (v1577_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1574_data, 3))));
                float v1580_data = r7[5];
                float v1583_data = ir8[5];
                ir8[5] = (v1583_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1580_data, 3))));
                float v1586_data = r7[6];
                float v1589_data = ir8[6];
                ir8[6] = (v1589_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1586_data, 3))));
                float v1592_data = r7[7];
                float v1595_data = ir8[7];
                ir8[7] = (v1595_data + (v1549_data * (sycl::group_broadcast(item.get_sub_group(), v1592_data, 3))));
              }
              if (v14_lead < 12) {
                float v1601_data = r6[4];
                float v1602_data = r7[0];
                float v1605_data = ir8[0];
                ir8[0] = (v1605_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1602_data, 4))));
                float v1608_data = r7[1];
                float v1611_data = ir8[1];
                ir8[1] = (v1611_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1608_data, 4))));
                float v1614_data = r7[2];
                float v1617_data = ir8[2];
                ir8[2] = (v1617_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1614_data, 4))));
                float v1620_data = r7[3];
                float v1623_data = ir8[3];
                ir8[3] = (v1623_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1620_data, 4))));
                float v1626_data = r7[4];
                float v1629_data = ir8[4];
                ir8[4] = (v1629_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1626_data, 4))));
                float v1632_data = r7[5];
                float v1635_data = ir8[5];
                ir8[5] = (v1635_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1632_data, 4))));
                float v1638_data = r7[6];
                float v1641_data = ir8[6];
                ir8[6] = (v1641_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1638_data, 4))));
                float v1644_data = r7[7];
                float v1647_data = ir8[7];
                ir8[7] = (v1647_data + (v1601_data * (sycl::group_broadcast(item.get_sub_group(), v1644_data, 4))));
              }
              if (v14_lead < 12) {
                float v1653_data = r6[5];
                float v1654_data = r7[0];
                float v1657_data = ir8[0];
                ir8[0] = (v1657_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1654_data, 5))));
                float v1660_data = r7[1];
                float v1663_data = ir8[1];
                ir8[1] = (v1663_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1660_data, 5))));
                float v1666_data = r7[2];
                float v1669_data = ir8[2];
                ir8[2] = (v1669_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1666_data, 5))));
                float v1672_data = r7[3];
                float v1675_data = ir8[3];
                ir8[3] = (v1675_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1672_data, 5))));
                float v1678_data = r7[4];
                float v1681_data = ir8[4];
                ir8[4] = (v1681_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1678_data, 5))));
                float v1684_data = r7[5];
                float v1687_data = ir8[5];
                ir8[5] = (v1687_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1684_data, 5))));
                float v1690_data = r7[6];
                float v1693_data = ir8[6];
                ir8[6] = (v1693_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1690_data, 5))));
                float v1696_data = r7[7];
                float v1699_data = ir8[7];
                ir8[7] = (v1699_data + (v1653_data * (sycl::group_broadcast(item.get_sub_group(), v1696_data, 5))));
              }
              if (v14_lead < 12) {
                float v1705_data = r6[6];
                float v1706_data = r7[0];
                float v1709_data = ir8[0];
                ir8[0] = (v1709_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1706_data, 6))));
                float v1712_data = r7[1];
                float v1715_data = ir8[1];
                ir8[1] = (v1715_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1712_data, 6))));
                float v1718_data = r7[2];
                float v1721_data = ir8[2];
                ir8[2] = (v1721_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1718_data, 6))));
                float v1724_data = r7[3];
                float v1727_data = ir8[3];
                ir8[3] = (v1727_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1724_data, 6))));
                float v1730_data = r7[4];
                float v1733_data = ir8[4];
                ir8[4] = (v1733_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1730_data, 6))));
                float v1736_data = r7[5];
                float v1739_data = ir8[5];
                ir8[5] = (v1739_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1736_data, 6))));
                float v1742_data = r7[6];
                float v1745_data = ir8[6];
                ir8[6] = (v1745_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 6))));
                float v1748_data = r7[7];
                float v1751_data = ir8[7];
                ir8[7] = (v1751_data + (v1705_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 6))));
              }
              if (v14_lead < 12) {
                float v1757_data = r6[7];
                float v1758_data = r7[0];
                float v1761_data = ir8[0];
                ir8[0] = (v1761_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1758_data, 7))));
                float v1764_data = r7[1];
                float v1767_data = ir8[1];
                ir8[1] = (v1767_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1764_data, 7))));
                float v1770_data = r7[2];
                float v1773_data = ir8[2];
                ir8[2] = (v1773_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1770_data, 7))));
                float v1776_data = r7[3];
                float v1779_data = ir8[3];
                ir8[3] = (v1779_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1776_data, 7))));
                float v1782_data = r7[4];
                float v1785_data = ir8[4];
                ir8[4] = (v1785_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1782_data, 7))));
                float v1788_data = r7[5];
                float v1791_data = ir8[5];
                ir8[5] = (v1791_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1788_data, 7))));
                float v1794_data = r7[6];
                float v1797_data = ir8[6];
                ir8[6] = (v1797_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1794_data, 7))));
                float v1800_data = r7[7];
                float v1803_data = ir8[7];
                ir8[7] = (v1803_data + (v1757_data * (sycl::group_broadcast(item.get_sub_group(), v1800_data, 7))));
              }
              if (v14_lead < 12) {
                float v1809_data = r6[8];
                float v1810_data = r7[0];
                float v1813_data = ir8[0];
                ir8[0] = (v1813_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1810_data, 8))));
                float v1816_data = r7[1];
                float v1819_data = ir8[1];
                ir8[1] = (v1819_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1816_data, 8))));
                float v1822_data = r7[2];
                float v1825_data = ir8[2];
                ir8[2] = (v1825_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1822_data, 8))));
                float v1828_data = r7[3];
                float v1831_data = ir8[3];
                ir8[3] = (v1831_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1828_data, 8))));
                float v1834_data = r7[4];
                float v1837_data = ir8[4];
                ir8[4] = (v1837_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1834_data, 8))));
                float v1840_data = r7[5];
                float v1843_data = ir8[5];
                ir8[5] = (v1843_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1840_data, 8))));
                float v1846_data = r7[6];
                float v1849_data = ir8[6];
                ir8[6] = (v1849_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 8))));
                float v1852_data = r7[7];
                float v1855_data = ir8[7];
                ir8[7] = (v1855_data + (v1809_data * (sycl::group_broadcast(item.get_sub_group(), v1852_data, 8))));
              }
              if (v14_lead < 12) {
                float v1861_data = r6[9];
                float v1862_data = r7[0];
                float v1865_data = ir8[0];
                ir8[0] = (v1865_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1862_data, 9))));
                float v1868_data = r7[1];
                float v1871_data = ir8[1];
                ir8[1] = (v1871_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1868_data, 9))));
                float v1874_data = r7[2];
                float v1877_data = ir8[2];
                ir8[2] = (v1877_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1874_data, 9))));
                float v1880_data = r7[3];
                float v1883_data = ir8[3];
                ir8[3] = (v1883_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1880_data, 9))));
                float v1886_data = r7[4];
                float v1889_data = ir8[4];
                ir8[4] = (v1889_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1886_data, 9))));
                float v1892_data = r7[5];
                float v1895_data = ir8[5];
                ir8[5] = (v1895_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1892_data, 9))));
                float v1898_data = r7[6];
                float v1901_data = ir8[6];
                ir8[6] = (v1901_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1898_data, 9))));
                float v1904_data = r7[7];
                float v1907_data = ir8[7];
                ir8[7] = (v1907_data + (v1861_data * (sycl::group_broadcast(item.get_sub_group(), v1904_data, 9))));
              }
              if (v14_lead < 12) {
                float v1913_data = r6[10];
                float v1914_data = r7[0];
                float v1917_data = ir8[0];
                ir8[0] = (v1917_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1914_data, 10))));
                float v1920_data = r7[1];
                float v1923_data = ir8[1];
                ir8[1] = (v1923_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1920_data, 10))));
                float v1926_data = r7[2];
                float v1929_data = ir8[2];
                ir8[2] = (v1929_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1926_data, 10))));
                float v1932_data = r7[3];
                float v1935_data = ir8[3];
                ir8[3] = (v1935_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1932_data, 10))));
                float v1938_data = r7[4];
                float v1941_data = ir8[4];
                ir8[4] = (v1941_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1938_data, 10))));
                float v1944_data = r7[5];
                float v1947_data = ir8[5];
                ir8[5] = (v1947_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1944_data, 10))));
                float v1950_data = r7[6];
                float v1953_data = ir8[6];
                ir8[6] = (v1953_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1950_data, 10))));
                float v1956_data = r7[7];
                float v1959_data = ir8[7];
                ir8[7] = (v1959_data + (v1913_data * (sycl::group_broadcast(item.get_sub_group(), v1956_data, 10))));
              }
              if (v14_lead < 12) {
                float v1965_data = r6[11];
                float v1966_data = r7[0];
                float v1969_data = ir8[0];
                ir8[0] = (v1969_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v1966_data, 11))));
                float v1972_data = r7[1];
                float v1975_data = ir8[1];
                ir8[1] = (v1975_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v1972_data, 11))));
                float v1978_data = r7[2];
                float v1981_data = ir8[2];
                ir8[2] = (v1981_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v1978_data, 11))));
                float v1984_data = r7[3];
                float v1987_data = ir8[3];
                ir8[3] = (v1987_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v1984_data, 11))));
                float v1990_data = r7[4];
                float v1993_data = ir8[4];
                ir8[4] = (v1993_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v1990_data, 11))));
                float v1996_data = r7[5];
                float v1999_data = ir8[5];
                ir8[5] = (v1999_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v1996_data, 11))));
                float v2002_data = r7[6];
                float v2005_data = ir8[6];
                ir8[6] = (v2005_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v2002_data, 11))));
                float v2008_data = r7[7];
                float v2011_data = ir8[7];
                ir8[7] = (v2011_data + (v1965_data * (sycl::group_broadcast(item.get_sub_group(), v2008_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2017_n1 = 0; v2017_n1 < 8; ++v2017_n1) {
                  float v2019_data = ir8[v2017_n1];
                  float v2021_data = r5[v2017_n1];
                  r8[v2017_n1] = (v2021_data + v2019_data);
                }
              }
              float r10[8]{};
              // r10 = load{g>r}(glb_m8);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2029_i1 = 0; v2029_i1 < 8; ++v2029_i1) {
                  float v2037_data = glb_m8[(v14_lead + (v2029_i1 * 12))];
                  r10[v2029_i1] = v2037_data;
                }
              }
              // wait(r9 = load{g>r}(glb_m7););
              // wait(r10 = load{g>r}(glb_m8););
              float r11[8]{};
              // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir11[8]{};
              if (v14_lead < 12) {
                float v2045_data = r9[0];
                float v2046_data = r10[0];
                float v2049_data = ir11[0];
                ir11[0] = (v2049_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2046_data, 0))));
                float v2052_data = r10[1];
                float v2055_data = ir11[1];
                ir11[1] = (v2055_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2052_data, 0))));
                float v2058_data = r10[2];
                float v2061_data = ir11[2];
                ir11[2] = (v2061_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2058_data, 0))));
                float v2064_data = r10[3];
                float v2067_data = ir11[3];
                ir11[3] = (v2067_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2064_data, 0))));
                float v2070_data = r10[4];
                float v2073_data = ir11[4];
                ir11[4] = (v2073_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2070_data, 0))));
                float v2076_data = r10[5];
                float v2079_data = ir11[5];
                ir11[5] = (v2079_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2076_data, 0))));
                float v2082_data = r10[6];
                float v2085_data = ir11[6];
                ir11[6] = (v2085_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2082_data, 0))));
                float v2088_data = r10[7];
                float v2091_data = ir11[7];
                ir11[7] = (v2091_data + (v2045_data * (sycl::group_broadcast(item.get_sub_group(), v2088_data, 0))));
              }
              if (v14_lead < 12) {
                float v2097_data = r9[1];
                float v2098_data = r10[0];
                float v2101_data = ir11[0];
                ir11[0] = (v2101_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2098_data, 1))));
                float v2104_data = r10[1];
                float v2107_data = ir11[1];
                ir11[1] = (v2107_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2104_data, 1))));
                float v2110_data = r10[2];
                float v2113_data = ir11[2];
                ir11[2] = (v2113_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2110_data, 1))));
                float v2116_data = r10[3];
                float v2119_data = ir11[3];
                ir11[3] = (v2119_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2116_data, 1))));
                float v2122_data = r10[4];
                float v2125_data = ir11[4];
                ir11[4] = (v2125_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2122_data, 1))));
                float v2128_data = r10[5];
                float v2131_data = ir11[5];
                ir11[5] = (v2131_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2128_data, 1))));
                float v2134_data = r10[6];
                float v2137_data = ir11[6];
                ir11[6] = (v2137_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2134_data, 1))));
                float v2140_data = r10[7];
                float v2143_data = ir11[7];
                ir11[7] = (v2143_data + (v2097_data * (sycl::group_broadcast(item.get_sub_group(), v2140_data, 1))));
              }
              if (v14_lead < 12) {
                float v2149_data = r9[2];
                float v2150_data = r10[0];
                float v2153_data = ir11[0];
                ir11[0] = (v2153_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2150_data, 2))));
                float v2156_data = r10[1];
                float v2159_data = ir11[1];
                ir11[1] = (v2159_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2156_data, 2))));
                float v2162_data = r10[2];
                float v2165_data = ir11[2];
                ir11[2] = (v2165_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2162_data, 2))));
                float v2168_data = r10[3];
                float v2171_data = ir11[3];
                ir11[3] = (v2171_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2168_data, 2))));
                float v2174_data = r10[4];
                float v2177_data = ir11[4];
                ir11[4] = (v2177_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2174_data, 2))));
                float v2180_data = r10[5];
                float v2183_data = ir11[5];
                ir11[5] = (v2183_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2180_data, 2))));
                float v2186_data = r10[6];
                float v2189_data = ir11[6];
                ir11[6] = (v2189_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2186_data, 2))));
                float v2192_data = r10[7];
                float v2195_data = ir11[7];
                ir11[7] = (v2195_data + (v2149_data * (sycl::group_broadcast(item.get_sub_group(), v2192_data, 2))));
              }
              if (v14_lead < 12) {
                float v2201_data = r9[3];
                float v2202_data = r10[0];
                float v2205_data = ir11[0];
                ir11[0] = (v2205_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2202_data, 3))));
                float v2208_data = r10[1];
                float v2211_data = ir11[1];
                ir11[1] = (v2211_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2208_data, 3))));
                float v2214_data = r10[2];
                float v2217_data = ir11[2];
                ir11[2] = (v2217_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2214_data, 3))));
                float v2220_data = r10[3];
                float v2223_data = ir11[3];
                ir11[3] = (v2223_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2220_data, 3))));
                float v2226_data = r10[4];
                float v2229_data = ir11[4];
                ir11[4] = (v2229_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2226_data, 3))));
                float v2232_data = r10[5];
                float v2235_data = ir11[5];
                ir11[5] = (v2235_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2232_data, 3))));
                float v2238_data = r10[6];
                float v2241_data = ir11[6];
                ir11[6] = (v2241_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2238_data, 3))));
                float v2244_data = r10[7];
                float v2247_data = ir11[7];
                ir11[7] = (v2247_data + (v2201_data * (sycl::group_broadcast(item.get_sub_group(), v2244_data, 3))));
              }
              if (v14_lead < 12) {
                float v2253_data = r9[4];
                float v2254_data = r10[0];
                float v2257_data = ir11[0];
                ir11[0] = (v2257_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2254_data, 4))));
                float v2260_data = r10[1];
                float v2263_data = ir11[1];
                ir11[1] = (v2263_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2260_data, 4))));
                float v2266_data = r10[2];
                float v2269_data = ir11[2];
                ir11[2] = (v2269_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2266_data, 4))));
                float v2272_data = r10[3];
                float v2275_data = ir11[3];
                ir11[3] = (v2275_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2272_data, 4))));
                float v2278_data = r10[4];
                float v2281_data = ir11[4];
                ir11[4] = (v2281_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2278_data, 4))));
                float v2284_data = r10[5];
                float v2287_data = ir11[5];
                ir11[5] = (v2287_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2284_data, 4))));
                float v2290_data = r10[6];
                float v2293_data = ir11[6];
                ir11[6] = (v2293_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2290_data, 4))));
                float v2296_data = r10[7];
                float v2299_data = ir11[7];
                ir11[7] = (v2299_data + (v2253_data * (sycl::group_broadcast(item.get_sub_group(), v2296_data, 4))));
              }
              if (v14_lead < 12) {
                float v2305_data = r9[5];
                float v2306_data = r10[0];
                float v2309_data = ir11[0];
                ir11[0] = (v2309_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2306_data, 5))));
                float v2312_data = r10[1];
                float v2315_data = ir11[1];
                ir11[1] = (v2315_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2312_data, 5))));
                float v2318_data = r10[2];
                float v2321_data = ir11[2];
                ir11[2] = (v2321_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2318_data, 5))));
                float v2324_data = r10[3];
                float v2327_data = ir11[3];
                ir11[3] = (v2327_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2324_data, 5))));
                float v2330_data = r10[4];
                float v2333_data = ir11[4];
                ir11[4] = (v2333_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2330_data, 5))));
                float v2336_data = r10[5];
                float v2339_data = ir11[5];
                ir11[5] = (v2339_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2336_data, 5))));
                float v2342_data = r10[6];
                float v2345_data = ir11[6];
                ir11[6] = (v2345_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2342_data, 5))));
                float v2348_data = r10[7];
                float v2351_data = ir11[7];
                ir11[7] = (v2351_data + (v2305_data * (sycl::group_broadcast(item.get_sub_group(), v2348_data, 5))));
              }
              if (v14_lead < 12) {
                float v2357_data = r9[6];
                float v2358_data = r10[0];
                float v2361_data = ir11[0];
                ir11[0] = (v2361_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2358_data, 6))));
                float v2364_data = r10[1];
                float v2367_data = ir11[1];
                ir11[1] = (v2367_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2364_data, 6))));
                float v2370_data = r10[2];
                float v2373_data = ir11[2];
                ir11[2] = (v2373_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2370_data, 6))));
                float v2376_data = r10[3];
                float v2379_data = ir11[3];
                ir11[3] = (v2379_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2376_data, 6))));
                float v2382_data = r10[4];
                float v2385_data = ir11[4];
                ir11[4] = (v2385_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2382_data, 6))));
                float v2388_data = r10[5];
                float v2391_data = ir11[5];
                ir11[5] = (v2391_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2388_data, 6))));
                float v2394_data = r10[6];
                float v2397_data = ir11[6];
                ir11[6] = (v2397_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2394_data, 6))));
                float v2400_data = r10[7];
                float v2403_data = ir11[7];
                ir11[7] = (v2403_data + (v2357_data * (sycl::group_broadcast(item.get_sub_group(), v2400_data, 6))));
              }
              if (v14_lead < 12) {
                float v2409_data = r9[7];
                float v2410_data = r10[0];
                float v2413_data = ir11[0];
                ir11[0] = (v2413_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2410_data, 7))));
                float v2416_data = r10[1];
                float v2419_data = ir11[1];
                ir11[1] = (v2419_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2416_data, 7))));
                float v2422_data = r10[2];
                float v2425_data = ir11[2];
                ir11[2] = (v2425_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2422_data, 7))));
                float v2428_data = r10[3];
                float v2431_data = ir11[3];
                ir11[3] = (v2431_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2428_data, 7))));
                float v2434_data = r10[4];
                float v2437_data = ir11[4];
                ir11[4] = (v2437_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2434_data, 7))));
                float v2440_data = r10[5];
                float v2443_data = ir11[5];
                ir11[5] = (v2443_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2440_data, 7))));
                float v2446_data = r10[6];
                float v2449_data = ir11[6];
                ir11[6] = (v2449_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2446_data, 7))));
                float v2452_data = r10[7];
                float v2455_data = ir11[7];
                ir11[7] = (v2455_data + (v2409_data * (sycl::group_broadcast(item.get_sub_group(), v2452_data, 7))));
              }
              if (v14_lead < 12) {
                float v2461_data = r9[8];
                float v2462_data = r10[0];
                float v2465_data = ir11[0];
                ir11[0] = (v2465_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2462_data, 8))));
                float v2468_data = r10[1];
                float v2471_data = ir11[1];
                ir11[1] = (v2471_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2468_data, 8))));
                float v2474_data = r10[2];
                float v2477_data = ir11[2];
                ir11[2] = (v2477_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2474_data, 8))));
                float v2480_data = r10[3];
                float v2483_data = ir11[3];
                ir11[3] = (v2483_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2480_data, 8))));
                float v2486_data = r10[4];
                float v2489_data = ir11[4];
                ir11[4] = (v2489_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2486_data, 8))));
                float v2492_data = r10[5];
                float v2495_data = ir11[5];
                ir11[5] = (v2495_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2492_data, 8))));
                float v2498_data = r10[6];
                float v2501_data = ir11[6];
                ir11[6] = (v2501_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2498_data, 8))));
                float v2504_data = r10[7];
                float v2507_data = ir11[7];
                ir11[7] = (v2507_data + (v2461_data * (sycl::group_broadcast(item.get_sub_group(), v2504_data, 8))));
              }
              if (v14_lead < 12) {
                float v2513_data = r9[9];
                float v2514_data = r10[0];
                float v2517_data = ir11[0];
                ir11[0] = (v2517_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2514_data, 9))));
                float v2520_data = r10[1];
                float v2523_data = ir11[1];
                ir11[1] = (v2523_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2520_data, 9))));
                float v2526_data = r10[2];
                float v2529_data = ir11[2];
                ir11[2] = (v2529_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2526_data, 9))));
                float v2532_data = r10[3];
                float v2535_data = ir11[3];
                ir11[3] = (v2535_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2532_data, 9))));
                float v2538_data = r10[4];
                float v2541_data = ir11[4];
                ir11[4] = (v2541_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2538_data, 9))));
                float v2544_data = r10[5];
                float v2547_data = ir11[5];
                ir11[5] = (v2547_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2544_data, 9))));
                float v2550_data = r10[6];
                float v2553_data = ir11[6];
                ir11[6] = (v2553_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2550_data, 9))));
                float v2556_data = r10[7];
                float v2559_data = ir11[7];
                ir11[7] = (v2559_data + (v2513_data * (sycl::group_broadcast(item.get_sub_group(), v2556_data, 9))));
              }
              if (v14_lead < 12) {
                float v2565_data = r9[10];
                float v2566_data = r10[0];
                float v2569_data = ir11[0];
                ir11[0] = (v2569_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2566_data, 10))));
                float v2572_data = r10[1];
                float v2575_data = ir11[1];
                ir11[1] = (v2575_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2572_data, 10))));
                float v2578_data = r10[2];
                float v2581_data = ir11[2];
                ir11[2] = (v2581_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2578_data, 10))));
                float v2584_data = r10[3];
                float v2587_data = ir11[3];
                ir11[3] = (v2587_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2584_data, 10))));
                float v2590_data = r10[4];
                float v2593_data = ir11[4];
                ir11[4] = (v2593_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2590_data, 10))));
                float v2596_data = r10[5];
                float v2599_data = ir11[5];
                ir11[5] = (v2599_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2596_data, 10))));
                float v2602_data = r10[6];
                float v2605_data = ir11[6];
                ir11[6] = (v2605_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2602_data, 10))));
                float v2608_data = r10[7];
                float v2611_data = ir11[7];
                ir11[7] = (v2611_data + (v2565_data * (sycl::group_broadcast(item.get_sub_group(), v2608_data, 10))));
              }
              if (v14_lead < 12) {
                float v2617_data = r9[11];
                float v2618_data = r10[0];
                float v2621_data = ir11[0];
                ir11[0] = (v2621_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2618_data, 11))));
                float v2624_data = r10[1];
                float v2627_data = ir11[1];
                ir11[1] = (v2627_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2624_data, 11))));
                float v2630_data = r10[2];
                float v2633_data = ir11[2];
                ir11[2] = (v2633_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2630_data, 11))));
                float v2636_data = r10[3];
                float v2639_data = ir11[3];
                ir11[3] = (v2639_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2636_data, 11))));
                float v2642_data = r10[4];
                float v2645_data = ir11[4];
                ir11[4] = (v2645_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2642_data, 11))));
                float v2648_data = r10[5];
                float v2651_data = ir11[5];
                ir11[5] = (v2651_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2648_data, 11))));
                float v2654_data = r10[6];
                float v2657_data = ir11[6];
                ir11[6] = (v2657_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2654_data, 11))));
                float v2660_data = r10[7];
                float v2663_data = ir11[7];
                ir11[7] = (v2663_data + (v2617_data * (sycl::group_broadcast(item.get_sub_group(), v2660_data, 11))));
              }
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2669_n1 = 0; v2669_n1 < 8; ++v2669_n1) {
                  float v2671_data = ir11[v2669_n1];
                  float v2673_data = r8[v2669_n1];
                  r11[v2669_n1] = (v2673_data + v2671_data);
                }
              }
              // glb_m0 = store{r>g}(r11);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v2680_i1 = 0; v2680_i1 < 8; ++v2680_i1) {
                  float v2682_data = r11[v2680_i1];
                  glb_m0[(v14_lead + (v2680_i1 * 12))] = v2682_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

