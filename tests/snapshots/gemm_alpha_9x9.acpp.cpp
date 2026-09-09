// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08a27dccde(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08a27dccde(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              bool v9_g = v8_lead < 9;
              if (v9_g) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 9; ++v10_i1) {
                  float v18_data = glb_m1[(v8_lead + (v10_i1 * 9))];
                  r0[v10_i1] = v18_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v9_g) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
                  float v33_data = glb_m2[(v8_lead + (v25_i1 * 9))];
                  r1[v25_i1] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir2[9]{};
              if (v9_g) {
                float v41_data = r0[0];
                float v42_data = r1[0];
                float v45_data = ir2[0];
                ir2[0] = (v45_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 0))));
                float v48_data = r1[1];
                float v51_data = ir2[1];
                ir2[1] = (v51_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 0))));
                float v54_data = r1[2];
                float v57_data = ir2[2];
                ir2[2] = (v57_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v54_data, 0))));
                float v60_data = r1[3];
                float v63_data = ir2[3];
                ir2[3] = (v63_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v60_data, 0))));
                float v66_data = r1[4];
                float v69_data = ir2[4];
                ir2[4] = (v69_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 0))));
                float v72_data = r1[5];
                float v75_data = ir2[5];
                ir2[5] = (v75_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 0))));
                float v78_data = r1[6];
                float v81_data = ir2[6];
                ir2[6] = (v81_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 0))));
                float v84_data = r1[7];
                float v87_data = ir2[7];
                ir2[7] = (v87_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 0))));
                float v90_data = r1[8];
                float v93_data = ir2[8];
                ir2[8] = (v93_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 0))));
              }
              if (v9_g) {
                float v99_data = r0[1];
                float v100_data = r1[0];
                float v103_data = ir2[0];
                ir2[0] = (v103_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 1))));
                float v106_data = r1[1];
                float v109_data = ir2[1];
                ir2[1] = (v109_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 1))));
                float v112_data = r1[2];
                float v115_data = ir2[2];
                ir2[2] = (v115_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 1))));
                float v118_data = r1[3];
                float v121_data = ir2[3];
                ir2[3] = (v121_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 1))));
                float v124_data = r1[4];
                float v127_data = ir2[4];
                ir2[4] = (v127_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 1))));
                float v130_data = r1[5];
                float v133_data = ir2[5];
                ir2[5] = (v133_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 1))));
                float v136_data = r1[6];
                float v139_data = ir2[6];
                ir2[6] = (v139_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 1))));
                float v142_data = r1[7];
                float v145_data = ir2[7];
                ir2[7] = (v145_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 1))));
                float v148_data = r1[8];
                float v151_data = ir2[8];
                ir2[8] = (v151_data + (v99_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 1))));
              }
              if (v9_g) {
                float v157_data = r0[2];
                float v158_data = r1[0];
                float v161_data = ir2[0];
                ir2[0] = (v161_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v158_data, 2))));
                float v164_data = r1[1];
                float v167_data = ir2[1];
                ir2[1] = (v167_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v164_data, 2))));
                float v170_data = r1[2];
                float v173_data = ir2[2];
                ir2[2] = (v173_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v170_data, 2))));
                float v176_data = r1[3];
                float v179_data = ir2[3];
                ir2[3] = (v179_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v176_data, 2))));
                float v182_data = r1[4];
                float v185_data = ir2[4];
                ir2[4] = (v185_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v182_data, 2))));
                float v188_data = r1[5];
                float v191_data = ir2[5];
                ir2[5] = (v191_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v188_data, 2))));
                float v194_data = r1[6];
                float v197_data = ir2[6];
                ir2[6] = (v197_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v194_data, 2))));
                float v200_data = r1[7];
                float v203_data = ir2[7];
                ir2[7] = (v203_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v200_data, 2))));
                float v206_data = r1[8];
                float v209_data = ir2[8];
                ir2[8] = (v209_data + (v157_data * (sycl::group_broadcast(item.get_sub_group(), v206_data, 2))));
              }
              if (v9_g) {
                float v215_data = r0[3];
                float v216_data = r1[0];
                float v219_data = ir2[0];
                ir2[0] = (v219_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v216_data, 3))));
                float v222_data = r1[1];
                float v225_data = ir2[1];
                ir2[1] = (v225_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v222_data, 3))));
                float v228_data = r1[2];
                float v231_data = ir2[2];
                ir2[2] = (v231_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v228_data, 3))));
                float v234_data = r1[3];
                float v237_data = ir2[3];
                ir2[3] = (v237_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v234_data, 3))));
                float v240_data = r1[4];
                float v243_data = ir2[4];
                ir2[4] = (v243_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v240_data, 3))));
                float v246_data = r1[5];
                float v249_data = ir2[5];
                ir2[5] = (v249_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 3))));
                float v252_data = r1[6];
                float v255_data = ir2[6];
                ir2[6] = (v255_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v252_data, 3))));
                float v258_data = r1[7];
                float v261_data = ir2[7];
                ir2[7] = (v261_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v258_data, 3))));
                float v264_data = r1[8];
                float v267_data = ir2[8];
                ir2[8] = (v267_data + (v215_data * (sycl::group_broadcast(item.get_sub_group(), v264_data, 3))));
              }
              if (v9_g) {
                float v273_data = r0[4];
                float v274_data = r1[0];
                float v277_data = ir2[0];
                ir2[0] = (v277_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v274_data, 4))));
                float v280_data = r1[1];
                float v283_data = ir2[1];
                ir2[1] = (v283_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v280_data, 4))));
                float v286_data = r1[2];
                float v289_data = ir2[2];
                ir2[2] = (v289_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v286_data, 4))));
                float v292_data = r1[3];
                float v295_data = ir2[3];
                ir2[3] = (v295_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v292_data, 4))));
                float v298_data = r1[4];
                float v301_data = ir2[4];
                ir2[4] = (v301_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v298_data, 4))));
                float v304_data = r1[5];
                float v307_data = ir2[5];
                ir2[5] = (v307_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v304_data, 4))));
                float v310_data = r1[6];
                float v313_data = ir2[6];
                ir2[6] = (v313_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v310_data, 4))));
                float v316_data = r1[7];
                float v319_data = ir2[7];
                ir2[7] = (v319_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v316_data, 4))));
                float v322_data = r1[8];
                float v325_data = ir2[8];
                ir2[8] = (v325_data + (v273_data * (sycl::group_broadcast(item.get_sub_group(), v322_data, 4))));
              }
              if (v9_g) {
                float v331_data = r0[5];
                float v332_data = r1[0];
                float v335_data = ir2[0];
                ir2[0] = (v335_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v332_data, 5))));
                float v338_data = r1[1];
                float v341_data = ir2[1];
                ir2[1] = (v341_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v338_data, 5))));
                float v344_data = r1[2];
                float v347_data = ir2[2];
                ir2[2] = (v347_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v344_data, 5))));
                float v350_data = r1[3];
                float v353_data = ir2[3];
                ir2[3] = (v353_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v350_data, 5))));
                float v356_data = r1[4];
                float v359_data = ir2[4];
                ir2[4] = (v359_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v356_data, 5))));
                float v362_data = r1[5];
                float v365_data = ir2[5];
                ir2[5] = (v365_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v362_data, 5))));
                float v368_data = r1[6];
                float v371_data = ir2[6];
                ir2[6] = (v371_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v368_data, 5))));
                float v374_data = r1[7];
                float v377_data = ir2[7];
                ir2[7] = (v377_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v374_data, 5))));
                float v380_data = r1[8];
                float v383_data = ir2[8];
                ir2[8] = (v383_data + (v331_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 5))));
              }
              if (v9_g) {
                float v389_data = r0[6];
                float v390_data = r1[0];
                float v393_data = ir2[0];
                ir2[0] = (v393_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v390_data, 6))));
                float v396_data = r1[1];
                float v399_data = ir2[1];
                ir2[1] = (v399_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v396_data, 6))));
                float v402_data = r1[2];
                float v405_data = ir2[2];
                ir2[2] = (v405_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v402_data, 6))));
                float v408_data = r1[3];
                float v411_data = ir2[3];
                ir2[3] = (v411_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v408_data, 6))));
                float v414_data = r1[4];
                float v417_data = ir2[4];
                ir2[4] = (v417_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v414_data, 6))));
                float v420_data = r1[5];
                float v423_data = ir2[5];
                ir2[5] = (v423_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v420_data, 6))));
                float v426_data = r1[6];
                float v429_data = ir2[6];
                ir2[6] = (v429_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v426_data, 6))));
                float v432_data = r1[7];
                float v435_data = ir2[7];
                ir2[7] = (v435_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v432_data, 6))));
                float v438_data = r1[8];
                float v441_data = ir2[8];
                ir2[8] = (v441_data + (v389_data * (sycl::group_broadcast(item.get_sub_group(), v438_data, 6))));
              }
              if (v9_g) {
                float v447_data = r0[7];
                float v448_data = r1[0];
                float v451_data = ir2[0];
                ir2[0] = (v451_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 7))));
                float v454_data = r1[1];
                float v457_data = ir2[1];
                ir2[1] = (v457_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v454_data, 7))));
                float v460_data = r1[2];
                float v463_data = ir2[2];
                ir2[2] = (v463_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v460_data, 7))));
                float v466_data = r1[3];
                float v469_data = ir2[3];
                ir2[3] = (v469_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v466_data, 7))));
                float v472_data = r1[4];
                float v475_data = ir2[4];
                ir2[4] = (v475_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v472_data, 7))));
                float v478_data = r1[5];
                float v481_data = ir2[5];
                ir2[5] = (v481_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v478_data, 7))));
                float v484_data = r1[6];
                float v487_data = ir2[6];
                ir2[6] = (v487_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v484_data, 7))));
                float v490_data = r1[7];
                float v493_data = ir2[7];
                ir2[7] = (v493_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v490_data, 7))));
                float v496_data = r1[8];
                float v499_data = ir2[8];
                ir2[8] = (v499_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v496_data, 7))));
              }
              if (v9_g) {
                float v505_data = r0[8];
                float v506_data = r1[0];
                float v509_data = ir2[0];
                ir2[0] = (v509_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v506_data, 8))));
                float v512_data = r1[1];
                float v515_data = ir2[1];
                ir2[1] = (v515_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v512_data, 8))));
                float v518_data = r1[2];
                float v521_data = ir2[2];
                ir2[2] = (v521_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v518_data, 8))));
                float v524_data = r1[3];
                float v527_data = ir2[3];
                ir2[3] = (v527_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v524_data, 8))));
                float v530_data = r1[4];
                float v533_data = ir2[4];
                ir2[4] = (v533_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v530_data, 8))));
                float v536_data = r1[5];
                float v539_data = ir2[5];
                ir2[5] = (v539_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v536_data, 8))));
                float v542_data = r1[6];
                float v545_data = ir2[6];
                ir2[6] = (v545_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 8))));
                float v548_data = r1[7];
                float v551_data = ir2[7];
                ir2[7] = (v551_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 8))));
                float v554_data = r1[8];
                float v557_data = ir2[8];
                ir2[8] = (v557_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v554_data, 8))));
              }
              if (v9_g) {
                #pragma unroll
                for (int32_t v564_n1 = 0; v564_n1 < 9; ++v564_n1) {
                  float v566_data = ir2[v564_n1];
                  r2[v564_n1] = (v566_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v9_g) {
                #pragma unroll
                for (int32_t v573_i1 = 0; v573_i1 < 9; ++v573_i1) {
                  float v575_data = r2[v573_i1];
                  glb_m0[(v8_lead + (v573_i1 * 9))] = v575_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

