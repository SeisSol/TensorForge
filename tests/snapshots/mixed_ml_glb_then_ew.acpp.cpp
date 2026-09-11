// === base name ===
kernel_2a58660a99f95cc8

// === header ===
void launcher_kernel_2a58660a99f95cc8(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_2a58660a99f95cc8(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_2a58660a99f95cc8(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2a58660a99f95cc8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(M)
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v2_batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(0) % 16;
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                  float v27_data = glb_m1[(v17_lead + (v19_i1 * 8))];
                  r0[v19_i1] = v27_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                  float v42_data = glb_m2[(v17_lead + (v34_i1 * 8))];
                  r1[v34_i1] = v42_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir2[8]{};
              if (v17_lead < 8) {
                float v50_data = r0[0];
                float v51_data = r1[0];
                float v54_data = ir2[0];
                ir2[0] = (v54_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 0))));
                float v57_data = r1[1];
                float v60_data = ir2[1];
                ir2[1] = (v60_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 0))));
                float v63_data = r1[2];
                float v66_data = ir2[2];
                ir2[2] = (v66_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 0))));
                float v69_data = r1[3];
                float v72_data = ir2[3];
                ir2[3] = (v72_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 0))));
                float v75_data = r1[4];
                float v78_data = ir2[4];
                ir2[4] = (v78_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 0))));
                float v81_data = r1[5];
                float v84_data = ir2[5];
                ir2[5] = (v84_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 0))));
                float v87_data = r1[6];
                float v90_data = ir2[6];
                ir2[6] = (v90_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 0))));
                float v93_data = r1[7];
                float v96_data = ir2[7];
                ir2[7] = (v96_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v93_data, 0))));
              }
              if (v17_lead < 8) {
                float v102_data = r0[1];
                float v103_data = r1[0];
                float v106_data = ir2[0];
                ir2[0] = (v106_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 1))));
                float v109_data = r1[1];
                float v112_data = ir2[1];
                ir2[1] = (v112_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 1))));
                float v115_data = r1[2];
                float v118_data = ir2[2];
                ir2[2] = (v118_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 1))));
                float v121_data = r1[3];
                float v124_data = ir2[3];
                ir2[3] = (v124_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 1))));
                float v127_data = r1[4];
                float v130_data = ir2[4];
                ir2[4] = (v130_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
                float v133_data = r1[5];
                float v136_data = ir2[5];
                ir2[5] = (v136_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 1))));
                float v139_data = r1[6];
                float v142_data = ir2[6];
                ir2[6] = (v142_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 1))));
                float v145_data = r1[7];
                float v148_data = ir2[7];
                ir2[7] = (v148_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 1))));
              }
              if (v17_lead < 8) {
                float v154_data = r0[2];
                float v155_data = r1[0];
                float v158_data = ir2[0];
                ir2[0] = (v158_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 2))));
                float v161_data = r1[1];
                float v164_data = ir2[1];
                ir2[1] = (v164_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 2))));
                float v167_data = r1[2];
                float v170_data = ir2[2];
                ir2[2] = (v170_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
                float v173_data = r1[3];
                float v176_data = ir2[3];
                ir2[3] = (v176_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 2))));
                float v179_data = r1[4];
                float v182_data = ir2[4];
                ir2[4] = (v182_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 2))));
                float v185_data = r1[5];
                float v188_data = ir2[5];
                ir2[5] = (v188_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 2))));
                float v191_data = r1[6];
                float v194_data = ir2[6];
                ir2[6] = (v194_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 2))));
                float v197_data = r1[7];
                float v200_data = ir2[7];
                ir2[7] = (v200_data + (v154_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 2))));
              }
              if (v17_lead < 8) {
                float v206_data = r0[3];
                float v207_data = r1[0];
                float v210_data = ir2[0];
                ir2[0] = (v210_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 3))));
                float v213_data = r1[1];
                float v216_data = ir2[1];
                ir2[1] = (v216_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v213_data, 3))));
                float v219_data = r1[2];
                float v222_data = ir2[2];
                ir2[2] = (v222_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v219_data, 3))));
                float v225_data = r1[3];
                float v228_data = ir2[3];
                ir2[3] = (v228_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 3))));
                float v231_data = r1[4];
                float v234_data = ir2[4];
                ir2[4] = (v234_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 3))));
                float v237_data = r1[5];
                float v240_data = ir2[5];
                ir2[5] = (v240_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v237_data, 3))));
                float v243_data = r1[6];
                float v246_data = ir2[6];
                ir2[6] = (v246_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v243_data, 3))));
                float v249_data = r1[7];
                float v252_data = ir2[7];
                ir2[7] = (v252_data + (v206_data * (sycl::group_broadcast(item.get_sub_group(), v249_data, 3))));
              }
              if (v17_lead < 8) {
                float v258_data = r0[4];
                float v259_data = r1[0];
                float v262_data = ir2[0];
                ir2[0] = (v262_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v259_data, 4))));
                float v265_data = r1[1];
                float v268_data = ir2[1];
                ir2[1] = (v268_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v265_data, 4))));
                float v271_data = r1[2];
                float v274_data = ir2[2];
                ir2[2] = (v274_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v271_data, 4))));
                float v277_data = r1[3];
                float v280_data = ir2[3];
                ir2[3] = (v280_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v277_data, 4))));
                float v283_data = r1[4];
                float v286_data = ir2[4];
                ir2[4] = (v286_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v283_data, 4))));
                float v289_data = r1[5];
                float v292_data = ir2[5];
                ir2[5] = (v292_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v289_data, 4))));
                float v295_data = r1[6];
                float v298_data = ir2[6];
                ir2[6] = (v298_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v295_data, 4))));
                float v301_data = r1[7];
                float v304_data = ir2[7];
                ir2[7] = (v304_data + (v258_data * (sycl::group_broadcast(item.get_sub_group(), v301_data, 4))));
              }
              if (v17_lead < 8) {
                float v310_data = r0[5];
                float v311_data = r1[0];
                float v314_data = ir2[0];
                ir2[0] = (v314_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 5))));
                float v317_data = r1[1];
                float v320_data = ir2[1];
                ir2[1] = (v320_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 5))));
                float v323_data = r1[2];
                float v326_data = ir2[2];
                ir2[2] = (v326_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 5))));
                float v329_data = r1[3];
                float v332_data = ir2[3];
                ir2[3] = (v332_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v329_data, 5))));
                float v335_data = r1[4];
                float v338_data = ir2[4];
                ir2[4] = (v338_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 5))));
                float v341_data = r1[5];
                float v344_data = ir2[5];
                ir2[5] = (v344_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v341_data, 5))));
                float v347_data = r1[6];
                float v350_data = ir2[6];
                ir2[6] = (v350_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v347_data, 5))));
                float v353_data = r1[7];
                float v356_data = ir2[7];
                ir2[7] = (v356_data + (v310_data * (sycl::group_broadcast(item.get_sub_group(), v353_data, 5))));
              }
              if (v17_lead < 8) {
                float v362_data = r0[6];
                float v363_data = r1[0];
                float v366_data = ir2[0];
                ir2[0] = (v366_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 6))));
                float v369_data = r1[1];
                float v372_data = ir2[1];
                ir2[1] = (v372_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 6))));
                float v375_data = r1[2];
                float v378_data = ir2[2];
                ir2[2] = (v378_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v375_data, 6))));
                float v381_data = r1[3];
                float v384_data = ir2[3];
                ir2[3] = (v384_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v381_data, 6))));
                float v387_data = r1[4];
                float v390_data = ir2[4];
                ir2[4] = (v390_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v387_data, 6))));
                float v393_data = r1[5];
                float v396_data = ir2[5];
                ir2[5] = (v396_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v393_data, 6))));
                float v399_data = r1[6];
                float v402_data = ir2[6];
                ir2[6] = (v402_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 6))));
                float v405_data = r1[7];
                float v408_data = ir2[7];
                ir2[7] = (v408_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v405_data, 6))));
              }
              if (v17_lead < 8) {
                float v414_data = r0[7];
                float v415_data = r1[0];
                float v418_data = ir2[0];
                ir2[0] = (v418_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v415_data, 7))));
                float v421_data = r1[1];
                float v424_data = ir2[1];
                ir2[1] = (v424_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v421_data, 7))));
                float v427_data = r1[2];
                float v430_data = ir2[2];
                ir2[2] = (v430_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 7))));
                float v433_data = r1[3];
                float v436_data = ir2[3];
                ir2[3] = (v436_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 7))));
                float v439_data = r1[4];
                float v442_data = ir2[4];
                ir2[4] = (v442_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v439_data, 7))));
                float v445_data = r1[5];
                float v448_data = ir2[5];
                ir2[5] = (v448_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v445_data, 7))));
                float v451_data = r1[6];
                float v454_data = ir2[6];
                ir2[6] = (v454_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v451_data, 7))));
                float v457_data = r1[7];
                float v460_data = ir2[7];
                ir2[7] = (v460_data + (v414_data * (sycl::group_broadcast(item.get_sub_group(), v457_data, 7))));
              }
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v466_n1 = 0; v466_n1 < 8; ++v466_n1) {
                  float v468_data = ir2[v466_n1];
                  r2[v466_n1] = v468_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v474_i1 = 0; v474_i1 < 8; ++v474_i1) {
                  float v476_data = r2[v474_i1];
                  glb_m0[(v17_lead + (v474_i1 * 8))] = v476_data;
                }
              }
              // glb_m3 = abs(glb_m0)
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v488_k1 = 0; v488_k1 < 8; ++v488_k1) {
                  int32_t v494_a = v488_k1 * 8;
                  float v496_data = glb_m0[(v17_lead + v494_a)];
                  glb_m3[(v17_lead + v494_a)] = (sycl::fabs(v496_data));
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

