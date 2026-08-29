// === base name ===
kernel_924fd3d329

// === header ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_924fd3d329(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_924fd3d329(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×4(8×4) {0..8}×{0..4} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v9_lead = item.get_local_id(0) % 16;
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v11_i1 = 0; v11_i1 < 8; ++v11_i1) {
                  float v19_data = glb_m0[(v9_lead + (v11_i1 * 8))];
                  r0[v11_i1] = v19_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m1);
              float v22_lin = glb_m1[0 + item.get_local_id(0) * 1];
              r1[0] = v22_lin;
              float v23_lin = glb_m1[16 + item.get_local_id(0) * 1];
              r1[1] = v23_lin;
              // wait(r0 = load{g>r}(glb_m0););
              float r3[4]{};
              // r3 = load{g>r}(glb_m2);
              float v25_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r3[0] = v25_lin;
              float v26_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r3[1] = v26_lin;
              // wait(r1 = load{g>r}(glb_m1););
              float r2[4]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              if (v9_lead < 8) {
                float v32_data = r0[0];
                float v33_data = r1[0];
                float v36_data = r2[0];
                r2[0] = (v36_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v33_data, 0))));
                float v39_data = r1[1];
                float v42_data = r2[1];
                r2[1] = (v42_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v39_data, 0))));
                float v45_data = r1[2];
                float v48_data = r2[2];
                r2[2] = (v48_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v45_data, 0))));
                float v51_data = r1[3];
                float v54_data = r2[3];
                r2[3] = (v54_data + (v32_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 0))));
              }
              if (v9_lead < 8) {
                float v60_data = r0[1];
                float v61_data = r1[0];
                float v64_data = r2[0];
                r2[0] = (v64_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 1))));
                float v67_data = r1[1];
                float v70_data = r2[1];
                r2[1] = (v70_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 1))));
                float v73_data = r1[2];
                float v76_data = r2[2];
                r2[2] = (v76_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 1))));
                float v79_data = r1[3];
                float v82_data = r2[3];
                r2[3] = (v82_data + (v60_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 1))));
              }
              if (v9_lead < 8) {
                float v88_data = r0[2];
                float v89_data = r1[0];
                float v92_data = r2[0];
                r2[0] = (v92_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 2))));
                float v95_data = r1[1];
                float v98_data = r2[1];
                r2[1] = (v98_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 2))));
                float v101_data = r1[2];
                float v104_data = r2[2];
                r2[2] = (v104_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 2))));
                float v107_data = r1[3];
                float v110_data = r2[3];
                r2[3] = (v110_data + (v88_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 2))));
              }
              if (v9_lead < 8) {
                float v116_data = r0[3];
                float v117_data = r1[0];
                float v120_data = r2[0];
                r2[0] = (v120_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 3))));
                float v123_data = r1[1];
                float v126_data = r2[1];
                r2[1] = (v126_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 3))));
                float v129_data = r1[2];
                float v132_data = r2[2];
                r2[2] = (v132_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 3))));
                float v135_data = r1[3];
                float v138_data = r2[3];
                r2[3] = (v138_data + (v116_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 3))));
              }
              if (v9_lead < 8) {
                float v144_data = r0[4];
                float v145_data = r1[0];
                float v148_data = r2[0];
                r2[0] = (v148_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 4))));
                float v151_data = r1[1];
                float v154_data = r2[1];
                r2[1] = (v154_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 4))));
                float v157_data = r1[2];
                float v160_data = r2[2];
                r2[2] = (v160_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 4))));
                float v163_data = r1[3];
                float v166_data = r2[3];
                r2[3] = (v166_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 4))));
              }
              if (v9_lead < 8) {
                float v172_data = r0[5];
                float v173_data = r1[0];
                float v176_data = r2[0];
                r2[0] = (v176_data + (v172_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 5))));
                float v179_data = r1[1];
                float v182_data = r2[1];
                r2[1] = (v182_data + (v172_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 5))));
                float v185_data = r1[2];
                float v188_data = r2[2];
                r2[2] = (v188_data + (v172_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 5))));
                float v191_data = r1[3];
                float v194_data = r2[3];
                r2[3] = (v194_data + (v172_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 5))));
              }
              if (v9_lead < 8) {
                float v200_data = r0[6];
                float v201_data = r1[0];
                float v204_data = r2[0];
                r2[0] = (v204_data + (v200_data * (sycl::group_broadcast(item.get_sub_group(), v201_data, 6))));
                float v207_data = r1[1];
                float v210_data = r2[1];
                r2[1] = (v210_data + (v200_data * (sycl::group_broadcast(item.get_sub_group(), v207_data, 6))));
                float v213_data = r1[2];
                float v216_data = r2[2];
                r2[2] = (v216_data + (v200_data * (sycl::group_broadcast(item.get_sub_group(), v213_data, 6))));
                float v219_data = r1[3];
                float v222_data = r2[3];
                r2[3] = (v222_data + (v200_data * (sycl::group_broadcast(item.get_sub_group(), v219_data, 6))));
              }
              if (v9_lead < 8) {
                float v228_data = r0[7];
                float v229_data = r1[0];
                float v232_data = r2[0];
                r2[0] = (v232_data + (v228_data * (sycl::group_broadcast(item.get_sub_group(), v229_data, 7))));
                float v235_data = r1[1];
                float v238_data = r2[1];
                r2[1] = (v238_data + (v228_data * (sycl::group_broadcast(item.get_sub_group(), v235_data, 7))));
                float v241_data = r1[2];
                float v244_data = r2[2];
                r2[2] = (v244_data + (v228_data * (sycl::group_broadcast(item.get_sub_group(), v241_data, 7))));
                float v247_data = r1[3];
                float v250_data = r2[3];
                r2[3] = (v250_data + (v228_data * (sycl::group_broadcast(item.get_sub_group(), v247_data, 7))));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r2);
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v257_i1 = 0; v257_i1 < 4; ++v257_i1) {
                  float v259_data = r2[v257_i1];
                  int32_t v266_a = v9_lead + (v257_i1 * 8);
                  s0[(v266_a ^ ((v266_a >> 5) & 31))] = v259_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[4]{};
              // r4 = +(r0 * r3) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir4[4]{};
              if (v9_lead < 8) {
                float v276_data = r0[0];
                float v277_data = r3[0];
                float v280_data = ir4[0];
                ir4[0] = (v280_data + (v276_data * (sycl::group_broadcast(item.get_sub_group(), v277_data, 0))));
                float v283_data = r3[1];
                float v286_data = ir4[1];
                ir4[1] = (v286_data + (v276_data * (sycl::group_broadcast(item.get_sub_group(), v283_data, 0))));
                float v289_data = r3[2];
                float v292_data = ir4[2];
                ir4[2] = (v292_data + (v276_data * (sycl::group_broadcast(item.get_sub_group(), v289_data, 0))));
                float v295_data = r3[3];
                float v298_data = ir4[3];
                ir4[3] = (v298_data + (v276_data * (sycl::group_broadcast(item.get_sub_group(), v295_data, 0))));
              }
              if (v9_lead < 8) {
                float v304_data = r0[1];
                float v305_data = r3[0];
                float v308_data = ir4[0];
                ir4[0] = (v308_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 1))));
                float v311_data = r3[1];
                float v314_data = ir4[1];
                ir4[1] = (v314_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 1))));
                float v317_data = r3[2];
                float v320_data = ir4[2];
                ir4[2] = (v320_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 1))));
                float v323_data = r3[3];
                float v326_data = ir4[3];
                ir4[3] = (v326_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 1))));
              }
              if (v9_lead < 8) {
                float v332_data = r0[2];
                float v333_data = r3[0];
                float v336_data = ir4[0];
                ir4[0] = (v336_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 2))));
                float v339_data = r3[1];
                float v342_data = ir4[1];
                ir4[1] = (v342_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 2))));
                float v345_data = r3[2];
                float v348_data = ir4[2];
                ir4[2] = (v348_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 2))));
                float v351_data = r3[3];
                float v354_data = ir4[3];
                ir4[3] = (v354_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 2))));
              }
              if (v9_lead < 8) {
                float v360_data = r0[3];
                float v361_data = r3[0];
                float v364_data = ir4[0];
                ir4[0] = (v364_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 3))));
                float v367_data = r3[1];
                float v370_data = ir4[1];
                ir4[1] = (v370_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 3))));
                float v373_data = r3[2];
                float v376_data = ir4[2];
                ir4[2] = (v376_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 3))));
                float v379_data = r3[3];
                float v382_data = ir4[3];
                ir4[3] = (v382_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 3))));
              }
              if (v9_lead < 8) {
                float v388_data = r0[4];
                float v389_data = r3[0];
                float v392_data = ir4[0];
                ir4[0] = (v392_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 4))));
                float v395_data = r3[1];
                float v398_data = ir4[1];
                ir4[1] = (v398_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 4))));
                float v401_data = r3[2];
                float v404_data = ir4[2];
                ir4[2] = (v404_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 4))));
                float v407_data = r3[3];
                float v410_data = ir4[3];
                ir4[3] = (v410_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 4))));
              }
              if (v9_lead < 8) {
                float v416_data = r0[5];
                float v417_data = r3[0];
                float v420_data = ir4[0];
                ir4[0] = (v420_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v417_data, 5))));
                float v423_data = r3[1];
                float v426_data = ir4[1];
                ir4[1] = (v426_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v423_data, 5))));
                float v429_data = r3[2];
                float v432_data = ir4[2];
                ir4[2] = (v432_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 5))));
                float v435_data = r3[3];
                float v438_data = ir4[3];
                ir4[3] = (v438_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 5))));
              }
              if (v9_lead < 8) {
                float v444_data = r0[6];
                float v445_data = r3[0];
                float v448_data = ir4[0];
                ir4[0] = (v448_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v445_data, 6))));
                float v451_data = r3[1];
                float v454_data = ir4[1];
                ir4[1] = (v454_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v451_data, 6))));
                float v457_data = r3[2];
                float v460_data = ir4[2];
                ir4[2] = (v460_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v457_data, 6))));
                float v463_data = r3[3];
                float v466_data = ir4[3];
                ir4[3] = (v466_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v463_data, 6))));
              }
              if (v9_lead < 8) {
                float v472_data = r0[7];
                float v473_data = r3[0];
                float v476_data = ir4[0];
                ir4[0] = (v476_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v473_data, 7))));
                float v479_data = r3[1];
                float v482_data = ir4[1];
                ir4[1] = (v482_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v479_data, 7))));
                float v485_data = r3[2];
                float v488_data = ir4[2];
                ir4[2] = (v488_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v485_data, 7))));
                float v491_data = r3[3];
                float v494_data = ir4[3];
                ir4[3] = (v494_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v491_data, 7))));
              }
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v500_n1 = 0; v500_n1 < 4; ++v500_n1) {
                  float v502_data = ir4[v500_n1];
                  r4[v500_n1] = v502_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v508_i1 = 0; v508_i1 < 4; ++v508_i1) {
                  float v510_data = r4[v508_i1];
                  int32_t v518_a = v9_lead + ((v508_i1 + 4) * 8);
                  s0[(v518_a ^ ((v518_a >> 5) & 31))] = v510_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m3 = abs(s0)
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v526_k1 = 0; v526_k1 < 8; ++v526_k1) {
                  int32_t v532_a = v526_k1 * 8;
                  int32_t v533_a = v9_lead + v532_a;
                  float v537_data = s0[(v533_a ^ ((v533_a >> 5) & 31))];
                  glb_m3[(v9_lead + v532_a)] = (sycl::fabs(v537_data));
                }
              }
            }
          }
        }
      });
    }
  });
}

