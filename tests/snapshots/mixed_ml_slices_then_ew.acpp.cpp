// === base name ===
kernel_550ef5526fb3d0b2

// === header ===
void launcher_kernel_550ef5526fb3d0b2(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_550ef5526fb3d0b2(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_550ef5526fb3d0b2(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_550ef5526fb3d0b2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v14_lead = item.get_local_id(0) % 16;
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                  float v24_data = glb_m0[(v14_lead + (v16_i1 * 8))];
                  r0[v16_i1] = v24_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m1);
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 4; ++v31_i1) {
                  float v39_data = glb_m1[(v14_lead + (v31_i1 * 8))];
                  r1[v31_i1] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[4]{};
              // r3 = load{g>r}(glb_m2);
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v46_i1 = 0; v46_i1 < 4; ++v46_i1) {
                  float v54_data = glb_m2[(v14_lead + (v46_i1 * 8))];
                  r3[v46_i1] = v54_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[4]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              if (v14_lead < 8) {
                float v61_data = r0[0];
                float v62_data = r1[0];
                float v65_data = r2[0];
                r2[0] = (v65_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 0))));
                float v68_data = r1[1];
                float v71_data = r2[1];
                r2[1] = (v71_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 0))));
                float v74_data = r1[2];
                float v77_data = r2[2];
                r2[2] = (v77_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 0))));
                float v80_data = r1[3];
                float v83_data = r2[3];
                r2[3] = (v83_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 0))));
              }
              if (v14_lead < 8) {
                float v89_data = r0[1];
                float v90_data = r1[0];
                float v93_data = r2[0];
                r2[0] = (v93_data + (v89_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 1))));
                float v96_data = r1[1];
                float v99_data = r2[1];
                r2[1] = (v99_data + (v89_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 1))));
                float v102_data = r1[2];
                float v105_data = r2[2];
                r2[2] = (v105_data + (v89_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 1))));
                float v108_data = r1[3];
                float v111_data = r2[3];
                r2[3] = (v111_data + (v89_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 1))));
              }
              if (v14_lead < 8) {
                float v117_data = r0[2];
                float v118_data = r1[0];
                float v121_data = r2[0];
                r2[0] = (v121_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 2))));
                float v124_data = r1[1];
                float v127_data = r2[1];
                r2[1] = (v127_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 2))));
                float v130_data = r1[2];
                float v133_data = r2[2];
                r2[2] = (v133_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 2))));
                float v136_data = r1[3];
                float v139_data = r2[3];
                r2[3] = (v139_data + (v117_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 2))));
              }
              if (v14_lead < 8) {
                float v145_data = r0[3];
                float v146_data = r1[0];
                float v149_data = r2[0];
                r2[0] = (v149_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 3))));
                float v152_data = r1[1];
                float v155_data = r2[1];
                r2[1] = (v155_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 3))));
                float v158_data = r1[2];
                float v161_data = r2[2];
                r2[2] = (v161_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v158_data, 3))));
                float v164_data = r1[3];
                float v167_data = r2[3];
                r2[3] = (v167_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v164_data, 3))));
              }
              if (v14_lead < 8) {
                float v173_data = r0[4];
                float v174_data = r1[0];
                float v177_data = r2[0];
                r2[0] = (v177_data + (v173_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 4))));
                float v180_data = r1[1];
                float v183_data = r2[1];
                r2[1] = (v183_data + (v173_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 4))));
                float v186_data = r1[2];
                float v189_data = r2[2];
                r2[2] = (v189_data + (v173_data * (sycl::group_broadcast(item.get_sub_group(), v186_data, 4))));
                float v192_data = r1[3];
                float v195_data = r2[3];
                r2[3] = (v195_data + (v173_data * (sycl::group_broadcast(item.get_sub_group(), v192_data, 4))));
              }
              if (v14_lead < 8) {
                float v201_data = r0[5];
                float v202_data = r1[0];
                float v205_data = r2[0];
                r2[0] = (v205_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v202_data, 5))));
                float v208_data = r1[1];
                float v211_data = r2[1];
                r2[1] = (v211_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v208_data, 5))));
                float v214_data = r1[2];
                float v217_data = r2[2];
                r2[2] = (v217_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v214_data, 5))));
                float v220_data = r1[3];
                float v223_data = r2[3];
                r2[3] = (v223_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v220_data, 5))));
              }
              if (v14_lead < 8) {
                float v229_data = r0[6];
                float v230_data = r1[0];
                float v233_data = r2[0];
                r2[0] = (v233_data + (v229_data * (sycl::group_broadcast(item.get_sub_group(), v230_data, 6))));
                float v236_data = r1[1];
                float v239_data = r2[1];
                r2[1] = (v239_data + (v229_data * (sycl::group_broadcast(item.get_sub_group(), v236_data, 6))));
                float v242_data = r1[2];
                float v245_data = r2[2];
                r2[2] = (v245_data + (v229_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 6))));
                float v248_data = r1[3];
                float v251_data = r2[3];
                r2[3] = (v251_data + (v229_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 6))));
              }
              if (v14_lead < 8) {
                float v257_data = r0[7];
                float v258_data = r1[0];
                float v261_data = r2[0];
                r2[0] = (v261_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v258_data, 7))));
                float v264_data = r1[1];
                float v267_data = r2[1];
                r2[1] = (v267_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v264_data, 7))));
                float v270_data = r1[2];
                float v273_data = r2[2];
                r2[2] = (v273_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v270_data, 7))));
                float v276_data = r1[3];
                float v279_data = r2[3];
                r2[3] = (v279_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v276_data, 7))));
              }
              // s0 = store{r>s}(localShrMem0, r2);
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v285_i1 = 0; v285_i1 < 4; ++v285_i1) {
                  float v287_data = r2[v285_i1];
                  int32_t v294_a = v14_lead + (v285_i1 * 8);
                  s0[(v294_a ^ ((v294_a >> 5) & 31))] = v287_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[4]{};
              // r4 = +(r0 * r3) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir4[4]{};
              if (v14_lead < 8) {
                float v304_data = r0[0];
                float v305_data = r3[0];
                float v308_data = ir4[0];
                ir4[0] = (v308_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v305_data, 0))));
                float v311_data = r3[1];
                float v314_data = ir4[1];
                ir4[1] = (v314_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v311_data, 0))));
                float v317_data = r3[2];
                float v320_data = ir4[2];
                ir4[2] = (v320_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v317_data, 0))));
                float v323_data = r3[3];
                float v326_data = ir4[3];
                ir4[3] = (v326_data + (v304_data * (sycl::group_broadcast(item.get_sub_group(), v323_data, 0))));
              }
              if (v14_lead < 8) {
                float v332_data = r0[1];
                float v333_data = r3[0];
                float v336_data = ir4[0];
                ir4[0] = (v336_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v333_data, 1))));
                float v339_data = r3[1];
                float v342_data = ir4[1];
                ir4[1] = (v342_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 1))));
                float v345_data = r3[2];
                float v348_data = ir4[2];
                ir4[2] = (v348_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v345_data, 1))));
                float v351_data = r3[3];
                float v354_data = ir4[3];
                ir4[3] = (v354_data + (v332_data * (sycl::group_broadcast(item.get_sub_group(), v351_data, 1))));
              }
              if (v14_lead < 8) {
                float v360_data = r0[2];
                float v361_data = r3[0];
                float v364_data = ir4[0];
                ir4[0] = (v364_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v361_data, 2))));
                float v367_data = r3[1];
                float v370_data = ir4[1];
                ir4[1] = (v370_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v367_data, 2))));
                float v373_data = r3[2];
                float v376_data = ir4[2];
                ir4[2] = (v376_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v373_data, 2))));
                float v379_data = r3[3];
                float v382_data = ir4[3];
                ir4[3] = (v382_data + (v360_data * (sycl::group_broadcast(item.get_sub_group(), v379_data, 2))));
              }
              if (v14_lead < 8) {
                float v388_data = r0[3];
                float v389_data = r3[0];
                float v392_data = ir4[0];
                ir4[0] = (v392_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v389_data, 3))));
                float v395_data = r3[1];
                float v398_data = ir4[1];
                ir4[1] = (v398_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v395_data, 3))));
                float v401_data = r3[2];
                float v404_data = ir4[2];
                ir4[2] = (v404_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v401_data, 3))));
                float v407_data = r3[3];
                float v410_data = ir4[3];
                ir4[3] = (v410_data + (v388_data * (sycl::group_broadcast(item.get_sub_group(), v407_data, 3))));
              }
              if (v14_lead < 8) {
                float v416_data = r0[4];
                float v417_data = r3[0];
                float v420_data = ir4[0];
                ir4[0] = (v420_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v417_data, 4))));
                float v423_data = r3[1];
                float v426_data = ir4[1];
                ir4[1] = (v426_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v423_data, 4))));
                float v429_data = r3[2];
                float v432_data = ir4[2];
                ir4[2] = (v432_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v429_data, 4))));
                float v435_data = r3[3];
                float v438_data = ir4[3];
                ir4[3] = (v438_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v435_data, 4))));
              }
              if (v14_lead < 8) {
                float v444_data = r0[5];
                float v445_data = r3[0];
                float v448_data = ir4[0];
                ir4[0] = (v448_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v445_data, 5))));
                float v451_data = r3[1];
                float v454_data = ir4[1];
                ir4[1] = (v454_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v451_data, 5))));
                float v457_data = r3[2];
                float v460_data = ir4[2];
                ir4[2] = (v460_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v457_data, 5))));
                float v463_data = r3[3];
                float v466_data = ir4[3];
                ir4[3] = (v466_data + (v444_data * (sycl::group_broadcast(item.get_sub_group(), v463_data, 5))));
              }
              if (v14_lead < 8) {
                float v472_data = r0[6];
                float v473_data = r3[0];
                float v476_data = ir4[0];
                ir4[0] = (v476_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v473_data, 6))));
                float v479_data = r3[1];
                float v482_data = ir4[1];
                ir4[1] = (v482_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v479_data, 6))));
                float v485_data = r3[2];
                float v488_data = ir4[2];
                ir4[2] = (v488_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v485_data, 6))));
                float v491_data = r3[3];
                float v494_data = ir4[3];
                ir4[3] = (v494_data + (v472_data * (sycl::group_broadcast(item.get_sub_group(), v491_data, 6))));
              }
              if (v14_lead < 8) {
                float v500_data = r0[7];
                float v501_data = r3[0];
                float v504_data = ir4[0];
                ir4[0] = (v504_data + (v500_data * (sycl::group_broadcast(item.get_sub_group(), v501_data, 7))));
                float v507_data = r3[1];
                float v510_data = ir4[1];
                ir4[1] = (v510_data + (v500_data * (sycl::group_broadcast(item.get_sub_group(), v507_data, 7))));
                float v513_data = r3[2];
                float v516_data = ir4[2];
                ir4[2] = (v516_data + (v500_data * (sycl::group_broadcast(item.get_sub_group(), v513_data, 7))));
                float v519_data = r3[3];
                float v522_data = ir4[3];
                ir4[3] = (v522_data + (v500_data * (sycl::group_broadcast(item.get_sub_group(), v519_data, 7))));
              }
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v528_n1 = 0; v528_n1 < 4; ++v528_n1) {
                  float v530_data = ir4[v528_n1];
                  r4[v528_n1] = v530_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v536_i1 = 0; v536_i1 < 4; ++v536_i1) {
                  float v538_data = r4[v536_i1];
                  int32_t v546_a = v14_lead + ((v536_i1 + 4) * 8);
                  s0[(v546_a ^ ((v546_a >> 5) & 31))] = v538_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m3 = abs(s0)
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v554_k1 = 0; v554_k1 < 8; ++v554_k1) {
                  int32_t v560_a = v554_k1 * 8;
                  int32_t v561_a = v14_lead + v560_a;
                  float v565_data = s0[(v561_a ^ ((v561_a >> 5) & 31))];
                  glb_m3[(v14_lead + v560_a)] = (sycl::fabs(v565_data));
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

