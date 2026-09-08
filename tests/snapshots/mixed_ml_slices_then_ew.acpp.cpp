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
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 4; ++v26_i1) {
                  float v34_data = glb_m1[(v9_lead + (v26_i1 * 8))];
                  r1[v26_i1] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[4]{};
              // r3 = load{g>r}(glb_m2);
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v41_i1 = 0; v41_i1 < 4; ++v41_i1) {
                  float v49_data = glb_m2[(v9_lead + (v41_i1 * 8))];
                  r3[v41_i1] = v49_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[4]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              if (v9_lead < 8) {
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
              }
              if (v9_lead < 8) {
                float v84_data = r0[1];
                float v85_data = r1[0];
                float v88_data = r2[0];
                r2[0] = (v88_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v85_data, 1))));
                float v91_data = r1[1];
                float v94_data = r2[1];
                r2[1] = (v94_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v91_data, 1))));
                float v97_data = r1[2];
                float v100_data = r2[2];
                r2[2] = (v100_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v97_data, 1))));
                float v103_data = r1[3];
                float v106_data = r2[3];
                r2[3] = (v106_data + (v84_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 1))));
              }
              if (v9_lead < 8) {
                float v112_data = r0[2];
                float v113_data = r1[0];
                float v116_data = r2[0];
                r2[0] = (v116_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 2))));
                float v119_data = r1[1];
                float v122_data = r2[1];
                r2[1] = (v122_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 2))));
                float v125_data = r1[2];
                float v128_data = r2[2];
                r2[2] = (v128_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 2))));
                float v131_data = r1[3];
                float v134_data = r2[3];
                r2[3] = (v134_data + (v112_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 2))));
              }
              if (v9_lead < 8) {
                float v140_data = r0[3];
                float v141_data = r1[0];
                float v144_data = r2[0];
                r2[0] = (v144_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v141_data, 3))));
                float v147_data = r1[1];
                float v150_data = r2[1];
                r2[1] = (v150_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v147_data, 3))));
                float v153_data = r1[2];
                float v156_data = r2[2];
                r2[2] = (v156_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v153_data, 3))));
                float v159_data = r1[3];
                float v162_data = r2[3];
                r2[3] = (v162_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v159_data, 3))));
              }
              if (v9_lead < 8) {
                float v168_data = r0[4];
                float v169_data = r1[0];
                float v172_data = r2[0];
                r2[0] = (v172_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 4))));
                float v175_data = r1[1];
                float v178_data = r2[1];
                r2[1] = (v178_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 4))));
                float v181_data = r1[2];
                float v184_data = r2[2];
                r2[2] = (v184_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v181_data, 4))));
                float v187_data = r1[3];
                float v190_data = r2[3];
                r2[3] = (v190_data + (v168_data * (sycl::group_broadcast(item.get_sub_group(), v187_data, 4))));
              }
              if (v9_lead < 8) {
                float v196_data = r0[5];
                float v197_data = r1[0];
                float v200_data = r2[0];
                r2[0] = (v200_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 5))));
                float v203_data = r1[1];
                float v206_data = r2[1];
                r2[1] = (v206_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 5))));
                float v209_data = r1[2];
                float v212_data = r2[2];
                r2[2] = (v212_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 5))));
                float v215_data = r1[3];
                float v218_data = r2[3];
                r2[3] = (v218_data + (v196_data * (sycl::group_broadcast(item.get_sub_group(), v215_data, 5))));
              }
              if (v9_lead < 8) {
                float v224_data = r0[6];
                float v225_data = r1[0];
                float v228_data = r2[0];
                r2[0] = (v228_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 6))));
                float v231_data = r1[1];
                float v234_data = r2[1];
                r2[1] = (v234_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 6))));
                float v237_data = r1[2];
                float v240_data = r2[2];
                r2[2] = (v240_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v237_data, 6))));
                float v243_data = r1[3];
                float v246_data = r2[3];
                r2[3] = (v246_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v243_data, 6))));
              }
              if (v9_lead < 8) {
                float v252_data = r0[7];
                float v253_data = r1[0];
                float v256_data = r2[0];
                r2[0] = (v256_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v253_data, 7))));
                float v259_data = r1[1];
                float v262_data = r2[1];
                r2[1] = (v262_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v259_data, 7))));
                float v265_data = r1[2];
                float v268_data = r2[2];
                r2[2] = (v268_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v265_data, 7))));
                float v271_data = r1[3];
                float v274_data = r2[3];
                r2[3] = (v274_data + (v252_data * (sycl::group_broadcast(item.get_sub_group(), v271_data, 7))));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r2);
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v281_i1 = 0; v281_i1 < 4; ++v281_i1) {
                  float v283_data = r2[v281_i1];
                  int32_t v290_a = v9_lead + (v281_i1 * 8);
                  s0[(v290_a ^ ((v290_a >> 5) & 31))] = v283_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[4]{};
              // r4 = +(r0 * r3) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir4[4]{};
              if (v9_lead < 8) {
                float v300_data = r0[0];
                float v301_data = r3[0];
                float v304_data = ir4[0];
                ir4[0] = (v304_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v301_data, 0))));
                float v307_data = r3[1];
                float v310_data = ir4[1];
                ir4[1] = (v310_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 0))));
                float v313_data = r3[2];
                float v316_data = ir4[2];
                ir4[2] = (v316_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 0))));
                float v319_data = r3[3];
                float v322_data = ir4[3];
                ir4[3] = (v322_data + (v300_data * (sycl::group_broadcast(item.get_sub_group(), v319_data, 0))));
              }
              if (v9_lead < 8) {
                float v328_data = r0[1];
                float v329_data = r3[0];
                float v332_data = ir4[0];
                ir4[0] = (v332_data + (v328_data * (sycl::group_broadcast(item.get_sub_group(), v329_data, 1))));
                float v335_data = r3[1];
                float v338_data = ir4[1];
                ir4[1] = (v338_data + (v328_data * (sycl::group_broadcast(item.get_sub_group(), v335_data, 1))));
                float v341_data = r3[2];
                float v344_data = ir4[2];
                ir4[2] = (v344_data + (v328_data * (sycl::group_broadcast(item.get_sub_group(), v341_data, 1))));
                float v347_data = r3[3];
                float v350_data = ir4[3];
                ir4[3] = (v350_data + (v328_data * (sycl::group_broadcast(item.get_sub_group(), v347_data, 1))));
              }
              if (v9_lead < 8) {
                float v356_data = r0[2];
                float v357_data = r3[0];
                float v360_data = ir4[0];
                ir4[0] = (v360_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v357_data, 2))));
                float v363_data = r3[1];
                float v366_data = ir4[1];
                ir4[1] = (v366_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v363_data, 2))));
                float v369_data = r3[2];
                float v372_data = ir4[2];
                ir4[2] = (v372_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 2))));
                float v375_data = r3[3];
                float v378_data = ir4[3];
                ir4[3] = (v378_data + (v356_data * (sycl::group_broadcast(item.get_sub_group(), v375_data, 2))));
              }
              if (v9_lead < 8) {
                float v384_data = r0[3];
                float v385_data = r3[0];
                float v388_data = ir4[0];
                ir4[0] = (v388_data + (v384_data * (sycl::group_broadcast(item.get_sub_group(), v385_data, 3))));
                float v391_data = r3[1];
                float v394_data = ir4[1];
                ir4[1] = (v394_data + (v384_data * (sycl::group_broadcast(item.get_sub_group(), v391_data, 3))));
                float v397_data = r3[2];
                float v400_data = ir4[2];
                ir4[2] = (v400_data + (v384_data * (sycl::group_broadcast(item.get_sub_group(), v397_data, 3))));
                float v403_data = r3[3];
                float v406_data = ir4[3];
                ir4[3] = (v406_data + (v384_data * (sycl::group_broadcast(item.get_sub_group(), v403_data, 3))));
              }
              if (v9_lead < 8) {
                float v412_data = r0[4];
                float v413_data = r3[0];
                float v416_data = ir4[0];
                ir4[0] = (v416_data + (v412_data * (sycl::group_broadcast(item.get_sub_group(), v413_data, 4))));
                float v419_data = r3[1];
                float v422_data = ir4[1];
                ir4[1] = (v422_data + (v412_data * (sycl::group_broadcast(item.get_sub_group(), v419_data, 4))));
                float v425_data = r3[2];
                float v428_data = ir4[2];
                ir4[2] = (v428_data + (v412_data * (sycl::group_broadcast(item.get_sub_group(), v425_data, 4))));
                float v431_data = r3[3];
                float v434_data = ir4[3];
                ir4[3] = (v434_data + (v412_data * (sycl::group_broadcast(item.get_sub_group(), v431_data, 4))));
              }
              if (v9_lead < 8) {
                float v440_data = r0[5];
                float v441_data = r3[0];
                float v444_data = ir4[0];
                ir4[0] = (v444_data + (v440_data * (sycl::group_broadcast(item.get_sub_group(), v441_data, 5))));
                float v447_data = r3[1];
                float v450_data = ir4[1];
                ir4[1] = (v450_data + (v440_data * (sycl::group_broadcast(item.get_sub_group(), v447_data, 5))));
                float v453_data = r3[2];
                float v456_data = ir4[2];
                ir4[2] = (v456_data + (v440_data * (sycl::group_broadcast(item.get_sub_group(), v453_data, 5))));
                float v459_data = r3[3];
                float v462_data = ir4[3];
                ir4[3] = (v462_data + (v440_data * (sycl::group_broadcast(item.get_sub_group(), v459_data, 5))));
              }
              if (v9_lead < 8) {
                float v468_data = r0[6];
                float v469_data = r3[0];
                float v472_data = ir4[0];
                ir4[0] = (v472_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v469_data, 6))));
                float v475_data = r3[1];
                float v478_data = ir4[1];
                ir4[1] = (v478_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v475_data, 6))));
                float v481_data = r3[2];
                float v484_data = ir4[2];
                ir4[2] = (v484_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v481_data, 6))));
                float v487_data = r3[3];
                float v490_data = ir4[3];
                ir4[3] = (v490_data + (v468_data * (sycl::group_broadcast(item.get_sub_group(), v487_data, 6))));
              }
              if (v9_lead < 8) {
                float v496_data = r0[7];
                float v497_data = r3[0];
                float v500_data = ir4[0];
                ir4[0] = (v500_data + (v496_data * (sycl::group_broadcast(item.get_sub_group(), v497_data, 7))));
                float v503_data = r3[1];
                float v506_data = ir4[1];
                ir4[1] = (v506_data + (v496_data * (sycl::group_broadcast(item.get_sub_group(), v503_data, 7))));
                float v509_data = r3[2];
                float v512_data = ir4[2];
                ir4[2] = (v512_data + (v496_data * (sycl::group_broadcast(item.get_sub_group(), v509_data, 7))));
                float v515_data = r3[3];
                float v518_data = ir4[3];
                ir4[3] = (v518_data + (v496_data * (sycl::group_broadcast(item.get_sub_group(), v515_data, 7))));
              }
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v524_n1 = 0; v524_n1 < 4; ++v524_n1) {
                  float v526_data = ir4[v524_n1];
                  r4[v524_n1] = v526_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v532_i1 = 0; v532_i1 < 4; ++v532_i1) {
                  float v534_data = r4[v532_i1];
                  int32_t v542_a = v9_lead + ((v532_i1 + 4) * 8);
                  s0[(v542_a ^ ((v542_a >> 5) & 31))] = v534_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m3 = abs(s0)
              if (v9_lead < 8) {
                #pragma unroll
                for (int32_t v550_k1 = 0; v550_k1 < 8; ++v550_k1) {
                  int32_t v556_a = v550_k1 * 8;
                  int32_t v557_a = v9_lead + v556_a;
                  float v561_data = s0[(v557_a ^ ((v557_a >> 5) & 31))];
                  glb_m3[(v9_lead + v556_a)] = (sycl::fabs(v561_data));
                }
              }
            }
          }
        }
      });
    }
  });
}

