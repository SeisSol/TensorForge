// === base name ===
kernel_95155588a3ec97fe

// === header ===
void launcher_kernel_95155588a3ec97fe(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_95155588a3ec97fe(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_95155588a3ec97fe(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_95155588a3ec97fe(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v18_lead = item.get_local_id(0) % 16;
              if (v18_lead < 8) {
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
                  float v28_data = glb_m0[(v18_lead + (v20_i1 * 8))];
                  r0[v20_i1] = v28_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m1);
              if (v18_lead < 8) {
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 4; ++v35_i1) {
                  float v43_data = glb_m1[(v18_lead + (v35_i1 * 8))];
                  r1[v35_i1] = v43_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[4]{};
              // r3 = load{g>r}(glb_m2);
              if (v18_lead < 8) {
                #pragma unroll
                for (int32_t v50_i1 = 0; v50_i1 < 4; ++v50_i1) {
                  float v58_data = glb_m2[(v18_lead + (v50_i1 * 8))];
                  r3[v50_i1] = v58_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[4]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              if (v18_lead < 8) {
                float v65_data = r0[0];
                float v66_data = r1[0];
                float v69_data = r2[0];
                r2[0] = (v69_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 0))));
                float v72_data = r1[1];
                float v75_data = r2[1];
                r2[1] = (v75_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 0))));
                float v78_data = r1[2];
                float v81_data = r2[2];
                r2[2] = (v81_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 0))));
                float v84_data = r1[3];
                float v87_data = r2[3];
                r2[3] = (v87_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 0))));
              }
              if (v18_lead < 8) {
                float v93_data = r0[1];
                float v94_data = r1[0];
                float v97_data = r2[0];
                r2[0] = (v97_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 1))));
                float v100_data = r1[1];
                float v103_data = r2[1];
                r2[1] = (v103_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 1))));
                float v106_data = r1[2];
                float v109_data = r2[2];
                r2[2] = (v109_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 1))));
                float v112_data = r1[3];
                float v115_data = r2[3];
                r2[3] = (v115_data + (v93_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 1))));
              }
              if (v18_lead < 8) {
                float v121_data = r0[2];
                float v122_data = r1[0];
                float v125_data = r2[0];
                r2[0] = (v125_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 2))));
                float v128_data = r1[1];
                float v131_data = r2[1];
                r2[1] = (v131_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 2))));
                float v134_data = r1[2];
                float v137_data = r2[2];
                r2[2] = (v137_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 2))));
                float v140_data = r1[3];
                float v143_data = r2[3];
                r2[3] = (v143_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 2))));
              }
              if (v18_lead < 8) {
                float v149_data = r0[3];
                float v150_data = r1[0];
                float v153_data = r2[0];
                r2[0] = (v153_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 3))));
                float v156_data = r1[1];
                float v159_data = r2[1];
                r2[1] = (v159_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 3))));
                float v162_data = r1[2];
                float v165_data = r2[2];
                r2[2] = (v165_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 3))));
                float v168_data = r1[3];
                float v171_data = r2[3];
                r2[3] = (v171_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 3))));
              }
              if (v18_lead < 8) {
                float v177_data = r0[4];
                float v178_data = r1[0];
                float v181_data = r2[0];
                r2[0] = (v181_data + (v177_data * (sycl::group_broadcast(item.get_sub_group(), v178_data, 4))));
                float v184_data = r1[1];
                float v187_data = r2[1];
                r2[1] = (v187_data + (v177_data * (sycl::group_broadcast(item.get_sub_group(), v184_data, 4))));
                float v190_data = r1[2];
                float v193_data = r2[2];
                r2[2] = (v193_data + (v177_data * (sycl::group_broadcast(item.get_sub_group(), v190_data, 4))));
                float v196_data = r1[3];
                float v199_data = r2[3];
                r2[3] = (v199_data + (v177_data * (sycl::group_broadcast(item.get_sub_group(), v196_data, 4))));
              }
              if (v18_lead < 8) {
                float v205_data = r0[5];
                float v206_data = r1[0];
                float v209_data = r2[0];
                r2[0] = (v209_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v206_data, 5))));
                float v212_data = r1[1];
                float v215_data = r2[1];
                r2[1] = (v215_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v212_data, 5))));
                float v218_data = r1[2];
                float v221_data = r2[2];
                r2[2] = (v221_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v218_data, 5))));
                float v224_data = r1[3];
                float v227_data = r2[3];
                r2[3] = (v227_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v224_data, 5))));
              }
              if (v18_lead < 8) {
                float v233_data = r0[6];
                float v234_data = r1[0];
                float v237_data = r2[0];
                r2[0] = (v237_data + (v233_data * (sycl::group_broadcast(item.get_sub_group(), v234_data, 6))));
                float v240_data = r1[1];
                float v243_data = r2[1];
                r2[1] = (v243_data + (v233_data * (sycl::group_broadcast(item.get_sub_group(), v240_data, 6))));
                float v246_data = r1[2];
                float v249_data = r2[2];
                r2[2] = (v249_data + (v233_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 6))));
                float v252_data = r1[3];
                float v255_data = r2[3];
                r2[3] = (v255_data + (v233_data * (sycl::group_broadcast(item.get_sub_group(), v252_data, 6))));
              }
              if (v18_lead < 8) {
                float v261_data = r0[7];
                float v262_data = r1[0];
                float v265_data = r2[0];
                r2[0] = (v265_data + (v261_data * (sycl::group_broadcast(item.get_sub_group(), v262_data, 7))));
                float v268_data = r1[1];
                float v271_data = r2[1];
                r2[1] = (v271_data + (v261_data * (sycl::group_broadcast(item.get_sub_group(), v268_data, 7))));
                float v274_data = r1[2];
                float v277_data = r2[2];
                r2[2] = (v277_data + (v261_data * (sycl::group_broadcast(item.get_sub_group(), v274_data, 7))));
                float v280_data = r1[3];
                float v283_data = r2[3];
                r2[3] = (v283_data + (v261_data * (sycl::group_broadcast(item.get_sub_group(), v280_data, 7))));
              }
              // s0 = store{r>s}(localShrMem0, r2);
              if (v18_lead < 8) {
                #pragma unroll
                for (int32_t v289_i1 = 0; v289_i1 < 4; ++v289_i1) {
                  float v291_data = r2[v289_i1];
                  int32_t v298_a = v18_lead + (v289_i1 * 8);
                  s0[(v298_a ^ ((v298_a >> 5) & 31))] = v291_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[4]{};
              // r4 = +(r0 * r3) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir4[4]{};
              if (v18_lead < 8) {
                float v308_data = r0[0];
                float v309_data = r3[0];
                float v312_data = ir4[0];
                ir4[0] = (v312_data + (v308_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 0))));
                float v315_data = r3[1];
                float v318_data = ir4[1];
                ir4[1] = (v318_data + (v308_data * (sycl::group_broadcast(item.get_sub_group(), v315_data, 0))));
                float v321_data = r3[2];
                float v324_data = ir4[2];
                ir4[2] = (v324_data + (v308_data * (sycl::group_broadcast(item.get_sub_group(), v321_data, 0))));
                float v327_data = r3[3];
                float v330_data = ir4[3];
                ir4[3] = (v330_data + (v308_data * (sycl::group_broadcast(item.get_sub_group(), v327_data, 0))));
              }
              if (v18_lead < 8) {
                float v336_data = r0[1];
                float v337_data = r3[0];
                float v340_data = ir4[0];
                ir4[0] = (v340_data + (v336_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 1))));
                float v343_data = r3[1];
                float v346_data = ir4[1];
                ir4[1] = (v346_data + (v336_data * (sycl::group_broadcast(item.get_sub_group(), v343_data, 1))));
                float v349_data = r3[2];
                float v352_data = ir4[2];
                ir4[2] = (v352_data + (v336_data * (sycl::group_broadcast(item.get_sub_group(), v349_data, 1))));
                float v355_data = r3[3];
                float v358_data = ir4[3];
                ir4[3] = (v358_data + (v336_data * (sycl::group_broadcast(item.get_sub_group(), v355_data, 1))));
              }
              if (v18_lead < 8) {
                float v364_data = r0[2];
                float v365_data = r3[0];
                float v368_data = ir4[0];
                ir4[0] = (v368_data + (v364_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 2))));
                float v371_data = r3[1];
                float v374_data = ir4[1];
                ir4[1] = (v374_data + (v364_data * (sycl::group_broadcast(item.get_sub_group(), v371_data, 2))));
                float v377_data = r3[2];
                float v380_data = ir4[2];
                ir4[2] = (v380_data + (v364_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 2))));
                float v383_data = r3[3];
                float v386_data = ir4[3];
                ir4[3] = (v386_data + (v364_data * (sycl::group_broadcast(item.get_sub_group(), v383_data, 2))));
              }
              if (v18_lead < 8) {
                float v392_data = r0[3];
                float v393_data = r3[0];
                float v396_data = ir4[0];
                ir4[0] = (v396_data + (v392_data * (sycl::group_broadcast(item.get_sub_group(), v393_data, 3))));
                float v399_data = r3[1];
                float v402_data = ir4[1];
                ir4[1] = (v402_data + (v392_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 3))));
                float v405_data = r3[2];
                float v408_data = ir4[2];
                ir4[2] = (v408_data + (v392_data * (sycl::group_broadcast(item.get_sub_group(), v405_data, 3))));
                float v411_data = r3[3];
                float v414_data = ir4[3];
                ir4[3] = (v414_data + (v392_data * (sycl::group_broadcast(item.get_sub_group(), v411_data, 3))));
              }
              if (v18_lead < 8) {
                float v420_data = r0[4];
                float v421_data = r3[0];
                float v424_data = ir4[0];
                ir4[0] = (v424_data + (v420_data * (sycl::group_broadcast(item.get_sub_group(), v421_data, 4))));
                float v427_data = r3[1];
                float v430_data = ir4[1];
                ir4[1] = (v430_data + (v420_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 4))));
                float v433_data = r3[2];
                float v436_data = ir4[2];
                ir4[2] = (v436_data + (v420_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 4))));
                float v439_data = r3[3];
                float v442_data = ir4[3];
                ir4[3] = (v442_data + (v420_data * (sycl::group_broadcast(item.get_sub_group(), v439_data, 4))));
              }
              if (v18_lead < 8) {
                float v448_data = r0[5];
                float v449_data = r3[0];
                float v452_data = ir4[0];
                ir4[0] = (v452_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v449_data, 5))));
                float v455_data = r3[1];
                float v458_data = ir4[1];
                ir4[1] = (v458_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v455_data, 5))));
                float v461_data = r3[2];
                float v464_data = ir4[2];
                ir4[2] = (v464_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v461_data, 5))));
                float v467_data = r3[3];
                float v470_data = ir4[3];
                ir4[3] = (v470_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v467_data, 5))));
              }
              if (v18_lead < 8) {
                float v476_data = r0[6];
                float v477_data = r3[0];
                float v480_data = ir4[0];
                ir4[0] = (v480_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v477_data, 6))));
                float v483_data = r3[1];
                float v486_data = ir4[1];
                ir4[1] = (v486_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v483_data, 6))));
                float v489_data = r3[2];
                float v492_data = ir4[2];
                ir4[2] = (v492_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v489_data, 6))));
                float v495_data = r3[3];
                float v498_data = ir4[3];
                ir4[3] = (v498_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v495_data, 6))));
              }
              if (v18_lead < 8) {
                float v504_data = r0[7];
                float v505_data = r3[0];
                float v508_data = ir4[0];
                ir4[0] = (v508_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v505_data, 7))));
                float v511_data = r3[1];
                float v514_data = ir4[1];
                ir4[1] = (v514_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v511_data, 7))));
                float v517_data = r3[2];
                float v520_data = ir4[2];
                ir4[2] = (v520_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v517_data, 7))));
                float v523_data = r3[3];
                float v526_data = ir4[3];
                ir4[3] = (v526_data + (v504_data * (sycl::group_broadcast(item.get_sub_group(), v523_data, 7))));
              }
              if (v18_lead < 8) {
                #pragma unroll
                for (int32_t v532_n1 = 0; v532_n1 < 4; ++v532_n1) {
                  float v534_data = ir4[v532_n1];
                  r4[v532_n1] = v534_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v18_lead < 8) {
                #pragma unroll
                for (int32_t v540_i1 = 0; v540_i1 < 4; ++v540_i1) {
                  float v542_data = r4[v540_i1];
                  int32_t v550_a = v18_lead + ((v540_i1 + 4) * 8);
                  s0[(v550_a ^ ((v550_a >> 5) & 31))] = v542_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m3 = abs(s0)
              if (v18_lead < 8) {
                #pragma unroll
                for (int32_t v558_k1 = 0; v558_k1 < 8; ++v558_k1) {
                  int32_t v564_a = v558_k1 * 8;
                  int32_t v565_a = v18_lead + v564_a;
                  float v569_data = s0[(v565_a ^ ((v565_a >> 5) & 31))];
                  glb_m3[(v18_lead + v564_a)] = (sycl::fabs(v569_data));
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

