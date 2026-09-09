// === base name ===
kernel_aaf4c4c1de1fe420

// === header ===
void launcher_kernel_aaf4c4c1de1fe420(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_aaf4c4c1de1fe420(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_aaf4c4c1de1fe420(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_aaf4c4c1de1fe420(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 35×4(35×4) {0..35}×{0..4} strided
        // m1 35×8(35×8) {0..35}×{0..8} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v10_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v17_lead = v10_lead + (v11_i0 * 32);
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
                  float v20_data = glb_m1[(v17_lead + (v12_i1 * 35))];
                  r0[(v11_i0 + (v12_i1 * 2))] = v20_data;
                }
              }
              if (v10_lead < 3) {
                int32_t v29_lead = v10_lead + 32_i32;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
                  float v32_data = glb_m1[(v29_lead + (v24_i1 * 35))];
                  r0[(1 + (v24_i1 * 2))] = v32_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m2);
              if (v10_lead < 8) {
                #pragma unroll
                for (int32_t v40_i1 = 0; v40_i1 < 4; ++v40_i1) {
                  float v48_data = glb_m2[(v10_lead + (v40_i1 * 8))];
                  r1[v40_i1] = v48_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir2[8]{};
              float v55_data = r0[0];
              float v56_data = r1[0];
              float v57_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 0);
              float v59_data = ir2[0];
              ir2[0] = (v59_data + (v55_data * v57_bc));
              float v62_data = r1[1];
              float v63_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 0);
              float v65_data = ir2[2];
              ir2[2] = (v65_data + (v55_data * v63_bc));
              float v68_data = r1[2];
              float v69_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 0);
              float v71_data = ir2[4];
              ir2[4] = (v71_data + (v55_data * v69_bc));
              float v74_data = r1[3];
              float v75_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 0);
              float v77_data = ir2[6];
              ir2[6] = (v77_data + (v55_data * v75_bc));
              if (v10_lead < 3) {
                float v80_data = r0[1];
                float v84_data = ir2[1];
                ir2[1] = (v84_data + (v80_data * v57_bc));
                float v90_data = ir2[3];
                ir2[3] = (v90_data + (v80_data * v63_bc));
                float v96_data = ir2[5];
                ir2[5] = (v96_data + (v80_data * v69_bc));
                float v102_data = ir2[7];
                ir2[7] = (v102_data + (v80_data * v75_bc));
              }
              float v107_data = r0[2];
              float v109_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 1);
              float v111_data = ir2[0];
              ir2[0] = (v111_data + (v107_data * v109_bc));
              float v115_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 1);
              float v117_data = ir2[2];
              ir2[2] = (v117_data + (v107_data * v115_bc));
              float v121_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 1);
              float v123_data = ir2[4];
              ir2[4] = (v123_data + (v107_data * v121_bc));
              float v127_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 1);
              float v129_data = ir2[6];
              ir2[6] = (v129_data + (v107_data * v127_bc));
              if (v10_lead < 3) {
                float v132_data = r0[3];
                float v136_data = ir2[1];
                ir2[1] = (v136_data + (v132_data * v109_bc));
                float v142_data = ir2[3];
                ir2[3] = (v142_data + (v132_data * v115_bc));
                float v148_data = ir2[5];
                ir2[5] = (v148_data + (v132_data * v121_bc));
                float v154_data = ir2[7];
                ir2[7] = (v154_data + (v132_data * v127_bc));
              }
              float v159_data = r0[4];
              float v161_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 2);
              float v163_data = ir2[0];
              ir2[0] = (v163_data + (v159_data * v161_bc));
              float v167_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 2);
              float v169_data = ir2[2];
              ir2[2] = (v169_data + (v159_data * v167_bc));
              float v173_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 2);
              float v175_data = ir2[4];
              ir2[4] = (v175_data + (v159_data * v173_bc));
              float v179_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 2);
              float v181_data = ir2[6];
              ir2[6] = (v181_data + (v159_data * v179_bc));
              if (v10_lead < 3) {
                float v184_data = r0[5];
                float v188_data = ir2[1];
                ir2[1] = (v188_data + (v184_data * v161_bc));
                float v194_data = ir2[3];
                ir2[3] = (v194_data + (v184_data * v167_bc));
                float v200_data = ir2[5];
                ir2[5] = (v200_data + (v184_data * v173_bc));
                float v206_data = ir2[7];
                ir2[7] = (v206_data + (v184_data * v179_bc));
              }
              float v211_data = r0[6];
              float v213_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 3);
              float v215_data = ir2[0];
              ir2[0] = (v215_data + (v211_data * v213_bc));
              float v219_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 3);
              float v221_data = ir2[2];
              ir2[2] = (v221_data + (v211_data * v219_bc));
              float v225_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 3);
              float v227_data = ir2[4];
              ir2[4] = (v227_data + (v211_data * v225_bc));
              float v231_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 3);
              float v233_data = ir2[6];
              ir2[6] = (v233_data + (v211_data * v231_bc));
              if (v10_lead < 3) {
                float v236_data = r0[7];
                float v240_data = ir2[1];
                ir2[1] = (v240_data + (v236_data * v213_bc));
                float v246_data = ir2[3];
                ir2[3] = (v246_data + (v236_data * v219_bc));
                float v252_data = ir2[5];
                ir2[5] = (v252_data + (v236_data * v225_bc));
                float v258_data = ir2[7];
                ir2[7] = (v258_data + (v236_data * v231_bc));
              }
              float v263_data = r0[8];
              float v265_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 4);
              float v267_data = ir2[0];
              ir2[0] = (v267_data + (v263_data * v265_bc));
              float v271_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 4);
              float v273_data = ir2[2];
              ir2[2] = (v273_data + (v263_data * v271_bc));
              float v277_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 4);
              float v279_data = ir2[4];
              ir2[4] = (v279_data + (v263_data * v277_bc));
              float v283_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 4);
              float v285_data = ir2[6];
              ir2[6] = (v285_data + (v263_data * v283_bc));
              if (v10_lead < 3) {
                float v288_data = r0[9];
                float v292_data = ir2[1];
                ir2[1] = (v292_data + (v288_data * v265_bc));
                float v298_data = ir2[3];
                ir2[3] = (v298_data + (v288_data * v271_bc));
                float v304_data = ir2[5];
                ir2[5] = (v304_data + (v288_data * v277_bc));
                float v310_data = ir2[7];
                ir2[7] = (v310_data + (v288_data * v283_bc));
              }
              float v315_data = r0[10];
              float v317_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 5);
              float v319_data = ir2[0];
              ir2[0] = (v319_data + (v315_data * v317_bc));
              float v323_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 5);
              float v325_data = ir2[2];
              ir2[2] = (v325_data + (v315_data * v323_bc));
              float v329_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 5);
              float v331_data = ir2[4];
              ir2[4] = (v331_data + (v315_data * v329_bc));
              float v335_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 5);
              float v337_data = ir2[6];
              ir2[6] = (v337_data + (v315_data * v335_bc));
              if (v10_lead < 3) {
                float v340_data = r0[11];
                float v344_data = ir2[1];
                ir2[1] = (v344_data + (v340_data * v317_bc));
                float v350_data = ir2[3];
                ir2[3] = (v350_data + (v340_data * v323_bc));
                float v356_data = ir2[5];
                ir2[5] = (v356_data + (v340_data * v329_bc));
                float v362_data = ir2[7];
                ir2[7] = (v362_data + (v340_data * v335_bc));
              }
              float v367_data = r0[12];
              float v369_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 6);
              float v371_data = ir2[0];
              ir2[0] = (v371_data + (v367_data * v369_bc));
              float v375_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 6);
              float v377_data = ir2[2];
              ir2[2] = (v377_data + (v367_data * v375_bc));
              float v381_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 6);
              float v383_data = ir2[4];
              ir2[4] = (v383_data + (v367_data * v381_bc));
              float v387_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 6);
              float v389_data = ir2[6];
              ir2[6] = (v389_data + (v367_data * v387_bc));
              if (v10_lead < 3) {
                float v392_data = r0[13];
                float v396_data = ir2[1];
                ir2[1] = (v396_data + (v392_data * v369_bc));
                float v402_data = ir2[3];
                ir2[3] = (v402_data + (v392_data * v375_bc));
                float v408_data = ir2[5];
                ir2[5] = (v408_data + (v392_data * v381_bc));
                float v414_data = ir2[7];
                ir2[7] = (v414_data + (v392_data * v387_bc));
              }
              float v419_data = r0[14];
              float v421_bc = sycl::group_broadcast(item.get_sub_group(), v56_data, 7);
              float v423_data = ir2[0];
              ir2[0] = (v423_data + (v419_data * v421_bc));
              float v427_bc = sycl::group_broadcast(item.get_sub_group(), v62_data, 7);
              float v429_data = ir2[2];
              ir2[2] = (v429_data + (v419_data * v427_bc));
              float v433_bc = sycl::group_broadcast(item.get_sub_group(), v68_data, 7);
              float v435_data = ir2[4];
              ir2[4] = (v435_data + (v419_data * v433_bc));
              float v439_bc = sycl::group_broadcast(item.get_sub_group(), v74_data, 7);
              float v441_data = ir2[6];
              ir2[6] = (v441_data + (v419_data * v439_bc));
              if (v10_lead < 3) {
                float v444_data = r0[15];
                float v448_data = ir2[1];
                ir2[1] = (v448_data + (v444_data * v421_bc));
                float v454_data = ir2[3];
                ir2[3] = (v454_data + (v444_data * v427_bc));
                float v460_data = ir2[5];
                ir2[5] = (v460_data + (v444_data * v433_bc));
                float v466_data = ir2[7];
                ir2[7] = (v466_data + (v444_data * v439_bc));
              }
              #pragma unroll
              for (int32_t v471_n0 = 0; v471_n0 < 1; ++v471_n0) {
                #pragma unroll
                for (int32_t v472_n1 = 0; v472_n1 < 4; ++v472_n1) {
                  int32_t v474_a = v471_n0 + (v472_n1 * 2);
                  float v475_data = ir2[v474_a];
                  r2[v474_a] = v475_data;
                }
              }
              if (v10_lead < 3) {
                #pragma unroll
                for (int32_t v479_n1 = 0; v479_n1 < 4; ++v479_n1) {
                  int32_t v481_a = 1 + (v479_n1 * 2);
                  float v482_data = ir2[v481_a];
                  r2[v481_a] = v482_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v488_i0 = 0; v488_i0 < 1; ++v488_i0) {
                int32_t v497_lead = v10_lead + (v488_i0 * 32);
                #pragma unroll
                for (int32_t v489_i1 = 0; v489_i1 < 4; ++v489_i1) {
                  float v492_data = r2[(v488_i0 + (v489_i1 * 2))];
                  glb_m0[(v497_lead + (v489_i1 * 35))] = v492_data;
                }
              }
              if (v10_lead < 3) {
                int32_t v509_lead = v10_lead + 32_i32;
                #pragma unroll
                for (int32_t v501_i1 = 0; v501_i1 < 4; ++v501_i1) {
                  float v504_data = r2[(1 + (v501_i1 * 2))];
                  glb_m0[(v509_lead + (v501_i1 * 35))] = v504_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

