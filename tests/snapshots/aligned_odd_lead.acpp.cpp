// === base name ===
kernel_cdd5bfaf6c76f769

// === header ===
void launcher_kernel_cdd5bfaf6c76f769(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_cdd5bfaf6c76f769(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_cdd5bfaf6c76f769(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_cdd5bfaf6c76f769(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 32 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v14_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v21_lead = v14_lead + (v15_i0 * 32);
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                  float v24_data = glb_m1[(v21_lead + (v16_i1 * 35))];
                  r0[(v15_i0 + (v16_i1 * 2))] = v24_data;
                }
              }
              if (v14_lead < 3) {
                int32_t v33_lead = v14_lead + 32_i32;
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                  float v36_data = glb_m1[(v33_lead + (v28_i1 * 35))];
                  r0[(1 + (v28_i1 * 2))] = v36_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m2);
              if (v14_lead < 8) {
                #pragma unroll
                for (int32_t v44_i1 = 0; v44_i1 < 4; ++v44_i1) {
                  float v52_data = glb_m2[(v14_lead + (v44_i1 * 8))];
                  r1[v44_i1] = v52_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir2[8]{};
              float v59_data = r0[0];
              float v60_data = r1[0];
              float v61_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 0);
              float v63_data = ir2[0];
              ir2[0] = (v63_data + (v59_data * v61_bc));
              float v66_data = r1[1];
              float v67_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 0);
              float v69_data = ir2[2];
              ir2[2] = (v69_data + (v59_data * v67_bc));
              float v72_data = r1[2];
              float v73_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 0);
              float v75_data = ir2[4];
              ir2[4] = (v75_data + (v59_data * v73_bc));
              float v78_data = r1[3];
              float v79_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 0);
              float v81_data = ir2[6];
              ir2[6] = (v81_data + (v59_data * v79_bc));
              if (v14_lead < 3) {
                float v84_data = r0[1];
                float v88_data = ir2[1];
                ir2[1] = (v88_data + (v84_data * v61_bc));
                float v94_data = ir2[3];
                ir2[3] = (v94_data + (v84_data * v67_bc));
                float v100_data = ir2[5];
                ir2[5] = (v100_data + (v84_data * v73_bc));
                float v106_data = ir2[7];
                ir2[7] = (v106_data + (v84_data * v79_bc));
              }
              float v111_data = r0[2];
              float v113_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 1);
              float v115_data = ir2[0];
              ir2[0] = (v115_data + (v111_data * v113_bc));
              float v119_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 1);
              float v121_data = ir2[2];
              ir2[2] = (v121_data + (v111_data * v119_bc));
              float v125_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 1);
              float v127_data = ir2[4];
              ir2[4] = (v127_data + (v111_data * v125_bc));
              float v131_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 1);
              float v133_data = ir2[6];
              ir2[6] = (v133_data + (v111_data * v131_bc));
              if (v14_lead < 3) {
                float v136_data = r0[3];
                float v140_data = ir2[1];
                ir2[1] = (v140_data + (v136_data * v113_bc));
                float v146_data = ir2[3];
                ir2[3] = (v146_data + (v136_data * v119_bc));
                float v152_data = ir2[5];
                ir2[5] = (v152_data + (v136_data * v125_bc));
                float v158_data = ir2[7];
                ir2[7] = (v158_data + (v136_data * v131_bc));
              }
              float v163_data = r0[4];
              float v165_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 2);
              float v167_data = ir2[0];
              ir2[0] = (v167_data + (v163_data * v165_bc));
              float v171_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 2);
              float v173_data = ir2[2];
              ir2[2] = (v173_data + (v163_data * v171_bc));
              float v177_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 2);
              float v179_data = ir2[4];
              ir2[4] = (v179_data + (v163_data * v177_bc));
              float v183_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 2);
              float v185_data = ir2[6];
              ir2[6] = (v185_data + (v163_data * v183_bc));
              if (v14_lead < 3) {
                float v188_data = r0[5];
                float v192_data = ir2[1];
                ir2[1] = (v192_data + (v188_data * v165_bc));
                float v198_data = ir2[3];
                ir2[3] = (v198_data + (v188_data * v171_bc));
                float v204_data = ir2[5];
                ir2[5] = (v204_data + (v188_data * v177_bc));
                float v210_data = ir2[7];
                ir2[7] = (v210_data + (v188_data * v183_bc));
              }
              float v215_data = r0[6];
              float v217_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 3);
              float v219_data = ir2[0];
              ir2[0] = (v219_data + (v215_data * v217_bc));
              float v223_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 3);
              float v225_data = ir2[2];
              ir2[2] = (v225_data + (v215_data * v223_bc));
              float v229_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 3);
              float v231_data = ir2[4];
              ir2[4] = (v231_data + (v215_data * v229_bc));
              float v235_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 3);
              float v237_data = ir2[6];
              ir2[6] = (v237_data + (v215_data * v235_bc));
              if (v14_lead < 3) {
                float v240_data = r0[7];
                float v244_data = ir2[1];
                ir2[1] = (v244_data + (v240_data * v217_bc));
                float v250_data = ir2[3];
                ir2[3] = (v250_data + (v240_data * v223_bc));
                float v256_data = ir2[5];
                ir2[5] = (v256_data + (v240_data * v229_bc));
                float v262_data = ir2[7];
                ir2[7] = (v262_data + (v240_data * v235_bc));
              }
              float v267_data = r0[8];
              float v269_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 4);
              float v271_data = ir2[0];
              ir2[0] = (v271_data + (v267_data * v269_bc));
              float v275_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 4);
              float v277_data = ir2[2];
              ir2[2] = (v277_data + (v267_data * v275_bc));
              float v281_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 4);
              float v283_data = ir2[4];
              ir2[4] = (v283_data + (v267_data * v281_bc));
              float v287_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 4);
              float v289_data = ir2[6];
              ir2[6] = (v289_data + (v267_data * v287_bc));
              if (v14_lead < 3) {
                float v292_data = r0[9];
                float v296_data = ir2[1];
                ir2[1] = (v296_data + (v292_data * v269_bc));
                float v302_data = ir2[3];
                ir2[3] = (v302_data + (v292_data * v275_bc));
                float v308_data = ir2[5];
                ir2[5] = (v308_data + (v292_data * v281_bc));
                float v314_data = ir2[7];
                ir2[7] = (v314_data + (v292_data * v287_bc));
              }
              float v319_data = r0[10];
              float v321_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 5);
              float v323_data = ir2[0];
              ir2[0] = (v323_data + (v319_data * v321_bc));
              float v327_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 5);
              float v329_data = ir2[2];
              ir2[2] = (v329_data + (v319_data * v327_bc));
              float v333_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 5);
              float v335_data = ir2[4];
              ir2[4] = (v335_data + (v319_data * v333_bc));
              float v339_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 5);
              float v341_data = ir2[6];
              ir2[6] = (v341_data + (v319_data * v339_bc));
              if (v14_lead < 3) {
                float v344_data = r0[11];
                float v348_data = ir2[1];
                ir2[1] = (v348_data + (v344_data * v321_bc));
                float v354_data = ir2[3];
                ir2[3] = (v354_data + (v344_data * v327_bc));
                float v360_data = ir2[5];
                ir2[5] = (v360_data + (v344_data * v333_bc));
                float v366_data = ir2[7];
                ir2[7] = (v366_data + (v344_data * v339_bc));
              }
              float v371_data = r0[12];
              float v373_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 6);
              float v375_data = ir2[0];
              ir2[0] = (v375_data + (v371_data * v373_bc));
              float v379_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 6);
              float v381_data = ir2[2];
              ir2[2] = (v381_data + (v371_data * v379_bc));
              float v385_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 6);
              float v387_data = ir2[4];
              ir2[4] = (v387_data + (v371_data * v385_bc));
              float v391_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 6);
              float v393_data = ir2[6];
              ir2[6] = (v393_data + (v371_data * v391_bc));
              if (v14_lead < 3) {
                float v396_data = r0[13];
                float v400_data = ir2[1];
                ir2[1] = (v400_data + (v396_data * v373_bc));
                float v406_data = ir2[3];
                ir2[3] = (v406_data + (v396_data * v379_bc));
                float v412_data = ir2[5];
                ir2[5] = (v412_data + (v396_data * v385_bc));
                float v418_data = ir2[7];
                ir2[7] = (v418_data + (v396_data * v391_bc));
              }
              float v423_data = r0[14];
              float v425_bc = sycl::group_broadcast(item.get_sub_group(), v60_data, 7);
              float v427_data = ir2[0];
              ir2[0] = (v427_data + (v423_data * v425_bc));
              float v431_bc = sycl::group_broadcast(item.get_sub_group(), v66_data, 7);
              float v433_data = ir2[2];
              ir2[2] = (v433_data + (v423_data * v431_bc));
              float v437_bc = sycl::group_broadcast(item.get_sub_group(), v72_data, 7);
              float v439_data = ir2[4];
              ir2[4] = (v439_data + (v423_data * v437_bc));
              float v443_bc = sycl::group_broadcast(item.get_sub_group(), v78_data, 7);
              float v445_data = ir2[6];
              ir2[6] = (v445_data + (v423_data * v443_bc));
              if (v14_lead < 3) {
                float v448_data = r0[15];
                float v452_data = ir2[1];
                ir2[1] = (v452_data + (v448_data * v425_bc));
                float v458_data = ir2[3];
                ir2[3] = (v458_data + (v448_data * v431_bc));
                float v464_data = ir2[5];
                ir2[5] = (v464_data + (v448_data * v437_bc));
                float v470_data = ir2[7];
                ir2[7] = (v470_data + (v448_data * v443_bc));
              }
              #pragma unroll
              for (int32_t v475_n0 = 0; v475_n0 < 1; ++v475_n0) {
                #pragma unroll
                for (int32_t v476_n1 = 0; v476_n1 < 4; ++v476_n1) {
                  int32_t v478_a = v475_n0 + (v476_n1 * 2);
                  float v479_data = ir2[v478_a];
                  r2[v478_a] = v479_data;
                }
              }
              if (v14_lead < 3) {
                #pragma unroll
                for (int32_t v483_n1 = 0; v483_n1 < 4; ++v483_n1) {
                  int32_t v485_a = 1 + (v483_n1 * 2);
                  float v486_data = ir2[v485_a];
                  r2[v485_a] = v486_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v492_i0 = 0; v492_i0 < 1; ++v492_i0) {
                int32_t v501_lead = v14_lead + (v492_i0 * 32);
                #pragma unroll
                for (int32_t v493_i1 = 0; v493_i1 < 4; ++v493_i1) {
                  float v496_data = r2[(v492_i0 + (v493_i1 * 2))];
                  glb_m0[(v501_lead + (v493_i1 * 35))] = v496_data;
                }
              }
              if (v14_lead < 3) {
                int32_t v513_lead = v14_lead + 32_i32;
                #pragma unroll
                for (int32_t v505_i1 = 0; v505_i1 < 4; ++v505_i1) {
                  float v508_data = r2[(1 + (v505_i1 * 2))];
                  glb_m0[(v513_lead + (v505_i1 * 35))] = v508_data;
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

