// === base name ===
kernel_69f2bb9311

// === header ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_69f2bb9311(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_69f2bb9311(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 35×4(35×4) {0..35}×{0..4} strided
        // m1 35×8(35×8) {0..35}×{0..8} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v6_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
                int32_t v13_lead = v6_lead + (v7_i0 * 32);
                #pragma unroll
                for (int32_t v8_i1 = 0; v8_i1 < 8; ++v8_i1) {
                  float v16_data = glb_m1[(v13_lead + (v8_i1 * 35))];
                  r0[(v7_i0 + (v8_i1 * 2))] = v16_data;
                }
              }
              if (v6_lead < 3) {
                int32_t v25_lead = v6_lead + 32_i32;
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
                  float v28_data = glb_m1[(v25_lead + (v20_i1 * 35))];
                  r0[(1 + (v20_i1 * 2))] = v28_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m2);
              if (v6_lead < 8) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 4; ++v36_i1) {
                  float v44_data = glb_m2[(v6_lead + (v36_i1 * 8))];
                  r1[v36_i1] = v44_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir2[8]{};
              float v51_data = r0[0];
              float v52_data = r1[0];
              float v53_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 0);
              float v55_data = ir2[0];
              ir2[0] = (v55_data + (v51_data * v53_bc));
              float v58_data = r1[1];
              float v59_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 0);
              float v61_data = ir2[2];
              ir2[2] = (v61_data + (v51_data * v59_bc));
              float v64_data = r1[2];
              float v65_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 0);
              float v67_data = ir2[4];
              ir2[4] = (v67_data + (v51_data * v65_bc));
              float v70_data = r1[3];
              float v71_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 0);
              float v73_data = ir2[6];
              ir2[6] = (v73_data + (v51_data * v71_bc));
              if (v6_lead < 3) {
                float v76_data = r0[1];
                float v80_data = ir2[1];
                ir2[1] = (v80_data + (v76_data * v53_bc));
                float v86_data = ir2[3];
                ir2[3] = (v86_data + (v76_data * v59_bc));
                float v92_data = ir2[5];
                ir2[5] = (v92_data + (v76_data * v65_bc));
                float v98_data = ir2[7];
                ir2[7] = (v98_data + (v76_data * v71_bc));
              }
              float v103_data = r0[2];
              float v105_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 1);
              float v107_data = ir2[0];
              ir2[0] = (v107_data + (v103_data * v105_bc));
              float v111_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 1);
              float v113_data = ir2[2];
              ir2[2] = (v113_data + (v103_data * v111_bc));
              float v117_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 1);
              float v119_data = ir2[4];
              ir2[4] = (v119_data + (v103_data * v117_bc));
              float v123_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 1);
              float v125_data = ir2[6];
              ir2[6] = (v125_data + (v103_data * v123_bc));
              if (v6_lead < 3) {
                float v128_data = r0[3];
                float v132_data = ir2[1];
                ir2[1] = (v132_data + (v128_data * v105_bc));
                float v138_data = ir2[3];
                ir2[3] = (v138_data + (v128_data * v111_bc));
                float v144_data = ir2[5];
                ir2[5] = (v144_data + (v128_data * v117_bc));
                float v150_data = ir2[7];
                ir2[7] = (v150_data + (v128_data * v123_bc));
              }
              float v155_data = r0[4];
              float v157_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 2);
              float v159_data = ir2[0];
              ir2[0] = (v159_data + (v155_data * v157_bc));
              float v163_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 2);
              float v165_data = ir2[2];
              ir2[2] = (v165_data + (v155_data * v163_bc));
              float v169_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 2);
              float v171_data = ir2[4];
              ir2[4] = (v171_data + (v155_data * v169_bc));
              float v175_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 2);
              float v177_data = ir2[6];
              ir2[6] = (v177_data + (v155_data * v175_bc));
              if (v6_lead < 3) {
                float v180_data = r0[5];
                float v184_data = ir2[1];
                ir2[1] = (v184_data + (v180_data * v157_bc));
                float v190_data = ir2[3];
                ir2[3] = (v190_data + (v180_data * v163_bc));
                float v196_data = ir2[5];
                ir2[5] = (v196_data + (v180_data * v169_bc));
                float v202_data = ir2[7];
                ir2[7] = (v202_data + (v180_data * v175_bc));
              }
              float v207_data = r0[6];
              float v209_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 3);
              float v211_data = ir2[0];
              ir2[0] = (v211_data + (v207_data * v209_bc));
              float v215_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 3);
              float v217_data = ir2[2];
              ir2[2] = (v217_data + (v207_data * v215_bc));
              float v221_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 3);
              float v223_data = ir2[4];
              ir2[4] = (v223_data + (v207_data * v221_bc));
              float v227_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 3);
              float v229_data = ir2[6];
              ir2[6] = (v229_data + (v207_data * v227_bc));
              if (v6_lead < 3) {
                float v232_data = r0[7];
                float v236_data = ir2[1];
                ir2[1] = (v236_data + (v232_data * v209_bc));
                float v242_data = ir2[3];
                ir2[3] = (v242_data + (v232_data * v215_bc));
                float v248_data = ir2[5];
                ir2[5] = (v248_data + (v232_data * v221_bc));
                float v254_data = ir2[7];
                ir2[7] = (v254_data + (v232_data * v227_bc));
              }
              float v259_data = r0[8];
              float v261_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 4);
              float v263_data = ir2[0];
              ir2[0] = (v263_data + (v259_data * v261_bc));
              float v267_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 4);
              float v269_data = ir2[2];
              ir2[2] = (v269_data + (v259_data * v267_bc));
              float v273_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 4);
              float v275_data = ir2[4];
              ir2[4] = (v275_data + (v259_data * v273_bc));
              float v279_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 4);
              float v281_data = ir2[6];
              ir2[6] = (v281_data + (v259_data * v279_bc));
              if (v6_lead < 3) {
                float v284_data = r0[9];
                float v288_data = ir2[1];
                ir2[1] = (v288_data + (v284_data * v261_bc));
                float v294_data = ir2[3];
                ir2[3] = (v294_data + (v284_data * v267_bc));
                float v300_data = ir2[5];
                ir2[5] = (v300_data + (v284_data * v273_bc));
                float v306_data = ir2[7];
                ir2[7] = (v306_data + (v284_data * v279_bc));
              }
              float v311_data = r0[10];
              float v313_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 5);
              float v315_data = ir2[0];
              ir2[0] = (v315_data + (v311_data * v313_bc));
              float v319_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 5);
              float v321_data = ir2[2];
              ir2[2] = (v321_data + (v311_data * v319_bc));
              float v325_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 5);
              float v327_data = ir2[4];
              ir2[4] = (v327_data + (v311_data * v325_bc));
              float v331_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 5);
              float v333_data = ir2[6];
              ir2[6] = (v333_data + (v311_data * v331_bc));
              if (v6_lead < 3) {
                float v336_data = r0[11];
                float v340_data = ir2[1];
                ir2[1] = (v340_data + (v336_data * v313_bc));
                float v346_data = ir2[3];
                ir2[3] = (v346_data + (v336_data * v319_bc));
                float v352_data = ir2[5];
                ir2[5] = (v352_data + (v336_data * v325_bc));
                float v358_data = ir2[7];
                ir2[7] = (v358_data + (v336_data * v331_bc));
              }
              float v363_data = r0[12];
              float v365_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 6);
              float v367_data = ir2[0];
              ir2[0] = (v367_data + (v363_data * v365_bc));
              float v371_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 6);
              float v373_data = ir2[2];
              ir2[2] = (v373_data + (v363_data * v371_bc));
              float v377_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 6);
              float v379_data = ir2[4];
              ir2[4] = (v379_data + (v363_data * v377_bc));
              float v383_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 6);
              float v385_data = ir2[6];
              ir2[6] = (v385_data + (v363_data * v383_bc));
              if (v6_lead < 3) {
                float v388_data = r0[13];
                float v392_data = ir2[1];
                ir2[1] = (v392_data + (v388_data * v365_bc));
                float v398_data = ir2[3];
                ir2[3] = (v398_data + (v388_data * v371_bc));
                float v404_data = ir2[5];
                ir2[5] = (v404_data + (v388_data * v377_bc));
                float v410_data = ir2[7];
                ir2[7] = (v410_data + (v388_data * v383_bc));
              }
              float v415_data = r0[14];
              float v417_bc = sycl::group_broadcast(item.get_sub_group(), v52_data, 7);
              float v419_data = ir2[0];
              ir2[0] = (v419_data + (v415_data * v417_bc));
              float v423_bc = sycl::group_broadcast(item.get_sub_group(), v58_data, 7);
              float v425_data = ir2[2];
              ir2[2] = (v425_data + (v415_data * v423_bc));
              float v429_bc = sycl::group_broadcast(item.get_sub_group(), v64_data, 7);
              float v431_data = ir2[4];
              ir2[4] = (v431_data + (v415_data * v429_bc));
              float v435_bc = sycl::group_broadcast(item.get_sub_group(), v70_data, 7);
              float v437_data = ir2[6];
              ir2[6] = (v437_data + (v415_data * v435_bc));
              if (v6_lead < 3) {
                float v440_data = r0[15];
                float v444_data = ir2[1];
                ir2[1] = (v444_data + (v440_data * v417_bc));
                float v450_data = ir2[3];
                ir2[3] = (v450_data + (v440_data * v423_bc));
                float v456_data = ir2[5];
                ir2[5] = (v456_data + (v440_data * v429_bc));
                float v462_data = ir2[7];
                ir2[7] = (v462_data + (v440_data * v435_bc));
              }
              #pragma unroll
              for (int32_t v467_n0 = 0; v467_n0 < 1; ++v467_n0) {
                #pragma unroll
                for (int32_t v468_n1 = 0; v468_n1 < 4; ++v468_n1) {
                  int32_t v470_a = v467_n0 + (v468_n1 * 2);
                  float v471_data = ir2[v470_a];
                  r2[v470_a] = v471_data;
                }
              }
              if (v6_lead < 3) {
                #pragma unroll
                for (int32_t v475_n1 = 0; v475_n1 < 4; ++v475_n1) {
                  int32_t v477_a = 1 + (v475_n1 * 2);
                  float v478_data = ir2[v477_a];
                  r2[v477_a] = v478_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v484_i0 = 0; v484_i0 < 1; ++v484_i0) {
                int32_t v493_lead = v6_lead + (v484_i0 * 32);
                #pragma unroll
                for (int32_t v485_i1 = 0; v485_i1 < 4; ++v485_i1) {
                  float v488_data = r2[(v484_i0 + (v485_i1 * 2))];
                  glb_m0[(v493_lead + (v485_i1 * 35))] = v488_data;
                }
              }
              if (v6_lead < 3) {
                int32_t v505_lead = v6_lead + 32_i32;
                #pragma unroll
                for (int32_t v497_i1 = 0; v497_i1 < 4; ++v497_i1) {
                  float v500_data = r2[(1 + (v497_i1 * 2))];
                  glb_m0[(v505_lead + (v497_i1 * 35))] = v500_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

