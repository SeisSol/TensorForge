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
              float v32_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v32_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir2[8]{};
              float v38_data = r0[0];
              float v39_data = r1[0];
              float v40_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 0);
              float v42_data = ir2[0];
              ir2[0] = (v42_data + (v38_data * v40_bc));
              float v45_data = r1[1];
              float v46_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 0);
              float v48_data = ir2[2];
              ir2[2] = (v48_data + (v38_data * v46_bc));
              float v51_data = r1[2];
              float v52_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 0);
              float v54_data = ir2[4];
              ir2[4] = (v54_data + (v38_data * v52_bc));
              float v57_data = r1[3];
              float v58_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 0);
              float v60_data = ir2[6];
              ir2[6] = (v60_data + (v38_data * v58_bc));
              if (v6_lead < 3) {
                float v63_data = r0[1];
                float v67_data = ir2[1];
                ir2[1] = (v67_data + (v63_data * v40_bc));
                float v73_data = ir2[3];
                ir2[3] = (v73_data + (v63_data * v46_bc));
                float v79_data = ir2[5];
                ir2[5] = (v79_data + (v63_data * v52_bc));
                float v85_data = ir2[7];
                ir2[7] = (v85_data + (v63_data * v58_bc));
              }
              float v90_data = r0[2];
              float v92_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 1);
              float v94_data = ir2[0];
              ir2[0] = (v94_data + (v90_data * v92_bc));
              float v98_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 1);
              float v100_data = ir2[2];
              ir2[2] = (v100_data + (v90_data * v98_bc));
              float v104_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 1);
              float v106_data = ir2[4];
              ir2[4] = (v106_data + (v90_data * v104_bc));
              float v110_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 1);
              float v112_data = ir2[6];
              ir2[6] = (v112_data + (v90_data * v110_bc));
              if (v6_lead < 3) {
                float v115_data = r0[3];
                float v119_data = ir2[1];
                ir2[1] = (v119_data + (v115_data * v92_bc));
                float v125_data = ir2[3];
                ir2[3] = (v125_data + (v115_data * v98_bc));
                float v131_data = ir2[5];
                ir2[5] = (v131_data + (v115_data * v104_bc));
                float v137_data = ir2[7];
                ir2[7] = (v137_data + (v115_data * v110_bc));
              }
              float v142_data = r0[4];
              float v144_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 2);
              float v146_data = ir2[0];
              ir2[0] = (v146_data + (v142_data * v144_bc));
              float v150_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 2);
              float v152_data = ir2[2];
              ir2[2] = (v152_data + (v142_data * v150_bc));
              float v156_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 2);
              float v158_data = ir2[4];
              ir2[4] = (v158_data + (v142_data * v156_bc));
              float v162_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 2);
              float v164_data = ir2[6];
              ir2[6] = (v164_data + (v142_data * v162_bc));
              if (v6_lead < 3) {
                float v167_data = r0[5];
                float v171_data = ir2[1];
                ir2[1] = (v171_data + (v167_data * v144_bc));
                float v177_data = ir2[3];
                ir2[3] = (v177_data + (v167_data * v150_bc));
                float v183_data = ir2[5];
                ir2[5] = (v183_data + (v167_data * v156_bc));
                float v189_data = ir2[7];
                ir2[7] = (v189_data + (v167_data * v162_bc));
              }
              float v194_data = r0[6];
              float v196_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 3);
              float v198_data = ir2[0];
              ir2[0] = (v198_data + (v194_data * v196_bc));
              float v202_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 3);
              float v204_data = ir2[2];
              ir2[2] = (v204_data + (v194_data * v202_bc));
              float v208_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 3);
              float v210_data = ir2[4];
              ir2[4] = (v210_data + (v194_data * v208_bc));
              float v214_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 3);
              float v216_data = ir2[6];
              ir2[6] = (v216_data + (v194_data * v214_bc));
              if (v6_lead < 3) {
                float v219_data = r0[7];
                float v223_data = ir2[1];
                ir2[1] = (v223_data + (v219_data * v196_bc));
                float v229_data = ir2[3];
                ir2[3] = (v229_data + (v219_data * v202_bc));
                float v235_data = ir2[5];
                ir2[5] = (v235_data + (v219_data * v208_bc));
                float v241_data = ir2[7];
                ir2[7] = (v241_data + (v219_data * v214_bc));
              }
              float v246_data = r0[8];
              float v248_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 4);
              float v250_data = ir2[0];
              ir2[0] = (v250_data + (v246_data * v248_bc));
              float v254_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 4);
              float v256_data = ir2[2];
              ir2[2] = (v256_data + (v246_data * v254_bc));
              float v260_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 4);
              float v262_data = ir2[4];
              ir2[4] = (v262_data + (v246_data * v260_bc));
              float v266_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 4);
              float v268_data = ir2[6];
              ir2[6] = (v268_data + (v246_data * v266_bc));
              if (v6_lead < 3) {
                float v271_data = r0[9];
                float v275_data = ir2[1];
                ir2[1] = (v275_data + (v271_data * v248_bc));
                float v281_data = ir2[3];
                ir2[3] = (v281_data + (v271_data * v254_bc));
                float v287_data = ir2[5];
                ir2[5] = (v287_data + (v271_data * v260_bc));
                float v293_data = ir2[7];
                ir2[7] = (v293_data + (v271_data * v266_bc));
              }
              float v298_data = r0[10];
              float v300_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 5);
              float v302_data = ir2[0];
              ir2[0] = (v302_data + (v298_data * v300_bc));
              float v306_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 5);
              float v308_data = ir2[2];
              ir2[2] = (v308_data + (v298_data * v306_bc));
              float v312_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 5);
              float v314_data = ir2[4];
              ir2[4] = (v314_data + (v298_data * v312_bc));
              float v318_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 5);
              float v320_data = ir2[6];
              ir2[6] = (v320_data + (v298_data * v318_bc));
              if (v6_lead < 3) {
                float v323_data = r0[11];
                float v327_data = ir2[1];
                ir2[1] = (v327_data + (v323_data * v300_bc));
                float v333_data = ir2[3];
                ir2[3] = (v333_data + (v323_data * v306_bc));
                float v339_data = ir2[5];
                ir2[5] = (v339_data + (v323_data * v312_bc));
                float v345_data = ir2[7];
                ir2[7] = (v345_data + (v323_data * v318_bc));
              }
              float v350_data = r0[12];
              float v352_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 6);
              float v354_data = ir2[0];
              ir2[0] = (v354_data + (v350_data * v352_bc));
              float v358_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 6);
              float v360_data = ir2[2];
              ir2[2] = (v360_data + (v350_data * v358_bc));
              float v364_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 6);
              float v366_data = ir2[4];
              ir2[4] = (v366_data + (v350_data * v364_bc));
              float v370_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 6);
              float v372_data = ir2[6];
              ir2[6] = (v372_data + (v350_data * v370_bc));
              if (v6_lead < 3) {
                float v375_data = r0[13];
                float v379_data = ir2[1];
                ir2[1] = (v379_data + (v375_data * v352_bc));
                float v385_data = ir2[3];
                ir2[3] = (v385_data + (v375_data * v358_bc));
                float v391_data = ir2[5];
                ir2[5] = (v391_data + (v375_data * v364_bc));
                float v397_data = ir2[7];
                ir2[7] = (v397_data + (v375_data * v370_bc));
              }
              float v402_data = r0[14];
              float v404_bc = sycl::group_broadcast(item.get_sub_group(), v39_data, 7);
              float v406_data = ir2[0];
              ir2[0] = (v406_data + (v402_data * v404_bc));
              float v410_bc = sycl::group_broadcast(item.get_sub_group(), v45_data, 7);
              float v412_data = ir2[2];
              ir2[2] = (v412_data + (v402_data * v410_bc));
              float v416_bc = sycl::group_broadcast(item.get_sub_group(), v51_data, 7);
              float v418_data = ir2[4];
              ir2[4] = (v418_data + (v402_data * v416_bc));
              float v422_bc = sycl::group_broadcast(item.get_sub_group(), v57_data, 7);
              float v424_data = ir2[6];
              ir2[6] = (v424_data + (v402_data * v422_bc));
              if (v6_lead < 3) {
                float v427_data = r0[15];
                float v431_data = ir2[1];
                ir2[1] = (v431_data + (v427_data * v404_bc));
                float v437_data = ir2[3];
                ir2[3] = (v437_data + (v427_data * v410_bc));
                float v443_data = ir2[5];
                ir2[5] = (v443_data + (v427_data * v416_bc));
                float v449_data = ir2[7];
                ir2[7] = (v449_data + (v427_data * v422_bc));
              }
              #pragma unroll
              for (int32_t v454_n0 = 0; v454_n0 < 1; ++v454_n0) {
                #pragma unroll
                for (int32_t v455_n1 = 0; v455_n1 < 4; ++v455_n1) {
                  int32_t v457_a = v454_n0 + (v455_n1 * 2);
                  float v458_data = ir2[v457_a];
                  r2[v457_a] = v458_data;
                }
              }
              if (v6_lead < 3) {
                #pragma unroll
                for (int32_t v462_n1 = 0; v462_n1 < 4; ++v462_n1) {
                  int32_t v464_a = 1 + (v462_n1 * 2);
                  float v465_data = ir2[v464_a];
                  r2[v464_a] = v465_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v471_i0 = 0; v471_i0 < 1; ++v471_i0) {
                int32_t v480_lead = v6_lead + (v471_i0 * 32);
                #pragma unroll
                for (int32_t v472_i1 = 0; v472_i1 < 4; ++v472_i1) {
                  float v475_data = r2[(v471_i0 + (v472_i1 * 2))];
                  glb_m0[(v480_lead + (v472_i1 * 35))] = v475_data;
                }
              }
              if (v6_lead < 3) {
                int32_t v492_lead = v6_lead + 32_i32;
                #pragma unroll
                for (int32_t v484_i1 = 0; v484_i1 < 4; ++v484_i1) {
                  float v487_data = r2[(1 + (v484_i1 * 2))];
                  glb_m0[(v492_lead + (v484_i1 * 35))] = v487_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

