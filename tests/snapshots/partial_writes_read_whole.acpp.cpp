// === base name ===
kernel_5d05ae1888f3849d

// === header ===
void launcher_kernel_5d05ae1888f3849d(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5d05ae1888f3849d(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_5d05ae1888f3849d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5d05ae1888f3849d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (288, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×9(32×9) {0..32}×{0..9} pointer_based
        // m1 16×9(16×9) {0..16}×{0..9} pointer_based
        // m2 16×9(16×9) {0..16}×{0..9} pointer_based
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based
        // m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[288 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[288];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v3_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v3_batchId0][0 + m4_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v19_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v26_lead = v19_lead + (v20_i0 * 32);
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 9; ++v21_i1) {
                  float v29_data = glb_m0[(v26_lead + (v21_i1 * 32))];
                  r0[(v20_i0 + v21_i1)] = v29_data;
                }
              }
              float r2[9]{};
              // r2 = load{g>r}(glb_m1);
              if (v19_lead < 16) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
                  float v44_data = glb_m1[(v19_lead + (v36_i1 * 16))];
                  r2[v36_i1] = v44_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[9]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              float v50_data = r0[0];
              float v51_data = r1[0];
              r1[0] = (v51_data + v50_data);
              float v53_data = r0[1];
              float v54_data = r1[1];
              r1[1] = (v54_data + v53_data);
              float v56_data = r0[2];
              float v57_data = r1[2];
              r1[2] = (v57_data + v56_data);
              float v59_data = r0[3];
              float v60_data = r1[3];
              r1[3] = (v60_data + v59_data);
              float v62_data = r0[4];
              float v63_data = r1[4];
              r1[4] = (v63_data + v62_data);
              float v65_data = r0[5];
              float v66_data = r1[5];
              r1[5] = (v66_data + v65_data);
              float v68_data = r0[6];
              float v69_data = r1[6];
              r1[6] = (v69_data + v68_data);
              float v71_data = r0[7];
              float v72_data = r1[7];
              r1[7] = (v72_data + v71_data);
              float v74_data = r0[8];
              float v75_data = r1[8];
              r1[8] = (v75_data + v74_data);
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v80_i0 = 0; v80_i0 < 1; ++v80_i0) {
                int32_t v88_lead = v19_lead + (v80_i0 * 32);
                #pragma unroll
                for (int32_t v81_i1 = 0; v81_i1 < 9; ++v81_i1) {
                  float v83_data = r1[(v80_i0 + v81_i1)];
                  int32_t v90_a = v88_lead + (v81_i1 * 32);
                  s0[(v90_a ^ ((v90_a >> 5) & 31))] = v83_data;
                }
              }
              float r4[9]{};
              // r4 = load{g>r}(glb_m2);
              if (v19_lead < 16) {
                #pragma unroll
                for (int32_t v99_i1 = 0; v99_i1 < 9; ++v99_i1) {
                  float v107_data = glb_m2[(v19_lead + (v99_i1 * 16))];
                  r4[v99_i1] = v107_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[9]{};
              item.barrier();
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir3[9]{};
              if (v19_lead < 16) {
                float v115_data = r2[0];
                float v116_data = ir3[0];
                ir3[0] = (v116_data + v115_data);
                float v118_data = r2[1];
                float v119_data = ir3[1];
                ir3[1] = (v119_data + v118_data);
                float v121_data = r2[2];
                float v122_data = ir3[2];
                ir3[2] = (v122_data + v121_data);
                float v124_data = r2[3];
                float v125_data = ir3[3];
                ir3[3] = (v125_data + v124_data);
                float v127_data = r2[4];
                float v128_data = ir3[4];
                ir3[4] = (v128_data + v127_data);
                float v130_data = r2[5];
                float v131_data = ir3[5];
                ir3[5] = (v131_data + v130_data);
                float v133_data = r2[6];
                float v134_data = ir3[6];
                ir3[6] = (v134_data + v133_data);
                float v136_data = r2[7];
                float v137_data = ir3[7];
                ir3[7] = (v137_data + v136_data);
                float v139_data = r2[8];
                float v140_data = ir3[8];
                ir3[8] = (v140_data + v139_data);
              }
              if (v19_lead < 16) {
                #pragma unroll
                for (int32_t v146_n1 = 0; v146_n1 < 9; ++v146_n1) {
                  float v148_data = ir3[v146_n1];
                  int32_t v155_a = v19_lead + (v146_n1 * 32);
                  float v159_data = s0[(v155_a ^ ((v155_a >> 5) & 31))];
                  r3[v146_n1] = (v159_data + v148_data);
                }
              }
              item.barrier();
              // s0 = store{r>s}(localShrMem0, r3);
              if (v19_lead < 16) {
                #pragma unroll
                for (int32_t v166_i1 = 0; v166_i1 < 9; ++v166_i1) {
                  float v168_data = r3[v166_i1];
                  int32_t v175_a = v19_lead + (v166_i1 * 32);
                  s0[(v175_a ^ ((v175_a >> 5) & 31))] = v168_data;
                }
              }
              float r6[9]{};
              // r6 = load{g>r}(glb_m4);
              if (v19_lead < 9) {
                #pragma unroll
                for (int32_t v184_i1 = 0; v184_i1 < 9; ++v184_i1) {
                  float v192_data = glb_m4[(v19_lead + (v184_i1 * 9))];
                  r6[v184_i1] = v192_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[9]{};
              item.barrier();
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir5[9]{};
              if (v19_lead < 16) {
                float v200_data = r4[0];
                float v201_data = ir5[0];
                ir5[0] = (v201_data + v200_data);
                float v203_data = r4[1];
                float v204_data = ir5[1];
                ir5[1] = (v204_data + v203_data);
                float v206_data = r4[2];
                float v207_data = ir5[2];
                ir5[2] = (v207_data + v206_data);
                float v209_data = r4[3];
                float v210_data = ir5[3];
                ir5[3] = (v210_data + v209_data);
                float v212_data = r4[4];
                float v213_data = ir5[4];
                ir5[4] = (v213_data + v212_data);
                float v215_data = r4[5];
                float v216_data = ir5[5];
                ir5[5] = (v216_data + v215_data);
                float v218_data = r4[6];
                float v219_data = ir5[6];
                ir5[6] = (v219_data + v218_data);
                float v221_data = r4[7];
                float v222_data = ir5[7];
                ir5[7] = (v222_data + v221_data);
                float v224_data = r4[8];
                float v225_data = ir5[8];
                ir5[8] = (v225_data + v224_data);
              }
              if (v19_lead < 16) {
                #pragma unroll
                for (int32_t v231_n1 = 0; v231_n1 < 9; ++v231_n1) {
                  float v233_data = ir5[v231_n1];
                  int32_t v240_a = v19_lead + (v231_n1 * 32);
                  float v244_data = s0[(v240_a ^ ((v240_a >> 5) & 31))];
                  r5[v231_n1] = (v244_data + v233_data);
                }
              }
              item.barrier();
              // s0 = store{r>s}(localShrMem0, r5);
              if (v19_lead < 16) {
                #pragma unroll
                for (int32_t v251_i1 = 0; v251_i1 < 9; ++v251_i1) {
                  float v253_data = r5[v251_i1];
                  int32_t v260_a = v19_lead + (v251_i1 * 32);
                  s0[(v260_a ^ ((v260_a >> 5) & 31))] = v253_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[9]{};
              item.barrier();
              // r7 = +(s0 * r6) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir7[9]{};
              float v278_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v279_data = r6[0];
              float v282_data = ir7[0];
              ir7[0] = (v282_data + (v278_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 0))));
              float v293_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v294_data = r6[1];
              float v297_data = ir7[1];
              ir7[1] = (v297_data + (v293_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 0))));
              float v308_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v309_data = r6[2];
              float v312_data = ir7[2];
              ir7[2] = (v312_data + (v308_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 0))));
              float v323_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v324_data = r6[3];
              float v327_data = ir7[3];
              ir7[3] = (v327_data + (v323_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 0))));
              float v338_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v339_data = r6[4];
              float v342_data = ir7[4];
              ir7[4] = (v342_data + (v338_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 0))));
              float v353_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v354_data = r6[5];
              float v357_data = ir7[5];
              ir7[5] = (v357_data + (v353_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 0))));
              float v368_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v369_data = r6[6];
              float v372_data = ir7[6];
              ir7[6] = (v372_data + (v368_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 0))));
              float v383_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v384_data = r6[7];
              float v387_data = ir7[7];
              ir7[7] = (v387_data + (v383_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 0))));
              float v398_data = s0[(v19_lead ^ ((v19_lead >> 5) & 31))];
              float v399_data = r6[8];
              float v402_data = ir7[8];
              ir7[8] = (v402_data + (v398_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 0))));
              int32_t v412_a = v19_lead + 32;
              float v416_data = s0[(v412_a ^ ((v412_a >> 5) & 31))];
              float v420_data = ir7[0];
              ir7[0] = (v420_data + (v416_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 1))));
              int32_t v427_a = v19_lead + 32;
              float v431_data = s0[(v427_a ^ ((v427_a >> 5) & 31))];
              float v435_data = ir7[1];
              ir7[1] = (v435_data + (v431_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 1))));
              int32_t v442_a = v19_lead + 32;
              float v446_data = s0[(v442_a ^ ((v442_a >> 5) & 31))];
              float v450_data = ir7[2];
              ir7[2] = (v450_data + (v446_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 1))));
              int32_t v457_a = v19_lead + 32;
              float v461_data = s0[(v457_a ^ ((v457_a >> 5) & 31))];
              float v465_data = ir7[3];
              ir7[3] = (v465_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 1))));
              int32_t v472_a = v19_lead + 32;
              float v476_data = s0[(v472_a ^ ((v472_a >> 5) & 31))];
              float v480_data = ir7[4];
              ir7[4] = (v480_data + (v476_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 1))));
              int32_t v487_a = v19_lead + 32;
              float v491_data = s0[(v487_a ^ ((v487_a >> 5) & 31))];
              float v495_data = ir7[5];
              ir7[5] = (v495_data + (v491_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 1))));
              int32_t v502_a = v19_lead + 32;
              float v506_data = s0[(v502_a ^ ((v502_a >> 5) & 31))];
              float v510_data = ir7[6];
              ir7[6] = (v510_data + (v506_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 1))));
              int32_t v517_a = v19_lead + 32;
              float v521_data = s0[(v517_a ^ ((v517_a >> 5) & 31))];
              float v525_data = ir7[7];
              ir7[7] = (v525_data + (v521_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 1))));
              int32_t v532_a = v19_lead + 32;
              float v536_data = s0[(v532_a ^ ((v532_a >> 5) & 31))];
              float v540_data = ir7[8];
              ir7[8] = (v540_data + (v536_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 1))));
              int32_t v550_a = v19_lead + 64;
              float v554_data = s0[(v550_a ^ ((v550_a >> 5) & 31))];
              float v558_data = ir7[0];
              ir7[0] = (v558_data + (v554_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 2))));
              int32_t v565_a = v19_lead + 64;
              float v569_data = s0[(v565_a ^ ((v565_a >> 5) & 31))];
              float v573_data = ir7[1];
              ir7[1] = (v573_data + (v569_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 2))));
              int32_t v580_a = v19_lead + 64;
              float v584_data = s0[(v580_a ^ ((v580_a >> 5) & 31))];
              float v588_data = ir7[2];
              ir7[2] = (v588_data + (v584_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 2))));
              int32_t v595_a = v19_lead + 64;
              float v599_data = s0[(v595_a ^ ((v595_a >> 5) & 31))];
              float v603_data = ir7[3];
              ir7[3] = (v603_data + (v599_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 2))));
              int32_t v610_a = v19_lead + 64;
              float v614_data = s0[(v610_a ^ ((v610_a >> 5) & 31))];
              float v618_data = ir7[4];
              ir7[4] = (v618_data + (v614_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 2))));
              int32_t v625_a = v19_lead + 64;
              float v629_data = s0[(v625_a ^ ((v625_a >> 5) & 31))];
              float v633_data = ir7[5];
              ir7[5] = (v633_data + (v629_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 2))));
              int32_t v640_a = v19_lead + 64;
              float v644_data = s0[(v640_a ^ ((v640_a >> 5) & 31))];
              float v648_data = ir7[6];
              ir7[6] = (v648_data + (v644_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 2))));
              int32_t v655_a = v19_lead + 64;
              float v659_data = s0[(v655_a ^ ((v655_a >> 5) & 31))];
              float v663_data = ir7[7];
              ir7[7] = (v663_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 2))));
              int32_t v670_a = v19_lead + 64;
              float v674_data = s0[(v670_a ^ ((v670_a >> 5) & 31))];
              float v678_data = ir7[8];
              ir7[8] = (v678_data + (v674_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 2))));
              int32_t v688_a = v19_lead + 96;
              float v692_data = s0[(v688_a ^ ((v688_a >> 5) & 31))];
              float v696_data = ir7[0];
              ir7[0] = (v696_data + (v692_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 3))));
              int32_t v703_a = v19_lead + 96;
              float v707_data = s0[(v703_a ^ ((v703_a >> 5) & 31))];
              float v711_data = ir7[1];
              ir7[1] = (v711_data + (v707_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 3))));
              int32_t v718_a = v19_lead + 96;
              float v722_data = s0[(v718_a ^ ((v718_a >> 5) & 31))];
              float v726_data = ir7[2];
              ir7[2] = (v726_data + (v722_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 3))));
              int32_t v733_a = v19_lead + 96;
              float v737_data = s0[(v733_a ^ ((v733_a >> 5) & 31))];
              float v741_data = ir7[3];
              ir7[3] = (v741_data + (v737_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 3))));
              int32_t v748_a = v19_lead + 96;
              float v752_data = s0[(v748_a ^ ((v748_a >> 5) & 31))];
              float v756_data = ir7[4];
              ir7[4] = (v756_data + (v752_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 3))));
              int32_t v763_a = v19_lead + 96;
              float v767_data = s0[(v763_a ^ ((v763_a >> 5) & 31))];
              float v771_data = ir7[5];
              ir7[5] = (v771_data + (v767_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 3))));
              int32_t v778_a = v19_lead + 96;
              float v782_data = s0[(v778_a ^ ((v778_a >> 5) & 31))];
              float v786_data = ir7[6];
              ir7[6] = (v786_data + (v782_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 3))));
              int32_t v793_a = v19_lead + 96;
              float v797_data = s0[(v793_a ^ ((v793_a >> 5) & 31))];
              float v801_data = ir7[7];
              ir7[7] = (v801_data + (v797_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 3))));
              int32_t v808_a = v19_lead + 96;
              float v812_data = s0[(v808_a ^ ((v808_a >> 5) & 31))];
              float v816_data = ir7[8];
              ir7[8] = (v816_data + (v812_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 3))));
              int32_t v826_a = v19_lead + 128;
              float v830_data = s0[(v826_a ^ ((v826_a >> 5) & 31))];
              float v834_data = ir7[0];
              ir7[0] = (v834_data + (v830_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 4))));
              int32_t v841_a = v19_lead + 128;
              float v845_data = s0[(v841_a ^ ((v841_a >> 5) & 31))];
              float v849_data = ir7[1];
              ir7[1] = (v849_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 4))));
              int32_t v856_a = v19_lead + 128;
              float v860_data = s0[(v856_a ^ ((v856_a >> 5) & 31))];
              float v864_data = ir7[2];
              ir7[2] = (v864_data + (v860_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 4))));
              int32_t v871_a = v19_lead + 128;
              float v875_data = s0[(v871_a ^ ((v871_a >> 5) & 31))];
              float v879_data = ir7[3];
              ir7[3] = (v879_data + (v875_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 4))));
              int32_t v886_a = v19_lead + 128;
              float v890_data = s0[(v886_a ^ ((v886_a >> 5) & 31))];
              float v894_data = ir7[4];
              ir7[4] = (v894_data + (v890_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 4))));
              int32_t v901_a = v19_lead + 128;
              float v905_data = s0[(v901_a ^ ((v901_a >> 5) & 31))];
              float v909_data = ir7[5];
              ir7[5] = (v909_data + (v905_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 4))));
              int32_t v916_a = v19_lead + 128;
              float v920_data = s0[(v916_a ^ ((v916_a >> 5) & 31))];
              float v924_data = ir7[6];
              ir7[6] = (v924_data + (v920_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 4))));
              int32_t v931_a = v19_lead + 128;
              float v935_data = s0[(v931_a ^ ((v931_a >> 5) & 31))];
              float v939_data = ir7[7];
              ir7[7] = (v939_data + (v935_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 4))));
              int32_t v946_a = v19_lead + 128;
              float v950_data = s0[(v946_a ^ ((v946_a >> 5) & 31))];
              float v954_data = ir7[8];
              ir7[8] = (v954_data + (v950_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 4))));
              int32_t v964_a = v19_lead + 160;
              float v968_data = s0[(v964_a ^ ((v964_a >> 5) & 31))];
              float v972_data = ir7[0];
              ir7[0] = (v972_data + (v968_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 5))));
              int32_t v979_a = v19_lead + 160;
              float v983_data = s0[(v979_a ^ ((v979_a >> 5) & 31))];
              float v987_data = ir7[1];
              ir7[1] = (v987_data + (v983_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 5))));
              int32_t v994_a = v19_lead + 160;
              float v998_data = s0[(v994_a ^ ((v994_a >> 5) & 31))];
              float v1002_data = ir7[2];
              ir7[2] = (v1002_data + (v998_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 5))));
              int32_t v1009_a = v19_lead + 160;
              float v1013_data = s0[(v1009_a ^ ((v1009_a >> 5) & 31))];
              float v1017_data = ir7[3];
              ir7[3] = (v1017_data + (v1013_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 5))));
              int32_t v1024_a = v19_lead + 160;
              float v1028_data = s0[(v1024_a ^ ((v1024_a >> 5) & 31))];
              float v1032_data = ir7[4];
              ir7[4] = (v1032_data + (v1028_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 5))));
              int32_t v1039_a = v19_lead + 160;
              float v1043_data = s0[(v1039_a ^ ((v1039_a >> 5) & 31))];
              float v1047_data = ir7[5];
              ir7[5] = (v1047_data + (v1043_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 5))));
              int32_t v1054_a = v19_lead + 160;
              float v1058_data = s0[(v1054_a ^ ((v1054_a >> 5) & 31))];
              float v1062_data = ir7[6];
              ir7[6] = (v1062_data + (v1058_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 5))));
              int32_t v1069_a = v19_lead + 160;
              float v1073_data = s0[(v1069_a ^ ((v1069_a >> 5) & 31))];
              float v1077_data = ir7[7];
              ir7[7] = (v1077_data + (v1073_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 5))));
              int32_t v1084_a = v19_lead + 160;
              float v1088_data = s0[(v1084_a ^ ((v1084_a >> 5) & 31))];
              float v1092_data = ir7[8];
              ir7[8] = (v1092_data + (v1088_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 5))));
              int32_t v1102_a = v19_lead + 192;
              float v1106_data = s0[(v1102_a ^ ((v1102_a >> 5) & 31))];
              float v1110_data = ir7[0];
              ir7[0] = (v1110_data + (v1106_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 6))));
              int32_t v1117_a = v19_lead + 192;
              float v1121_data = s0[(v1117_a ^ ((v1117_a >> 5) & 31))];
              float v1125_data = ir7[1];
              ir7[1] = (v1125_data + (v1121_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 6))));
              int32_t v1132_a = v19_lead + 192;
              float v1136_data = s0[(v1132_a ^ ((v1132_a >> 5) & 31))];
              float v1140_data = ir7[2];
              ir7[2] = (v1140_data + (v1136_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 6))));
              int32_t v1147_a = v19_lead + 192;
              float v1151_data = s0[(v1147_a ^ ((v1147_a >> 5) & 31))];
              float v1155_data = ir7[3];
              ir7[3] = (v1155_data + (v1151_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 6))));
              int32_t v1162_a = v19_lead + 192;
              float v1166_data = s0[(v1162_a ^ ((v1162_a >> 5) & 31))];
              float v1170_data = ir7[4];
              ir7[4] = (v1170_data + (v1166_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 6))));
              int32_t v1177_a = v19_lead + 192;
              float v1181_data = s0[(v1177_a ^ ((v1177_a >> 5) & 31))];
              float v1185_data = ir7[5];
              ir7[5] = (v1185_data + (v1181_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 6))));
              int32_t v1192_a = v19_lead + 192;
              float v1196_data = s0[(v1192_a ^ ((v1192_a >> 5) & 31))];
              float v1200_data = ir7[6];
              ir7[6] = (v1200_data + (v1196_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 6))));
              int32_t v1207_a = v19_lead + 192;
              float v1211_data = s0[(v1207_a ^ ((v1207_a >> 5) & 31))];
              float v1215_data = ir7[7];
              ir7[7] = (v1215_data + (v1211_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 6))));
              int32_t v1222_a = v19_lead + 192;
              float v1226_data = s0[(v1222_a ^ ((v1222_a >> 5) & 31))];
              float v1230_data = ir7[8];
              ir7[8] = (v1230_data + (v1226_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 6))));
              int32_t v1240_a = v19_lead + 224;
              float v1244_data = s0[(v1240_a ^ ((v1240_a >> 5) & 31))];
              float v1248_data = ir7[0];
              ir7[0] = (v1248_data + (v1244_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 7))));
              int32_t v1255_a = v19_lead + 224;
              float v1259_data = s0[(v1255_a ^ ((v1255_a >> 5) & 31))];
              float v1263_data = ir7[1];
              ir7[1] = (v1263_data + (v1259_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 7))));
              int32_t v1270_a = v19_lead + 224;
              float v1274_data = s0[(v1270_a ^ ((v1270_a >> 5) & 31))];
              float v1278_data = ir7[2];
              ir7[2] = (v1278_data + (v1274_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 7))));
              int32_t v1285_a = v19_lead + 224;
              float v1289_data = s0[(v1285_a ^ ((v1285_a >> 5) & 31))];
              float v1293_data = ir7[3];
              ir7[3] = (v1293_data + (v1289_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 7))));
              int32_t v1300_a = v19_lead + 224;
              float v1304_data = s0[(v1300_a ^ ((v1300_a >> 5) & 31))];
              float v1308_data = ir7[4];
              ir7[4] = (v1308_data + (v1304_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 7))));
              int32_t v1315_a = v19_lead + 224;
              float v1319_data = s0[(v1315_a ^ ((v1315_a >> 5) & 31))];
              float v1323_data = ir7[5];
              ir7[5] = (v1323_data + (v1319_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 7))));
              int32_t v1330_a = v19_lead + 224;
              float v1334_data = s0[(v1330_a ^ ((v1330_a >> 5) & 31))];
              float v1338_data = ir7[6];
              ir7[6] = (v1338_data + (v1334_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 7))));
              int32_t v1345_a = v19_lead + 224;
              float v1349_data = s0[(v1345_a ^ ((v1345_a >> 5) & 31))];
              float v1353_data = ir7[7];
              ir7[7] = (v1353_data + (v1349_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 7))));
              int32_t v1360_a = v19_lead + 224;
              float v1364_data = s0[(v1360_a ^ ((v1360_a >> 5) & 31))];
              float v1368_data = ir7[8];
              ir7[8] = (v1368_data + (v1364_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 7))));
              int32_t v1378_a = v19_lead + 256;
              float v1382_data = s0[(v1378_a ^ ((v1378_a >> 5) & 31))];
              float v1386_data = ir7[0];
              ir7[0] = (v1386_data + (v1382_data * (sycl::group_broadcast(item.get_sub_group(), v279_data, 8))));
              int32_t v1393_a = v19_lead + 256;
              float v1397_data = s0[(v1393_a ^ ((v1393_a >> 5) & 31))];
              float v1401_data = ir7[1];
              ir7[1] = (v1401_data + (v1397_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 8))));
              int32_t v1408_a = v19_lead + 256;
              float v1412_data = s0[(v1408_a ^ ((v1408_a >> 5) & 31))];
              float v1416_data = ir7[2];
              ir7[2] = (v1416_data + (v1412_data * (sycl::group_broadcast(item.get_sub_group(), v309_data, 8))));
              int32_t v1423_a = v19_lead + 256;
              float v1427_data = s0[(v1423_a ^ ((v1423_a >> 5) & 31))];
              float v1431_data = ir7[3];
              ir7[3] = (v1431_data + (v1427_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 8))));
              int32_t v1438_a = v19_lead + 256;
              float v1442_data = s0[(v1438_a ^ ((v1438_a >> 5) & 31))];
              float v1446_data = ir7[4];
              ir7[4] = (v1446_data + (v1442_data * (sycl::group_broadcast(item.get_sub_group(), v339_data, 8))));
              int32_t v1453_a = v19_lead + 256;
              float v1457_data = s0[(v1453_a ^ ((v1453_a >> 5) & 31))];
              float v1461_data = ir7[5];
              ir7[5] = (v1461_data + (v1457_data * (sycl::group_broadcast(item.get_sub_group(), v354_data, 8))));
              int32_t v1468_a = v19_lead + 256;
              float v1472_data = s0[(v1468_a ^ ((v1468_a >> 5) & 31))];
              float v1476_data = ir7[6];
              ir7[6] = (v1476_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v369_data, 8))));
              int32_t v1483_a = v19_lead + 256;
              float v1487_data = s0[(v1483_a ^ ((v1483_a >> 5) & 31))];
              float v1491_data = ir7[7];
              ir7[7] = (v1491_data + (v1487_data * (sycl::group_broadcast(item.get_sub_group(), v384_data, 8))));
              int32_t v1498_a = v19_lead + 256;
              float v1502_data = s0[(v1498_a ^ ((v1498_a >> 5) & 31))];
              float v1506_data = ir7[8];
              ir7[8] = (v1506_data + (v1502_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 8))));
              #pragma unroll
              for (int32_t v1511_n0 = 0; v1511_n0 < 1; ++v1511_n0) {
                #pragma unroll
                for (int32_t v1512_n1 = 0; v1512_n1 < 9; ++v1512_n1) {
                  int32_t v1513_a = v1511_n0 + v1512_n1;
                  float v1514_data = ir7[v1513_a];
                  r7[v1513_a] = v1514_data;
                }
              }
              // glb_m3 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1519_i0 = 0; v1519_i0 < 1; ++v1519_i0) {
                int32_t v1527_lead = v19_lead + (v1519_i0 * 32);
                #pragma unroll
                for (int32_t v1520_i1 = 0; v1520_i1 < 9; ++v1520_i1) {
                  float v1522_data = r7[(v1519_i0 + v1520_i1)];
                  glb_m3[(v1527_lead + (v1520_i1 * 32))] = v1522_data;
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

