// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5e7da3148f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5e7da3148f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              ;
              // r0 = +(glb_m1 * s0) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                int32_t v11_a = 0_i32 + 0;
                int32_t v13_a = 0_i32 + 0;
                int32_t v15_a = 0_i32 + 0;
                int32_t v17_a = 0_i32 + 0;
                int32_t v19_a = 0_i32 + 0;
                int32_t v21_a = 0_i32 + 0;
                int32_t v23_a = 0_i32 + 0;
                int32_t v25_a = 0_i32 + 0;
              }
              if (v8_lead < 12) {
                int32_t v31_a = 0_i32 + 12;
                int32_t v33_a = 0_i32 + 12;
                int32_t v35_a = 0_i32 + 12;
                int32_t v37_a = 0_i32 + 12;
                int32_t v39_a = 0_i32 + 12;
                int32_t v41_a = 0_i32 + 12;
                int32_t v43_a = 0_i32 + 12;
                int32_t v45_a = 0_i32 + 12;
              }
              if (v8_lead < 12) {
                int32_t v51_a = 0_i32 + 24;
                int32_t v53_a = 0_i32 + 24;
                int32_t v55_a = 0_i32 + 24;
                int32_t v57_a = 0_i32 + 24;
                int32_t v59_a = 0_i32 + 24;
                int32_t v61_a = 0_i32 + 24;
                int32_t v63_a = 0_i32 + 24;
                int32_t v65_a = 0_i32 + 24;
              }
              if (v8_lead < 12) {
                int32_t v71_a = 0_i32 + 36;
                int32_t v73_a = 0_i32 + 36;
                int32_t v75_a = 0_i32 + 36;
                int32_t v77_a = 0_i32 + 36;
                int32_t v79_a = 0_i32 + 36;
                int32_t v81_a = 0_i32 + 36;
                int32_t v83_a = 0_i32 + 36;
                int32_t v85_a = 0_i32 + 36;
              }
              if (v8_lead < 12) {
                int32_t v91_a = 0_i32 + 48;
                int32_t v93_a = 0_i32 + 48;
                int32_t v95_a = 0_i32 + 48;
                int32_t v97_a = 0_i32 + 48;
                int32_t v99_a = 0_i32 + 48;
                int32_t v101_a = 0_i32 + 48;
                int32_t v103_a = 0_i32 + 48;
                int32_t v105_a = 0_i32 + 48;
              }
              if (v8_lead < 12) {
                int32_t v111_a = 0_i32 + 60;
                int32_t v113_a = 0_i32 + 60;
                int32_t v115_a = 0_i32 + 60;
                int32_t v117_a = 0_i32 + 60;
                int32_t v119_a = 0_i32 + 60;
                int32_t v121_a = 0_i32 + 60;
                int32_t v123_a = 0_i32 + 60;
                int32_t v125_a = 0_i32 + 60;
              }
              if (v8_lead < 12) {
                int32_t v131_a = 0_i32 + 72;
                int32_t v133_a = 0_i32 + 72;
                int32_t v135_a = 0_i32 + 72;
                int32_t v137_a = 0_i32 + 72;
                int32_t v139_a = 0_i32 + 72;
                int32_t v141_a = 0_i32 + 72;
                int32_t v143_a = 0_i32 + 72;
                int32_t v145_a = 0_i32 + 72;
              }
              if (v8_lead < 12) {
                int32_t v151_a = 0_i32 + 84;
                int32_t v153_a = 0_i32 + 84;
                int32_t v155_a = 0_i32 + 84;
                int32_t v157_a = 0_i32 + 84;
                int32_t v159_a = 0_i32 + 84;
                int32_t v161_a = 0_i32 + 84;
                int32_t v163_a = 0_i32 + 84;
                int32_t v165_a = 0_i32 + 84;
              }
              if (v8_lead < 12) {
                int32_t v171_a = 0_i32 + 96;
                int32_t v173_a = 0_i32 + 96;
                int32_t v175_a = 0_i32 + 96;
                int32_t v177_a = 0_i32 + 96;
                int32_t v179_a = 0_i32 + 96;
                int32_t v181_a = 0_i32 + 96;
                int32_t v183_a = 0_i32 + 96;
                int32_t v185_a = 0_i32 + 96;
              }
              if (v8_lead < 12) {
                int32_t v191_a = 0_i32 + 108;
                int32_t v193_a = 0_i32 + 108;
                int32_t v195_a = 0_i32 + 108;
                int32_t v197_a = 0_i32 + 108;
                int32_t v199_a = 0_i32 + 108;
                int32_t v201_a = 0_i32 + 108;
                int32_t v203_a = 0_i32 + 108;
                int32_t v205_a = 0_i32 + 108;
              }
              if (v8_lead < 12) {
                int32_t v211_a = 0_i32 + 120;
                int32_t v213_a = 0_i32 + 120;
                int32_t v215_a = 0_i32 + 120;
                int32_t v217_a = 0_i32 + 120;
                int32_t v219_a = 0_i32 + 120;
                int32_t v221_a = 0_i32 + 120;
                int32_t v223_a = 0_i32 + 120;
                int32_t v225_a = 0_i32 + 120;
              }
              if (v8_lead < 12) {
                int32_t v231_a = 0_i32 + 132;
                int32_t v233_a = 0_i32 + 132;
                int32_t v235_a = 0_i32 + 132;
                int32_t v237_a = 0_i32 + 132;
                int32_t v239_a = 0_i32 + 132;
                int32_t v241_a = 0_i32 + 132;
                int32_t v243_a = 0_i32 + 132;
                int32_t v245_a = 0_i32 + 132;
              }
              if (v8_lead < 12) {
                int32_t v251_a = 0_i32 + 144;
                int32_t v253_a = 0_i32 + 144;
                int32_t v255_a = 0_i32 + 144;
                int32_t v257_a = 0_i32 + 144;
                int32_t v259_a = 0_i32 + 144;
                int32_t v261_a = 0_i32 + 144;
                int32_t v263_a = 0_i32 + 144;
                int32_t v265_a = 0_i32 + 144;
              }
              if (v8_lead < 12) {
                int32_t v271_a = 0_i32 + 156;
                int32_t v273_a = 0_i32 + 156;
                int32_t v275_a = 0_i32 + 156;
                int32_t v277_a = 0_i32 + 156;
                int32_t v279_a = 0_i32 + 156;
                int32_t v281_a = 0_i32 + 156;
                int32_t v283_a = 0_i32 + 156;
                int32_t v285_a = 0_i32 + 156;
              }
              if (v8_lead < 12) {
                int32_t v291_a = 0_i32 + 168;
                int32_t v293_a = 0_i32 + 168;
                int32_t v295_a = 0_i32 + 168;
                int32_t v297_a = 0_i32 + 168;
                int32_t v299_a = 0_i32 + 168;
                int32_t v301_a = 0_i32 + 168;
                int32_t v303_a = 0_i32 + 168;
                int32_t v305_a = 0_i32 + 168;
              }
              if (v8_lead < 12) {
                int32_t v311_a = 0_i32 + 180;
                int32_t v313_a = 0_i32 + 180;
                int32_t v315_a = 0_i32 + 180;
                int32_t v317_a = 0_i32 + 180;
                int32_t v319_a = 0_i32 + 180;
                int32_t v321_a = 0_i32 + 180;
                int32_t v323_a = 0_i32 + 180;
                int32_t v325_a = 0_i32 + 180;
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v330_n1 = 0; v330_n1 < 8; ++v330_n1) {
                  int32_t v331_a = 0 + v330_n1;
                  int32_t v334_a = 0_i32 + (v330_n1 * 12);
                  int32_t v336_a = 0 + v330_n1;
                  v335_p = r0[v336_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v341_i1 = 0; v341_i1 < 8; ++v341_i1) {
                  int32_t v342_a = 0 + v341_i1;
                  int32_t v345_a = 0_i32 + (v341_i1 * 12);
                  None.copy_to(glb_m0[v345_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}

