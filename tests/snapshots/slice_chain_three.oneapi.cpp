// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08703cce1d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08703cce1d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1536, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[96 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[80];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m1[0 + 0 + 2 * item.get_local_id(0) + 0];
              if (item.get_local_id(0) < 4) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 32] = glb_m1[0 + 0 + 1 * item.get_local_id(0) + 32];
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r0[6]{};
              ;
              // r0 = +(glb_m0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              auto& ir0 = r0;
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                int32_t v11_a = 0_i32 + 0;
                int32_t v13_a = 0_i32 + 0;
                int32_t v15_a = 0_i32 + 0;
                int32_t v17_a = 0_i32 + 0;
                int32_t v19_a = 0_i32 + 0;
                int32_t v21_a = 0_i32 + 0;
              }
              if (v8_lead < 12) {
                int32_t v27_a = 0_i32 + 12;
                int32_t v29_a = 0_i32 + 12;
                int32_t v31_a = 0_i32 + 12;
                int32_t v33_a = 0_i32 + 12;
                int32_t v35_a = 0_i32 + 12;
                int32_t v37_a = 0_i32 + 12;
              }
              if (v8_lead < 12) {
                int32_t v43_a = 0_i32 + 24;
                int32_t v45_a = 0_i32 + 24;
                int32_t v47_a = 0_i32 + 24;
                int32_t v49_a = 0_i32 + 24;
                int32_t v51_a = 0_i32 + 24;
                int32_t v53_a = 0_i32 + 24;
              }
              if (v8_lead < 12) {
                int32_t v59_a = 0_i32 + 36;
                int32_t v61_a = 0_i32 + 36;
                int32_t v63_a = 0_i32 + 36;
                int32_t v65_a = 0_i32 + 36;
                int32_t v67_a = 0_i32 + 36;
                int32_t v69_a = 0_i32 + 36;
              }
              if (v8_lead < 12) {
                int32_t v75_a = 0_i32 + 48;
                int32_t v77_a = 0_i32 + 48;
                int32_t v79_a = 0_i32 + 48;
                int32_t v81_a = 0_i32 + 48;
                int32_t v83_a = 0_i32 + 48;
                int32_t v85_a = 0_i32 + 48;
              }
              if (v8_lead < 12) {
                int32_t v91_a = 0_i32 + 60;
                int32_t v93_a = 0_i32 + 60;
                int32_t v95_a = 0_i32 + 60;
                int32_t v97_a = 0_i32 + 60;
                int32_t v99_a = 0_i32 + 60;
                int32_t v101_a = 0_i32 + 60;
              }
              ;
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v107_i1 = 0; v107_i1 < 6; ++v107_i1) {
                  int32_t v108_a = 0 + v107_i1;
                  int32_t v111_a = 0_i32 + (v107_i1 * 12);
                  None = s1[v111_a];
                }
              }
              float r1[6]{};
              ;
              // r1 = +(glb_m3 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir1[6]{};
              if (v8_lead < 12) {
                int32_t v119_a = 0_i32 + 0;
                int32_t v121_a = 0_i32 + 0;
                int32_t v123_a = 0_i32 + 0;
                int32_t v125_a = 0_i32 + 0;
                int32_t v127_a = 0_i32 + 0;
                int32_t v129_a = 0_i32 + 0;
              }
              if (v8_lead < 12) {
                int32_t v135_a = 0_i32 + 12;
                int32_t v137_a = 0_i32 + 12;
                int32_t v139_a = 0_i32 + 12;
                int32_t v141_a = 0_i32 + 12;
                int32_t v143_a = 0_i32 + 12;
                int32_t v145_a = 0_i32 + 12;
              }
              if (v8_lead < 12) {
                int32_t v151_a = 0_i32 + 24;
                int32_t v153_a = 0_i32 + 24;
                int32_t v155_a = 0_i32 + 24;
                int32_t v157_a = 0_i32 + 24;
                int32_t v159_a = 0_i32 + 24;
                int32_t v161_a = 0_i32 + 24;
              }
              if (v8_lead < 12) {
                int32_t v167_a = 0_i32 + 36;
                int32_t v169_a = 0_i32 + 36;
                int32_t v171_a = 0_i32 + 36;
                int32_t v173_a = 0_i32 + 36;
                int32_t v175_a = 0_i32 + 36;
                int32_t v177_a = 0_i32 + 36;
              }
              if (v8_lead < 12) {
                int32_t v183_a = 0_i32 + 48;
                int32_t v185_a = 0_i32 + 48;
                int32_t v187_a = 0_i32 + 48;
                int32_t v189_a = 0_i32 + 48;
                int32_t v191_a = 0_i32 + 48;
                int32_t v193_a = 0_i32 + 48;
              }
              if (v8_lead < 12) {
                int32_t v199_a = 0_i32 + 60;
                int32_t v201_a = 0_i32 + 60;
                int32_t v203_a = 0_i32 + 60;
                int32_t v205_a = 0_i32 + 60;
                int32_t v207_a = 0_i32 + 60;
                int32_t v209_a = 0_i32 + 60;
              }
              if (v8_lead < 12) {
                int32_t v215_a = 0_i32 + 72;
                int32_t v217_a = 0_i32 + 72;
                int32_t v219_a = 0_i32 + 72;
                int32_t v221_a = 0_i32 + 72;
                int32_t v223_a = 0_i32 + 72;
                int32_t v225_a = 0_i32 + 72;
              }
              if (v8_lead < 12) {
                int32_t v231_a = 0_i32 + 84;
                int32_t v233_a = 0_i32 + 84;
                int32_t v235_a = 0_i32 + 84;
                int32_t v237_a = 0_i32 + 84;
                int32_t v239_a = 0_i32 + 84;
                int32_t v241_a = 0_i32 + 84;
              }
              if (v8_lead < 12) {
                int32_t v247_a = 0_i32 + 96;
                int32_t v249_a = 0_i32 + 96;
                int32_t v251_a = 0_i32 + 96;
                int32_t v253_a = 0_i32 + 96;
                int32_t v255_a = 0_i32 + 96;
                int32_t v257_a = 0_i32 + 96;
              }
              if (v8_lead < 12) {
                int32_t v263_a = 0_i32 + 108;
                int32_t v265_a = 0_i32 + 108;
                int32_t v267_a = 0_i32 + 108;
                int32_t v269_a = 0_i32 + 108;
                int32_t v271_a = 0_i32 + 108;
                int32_t v273_a = 0_i32 + 108;
              }
              if (v8_lead < 12) {
                int32_t v279_a = 0_i32 + 120;
                int32_t v281_a = 0_i32 + 120;
                int32_t v283_a = 0_i32 + 120;
                int32_t v285_a = 0_i32 + 120;
                int32_t v287_a = 0_i32 + 120;
                int32_t v289_a = 0_i32 + 120;
              }
              if (v8_lead < 12) {
                int32_t v295_a = 0_i32 + 132;
                int32_t v297_a = 0_i32 + 132;
                int32_t v299_a = 0_i32 + 132;
                int32_t v301_a = 0_i32 + 132;
                int32_t v303_a = 0_i32 + 132;
                int32_t v305_a = 0_i32 + 132;
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v310_n1 = 0; v310_n1 < 6; ++v310_n1) {
                  int32_t v311_a = 0 + v310_n1;
                  int32_t v312_a = 0 + v310_n1;
                  None = r1[v312_a];
                }
              }
              // glb_m2 = store{r>g}(r1);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v317_i1 = 0; v317_i1 < 6; ++v317_i1) {
                  int32_t v318_a = 0 + v317_i1;
                  int32_t v321_a = 0_i32 + (v317_i1 * 12);
                  None.copy_to(glb_m2[v321_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}

