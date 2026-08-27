// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08a27dccde(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08a27dccde(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 64] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 64];
              if (item.get_local_id(0) < 1) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 80] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 80];
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[9]{};
              ;
              // r0 = +(glb_m1 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir0[9]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 9) {
                int32_t v11_a = 0_i32 + 0;
                int32_t v13_a = 0_i32 + 0;
                int32_t v15_a = 0_i32 + 0;
                int32_t v17_a = 0_i32 + 0;
                int32_t v19_a = 0_i32 + 0;
                int32_t v21_a = 0_i32 + 0;
                int32_t v23_a = 0_i32 + 0;
                int32_t v25_a = 0_i32 + 0;
                int32_t v27_a = 0_i32 + 0;
              }
              if (v8_lead < 9) {
                int32_t v33_a = 0_i32 + 9;
                int32_t v35_a = 0_i32 + 9;
                int32_t v37_a = 0_i32 + 9;
                int32_t v39_a = 0_i32 + 9;
                int32_t v41_a = 0_i32 + 9;
                int32_t v43_a = 0_i32 + 9;
                int32_t v45_a = 0_i32 + 9;
                int32_t v47_a = 0_i32 + 9;
                int32_t v49_a = 0_i32 + 9;
              }
              if (v8_lead < 9) {
                int32_t v55_a = 0_i32 + 18;
                int32_t v57_a = 0_i32 + 18;
                int32_t v59_a = 0_i32 + 18;
                int32_t v61_a = 0_i32 + 18;
                int32_t v63_a = 0_i32 + 18;
                int32_t v65_a = 0_i32 + 18;
                int32_t v67_a = 0_i32 + 18;
                int32_t v69_a = 0_i32 + 18;
                int32_t v71_a = 0_i32 + 18;
              }
              if (v8_lead < 9) {
                int32_t v77_a = 0_i32 + 27;
                int32_t v79_a = 0_i32 + 27;
                int32_t v81_a = 0_i32 + 27;
                int32_t v83_a = 0_i32 + 27;
                int32_t v85_a = 0_i32 + 27;
                int32_t v87_a = 0_i32 + 27;
                int32_t v89_a = 0_i32 + 27;
                int32_t v91_a = 0_i32 + 27;
                int32_t v93_a = 0_i32 + 27;
              }
              if (v8_lead < 9) {
                int32_t v99_a = 0_i32 + 36;
                int32_t v101_a = 0_i32 + 36;
                int32_t v103_a = 0_i32 + 36;
                int32_t v105_a = 0_i32 + 36;
                int32_t v107_a = 0_i32 + 36;
                int32_t v109_a = 0_i32 + 36;
                int32_t v111_a = 0_i32 + 36;
                int32_t v113_a = 0_i32 + 36;
                int32_t v115_a = 0_i32 + 36;
              }
              if (v8_lead < 9) {
                int32_t v121_a = 0_i32 + 45;
                int32_t v123_a = 0_i32 + 45;
                int32_t v125_a = 0_i32 + 45;
                int32_t v127_a = 0_i32 + 45;
                int32_t v129_a = 0_i32 + 45;
                int32_t v131_a = 0_i32 + 45;
                int32_t v133_a = 0_i32 + 45;
                int32_t v135_a = 0_i32 + 45;
                int32_t v137_a = 0_i32 + 45;
              }
              if (v8_lead < 9) {
                int32_t v143_a = 0_i32 + 54;
                int32_t v145_a = 0_i32 + 54;
                int32_t v147_a = 0_i32 + 54;
                int32_t v149_a = 0_i32 + 54;
                int32_t v151_a = 0_i32 + 54;
                int32_t v153_a = 0_i32 + 54;
                int32_t v155_a = 0_i32 + 54;
                int32_t v157_a = 0_i32 + 54;
                int32_t v159_a = 0_i32 + 54;
              }
              if (v8_lead < 9) {
                int32_t v165_a = 0_i32 + 63;
                int32_t v167_a = 0_i32 + 63;
                int32_t v169_a = 0_i32 + 63;
                int32_t v171_a = 0_i32 + 63;
                int32_t v173_a = 0_i32 + 63;
                int32_t v175_a = 0_i32 + 63;
                int32_t v177_a = 0_i32 + 63;
                int32_t v179_a = 0_i32 + 63;
                int32_t v181_a = 0_i32 + 63;
              }
              if (v8_lead < 9) {
                int32_t v187_a = 0_i32 + 72;
                int32_t v189_a = 0_i32 + 72;
                int32_t v191_a = 0_i32 + 72;
                int32_t v193_a = 0_i32 + 72;
                int32_t v195_a = 0_i32 + 72;
                int32_t v197_a = 0_i32 + 72;
                int32_t v199_a = 0_i32 + 72;
                int32_t v201_a = 0_i32 + 72;
                int32_t v203_a = 0_i32 + 72;
              }
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v209_n1 = 0; v209_n1 < 9; ++v209_n1) {
                  int32_t v210_a = 0 + v209_n1;
                  int32_t v212_a = 0 + v209_n1;
                  v211_p = r0[v212_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 9) {
                #pragma unroll
                for (int32_t v217_i1 = 0; v217_i1 < 9; ++v217_i1) {
                  int32_t v218_a = 0 + v217_i1;
                  int32_t v221_a = 0_i32 + (v217_i1 * 9);
                  None.copy_to(glb_m0[v221_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}

