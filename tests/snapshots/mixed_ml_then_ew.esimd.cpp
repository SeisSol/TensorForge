// === base name ===
kernel_8ab0d0fff0

// === header ===
void launcher_kernel_8ab0d0fff0(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8ab0d0fff0(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8ab0d0fff0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8ab0d0fff0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 8; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 8> v11_data;
                v11_data.copy_from(glb_m0 + ((v6_i1 * 8)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v15_ld;
              v15_ld.copy_from(glb_m1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v15_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v17_data;
              v17_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v18_data;
              v18_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v25_acc{};
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(s0 + (0_i32));
              v25_acc += ((v29_data[0]) * v17_data);
              v25_acc += ((v29_data[1]) * v18_data);
              v25_acc += ((v29_data[2]) * v19_data);
              v25_acc += ((v29_data[3]) * v20_data);
              v25_acc += ((v29_data[4]) * v21_data);
              v25_acc += ((v29_data[5]) * v22_data);
              v25_acc += ((v29_data[6]) * v23_data);
              v25_acc += ((v29_data[7]) * v24_data);
              v25_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v46_acc{};
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(s0 + (8_i32));
              v46_acc += ((v50_data[0]) * v17_data);
              v46_acc += ((v50_data[1]) * v18_data);
              v46_acc += ((v50_data[2]) * v19_data);
              v46_acc += ((v50_data[3]) * v20_data);
              v46_acc += ((v50_data[4]) * v21_data);
              v46_acc += ((v50_data[5]) * v22_data);
              v46_acc += ((v50_data[6]) * v23_data);
              v46_acc += ((v50_data[7]) * v24_data);
              v46_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v67_acc{};
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(s0 + (16_i32));
              v67_acc += ((v71_data[0]) * v17_data);
              v67_acc += ((v71_data[1]) * v18_data);
              v67_acc += ((v71_data[2]) * v19_data);
              v67_acc += ((v71_data[3]) * v20_data);
              v67_acc += ((v71_data[4]) * v21_data);
              v67_acc += ((v71_data[5]) * v22_data);
              v67_acc += ((v71_data[6]) * v23_data);
              v67_acc += ((v71_data[7]) * v24_data);
              v67_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v88_acc{};
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(s0 + (24_i32));
              v88_acc += ((v92_data[0]) * v17_data);
              v88_acc += ((v92_data[1]) * v18_data);
              v88_acc += ((v92_data[2]) * v19_data);
              v88_acc += ((v92_data[3]) * v20_data);
              v88_acc += ((v92_data[4]) * v21_data);
              v88_acc += ((v92_data[5]) * v22_data);
              v88_acc += ((v92_data[6]) * v23_data);
              v88_acc += ((v92_data[7]) * v24_data);
              v88_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v109_acc{};
              tensorforge::intel_esimd::simd<float, 16> v113_data;
              v113_data.copy_from(s0 + (32_i32));
              v109_acc += ((v113_data[0]) * v17_data);
              v109_acc += ((v113_data[1]) * v18_data);
              v109_acc += ((v113_data[2]) * v19_data);
              v109_acc += ((v113_data[3]) * v20_data);
              v109_acc += ((v113_data[4]) * v21_data);
              v109_acc += ((v113_data[5]) * v22_data);
              v109_acc += ((v113_data[6]) * v23_data);
              v109_acc += ((v113_data[7]) * v24_data);
              v109_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v134_data;
              v134_data.copy_from(s0 + (40_i32));
              v130_acc += ((v134_data[0]) * v17_data);
              v130_acc += ((v134_data[1]) * v18_data);
              v130_acc += ((v134_data[2]) * v19_data);
              v130_acc += ((v134_data[3]) * v20_data);
              v130_acc += ((v134_data[4]) * v21_data);
              v130_acc += ((v134_data[5]) * v22_data);
              v130_acc += ((v134_data[6]) * v23_data);
              v130_acc += ((v134_data[7]) * v24_data);
              v130_acc.copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v151_acc{};
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(s0 + (48_i32));
              v151_acc += ((v155_data[0]) * v17_data);
              v151_acc += ((v155_data[1]) * v18_data);
              v151_acc += ((v155_data[2]) * v19_data);
              v151_acc += ((v155_data[3]) * v20_data);
              v151_acc += ((v155_data[4]) * v21_data);
              v151_acc += ((v155_data[5]) * v22_data);
              v151_acc += ((v155_data[6]) * v23_data);
              v151_acc += ((v155_data[7]) * v24_data);
              v151_acc.copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v172_acc{};
              tensorforge::intel_esimd::simd<float, 16> v176_data;
              v176_data.copy_from(s0 + (56_i32));
              v172_acc += ((v176_data[0]) * v17_data);
              v172_acc += ((v176_data[1]) * v18_data);
              v172_acc += ((v176_data[2]) * v19_data);
              v172_acc += ((v176_data[3]) * v20_data);
              v172_acc += ((v176_data[4]) * v21_data);
              v172_acc += ((v176_data[5]) * v22_data);
              v172_acc += ((v176_data[6]) * v23_data);
              v172_acc += ((v176_data[7]) * v24_data);
              v172_acc.copy_to(r1 + (112));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v194_i1 = 0; v194_i1 < 8; ++v194_i1) {
                tensorforge::intel_esimd::simd<float, 8> v197_data;
                v197_data.copy_from(r1 + ((v194_i1 * 16)));
                v197_data.copy_to(s1 + ((v194_i1 * 8)));
              }
              // glb_m2 = abs(s1)
              #pragma unroll
              for (int32_t v202_k1 = 0; v202_k1 < 8; ++v202_k1) {
                int32_t v205_a = v202_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v207_data;
                v207_data.copy_from(s1 + (v205_a));
                (tensorforge::intel_esimd::abs(v207_data)).copy_to(glb_m2 + (v205_a));
              }
            }
          }
        }
      });
    }
  });
}

