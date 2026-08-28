// === base name ===
kernel_0b2fc070b9

// === header ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (2, 128, 1);
  sycl::range<3> grid ((numElements0 + 128 - 1) / 128, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0b2fc070b9(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0b2fc070b9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, __float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<__float128, 1> totalShrMem (768, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 2×2(2×2) {0..2}×{0..2} strided
        // m1 2×2(2×2) {0..2}×{0..2} strided
        // m2 2×2(2×2) {0..2}×{0..2} strided
        // m0 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, 1] = m1 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, -1]×m2 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          __float128* localShrMem0 = &totalShrMem[6 * item.get_local_id(1) + 0];
          __float128* tempShrMem = &localShrMem0[4];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
              const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
              const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
              __float128* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              s0[0 + 0 + 1 * item.get_local_id(0) + 0] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 2] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 2];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              __float128 r0[2]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 2), (0, 2)] [(0, 2)]
              __float128 ir0[2]{};
              tensorforge::intel_esimd::simd<int32_t, 2> v6_lead = tensorforge::intel_esimd::simd<int32_t, 2>(0, 1);
              int32_t v9_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<__float128, 2> v13_data;
              v13_data.copy_from(glb_m1 + (0_i32));
              __float128 v14_data = s0[0];
              tensorforge::intel_esimd::simd<__float128, 2> v16_data;
              v16_data.copy_from(ir0 + (0));
              (v16_data + (v13_data * v14_data)).copy_to(ir0 + (0));
              int32_t v20_a = 0_i32 + 0;
              __float128 v25_data = s0[2];
              tensorforge::intel_esimd::simd<__float128, 2> v27_data;
              v27_data.copy_from(ir0 + (1));
              (v27_data + (v13_data * v25_data)).copy_to(ir0 + (1));
              int32_t v32_a = 0_i32 + 2;
              tensorforge::intel_esimd::simd<__float128, 2> v36_data;
              v36_data.copy_from(glb_m1 + (2_i32));
              __float128 v37_data = s0[1];
              tensorforge::intel_esimd::simd<__float128, 2> v39_data;
              v39_data.copy_from(ir0 + (0));
              (v39_data + (v36_data * v37_data)).copy_to(ir0 + (0));
              int32_t v43_a = 0_i32 + 2;
              __float128 v48_data = s0[3];
              tensorforge::intel_esimd::simd<__float128, 2> v50_data;
              v50_data.copy_from(ir0 + (1));
              (v50_data + (v36_data * v48_data)).copy_to(ir0 + (1));
              #pragma unroll
              for (int32_t v53_n0 = 0; v53_n0 < 1; ++v53_n0) {
                #pragma unroll
                for (int32_t v54_n1 = 0; v54_n1 < 2; ++v54_n1) {
                  int32_t v55_a = v53_n0 + v54_n1;
                  int32_t v56_a = v53_n0 + v54_n1;
                  tensorforge::intel_esimd::simd<__float128, 2> v57_data;
                  v57_data.copy_from(ir0 + (v56_a));
                  v57_data.copy_to(r0 + (v56_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v60_i0 = 0; v60_i0 < 1; ++v60_i0) {
                int32_t v65_lead = v60_i0 * 2;
                #pragma unroll
                for (int32_t v61_i1 = 0; v61_i1 < 2; ++v61_i1) {
                  int32_t v62_a = v60_i0 + v61_i1;
                  tensorforge::intel_esimd::simd<__float128, 2> v64_data;
                  v64_data.copy_from(r0 + ((v60_i0 + v61_i1)));
                  v64_data.copy_to(glb_m0 + ((v65_lead + (v61_i1 * 2))));
                }
              }
            }
          }
        }
      });
    }
  });
}

