// === base name ===
kernel_e034fafc05e35e56

// === header ===
void launcher_kernel_e034fafc05e35e56(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e034fafc05e35e56(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (2, 128, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e034fafc05e35e56(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e034fafc05e35e56(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, __float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<__float128, 1> totalShrMem (768, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          __float128* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
              const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
              const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
              __float128 r0[4]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v13_lead = v11_i0 * 2;
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 2; ++v12_i1) {
                  int32_t v16_a = v13_lead + (v12_i1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v17_data;
                  v17_data.copy_from(glb_m1 + (v16_a));
                  v17_data.copy_to(r0 + (v16_a));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<__float128, 2> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              v21_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<__float128, 2> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 2));
              v22_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 2));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              __float128 r1[4]{};
              // r1 = +(r0 * s0) + None
              // [(0, 2), (0, 2)] [(0, 2)]
              __float128 ir1[4]{};
              tensorforge::intel_esimd::simd<__float128, 2> v25_data;
              v25_data.copy_from(r0 + (0));
              __float128 v26_data = s0[0];
              tensorforge::intel_esimd::simd<__float128, 2> v28_data;
              v28_data.copy_from(ir1 + (0));
              (v28_data + (v25_data * v26_data)).copy_to(ir1 + (0));
              __float128 v31_data = s0[2];
              tensorforge::intel_esimd::simd<__float128, 2> v33_data;
              v33_data.copy_from(ir1 + (2));
              (v33_data + (v25_data * v31_data)).copy_to(ir1 + (2));
              tensorforge::intel_esimd::simd<__float128, 2> v35_data;
              v35_data.copy_from(r0 + (2));
              __float128 v36_data = s0[1];
              tensorforge::intel_esimd::simd<__float128, 2> v38_data;
              v38_data.copy_from(ir1 + (0));
              (v38_data + (v35_data * v36_data)).copy_to(ir1 + (0));
              __float128 v41_data = s0[3];
              tensorforge::intel_esimd::simd<__float128, 2> v43_data;
              v43_data.copy_from(ir1 + (2));
              (v43_data + (v35_data * v41_data)).copy_to(ir1 + (2));
              #pragma unroll
              for (int32_t v45_n0 = 0; v45_n0 < 1; ++v45_n0) {
                int32_t v47_a = v45_n0 * 2;
                #pragma unroll
                for (int32_t v46_n1 = 0; v46_n1 < 2; ++v46_n1) {
                  int32_t v49_a = v47_a + (v46_n1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v50_data;
                  v50_data.copy_from(ir1 + (v49_a));
                  v50_data.copy_to(r1 + (v49_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v54_i0 = 0; v54_i0 < 1; ++v54_i0) {
                int32_t v56_a = v54_i0 * 2;
                #pragma unroll
                for (int32_t v55_i1 = 0; v55_i1 < 2; ++v55_i1) {
                  int32_t v58_a = v56_a + (v55_i1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v59_data;
                  v59_data.copy_from(r1 + (v58_a));
                  v59_data.copy_to(glb_m0 + (v58_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

