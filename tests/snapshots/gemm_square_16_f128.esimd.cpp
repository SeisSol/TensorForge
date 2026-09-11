// === base name ===
kernel_ffa4802ed2d54cc4

// === header ===
void launcher_kernel_ffa4802ed2d54cc4(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ffa4802ed2d54cc4(__float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (2, 128, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ffa4802ed2d54cc4(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ffa4802ed2d54cc4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, __float128* m0, size_t m0_extraOffset, const __float128* m1, size_t m1_extraOffset, const __float128* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<__float128, 1> totalShrMem (768, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 2×2(2×2) {0..2}×{0..2} strided
        // m1 2×2(2×2) {0..2}×{0..2} strided
        // m2 2×2(2×2) {0..2}×{0..2} strided
        // m0 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, 1] = m1 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, -1]×m2 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          __float128* localShrMem0 = &totalShrMem[6 * item.get_local_id(1) + 0];
          __float128* tempShrMem = &localShrMem0[4];
          __float128 * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              __float128 *const __restrict__ glb_m0 = &m0[v3_batchId0 * 4 + 0 + m0_extraOffset];
              const __float128 *const __restrict__ glb_m1 = &m1[v3_batchId0 * 4 + 0 + m1_extraOffset];
              const __float128 *const __restrict__ glb_m2 = &m2[v3_batchId0 * 4 + 0 + m2_extraOffset];
              __float128 r0[4]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v17_lead = v15_i0 * 2;
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 2; ++v16_i1) {
                  int32_t v20_a = v17_lead + (v16_i1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v21_data;
                  v21_data.copy_from(glb_m1 + (v20_a));
                  v21_data.copy_to(r0 + (v20_a));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<__float128, 2> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              v25_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<__float128, 2> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 2));
              v26_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 2));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              __float128 r1[4]{};
              // r1 = +(r0 * s0) + None
              // [(0, 2), (0, 2)] [(0, 2)]
              __float128 ir1[4]{};
              tensorforge::intel_esimd::simd<__float128, 2> v29_data;
              v29_data.copy_from(r0 + (0));
              __float128 v30_data = s0[0];
              tensorforge::intel_esimd::simd<__float128, 2> v32_data;
              v32_data.copy_from(ir1 + (0));
              (v32_data + (v29_data * v30_data)).copy_to(ir1 + (0));
              __float128 v35_data = s0[2];
              tensorforge::intel_esimd::simd<__float128, 2> v37_data;
              v37_data.copy_from(ir1 + (2));
              (v37_data + (v29_data * v35_data)).copy_to(ir1 + (2));
              tensorforge::intel_esimd::simd<__float128, 2> v39_data;
              v39_data.copy_from(r0 + (2));
              __float128 v40_data = s0[1];
              tensorforge::intel_esimd::simd<__float128, 2> v42_data;
              v42_data.copy_from(ir1 + (0));
              (v42_data + (v39_data * v40_data)).copy_to(ir1 + (0));
              __float128 v45_data = s0[3];
              tensorforge::intel_esimd::simd<__float128, 2> v47_data;
              v47_data.copy_from(ir1 + (2));
              (v47_data + (v39_data * v45_data)).copy_to(ir1 + (2));
              #pragma unroll
              for (int32_t v49_n0 = 0; v49_n0 < 1; ++v49_n0) {
                int32_t v51_a = v49_n0 * 2;
                #pragma unroll
                for (int32_t v50_n1 = 0; v50_n1 < 2; ++v50_n1) {
                  int32_t v53_a = v51_a + (v50_n1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v54_data;
                  v54_data.copy_from(ir1 + (v53_a));
                  v54_data.copy_to(r1 + (v53_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v58_i0 = 0; v58_i0 < 1; ++v58_i0) {
                int32_t v60_a = v58_i0 * 2;
                #pragma unroll
                for (int32_t v59_i1 = 0; v59_i1 < 2; ++v59_i1) {
                  int32_t v62_a = v60_a + (v59_i1 * 2);
                  tensorforge::intel_esimd::simd<__float128, 2> v63_data;
                  v63_data.copy_from(r1 + (v62_a));
                  v63_data.copy_to(glb_m0 + (v62_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

