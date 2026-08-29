// === base name ===
kernel_7cc2a3c5b0

// === header ===
void launcher_kernel_7cc2a3c5b0(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7cc2a3c5b0(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7cc2a3c5b0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7cc2a3c5b0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32(32) {0..32} pointer_based
        // m1 32×3(32×3) {0..32}×{0..3} pointer_based
        // m2 32×3(32×3) {0..32}×{0..3} pointer_based
        // t0 32(32) {0..32} strided({0..32})[0] = m0 32(32) {0..32} pointer_based({0..32})[0]
        // t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = m1 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1]
        // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = t0 32(32) {0..32} strided({0..32})[0]
        // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] += t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
        // m2 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1] = t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[32]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
                int32_t v5_lead = v4_i0 * 32;
                tensorforge::intel_esimd::simd<float, 32> v7_data;
                v7_data.copy_from(glb_m0 + (v5_lead));
                v7_data.copy_to(r0 + (v5_lead));
              }
              float r2[96]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
                int32_t v12_lead = v10_i0 * 32;
                #pragma unroll
                for (int32_t v11_i1 = 0; v11_i1 < 3; ++v11_i1) {
                  int32_t v15_a = v12_lead + (v11_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v16_data;
                  v16_data.copy_from(glb_m1 + (v15_a));
                  v16_data.copy_to(r2 + (v15_a));
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[32]{};
              // r1 = +(r0) + None
              // [(0, 32)] []
              tensorforge::intel_esimd::simd<float, 32> v21_data;
              v21_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v22_data;
              v22_data.copy_from(r1 + (0));
              (v22_data + v21_data).copy_to(r1 + (0));
              // wait(r2 = load{g>r}(glb_m1););
              float r3[96]{};
              // r3 = +(r2) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v25_data;
              v25_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 32> v26_data;
              v26_data.copy_from(r3 + (0));
              (v26_data + v25_data).copy_to(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v28_data;
              v28_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 32> v29_data;
              v29_data.copy_from(r3 + (32));
              (v29_data + v28_data).copy_to(r3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v31_data;
              v31_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 32> v32_data;
              v32_data.copy_from(r3 + (64));
              (v32_data + v31_data).copy_to(r3 + (64));
              float r4[96]{};
              // r4 = +(r1) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v35_data;
              v35_data.copy_from(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(r4 + (0));
              (v36_data + v35_data).copy_to(r4 + (0));
              tensorforge::intel_esimd::simd<float, 32> v39_data;
              v39_data.copy_from(r4 + (32));
              (v39_data + v35_data).copy_to(r4 + (32));
              tensorforge::intel_esimd::simd<float, 32> v42_data;
              v42_data.copy_from(r4 + (64));
              (v42_data + v35_data).copy_to(r4 + (64));
              float r5[96]{};
              // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir5[96]{};
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(ir5 + (0));
              (v47_data + v46_data).copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 32> v49_data;
              v49_data.copy_from(r3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(ir5 + (32));
              (v50_data + v49_data).copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 32> v52_data;
              v52_data.copy_from(r3 + (64));
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(ir5 + (64));
              (v53_data + v52_data).copy_to(ir5 + (64));
              #pragma unroll
              for (int32_t v55_n0 = 0; v55_n0 < 1; ++v55_n0) {
                int32_t v57_a = v55_n0 * 32;
                #pragma unroll
                for (int32_t v56_n1 = 0; v56_n1 < 3; ++v56_n1) {
                  int32_t v59_a = v57_a + (v56_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v60_data;
                  v60_data.copy_from(ir5 + (v59_a));
                  tensorforge::intel_esimd::simd<float, 32> v64_data;
                  v64_data.copy_from(r4 + (v59_a));
                  (v64_data + v60_data).copy_to(r5 + (v59_a));
                }
              }
              float r6[96]{};
              // r6 = +(r5) + None
              // [(0, 32), (0, 3)] []
              float ir6[96]{};
              tensorforge::intel_esimd::simd<float, 32> v71_data;
              v71_data.copy_from(r5 + (0));
              tensorforge::intel_esimd::simd<float, 32> v72_data;
              v72_data.copy_from(ir6 + (0));
              (v72_data + v71_data).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v74_data;
              v74_data.copy_from(r5 + (32));
              tensorforge::intel_esimd::simd<float, 32> v75_data;
              v75_data.copy_from(ir6 + (32));
              (v75_data + v74_data).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v77_data;
              v77_data.copy_from(r5 + (64));
              tensorforge::intel_esimd::simd<float, 32> v78_data;
              v78_data.copy_from(ir6 + (64));
              (v78_data + v77_data).copy_to(ir6 + (64));
              #pragma unroll
              for (int32_t v80_n0 = 0; v80_n0 < 1; ++v80_n0) {
                int32_t v82_a = v80_n0 * 32;
                #pragma unroll
                for (int32_t v81_n1 = 0; v81_n1 < 3; ++v81_n1) {
                  int32_t v84_a = v82_a + (v81_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v85_data;
                  v85_data.copy_from(ir6 + (v84_a));
                  v85_data.copy_to(r6 + (v84_a));
                }
              }
              // glb_m2 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v89_i0 = 0; v89_i0 < 1; ++v89_i0) {
                int32_t v91_a = v89_i0 * 32;
                #pragma unroll
                for (int32_t v90_i1 = 0; v90_i1 < 3; ++v90_i1) {
                  int32_t v93_a = v91_a + (v90_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v94_data;
                  v94_data.copy_from(r6 + (v93_a));
                  v94_data.copy_to(glb_m2 + (v93_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

