// === base name ===
kernel_86589abfea5c83f7

// === header ===
void launcher_kernel_86589abfea5c83f7(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_86589abfea5c83f7(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_86589abfea5c83f7(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_86589abfea5c83f7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[32]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
                int32_t v9_lead = v8_i0 * 32;
                tensorforge::intel_esimd::simd<float, 32> v11_data;
                v11_data.copy_from(glb_m0 + (v9_lead));
                v11_data.copy_to(r0 + (v9_lead));
              }
              float r2[96]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
                int32_t v16_lead = v14_i0 * 32;
                #pragma unroll
                for (int32_t v15_i1 = 0; v15_i1 < 3; ++v15_i1) {
                  int32_t v19_a = v16_lead + (v15_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v20_data;
                  v20_data.copy_from(glb_m1 + (v19_a));
                  v20_data.copy_to(r2 + (v19_a));
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[32]{};
              // r1 = +(r0) + None
              // [(0, 32)] []
              tensorforge::intel_esimd::simd<float, 32> v25_data;
              v25_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v26_data;
              v26_data.copy_from(r1 + (0));
              (v26_data + v25_data).copy_to(r1 + (0));
              // wait(r2 = load{g>r}(glb_m1););
              float r3[96]{};
              // r3 = +(r2) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v29_data;
              v29_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 32> v30_data;
              v30_data.copy_from(r3 + (0));
              (v30_data + v29_data).copy_to(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v32_data;
              v32_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 32> v33_data;
              v33_data.copy_from(r3 + (32));
              (v33_data + v32_data).copy_to(r3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v35_data;
              v35_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(r3 + (64));
              (v36_data + v35_data).copy_to(r3 + (64));
              float r4[96]{};
              // r4 = +(r1) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v39_data;
              v39_data.copy_from(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(r4 + (0));
              (v40_data + v39_data).copy_to(r4 + (0));
              tensorforge::intel_esimd::simd<float, 32> v43_data;
              v43_data.copy_from(r4 + (32));
              (v43_data + v39_data).copy_to(r4 + (32));
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(r4 + (64));
              (v46_data + v39_data).copy_to(r4 + (64));
              float r5[96]{};
              // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir5[96]{};
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v51_data;
              v51_data.copy_from(ir5 + (0));
              (v51_data + v50_data).copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(r3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v54_data;
              v54_data.copy_from(ir5 + (32));
              (v54_data + v53_data).copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 32> v56_data;
              v56_data.copy_from(r3 + (64));
              tensorforge::intel_esimd::simd<float, 32> v57_data;
              v57_data.copy_from(ir5 + (64));
              (v57_data + v56_data).copy_to(ir5 + (64));
              #pragma unroll
              for (int32_t v59_n0 = 0; v59_n0 < 1; ++v59_n0) {
                int32_t v61_a = v59_n0 * 32;
                #pragma unroll
                for (int32_t v60_n1 = 0; v60_n1 < 3; ++v60_n1) {
                  int32_t v63_a = v61_a + (v60_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v64_data;
                  v64_data.copy_from(ir5 + (v63_a));
                  tensorforge::intel_esimd::simd<float, 32> v68_data;
                  v68_data.copy_from(r4 + (v63_a));
                  (v68_data + v64_data).copy_to(r5 + (v63_a));
                }
              }
              float r6[96]{};
              // r6 = +(r5) + None
              // [(0, 32), (0, 3)] []
              float ir6[96]{};
              tensorforge::intel_esimd::simd<float, 32> v75_data;
              v75_data.copy_from(r5 + (0));
              tensorforge::intel_esimd::simd<float, 32> v76_data;
              v76_data.copy_from(ir6 + (0));
              (v76_data + v75_data).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v78_data;
              v78_data.copy_from(r5 + (32));
              tensorforge::intel_esimd::simd<float, 32> v79_data;
              v79_data.copy_from(ir6 + (32));
              (v79_data + v78_data).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v81_data;
              v81_data.copy_from(r5 + (64));
              tensorforge::intel_esimd::simd<float, 32> v82_data;
              v82_data.copy_from(ir6 + (64));
              (v82_data + v81_data).copy_to(ir6 + (64));
              #pragma unroll
              for (int32_t v84_n0 = 0; v84_n0 < 1; ++v84_n0) {
                int32_t v86_a = v84_n0 * 32;
                #pragma unroll
                for (int32_t v85_n1 = 0; v85_n1 < 3; ++v85_n1) {
                  int32_t v88_a = v86_a + (v85_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v89_data;
                  v89_data.copy_from(ir6 + (v88_a));
                  v89_data.copy_to(r6 + (v88_a));
                }
              }
              // glb_m2 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v93_i0 = 0; v93_i0 < 1; ++v93_i0) {
                int32_t v95_a = v93_i0 * 32;
                #pragma unroll
                for (int32_t v94_i1 = 0; v94_i1 < 3; ++v94_i1) {
                  int32_t v97_a = v95_a + (v94_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v98_data;
                  v98_data.copy_from(r6 + (v97_a));
                  v98_data.copy_to(glb_m2 + (v97_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

