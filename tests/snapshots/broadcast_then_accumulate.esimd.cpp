// === base name ===
kernel_a10af3b05b9d31a1

// === header ===
void launcher_kernel_a10af3b05b9d31a1(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a10af3b05b9d31a1(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a10af3b05b9d31a1(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a10af3b05b9d31a1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v0_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v0_batchId0][0 + m2_extraOffset];
              float r0[32]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v12_i0 = 0; v12_i0 < 1; ++v12_i0) {
                int32_t v13_lead = v12_i0 * 32;
                tensorforge::intel_esimd::simd<float, 32> v15_data;
                v15_data.copy_from(glb_m0 + (v13_lead));
                v15_data.copy_to(r0 + (v13_lead));
              }
              float r2[96]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v20_lead = v18_i0 * 32;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 3; ++v19_i1) {
                  int32_t v23_a = v20_lead + (v19_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v24_data;
                  v24_data.copy_from(glb_m1 + (v23_a));
                  v24_data.copy_to(r2 + (v23_a));
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[32]{};
              // r1 = +(r0) + None
              // [(0, 32)] []
              tensorforge::intel_esimd::simd<float, 32> v29_data;
              v29_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v30_data;
              v30_data.copy_from(r1 + (0));
              (v30_data + v29_data).copy_to(r1 + (0));
              // wait(r2 = load{g>r}(glb_m1););
              float r3[96]{};
              // r3 = +(r2) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v33_data;
              v33_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 32> v34_data;
              v34_data.copy_from(r3 + (0));
              (v34_data + v33_data).copy_to(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 32> v37_data;
              v37_data.copy_from(r3 + (32));
              (v37_data + v36_data).copy_to(r3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v39_data;
              v39_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(r3 + (64));
              (v40_data + v39_data).copy_to(r3 + (64));
              float r4[96]{};
              // r4 = +(r1) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v43_data;
              v43_data.copy_from(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(r4 + (0));
              (v44_data + v43_data).copy_to(r4 + (0));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r4 + (32));
              (v47_data + v43_data).copy_to(r4 + (32));
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r4 + (64));
              (v50_data + v43_data).copy_to(r4 + (64));
              float r5[96]{};
              // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir5[96]{};
              tensorforge::intel_esimd::simd<float, 32> v54_data;
              v54_data.copy_from(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v55_data;
              v55_data.copy_from(ir5 + (0));
              (v55_data + v54_data).copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 32> v57_data;
              v57_data.copy_from(r3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(ir5 + (32));
              (v58_data + v57_data).copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 32> v60_data;
              v60_data.copy_from(r3 + (64));
              tensorforge::intel_esimd::simd<float, 32> v61_data;
              v61_data.copy_from(ir5 + (64));
              (v61_data + v60_data).copy_to(ir5 + (64));
              #pragma unroll
              for (int32_t v63_n0 = 0; v63_n0 < 1; ++v63_n0) {
                int32_t v65_a = v63_n0 * 32;
                #pragma unroll
                for (int32_t v64_n1 = 0; v64_n1 < 3; ++v64_n1) {
                  int32_t v67_a = v65_a + (v64_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v68_data;
                  v68_data.copy_from(ir5 + (v67_a));
                  tensorforge::intel_esimd::simd<float, 32> v72_data;
                  v72_data.copy_from(r4 + (v67_a));
                  (v72_data + v68_data).copy_to(r5 + (v67_a));
                }
              }
              float r6[96]{};
              // r6 = +(r5) + None
              // [(0, 32), (0, 3)] []
              float ir6[96]{};
              tensorforge::intel_esimd::simd<float, 32> v79_data;
              v79_data.copy_from(r5 + (0));
              tensorforge::intel_esimd::simd<float, 32> v80_data;
              v80_data.copy_from(ir6 + (0));
              (v80_data + v79_data).copy_to(ir6 + (0));
              tensorforge::intel_esimd::simd<float, 32> v82_data;
              v82_data.copy_from(r5 + (32));
              tensorforge::intel_esimd::simd<float, 32> v83_data;
              v83_data.copy_from(ir6 + (32));
              (v83_data + v82_data).copy_to(ir6 + (32));
              tensorforge::intel_esimd::simd<float, 32> v85_data;
              v85_data.copy_from(r5 + (64));
              tensorforge::intel_esimd::simd<float, 32> v86_data;
              v86_data.copy_from(ir6 + (64));
              (v86_data + v85_data).copy_to(ir6 + (64));
              #pragma unroll
              for (int32_t v88_n0 = 0; v88_n0 < 1; ++v88_n0) {
                int32_t v90_a = v88_n0 * 32;
                #pragma unroll
                for (int32_t v89_n1 = 0; v89_n1 < 3; ++v89_n1) {
                  int32_t v92_a = v90_a + (v89_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v93_data;
                  v93_data.copy_from(ir6 + (v92_a));
                  v93_data.copy_to(r6 + (v92_a));
                }
              }
              // glb_m2 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v97_i0 = 0; v97_i0 < 1; ++v97_i0) {
                int32_t v99_a = v97_i0 * 32;
                #pragma unroll
                for (int32_t v98_i1 = 0; v98_i1 < 3; ++v98_i1) {
                  int32_t v101_a = v99_a + (v98_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v102_data;
                  v102_data.copy_from(r6 + (v101_a));
                  v102_data.copy_to(glb_m2 + (v101_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

