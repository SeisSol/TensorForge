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
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[32]{};
              // r0 = +(glb_m0) + None
              // [(0, 32)] []
              int32_t v5_lead = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 32> v8_data;
              v8_data.copy_from(glb_m0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 32> v9_data;
              v9_data.copy_from(r0 + (0));
              (v9_data + v8_data).copy_to(r0 + (0));
              float r1[96]{};
              // r1 = +(glb_m1) + None
              // [(0, 32), (0, 3)] []
              int32_t v14_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v18_data;
              v18_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 32> v19_data;
              v19_data.copy_from(r1 + (0));
              (v19_data + v18_data).copy_to(r1 + (0));
              int32_t v23_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 32> v27_data;
              v27_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 32> v28_data;
              v28_data.copy_from(r1 + (32));
              (v28_data + v27_data).copy_to(r1 + (32));
              int32_t v32_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 32> v37_data;
              v37_data.copy_from(r1 + (64));
              (v37_data + v36_data).copy_to(r1 + (64));
              float r2[96]{};
              // r2 = +(r0) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(r2 + (0));
              (v41_data + v40_data).copy_to(r2 + (0));
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(r2 + (32));
              (v44_data + v40_data).copy_to(r2 + (32));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r2 + (64));
              (v47_data + v40_data).copy_to(r2 + (64));
              float r3[96]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir3[96]{};
              tensorforge::intel_esimd::simd<float, 32> v51_data;
              v51_data.copy_from(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v52_data;
              v52_data.copy_from(ir3 + (0));
              (v52_data + v51_data).copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v54_data;
              v54_data.copy_from(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v55_data;
              v55_data.copy_from(ir3 + (32));
              (v55_data + v54_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v57_data;
              v57_data.copy_from(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(ir3 + (64));
              (v58_data + v57_data).copy_to(ir3 + (64));
              #pragma unroll
              for (int32_t v60_n0 = 0; v60_n0 < 1; ++v60_n0) {
                int32_t v62_a = v60_n0 * 32;
                #pragma unroll
                for (int32_t v61_n1 = 0; v61_n1 < 3; ++v61_n1) {
                  int32_t v63_a = v61_n1 * 32;
                  int32_t v64_a = v62_a + v63_a;
                  int32_t v67_a = v62_a + v63_a;
                  tensorforge::intel_esimd::simd<float, 32> v68_data;
                  v68_data.copy_from(ir3 + (v67_a));
                  int32_t v71_a = v62_a + v63_a;
                  tensorforge::intel_esimd::simd<float, 32> v75_data;
                  v75_data.copy_from(r2 + (v67_a));
                  (v75_data + v68_data).copy_to(r3 + (v67_a));
                }
              }
              float r4[96]{};
              // r4 = +(r3) + None
              // [(0, 32), (0, 3)] []
              float ir4[96]{};
              tensorforge::intel_esimd::simd<float, 32> v82_data;
              v82_data.copy_from(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v83_data;
              v83_data.copy_from(ir4 + (0));
              (v83_data + v82_data).copy_to(ir4 + (0));
              tensorforge::intel_esimd::simd<float, 32> v85_data;
              v85_data.copy_from(r3 + (32));
              tensorforge::intel_esimd::simd<float, 32> v86_data;
              v86_data.copy_from(ir4 + (32));
              (v86_data + v85_data).copy_to(ir4 + (32));
              tensorforge::intel_esimd::simd<float, 32> v88_data;
              v88_data.copy_from(r3 + (64));
              tensorforge::intel_esimd::simd<float, 32> v89_data;
              v89_data.copy_from(ir4 + (64));
              (v89_data + v88_data).copy_to(ir4 + (64));
              #pragma unroll
              for (int32_t v91_n0 = 0; v91_n0 < 1; ++v91_n0) {
                int32_t v93_a = v91_n0 * 32;
                #pragma unroll
                for (int32_t v92_n1 = 0; v92_n1 < 3; ++v92_n1) {
                  int32_t v94_a = v92_n1 * 32;
                  int32_t v95_a = v93_a + v94_a;
                  int32_t v98_a = v93_a + v94_a;
                  tensorforge::intel_esimd::simd<float, 32> v99_data;
                  v99_data.copy_from(ir4 + (v98_a));
                  v99_data.copy_to(r4 + (v98_a));
                }
              }
              // glb_m2 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v103_i0 = 0; v103_i0 < 1; ++v103_i0) {
                int32_t v105_a = v103_i0 * 32;
                #pragma unroll
                for (int32_t v104_i1 = 0; v104_i1 < 3; ++v104_i1) {
                  int32_t v106_a = v104_i1 * 32;
                  int32_t v107_a = v105_a + v106_a;
                  int32_t v110_a = v105_a + v106_a;
                  tensorforge::intel_esimd::simd<float, 32> v111_data;
                  v111_data.copy_from(r4 + (v110_a));
                  v111_data.copy_to(glb_m2 + (v110_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

