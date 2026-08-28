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
              float r0[1]{};
              // r0 = +(glb_m0) + None
              // [(0, 32)] []
              tensorforge::intel_esimd::simd<int32_t, 32> v4_lead = tensorforge::intel_esimd::simd<int32_t, 32>(0, 1);
              int32_t v6_lead = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 32> v9_data;
              v9_data.copy_from(glb_m0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 32> v10_data;
              v10_data.copy_from(r0 + (0));
              (v10_data + v9_data).copy_to(r0 + (0));
              float r1[3]{};
              // r1 = +(glb_m1) + None
              // [(0, 32), (0, 3)] []
              int32_t v16_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 32> v20_data;
              v20_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 32> v21_data;
              v21_data.copy_from(r1 + (0));
              (v21_data + v20_data).copy_to(r1 + (0));
              int32_t v25_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 32> v29_data;
              v29_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 32> v30_data;
              v30_data.copy_from(r1 + (1));
              (v30_data + v29_data).copy_to(r1 + (1));
              int32_t v34_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 32> v38_data;
              v38_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 32> v39_data;
              v39_data.copy_from(r1 + (2));
              (v39_data + v38_data).copy_to(r1 + (2));
              float r2[3]{};
              // r2 = +(r0) + None
              // [(0, 32), (0, 3)] []
              tensorforge::intel_esimd::simd<float, 32> v43_data;
              v43_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(r2 + (0));
              (v44_data + v43_data).copy_to(r2 + (0));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r2 + (1));
              (v47_data + v43_data).copy_to(r2 + (1));
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r2 + (2));
              (v50_data + v43_data).copy_to(r2 + (2));
              float r3[3]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir3[3]{};
              tensorforge::intel_esimd::simd<float, 32> v55_data;
              v55_data.copy_from(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v56_data;
              v56_data.copy_from(ir3 + (0));
              (v56_data + v55_data).copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(r1 + (1));
              tensorforge::intel_esimd::simd<float, 32> v59_data;
              v59_data.copy_from(ir3 + (1));
              (v59_data + v58_data).copy_to(ir3 + (1));
              tensorforge::intel_esimd::simd<float, 32> v61_data;
              v61_data.copy_from(r1 + (2));
              tensorforge::intel_esimd::simd<float, 32> v62_data;
              v62_data.copy_from(ir3 + (2));
              (v62_data + v61_data).copy_to(ir3 + (2));
              #pragma unroll
              for (int32_t v65_n0 = 0; v65_n0 < 1; ++v65_n0) {
                #pragma unroll
                for (int32_t v66_n1 = 0; v66_n1 < 3; ++v66_n1) {
                  int32_t v67_a = v65_n0 + v66_n1;
                  int32_t v68_a = v65_n0 + v66_n1;
                  tensorforge::intel_esimd::simd<float, 32> v69_data;
                  v69_data.copy_from(ir3 + (v68_a));
                  int32_t v70_a = v65_n0 + v66_n1;
                  tensorforge::intel_esimd::simd<float, 32> v72_data;
                  v72_data.copy_from(r2 + (v68_a));
                  (v72_data + v69_data).copy_to(r3 + (v68_a));
                }
              }
              float r4[3]{};
              // r4 = +(r3) + None
              // [(0, 32), (0, 3)] []
              float ir4[3]{};
              tensorforge::intel_esimd::simd<float, 32> v78_data;
              v78_data.copy_from(r3 + (0));
              tensorforge::intel_esimd::simd<float, 32> v79_data;
              v79_data.copy_from(ir4 + (0));
              (v79_data + v78_data).copy_to(ir4 + (0));
              tensorforge::intel_esimd::simd<float, 32> v81_data;
              v81_data.copy_from(r3 + (1));
              tensorforge::intel_esimd::simd<float, 32> v82_data;
              v82_data.copy_from(ir4 + (1));
              (v82_data + v81_data).copy_to(ir4 + (1));
              tensorforge::intel_esimd::simd<float, 32> v84_data;
              v84_data.copy_from(r3 + (2));
              tensorforge::intel_esimd::simd<float, 32> v85_data;
              v85_data.copy_from(ir4 + (2));
              (v85_data + v84_data).copy_to(ir4 + (2));
              #pragma unroll
              for (int32_t v88_n0 = 0; v88_n0 < 1; ++v88_n0) {
                #pragma unroll
                for (int32_t v89_n1 = 0; v89_n1 < 3; ++v89_n1) {
                  int32_t v90_a = v88_n0 + v89_n1;
                  int32_t v91_a = v88_n0 + v89_n1;
                  tensorforge::intel_esimd::simd<float, 32> v92_data;
                  v92_data.copy_from(ir4 + (v91_a));
                  v92_data.copy_to(r4 + (v91_a));
                }
              }
              // glb_m2 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v95_i0 = 0; v95_i0 < 1; ++v95_i0) {
                int32_t v100_lead = v95_i0 * 32;
                #pragma unroll
                for (int32_t v96_i1 = 0; v96_i1 < 3; ++v96_i1) {
                  int32_t v97_a = v95_i0 + v96_i1;
                  tensorforge::intel_esimd::simd<float, 32> v99_data;
                  v99_data.copy_from(r4 + ((v95_i0 + v96_i1)));
                  v99_data.copy_to(glb_m2 + ((v100_lead + (v96_i1 * 32))));
                }
              }
            }
          }
        }
      });
    }
  });
}

