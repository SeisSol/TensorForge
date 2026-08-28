// === base name ===
kernel_a7d5d30824

// === header ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a7d5d30824(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a7d5d30824(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a7d5d30824(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16(16) {0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              float r0[16]{};
              // r0 = +(glb_m1) + None
              // [(0, 16)] [(0, 16)]
              float ir0[16]{};
              tensorforge::intel_esimd::simd<float, 16> v7_data;
              v7_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v8_data;
              v8_data.copy_from(ir0 + (0));
              (v8_data + v7_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v13_data;
              v13_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v14_data;
              v14_data.copy_from(ir0 + (0));
              (v14_data + v13_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v20_data;
              v20_data.copy_from(ir0 + (0));
              (v20_data + v19_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(ir0 + (0));
              (v26_data + v25_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(ir0 + (0));
              (v32_data + v31_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(ir0 + (0));
              (v38_data + v37_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(ir0 + (0));
              (v44_data + v43_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(ir0 + (0));
              (v50_data + v49_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(ir0 + (0));
              (v56_data + v55_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data;
              v62_data.copy_from(ir0 + (0));
              (v62_data + v61_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data;
              v68_data.copy_from(ir0 + (0));
              (v68_data + v67_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v73_data;
              v73_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data;
              v74_data.copy_from(ir0 + (0));
              (v74_data + v73_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v79_data;
              v79_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(ir0 + (0));
              (v80_data + v79_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v85_data;
              v85_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(ir0 + (0));
              (v86_data + v85_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v91_data;
              v91_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(ir0 + (0));
              (v92_data + v91_data).copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v97_data;
              v97_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v98_data;
              v98_data.copy_from(ir0 + (0));
              (v98_data + v97_data).copy_to(ir0 + (0));
              #pragma unroll
              for (int32_t v100_n0 = 0; v100_n0 < 1; ++v100_n0) {
                int32_t v101_a = v100_n0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v102_data;
                v102_data.copy_from(ir0 + (v101_a));
                v102_data.copy_to(r0 + (v101_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v104_i0 = 0; v104_i0 < 1; ++v104_i0) {
                int32_t v105_a = v104_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v106_data;
                v106_data.copy_from(r0 + (v105_a));
                v106_data.copy_to(glb_m0 + (v105_a));
              }
            }
          }
        }
      });
    }
  });
}

