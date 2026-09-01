// === base name ===
kernel_f2b477f03e

// === header ===
void launcher_kernel_f2b477f03e(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f2b477f03e(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f2b477f03e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f2b477f03e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // TMP = abs(A)
        // C = neg(TMP)
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
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              float r0[128]{};
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v5_k1 = 0; v5_k1 < 8; ++v5_k1) {
                tensorforge::intel_esimd::simd<float, 8> v10_data;
                v10_data.copy_from(glb_m0 + ((v5_k1 * 8)));
                (tensorforge::intel_esimd::abs(v10_data)).copy_to(r0 + ((v5_k1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v15_i1 = 0; v15_i1 < 8; ++v15_i1) {
                tensorforge::intel_esimd::simd<float, 8> v18_data;
                v18_data.copy_from(r0 + ((v15_i1 * 16)));
                v18_data.copy_to(s0 + ((v15_i1 * 8)));
              }
              // glb_m1 = neg(s0)
              #pragma unroll
              for (int32_t v23_k1 = 0; v23_k1 < 8; ++v23_k1) {
                int32_t v26_a = v23_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v28_data;
                v28_data.copy_from(s0 + (v26_a));
                ((-v28_data)).copy_to(glb_m1 + (v26_a));
              }
            }
          }
        }
      });
    }
  });
}

