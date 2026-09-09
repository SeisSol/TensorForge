// === base name ===
kernel_d0a007e1b2a1c8e5

// === header ===
void launcher_kernel_d0a007e1b2a1c8e5(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_d0a007e1b2a1c8e5(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d0a007e1b2a1c8e5(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d0a007e1b2a1c8e5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              float r0[128]{};
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v10_k1 = 0; v10_k1 < 8; ++v10_k1) {
                tensorforge::intel_esimd::simd<float, 8> v15_data;
                v15_data.copy_from(glb_m0 + ((v10_k1 * 8)));
                (tensorforge::intel_esimd::abs(v15_data)).copy_to(r0 + ((v10_k1 * 16)));
              }
              // s0 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 8> v22_data;
                v22_data.copy_from(r0 + ((v19_i1 * 16)));
                v22_data.copy_to(s0 + ((v19_i1 * 8)));
              }
              // glb_m1 = neg(s0)
              #pragma unroll
              for (int32_t v27_k1 = 0; v27_k1 < 8; ++v27_k1) {
                int32_t v30_a = v27_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v32_data;
                v32_data.copy_from(s0 + (v30_a));
                ((-v32_data)).copy_to(glb_m1 + (v30_a));
              }
            }
          }
        }
      });
    }
  });
}

