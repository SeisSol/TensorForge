// === base name ===
kernel_ba8a743602

// === header ===
void launcher_kernel_ba8a743602(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ba8a743602(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ba8a743602(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ba8a743602(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // B = rcp(A)
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
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              // glb_m1 = rcp(glb_m0)
              #pragma unroll
              for (int32_t v2_k0 = 0; v2_k0 < 1; ++v2_k0) {
                int32_t v4_lead = v2_k0 * 16;
                #pragma unroll
                for (int32_t v3_k1 = 0; v3_k1 < 16; ++v3_k1) {
                  int32_t v6_a = v3_k1 * 16;
                  int32_t v7_a = v4_lead + v6_a;
                  int32_t v11_a = v4_lead + v6_a;
                  tensorforge::intel_esimd::simd<float, 16> v12_data;
                  v12_data.copy_from(glb_m0 + (v11_a));
                  (tensorforge::intel_esimd::inv(v12_data)).copy_to(glb_m1 + (v11_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

