// === base name ===
kernel_90d491a14e

// === header ===
void launcher_kernel_90d491a14e(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_90d491a14e(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_90d491a14e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_90d491a14e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 1(1) {0..1} strided
        // OUT = max(A, dims=[0, 1])
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
              float *const __restrict__ glb_m1 = &m1[batchId0 * 1 + 0 + m1_extraOffset];
              // glb_m1 = max(glb_m0, dims=[0, 1])
              tensorforge::intel_esimd::simd<int32_t, 16> v4_lead = tensorforge::intel_esimd::simd<int32_t, 16>(0, 1);
              tensorforge::intel_esimd::simd<float, 16> v6_acc0(-INFINITY);
              #pragma unroll
              for (int32_t v5_r1 = 0; v5_r1 < 16; ++v5_r1) {
                tensorforge::intel_esimd::simd<float, 16> v11_data;
                v11_data.copy_from(glb_m0 + ((v5_r1 * 16)));
                v6_acc0 = (tensorforge::intel_esimd::max(v6_acc0, v11_data));
              }
              glb_m1[0] = (tensorforge::intel_esimd::hmax<float>(v6_acc0));
            }
          }
        }
      });
    }
  });
}

