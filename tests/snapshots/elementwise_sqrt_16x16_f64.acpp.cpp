// === base name ===
kernel_7bad7afe30

// === header ===
void launcher_kernel_7bad7afe30(const double* m0, unsigned m0_extraOffset, double* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7bad7afe30(const double* m0, unsigned m0_extraOffset, double* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7bad7afe30(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7bad7afe30(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const double* m0, unsigned m0_extraOffset, double* m1, unsigned m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // B = sqrt(A)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              // glb_m1 = sqrt(glb_m0)
              int32_t v6_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v7_k0 = 0; v7_k0 < 1; ++v7_k0) {
                int32_t v12_lead = v7_k0 * 16;
                int32_t v13_lead = v6_lead + v12_lead;
                int32_t v22_lead = v6_lead + v12_lead;
                #pragma unroll
                for (int32_t v8_k1 = 0; v8_k1 < 16; ++v8_k1) {
                  int32_t v14_a = v8_k1 * 16;
                  double v16_data = glb_m0[(v13_lead + v14_a)];
                  glb_m1[(v22_lead + v14_a)] = (sycl::sqrt(v16_data));
                }
              }
            }
          }
        }
      });
    }
  });
}

