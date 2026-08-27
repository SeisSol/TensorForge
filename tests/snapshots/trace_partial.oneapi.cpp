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
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              float r0[1]{};
              // r0 = +(glb_m1) + None
              // [(0, 16)] [(0, 16)]
              float ir0[1]{};
              int32_t v8_a = 0_i32 + 0;
              int32_t v13_a = 0_i32 + 16;
              int32_t v18_a = 0_i32 + 32;
              int32_t v23_a = 0_i32 + 48;
              int32_t v28_a = 0_i32 + 64;
              int32_t v33_a = 0_i32 + 80;
              int32_t v38_a = 0_i32 + 96;
              int32_t v43_a = 0_i32 + 112;
              int32_t v48_a = 0_i32 + 128;
              int32_t v53_a = 0_i32 + 144;
              int32_t v58_a = 0_i32 + 160;
              int32_t v63_a = 0_i32 + 176;
              int32_t v68_a = 0_i32 + 192;
              int32_t v73_a = 0_i32 + 208;
              int32_t v78_a = 0_i32 + 224;
              int32_t v83_a = 0_i32 + 240;
              #pragma unroll
              for (int32_t v87_n0 = 0; v87_n0 < 1; ++v87_n0) {
                None = r0[v87_n0];
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v91_i0 = 0; v91_i0 < 1; ++v91_i0) {
                int32_t v92_lead = v91_i0 * 16;
                None.copy_to(glb_m0[v92_lead]);
              }
            }
          }
        }
      });
    }
  });
}

