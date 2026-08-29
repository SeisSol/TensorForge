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
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
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
              float r0[8]{};
              // r0 = abs(glb_m0)
              int32_t v7_lead = item.get_local_id(0) % 16;
              if (v7_lead < 8) {
                #pragma unroll
                for (int32_t v9_k1 = 0; v9_k1 < 8; ++v9_k1) {
                  float v17_data = glb_m0[(v7_lead + (v9_k1 * 8))];
                  r0[v9_k1] = (sycl::fabs(v17_data));
                }
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = store{r>s}(localShrMem0, r0);
              if (v7_lead < 8) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                  float v27_data = r0[v25_i1];
                  int32_t v34_a = v7_lead + (v25_i1 * 8);
                  s0[(v34_a ^ ((v34_a >> 5) & 31))] = v27_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m1 = neg(s0)
              if (v7_lead < 8) {
                #pragma unroll
                for (int32_t v42_k1 = 0; v42_k1 < 8; ++v42_k1) {
                  int32_t v48_a = v42_k1 * 8;
                  int32_t v49_a = v7_lead + v48_a;
                  float v53_data = s0[(v49_a ^ ((v49_a >> 5) & 31))];
                  glb_m1[(v7_lead + v48_a)] = ((-v53_data));
                }
              }
            }
          }
        }
      });
    }
  });
}

