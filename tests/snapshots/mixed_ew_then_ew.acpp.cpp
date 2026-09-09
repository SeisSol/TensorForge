// === base name ===
kernel_604919a790473eb7

// === header ===
void launcher_kernel_604919a790473eb7(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_604919a790473eb7(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_604919a790473eb7(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_604919a790473eb7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
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
              float r0[8]{};
              // r0 = abs(glb_m0)
              int32_t v12_lead = item.get_local_id(0) % 16;
              if (v12_lead < 8) {
                #pragma unroll
                for (int32_t v14_k1 = 0; v14_k1 < 8; ++v14_k1) {
                  float v22_data = glb_m0[(v12_lead + (v14_k1 * 8))];
                  r0[v14_k1] = (sycl::fabs(v22_data));
                }
              }
              // s0 = store{r>s}(localShrMem0, r0);
              if (v12_lead < 8) {
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                  float v31_data = r0[v29_i1];
                  int32_t v38_a = v12_lead + (v29_i1 * 8);
                  s0[(v38_a ^ ((v38_a >> 5) & 31))] = v31_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m1 = neg(s0)
              if (v12_lead < 8) {
                #pragma unroll
                for (int32_t v46_k1 = 0; v46_k1 < 8; ++v46_k1) {
                  int32_t v52_a = v46_k1 * 8;
                  int32_t v53_a = v12_lead + v52_a;
                  float v57_data = s0[(v53_a ^ ((v53_a >> 5) & 31))];
                  glb_m1[(v12_lead + v52_a)] = ((-v57_data));
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

