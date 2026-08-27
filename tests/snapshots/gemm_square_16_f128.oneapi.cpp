// === base name ===
kernel_0b2fc070b9

// === header ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0b2fc070b9(__float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (2, 128, 1);
  sycl::range<3> grid ((numElements0 + 128 - 1) / 128, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0b2fc070b9(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0b2fc070b9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, __float128* m0, unsigned m0_extraOffset, const __float128* m1, unsigned m1_extraOffset, const __float128* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<__float128, 1> totalShrMem (768, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 2×2(2×2) {0..2}×{0..2} strided
        // m1 2×2(2×2) {0..2}×{0..2} strided
        // m2 2×2(2×2) {0..2}×{0..2} strided
        // m0 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, 1] = m1 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[0, -1]×m2 2×2(2×2) {0..2}×{0..2} strided({0..2}×{0..2})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          __float128* localShrMem0 = &totalShrMem[6 * item.get_local_id(1) + 0];
          __float128* tempShrMem = &localShrMem0[4];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
              const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
              const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
              __float128* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              s0[0 + 0 + 1 * item.get_local_id(0) + 0] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 2] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 2];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              __float128 r0[2]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 2), (0, 2)] [(0, 2)]
              __float128 ir0[2]{};
              int32_t v8_lead = item.get_local_id(0) % 2;
              int32_t v14_a = v8_lead + 0;
              __float128 v21_data = glb_m1[v8_lead];
              __float128 v22_data = s0[0];
              __float128 v24_data = ir0[0];
              ir0[0] = (v24_data + (v21_data * v22_data));
              int32_t v31_a = v8_lead + 0;
              __float128 v38_data = glb_m1[v8_lead];
              __float128 v39_data = s0[2];
              __float128 v41_data = ir0[1];
              ir0[1] = (v41_data + (v38_data * v39_data));
              int32_t v51_a = v8_lead + 2;
              __float128 v58_data = glb_m1[(v8_lead + 2)];
              __float128 v59_data = s0[1];
              __float128 v61_data = ir0[0];
              ir0[0] = (v61_data + (v58_data * v59_data));
              int32_t v68_a = v8_lead + 2;
              __float128 v75_data = glb_m1[(v8_lead + 2)];
              __float128 v76_data = s0[3];
              __float128 v78_data = ir0[1];
              ir0[1] = (v78_data + (v75_data * v76_data));
              #pragma unroll
              for (int32_t v83_n0 = 0; v83_n0 < 1; ++v83_n0) {
                #pragma unroll
                for (int32_t v84_n1 = 0; v84_n1 < 2; ++v84_n1) {
                  int32_t v85_a = v83_n0 + v84_n1;
                  int32_t v86_a = v83_n0 + v84_n1;
                  __float128 v87_data = ir0[v86_a];
                  int32_t v88_a = v83_n0 + v84_n1;
                  r0[v86_a] = v87_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v93_i0 = 0; v93_i0 < 1; ++v93_i0) {
                int32_t v102_lead = v8_lead + (v93_i0 * 2);
                #pragma unroll
                for (int32_t v94_i1 = 0; v94_i1 < 2; ++v94_i1) {
                  int32_t v95_a = v93_i0 + v94_i1;
                  __float128 v97_data = r0[(v93_i0 + v94_i1)];
                  int32_t v104_a = v102_lead + (v94_i1 * 2);
                  glb_m0[v104_a] = v97_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

