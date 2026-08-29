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
    sycl::accessor<__float128, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (768, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
              __float128 v15_data = glb_m1[v8_lead];
              __float128 v16_data = s0[0];
              __float128 v18_data = ir0[0];
              ir0[0] = (v18_data + (v15_data * v16_data));
              __float128 v26_data = glb_m1[v8_lead];
              __float128 v27_data = s0[2];
              __float128 v29_data = ir0[1];
              ir0[1] = (v29_data + (v26_data * v27_data));
              __float128 v40_data = glb_m1[(v8_lead + 2)];
              __float128 v41_data = s0[1];
              __float128 v43_data = ir0[0];
              ir0[0] = (v43_data + (v40_data * v41_data));
              __float128 v51_data = glb_m1[(v8_lead + 2)];
              __float128 v52_data = s0[3];
              __float128 v54_data = ir0[1];
              ir0[1] = (v54_data + (v51_data * v52_data));
              #pragma unroll
              for (int32_t v59_n0 = 0; v59_n0 < 1; ++v59_n0) {
                #pragma unroll
                for (int32_t v60_n1 = 0; v60_n1 < 2; ++v60_n1) {
                  int32_t v61_a = v59_n0 + v60_n1;
                  __float128 v62_data = ir0[v61_a];
                  r0[v61_a] = v62_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v67_i0 = 0; v67_i0 < 1; ++v67_i0) {
                int32_t v75_lead = v8_lead + (v67_i0 * 2);
                #pragma unroll
                for (int32_t v68_i1 = 0; v68_i1 < 2; ++v68_i1) {
                  __float128 v70_data = r0[(v67_i0 + v68_i1)];
                  glb_m0[(v75_lead + (v68_i1 * 2))] = v70_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

