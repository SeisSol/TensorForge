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
    sycl::accessor<__float128, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
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
          __float128* localShrMem0 = &totalShrMem[2 * item.get_local_id(1) + 0];
          __float128* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              __float128 *const __restrict__ glb_m0 = &m0[batchId0 * 4 + 0 + m0_extraOffset];
              const __float128 *const __restrict__ glb_m1 = &m1[batchId0 * 4 + 0 + m1_extraOffset];
              const __float128 *const __restrict__ glb_m2 = &m2[batchId0 * 4 + 0 + m2_extraOffset];
              __float128 r0[2]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 2;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 2);
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 2; ++v10_i1) {
                  __float128 v18_data = glb_m1[(v15_lead + (v10_i1 * 2))];
                  r0[(v9_i0 + v10_i1)] = v18_data;
                }
              }
              __float128 r1[2]{};
              // r1 = load{g>r}(glb_m2);
              __float128 v21_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v21_lin;
              __float128 v22_lin = glb_m2[2 + item.get_local_id(0) * 1];
              r1[1] = v22_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              __float128 r2[2]{};
              // r2 = +(r0 * r1) + None
              // [(0, 2), (0, 2)] [(0, 2)]
              __float128 ir2[2]{};
              __float128 v28_data = r0[0];
              __float128 v29_data = r1[0];
              __float128 v32_data = ir2[0];
              ir2[0] = (v32_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 0))));
              __float128 v35_data = r1[1];
              __float128 v38_data = ir2[1];
              ir2[1] = (v38_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 0))));
              __float128 v43_data = r0[1];
              __float128 v47_data = ir2[0];
              ir2[0] = (v47_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 1))));
              __float128 v53_data = ir2[1];
              ir2[1] = (v53_data + (v43_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 1))));
              #pragma unroll
              for (int32_t v58_n0 = 0; v58_n0 < 1; ++v58_n0) {
                #pragma unroll
                for (int32_t v59_n1 = 0; v59_n1 < 2; ++v59_n1) {
                  int32_t v60_a = v58_n0 + v59_n1;
                  __float128 v61_data = ir2[v60_a];
                  r2[v60_a] = v61_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v66_i0 = 0; v66_i0 < 1; ++v66_i0) {
                int32_t v74_lead = v8_lead + (v66_i0 * 2);
                #pragma unroll
                for (int32_t v67_i1 = 0; v67_i1 < 2; ++v67_i1) {
                  __float128 v69_data = r2[(v66_i0 + v67_i1)];
                  glb_m0[(v74_lead + (v67_i1 * 2))] = v69_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

