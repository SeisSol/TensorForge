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
        // options: default
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
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v30_lead = v8_lead + (v24_i0 * 2);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 2; ++v25_i1) {
                  __float128 v33_data = glb_m2[(v30_lead + (v25_i1 * 2))];
                  r1[(v24_i0 + v25_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              __float128 r2[2]{};
              // r2 = +(r0 * r1) + None
              // [(0, 2), (0, 2)] [(0, 2)]
              __float128 ir2[2]{};
              __float128 v40_data = r0[0];
              __float128 v41_data = r1[0];
              __float128 v44_data = ir2[0];
              ir2[0] = (v44_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 0))));
              __float128 v47_data = r1[1];
              __float128 v50_data = ir2[1];
              ir2[1] = (v50_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 0))));
              __float128 v55_data = r0[1];
              __float128 v59_data = ir2[0];
              ir2[0] = (v59_data + (v55_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 1))));
              __float128 v65_data = ir2[1];
              ir2[1] = (v65_data + (v55_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 1))));
              #pragma unroll
              for (int32_t v70_n0 = 0; v70_n0 < 1; ++v70_n0) {
                #pragma unroll
                for (int32_t v71_n1 = 0; v71_n1 < 2; ++v71_n1) {
                  int32_t v72_a = v70_n0 + v71_n1;
                  __float128 v73_data = ir2[v72_a];
                  r2[v72_a] = v73_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v78_i0 = 0; v78_i0 < 1; ++v78_i0) {
                int32_t v86_lead = v8_lead + (v78_i0 * 2);
                #pragma unroll
                for (int32_t v79_i1 = 0; v79_i1 < 2; ++v79_i1) {
                  __float128 v81_data = r2[(v78_i0 + v79_i1)];
                  glb_m0[(v86_lead + (v79_i1 * 2))] = v81_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

