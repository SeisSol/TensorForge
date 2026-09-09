// === base name ===
kernel_657249b99def28fc

// === header ===
void launcher_kernel_657249b99def28fc(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_657249b99def28fc(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_657249b99def28fc(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_657249b99def28fc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // TMP = +(A, dims=[1])
        // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8(8) {0..8} pointer_based({0..8})[0]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v12_lead = item.get_local_id(0) % 16;
              if (v12_lead < 8) {
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 8; ++v14_i1) {
                  float v22_data = glb_m2[(v12_lead + (v14_i1 * 8))];
                  r1[v14_i1] = v22_data;
                }
              }
              float r0[1]{};
              // r0 = +(glb_m0, dims=[1])
              if (v12_lead < 8) {
                float v30_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v29_r1 = 0; v29_r1 < 8; ++v29_r1) {
                  float v38_data = glb_m0[(v12_lead + (v29_r1 * 8))];
                  v30_acc0 = (v30_acc0 + v38_data);
                }
                r0[0] = v30_acc0;
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] []
              float ir2[8]{};
              if (v12_lead < 8) {
                float v47_data = r0[0];
                float v48_data = r1[0];
                float v50_data = ir2[0];
                ir2[0] = (v50_data + (v47_data * v48_data));
                float v53_data = r1[1];
                float v55_data = ir2[1];
                ir2[1] = (v55_data + (v47_data * v53_data));
                float v58_data = r1[2];
                float v60_data = ir2[2];
                ir2[2] = (v60_data + (v47_data * v58_data));
                float v63_data = r1[3];
                float v65_data = ir2[3];
                ir2[3] = (v65_data + (v47_data * v63_data));
                float v68_data = r1[4];
                float v70_data = ir2[4];
                ir2[4] = (v70_data + (v47_data * v68_data));
                float v73_data = r1[5];
                float v75_data = ir2[5];
                ir2[5] = (v75_data + (v47_data * v73_data));
                float v78_data = r1[6];
                float v80_data = ir2[6];
                ir2[6] = (v80_data + (v47_data * v78_data));
                float v83_data = r1[7];
                float v85_data = ir2[7];
                ir2[7] = (v85_data + (v47_data * v83_data));
              }
              if (v12_lead < 8) {
                #pragma unroll
                for (int32_t v91_n1 = 0; v91_n1 < 8; ++v91_n1) {
                  float v93_data = ir2[v91_n1];
                  r2[v91_n1] = v93_data;
                }
              }
              // glb_m1 = store{r>g}(r2);
              if (v12_lead < 8) {
                #pragma unroll
                for (int32_t v99_i1 = 0; v99_i1 < 8; ++v99_i1) {
                  float v101_data = r2[v99_i1];
                  glb_m1[(v12_lead + (v99_i1 * 8))] = v101_data;
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

