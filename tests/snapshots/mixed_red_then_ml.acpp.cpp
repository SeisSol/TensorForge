// === base name ===
kernel_49337a255f

// === header ===
void launcher_kernel_49337a255f(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_49337a255f(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_49337a255f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_49337a255f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 8; ++v10_i1) {
                  float v18_data = glb_m2[(v8_lead + (v10_i1 * 8))];
                  r1[v10_i1] = v18_data;
                }
              }
              float r0[1]{};
              // r0 = +(glb_m0, dims=[1])
              if (v8_lead < 8) {
                float v26_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v25_r1 = 0; v25_r1 < 8; ++v25_r1) {
                  float v34_data = glb_m0[(v8_lead + (v25_r1 * 8))];
                  v26_acc0 = (v26_acc0 + v34_data);
                }
                r0[0] = v26_acc0;
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] []
              float ir2[8]{};
              if (v8_lead < 8) {
                float v43_data = r0[0];
                float v44_data = r1[0];
                float v46_data = ir2[0];
                ir2[0] = (v46_data + (v43_data * v44_data));
                float v49_data = r1[1];
                float v51_data = ir2[1];
                ir2[1] = (v51_data + (v43_data * v49_data));
                float v54_data = r1[2];
                float v56_data = ir2[2];
                ir2[2] = (v56_data + (v43_data * v54_data));
                float v59_data = r1[3];
                float v61_data = ir2[3];
                ir2[3] = (v61_data + (v43_data * v59_data));
                float v64_data = r1[4];
                float v66_data = ir2[4];
                ir2[4] = (v66_data + (v43_data * v64_data));
                float v69_data = r1[5];
                float v71_data = ir2[5];
                ir2[5] = (v71_data + (v43_data * v69_data));
                float v74_data = r1[6];
                float v76_data = ir2[6];
                ir2[6] = (v76_data + (v43_data * v74_data));
                float v79_data = r1[7];
                float v81_data = ir2[7];
                ir2[7] = (v81_data + (v43_data * v79_data));
              }
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v87_n1 = 0; v87_n1 < 8; ++v87_n1) {
                  float v89_data = ir2[v87_n1];
                  r2[v87_n1] = v89_data;
                }
              }
              // glb_m1 = store{r>g}(r2);
              if (v8_lead < 8) {
                #pragma unroll
                for (int32_t v95_i1 = 0; v95_i1 < 8; ++v95_i1) {
                  float v97_data = r2[v95_i1];
                  glb_m1[(v8_lead + (v95_i1 * 8))] = v97_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

