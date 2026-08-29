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
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v7_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
                int32_t v14_lead = v7_lead + (v8_i0 * 16);
                #pragma unroll
                for (int32_t v9_i1 = 0; v9_i1 < 16; ++v9_i1) {
                  float v17_data = glb_m1[(v14_lead + (v9_i1 * 16))];
                  r0[(v8_i0 + v9_i1)] = v17_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 16)] [(0, 16)]
              float ir1[1]{};
              float v24_data = r0[0];
              float v25_data = ir1[0];
              ir1[0] = (v25_data + v24_data);
              float v30_data = r0[1];
              float v31_data = ir1[0];
              ir1[0] = (v31_data + v30_data);
              float v36_data = r0[2];
              float v37_data = ir1[0];
              ir1[0] = (v37_data + v36_data);
              float v42_data = r0[3];
              float v43_data = ir1[0];
              ir1[0] = (v43_data + v42_data);
              float v48_data = r0[4];
              float v49_data = ir1[0];
              ir1[0] = (v49_data + v48_data);
              float v54_data = r0[5];
              float v55_data = ir1[0];
              ir1[0] = (v55_data + v54_data);
              float v60_data = r0[6];
              float v61_data = ir1[0];
              ir1[0] = (v61_data + v60_data);
              float v66_data = r0[7];
              float v67_data = ir1[0];
              ir1[0] = (v67_data + v66_data);
              float v72_data = r0[8];
              float v73_data = ir1[0];
              ir1[0] = (v73_data + v72_data);
              float v78_data = r0[9];
              float v79_data = ir1[0];
              ir1[0] = (v79_data + v78_data);
              float v84_data = r0[10];
              float v85_data = ir1[0];
              ir1[0] = (v85_data + v84_data);
              float v90_data = r0[11];
              float v91_data = ir1[0];
              ir1[0] = (v91_data + v90_data);
              float v96_data = r0[12];
              float v97_data = ir1[0];
              ir1[0] = (v97_data + v96_data);
              float v102_data = r0[13];
              float v103_data = ir1[0];
              ir1[0] = (v103_data + v102_data);
              float v108_data = r0[14];
              float v109_data = ir1[0];
              ir1[0] = (v109_data + v108_data);
              float v114_data = r0[15];
              float v115_data = ir1[0];
              ir1[0] = (v115_data + v114_data);
              #pragma unroll
              for (int32_t v120_n0 = 0; v120_n0 < 1; ++v120_n0) {
                float v121_data = ir1[v120_n0];
                r1[v120_n0] = v121_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v125_i0 = 0; v125_i0 < 1; ++v125_i0) {
                float v126_data = r1[v125_i0];
                glb_m0[(v7_lead + (v125_i0 * 16))] = v126_data;
              }
            }
          }
        }
      });
    }
  });
}

