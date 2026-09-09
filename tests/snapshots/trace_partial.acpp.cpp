// === base name ===
kernel_cf367ac9c5395360

// === header ===
void launcher_kernel_cf367ac9c5395360(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_cf367ac9c5395360(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_cf367ac9c5395360(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_cf367ac9c5395360(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v11_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v12_i0 = 0; v12_i0 < 1; ++v12_i0) {
                int32_t v18_lead = v11_lead + (v12_i0 * 16);
                #pragma unroll
                for (int32_t v13_i1 = 0; v13_i1 < 16; ++v13_i1) {
                  float v21_data = glb_m1[(v18_lead + (v13_i1 * 16))];
                  r0[(v12_i0 + v13_i1)] = v21_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 16)] [(0, 16)]
              float ir1[1]{};
              float v28_data = r0[0];
              float v29_data = ir1[0];
              ir1[0] = (v29_data + v28_data);
              float v34_data = r0[1];
              float v35_data = ir1[0];
              ir1[0] = (v35_data + v34_data);
              float v40_data = r0[2];
              float v41_data = ir1[0];
              ir1[0] = (v41_data + v40_data);
              float v46_data = r0[3];
              float v47_data = ir1[0];
              ir1[0] = (v47_data + v46_data);
              float v52_data = r0[4];
              float v53_data = ir1[0];
              ir1[0] = (v53_data + v52_data);
              float v58_data = r0[5];
              float v59_data = ir1[0];
              ir1[0] = (v59_data + v58_data);
              float v64_data = r0[6];
              float v65_data = ir1[0];
              ir1[0] = (v65_data + v64_data);
              float v70_data = r0[7];
              float v71_data = ir1[0];
              ir1[0] = (v71_data + v70_data);
              float v76_data = r0[8];
              float v77_data = ir1[0];
              ir1[0] = (v77_data + v76_data);
              float v82_data = r0[9];
              float v83_data = ir1[0];
              ir1[0] = (v83_data + v82_data);
              float v88_data = r0[10];
              float v89_data = ir1[0];
              ir1[0] = (v89_data + v88_data);
              float v94_data = r0[11];
              float v95_data = ir1[0];
              ir1[0] = (v95_data + v94_data);
              float v100_data = r0[12];
              float v101_data = ir1[0];
              ir1[0] = (v101_data + v100_data);
              float v106_data = r0[13];
              float v107_data = ir1[0];
              ir1[0] = (v107_data + v106_data);
              float v112_data = r0[14];
              float v113_data = ir1[0];
              ir1[0] = (v113_data + v112_data);
              float v118_data = r0[15];
              float v119_data = ir1[0];
              ir1[0] = (v119_data + v118_data);
              #pragma unroll
              for (int32_t v124_n0 = 0; v124_n0 < 1; ++v124_n0) {
                float v125_data = ir1[v124_n0];
                r1[v124_n0] = v125_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v129_i0 = 0; v129_i0 < 1; ++v129_i0) {
                float v130_data = r1[v129_i0];
                glb_m0[(v11_lead + (v129_i0 * 16))] = v130_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

