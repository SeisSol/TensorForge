// === base name ===
kernel_75d3097b00

// === header ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_75d3097b00(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_75d3097b00(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_75d3097b00(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 20×9(20×9) {0..20}×{0..9} strided
        // m1 1×20(1×20) {0..1}×{0..20} strided
        // m2 1×9(1×9) {0..1}×{0..9} strided
        // m0 20×9(20×9) {0..20}×{0..9} strided({0..20}×{0..9})[0, 1] = m1 1×20(1×20) {0..1}×{0..20} strided({0..1}×{0..20})[-1, 0]×m2 1×9(1×9) {0..1}×{0..9} strided({0..1}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 20 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 9 + 0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v7_lead = item.get_local_id(0) % 32;
              bool v8_g = v7_lead < 20;
              #pragma unroll
              for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
                if (v8_g) {
                  float v15_data = glb_m1[(v4_i0 + v7_lead)];
                  r0[v4_i0] = v15_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              float v18_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v18_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 20), (0, 9)] [(0, 1)]
              float ir2[9]{};
              int32_t v23_lead = item.get_local_id(0) % 32;
              if (v23_lead < 20) {
                float v25_data = r0[0];
                float v26_data = r1[0];
                float v29_data = ir2[0];
                ir2[0] = (v29_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v26_data, 0))));
                float v32_data = r1[1];
                float v35_data = ir2[1];
                ir2[1] = (v35_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v32_data, 0))));
                float v38_data = r1[2];
                float v41_data = ir2[2];
                ir2[2] = (v41_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v38_data, 0))));
                float v44_data = r1[3];
                float v47_data = ir2[3];
                ir2[3] = (v47_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v44_data, 0))));
                float v50_data = r1[4];
                float v53_data = ir2[4];
                ir2[4] = (v53_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 0))));
                float v56_data = r1[5];
                float v59_data = ir2[5];
                ir2[5] = (v59_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v56_data, 0))));
                float v62_data = r1[6];
                float v65_data = ir2[6];
                ir2[6] = (v65_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 0))));
                float v68_data = r1[7];
                float v71_data = ir2[7];
                ir2[7] = (v71_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 0))));
                float v74_data = r1[8];
                float v77_data = ir2[8];
                ir2[8] = (v77_data + (v25_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 0))));
              }
              if (v23_lead < 20) {
                #pragma unroll
                for (int32_t v83_n1 = 0; v83_n1 < 9; ++v83_n1) {
                  float v85_data = ir2[v83_n1];
                  r2[v83_n1] = v85_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v23_lead < 20) {
                #pragma unroll
                for (int32_t v91_i1 = 0; v91_i1 < 9; ++v91_i1) {
                  float v93_data = r2[v91_i1];
                  glb_m0[(v23_lead + (v91_i1 * 20))] = v93_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

