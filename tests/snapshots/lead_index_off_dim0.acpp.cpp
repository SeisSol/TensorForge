// === base name ===
kernel_c2773117d86e95e0

// === header ===
void launcher_kernel_c2773117d86e95e0(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c2773117d86e95e0(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c2773117d86e95e0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c2773117d86e95e0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 20×9(20×9) {0..20}×{0..9} strided
        // m1 1×20(1×20) {0..1}×{0..20} strided
        // m2 1×9(1×9) {0..1}×{0..9} strided
        // m0 20×9(20×9) {0..20}×{0..9} strided({0..20}×{0..9})[0, 1] = m1 1×20(1×20) {0..1}×{0..20} strided({0..1}×{0..20})[-1, 0]×m2 1×9(1×9) {0..1}×{0..9} strided({0..1}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 20 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 9 + 0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v11_lead = item.get_local_id(0) % 32;
              bool v12_g = v11_lead < 20;
              #pragma unroll
              for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
                if (v12_g) {
                  float v19_data = glb_m1[(v8_i0 + v11_lead)];
                  r0[v8_i0] = v19_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v24_lead = item.get_local_id(0) % 32;
              if (v24_lead < 1) {
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 9; ++v26_i1) {
                  float v33_data = glb_m2[(v24_lead + v26_i1)];
                  r1[v26_i1] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 20), (0, 9)] [(0, 1)]
              float ir2[9]{};
              if (v24_lead < 20) {
                float v41_data = r0[0];
                float v42_data = r1[0];
                float v45_data = ir2[0];
                ir2[0] = (v45_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 0))));
                float v48_data = r1[1];
                float v51_data = ir2[1];
                ir2[1] = (v51_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 0))));
                float v54_data = r1[2];
                float v57_data = ir2[2];
                ir2[2] = (v57_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v54_data, 0))));
                float v60_data = r1[3];
                float v63_data = ir2[3];
                ir2[3] = (v63_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v60_data, 0))));
                float v66_data = r1[4];
                float v69_data = ir2[4];
                ir2[4] = (v69_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 0))));
                float v72_data = r1[5];
                float v75_data = ir2[5];
                ir2[5] = (v75_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 0))));
                float v78_data = r1[6];
                float v81_data = ir2[6];
                ir2[6] = (v81_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 0))));
                float v84_data = r1[7];
                float v87_data = ir2[7];
                ir2[7] = (v87_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 0))));
                float v90_data = r1[8];
                float v93_data = ir2[8];
                ir2[8] = (v93_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 0))));
              }
              if (v24_lead < 20) {
                #pragma unroll
                for (int32_t v99_n1 = 0; v99_n1 < 9; ++v99_n1) {
                  float v101_data = ir2[v99_n1];
                  r2[v99_n1] = v101_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v24_lead < 20) {
                #pragma unroll
                for (int32_t v107_i1 = 0; v107_i1 < 9; ++v107_i1) {
                  float v109_data = r2[v107_i1];
                  glb_m0[(v24_lead + (v107_i1 * 20))] = v109_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

