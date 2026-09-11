// === base name ===
kernel_f25924e6afbb7947

// === header ===
void launcher_kernel_f25924e6afbb7947(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f25924e6afbb7947(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f25924e6afbb7947(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f25924e6afbb7947(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 20 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 9 + 0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v15_lead = item.get_local_id(0) % 32;
              bool v16_g = v15_lead < 20;
              #pragma unroll
              for (int32_t v12_i0 = 0; v12_i0 < 1; ++v12_i0) {
                if (v16_g) {
                  float v23_data = glb_m1[(v12_i0 + v15_lead)];
                  r0[v12_i0] = v23_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v28_lead = item.get_local_id(0) % 32;
              if (v28_lead < 1) {
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 9; ++v30_i1) {
                  float v37_data = glb_m2[(v28_lead + v30_i1)];
                  r1[v30_i1] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 20), (0, 9)] [(0, 1)]
              float ir2[9]{};
              if (v28_lead < 20) {
                float v45_data = r0[0];
                float v46_data = r1[0];
                float v49_data = ir2[0];
                ir2[0] = (v49_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 0))));
                float v52_data = r1[1];
                float v55_data = ir2[1];
                ir2[1] = (v55_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v52_data, 0))));
                float v58_data = r1[2];
                float v61_data = ir2[2];
                ir2[2] = (v61_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[3];
                float v67_data = ir2[3];
                ir2[3] = (v67_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[4];
                float v73_data = ir2[4];
                ir2[4] = (v73_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[5];
                float v79_data = ir2[5];
                ir2[5] = (v79_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[6];
                float v85_data = ir2[6];
                ir2[6] = (v85_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
                float v88_data = r1[7];
                float v91_data = ir2[7];
                ir2[7] = (v91_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
                float v94_data = r1[8];
                float v97_data = ir2[8];
                ir2[8] = (v97_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 0))));
              }
              if (v28_lead < 20) {
                #pragma unroll
                for (int32_t v103_n1 = 0; v103_n1 < 9; ++v103_n1) {
                  float v105_data = ir2[v103_n1];
                  r2[v103_n1] = v105_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v28_lead < 20) {
                #pragma unroll
                for (int32_t v111_i1 = 0; v111_i1 < 9; ++v111_i1) {
                  float v113_data = r2[v111_i1];
                  glb_m0[(v28_lead + (v111_i1 * 20))] = v113_data;
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

