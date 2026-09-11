// === base name ===
kernel_5b3e72737de51b9a

// === header ===
void launcher_kernel_5b3e72737de51b9a(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5b3e72737de51b9a(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5b3e72737de51b9a(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5b3e72737de51b9a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 64 + 0 + m2_extraOffset];
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v16_lead = item.get_local_id(0) % 16;
              if (v16_lead < 8) {
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 8; ++v18_i1) {
                  float v26_data = glb_m2[(v16_lead + (v18_i1 * 8))];
                  r1[v18_i1] = v26_data;
                }
              }
              float r0[1]{};
              // r0 = +(glb_m0, dims=[1])
              if (v16_lead < 8) {
                float v34_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v33_r1 = 0; v33_r1 < 8; ++v33_r1) {
                  float v42_data = glb_m0[(v16_lead + (v33_r1 * 8))];
                  v34_acc0 = (v34_acc0 + v42_data);
                }
                r0[0] = v34_acc0;
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] []
              float ir2[8]{};
              if (v16_lead < 8) {
                float v51_data = r0[0];
                float v52_data = r1[0];
                float v54_data = ir2[0];
                ir2[0] = (v54_data + (v51_data * v52_data));
                float v57_data = r1[1];
                float v59_data = ir2[1];
                ir2[1] = (v59_data + (v51_data * v57_data));
                float v62_data = r1[2];
                float v64_data = ir2[2];
                ir2[2] = (v64_data + (v51_data * v62_data));
                float v67_data = r1[3];
                float v69_data = ir2[3];
                ir2[3] = (v69_data + (v51_data * v67_data));
                float v72_data = r1[4];
                float v74_data = ir2[4];
                ir2[4] = (v74_data + (v51_data * v72_data));
                float v77_data = r1[5];
                float v79_data = ir2[5];
                ir2[5] = (v79_data + (v51_data * v77_data));
                float v82_data = r1[6];
                float v84_data = ir2[6];
                ir2[6] = (v84_data + (v51_data * v82_data));
                float v87_data = r1[7];
                float v89_data = ir2[7];
                ir2[7] = (v89_data + (v51_data * v87_data));
              }
              if (v16_lead < 8) {
                #pragma unroll
                for (int32_t v95_n1 = 0; v95_n1 < 8; ++v95_n1) {
                  float v97_data = ir2[v95_n1];
                  r2[v95_n1] = v97_data;
                }
              }
              // glb_m1 = store{r>g}(r2);
              if (v16_lead < 8) {
                #pragma unroll
                for (int32_t v103_i1 = 0; v103_i1 < 8; ++v103_i1) {
                  float v105_data = r2[v103_i1];
                  glb_m1[(v16_lead + (v103_i1 * 8))] = v105_data;
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

