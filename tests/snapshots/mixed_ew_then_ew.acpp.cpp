// === base name ===
kernel_dd7cc6859794a98a

// === header ===
void launcher_kernel_dd7cc6859794a98a(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_dd7cc6859794a98a(const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_dd7cc6859794a98a(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_dd7cc6859794a98a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // TMP = abs(A)
        // C = neg(TMP)
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 64 + 0 + m1_extraOffset];
              float r0[8]{};
              // r0 = abs(glb_m0)
              int32_t v16_lead = item.get_local_id(0) % 16;
              if (v16_lead < 8) {
                #pragma unroll
                for (int32_t v18_k1 = 0; v18_k1 < 8; ++v18_k1) {
                  float v26_data = glb_m0[(v16_lead + (v18_k1 * 8))];
                  r0[v18_k1] = (sycl::fabs(v26_data));
                }
              }
              // s0 = store{r>s}(localShrMem0, r0);
              if (v16_lead < 8) {
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                  float v35_data = r0[v33_i1];
                  int32_t v42_a = v16_lead + (v33_i1 * 8);
                  s0[(v42_a ^ ((v42_a >> 5) & 31))] = v35_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m1 = neg(s0)
              if (v16_lead < 8) {
                #pragma unroll
                for (int32_t v50_k1 = 0; v50_k1 < 8; ++v50_k1) {
                  int32_t v56_a = v50_k1 * 8;
                  int32_t v57_a = v16_lead + v56_a;
                  float v61_data = s0[(v57_a ^ ((v57_a >> 5) & 31))];
                  glb_m1[(v16_lead + v56_a)] = ((-v61_data));
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

