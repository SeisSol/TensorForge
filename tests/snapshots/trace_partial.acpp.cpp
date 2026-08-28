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
              float r0[1]{};
              // r0 = +(glb_m1) + None
              // [(0, 16)] [(0, 16)]
              float ir0[1]{};
              int32_t v6_lead = item.get_local_id(0) % 16;
              float v13_data = glb_m1[v6_lead];
              float v14_data = ir0[0];
              ir0[0] = (v14_data + v13_data);
              float v25_data = glb_m1[(v6_lead + 16)];
              float v26_data = ir0[0];
              ir0[0] = (v26_data + v25_data);
              float v37_data = glb_m1[(v6_lead + 32)];
              float v38_data = ir0[0];
              ir0[0] = (v38_data + v37_data);
              float v49_data = glb_m1[(v6_lead + 48)];
              float v50_data = ir0[0];
              ir0[0] = (v50_data + v49_data);
              float v61_data = glb_m1[(v6_lead + 64)];
              float v62_data = ir0[0];
              ir0[0] = (v62_data + v61_data);
              float v73_data = glb_m1[(v6_lead + 80)];
              float v74_data = ir0[0];
              ir0[0] = (v74_data + v73_data);
              float v85_data = glb_m1[(v6_lead + 96)];
              float v86_data = ir0[0];
              ir0[0] = (v86_data + v85_data);
              float v97_data = glb_m1[(v6_lead + 112)];
              float v98_data = ir0[0];
              ir0[0] = (v98_data + v97_data);
              float v109_data = glb_m1[(v6_lead + 128)];
              float v110_data = ir0[0];
              ir0[0] = (v110_data + v109_data);
              float v121_data = glb_m1[(v6_lead + 144)];
              float v122_data = ir0[0];
              ir0[0] = (v122_data + v121_data);
              float v133_data = glb_m1[(v6_lead + 160)];
              float v134_data = ir0[0];
              ir0[0] = (v134_data + v133_data);
              float v145_data = glb_m1[(v6_lead + 176)];
              float v146_data = ir0[0];
              ir0[0] = (v146_data + v145_data);
              float v157_data = glb_m1[(v6_lead + 192)];
              float v158_data = ir0[0];
              ir0[0] = (v158_data + v157_data);
              float v169_data = glb_m1[(v6_lead + 208)];
              float v170_data = ir0[0];
              ir0[0] = (v170_data + v169_data);
              float v181_data = glb_m1[(v6_lead + 224)];
              float v182_data = ir0[0];
              ir0[0] = (v182_data + v181_data);
              float v193_data = glb_m1[(v6_lead + 240)];
              float v194_data = ir0[0];
              ir0[0] = (v194_data + v193_data);
              #pragma unroll
              for (int32_t v199_n0 = 0; v199_n0 < 1; ++v199_n0) {
                float v200_data = ir0[v199_n0];
                r0[v199_n0] = v200_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v204_i0 = 0; v204_i0 < 1; ++v204_i0) {
                float v205_data = r0[v204_i0];
                glb_m0[(v6_lead + (v204_i0 * 16))] = v205_data;
              }
            }
          }
        }
      });
    }
  });
}

