// === base name ===
kernel_7cc2a3c5b0

// === header ===
void launcher_kernel_7cc2a3c5b0(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7cc2a3c5b0(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7cc2a3c5b0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7cc2a3c5b0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32(32) {0..32} pointer_based
        // m1 32×3(32×3) {0..32}×{0..3} pointer_based
        // m2 32×3(32×3) {0..32}×{0..3} pointer_based
        // t0 32(32) {0..32} strided({0..32})[0] = m0 32(32) {0..32} pointer_based({0..32})[0]
        // t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = m1 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1]
        // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] = t0 32(32) {0..32} strided({0..32})[0]
        // t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1] += t1 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
        // m2 32×3(32×3) {0..32}×{0..3} pointer_based({0..32}×{0..3})[0, 1] = t2 32×3(32×3) {0..32}×{0..3} strided({0..32}×{0..3})[0, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[1]{};
              // r0 = +(glb_m0) + None
              // [(0, 32)] []
              int32_t v6_lead = item.get_local_id(0) % 32;
              int32_t v11_lead = v6_lead + 0_i32;
              float v17_data = glb_m0[v6_lead];
              float v18_data = r0[0];
              r0[0] = (v18_data + v17_data);
              float r1[3]{};
              // r1 = +(glb_m1) + None
              // [(0, 32), (0, 3)] []
              int32_t v29_a = v6_lead + 0;
              float v36_data = glb_m1[v6_lead];
              float v37_data = r1[0];
              r1[0] = (v37_data + v36_data);
              int32_t v44_a = v6_lead + 32;
              float v51_data = glb_m1[(v6_lead + 32)];
              float v52_data = r1[1];
              r1[1] = (v52_data + v51_data);
              int32_t v59_a = v6_lead + 64;
              float v66_data = glb_m1[(v6_lead + 64)];
              float v67_data = r1[2];
              r1[2] = (v67_data + v66_data);
              float r2[3]{};
              // r2 = +(r0) + None
              // [(0, 32), (0, 3)] []
              float v73_data = r0[0];
              float v74_data = r2[0];
              r2[0] = (v74_data + v73_data);
              float v77_data = r2[1];
              r2[1] = (v77_data + v73_data);
              float v80_data = r2[2];
              r2[2] = (v80_data + v73_data);
              float r3[3]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir3[3]{};
              float v87_data = r1[0];
              float v88_data = ir3[0];
              ir3[0] = (v88_data + v87_data);
              float v90_data = r1[1];
              float v91_data = ir3[1];
              ir3[1] = (v91_data + v90_data);
              float v93_data = r1[2];
              float v94_data = ir3[2];
              ir3[2] = (v94_data + v93_data);
              #pragma unroll
              for (int32_t v99_n0 = 0; v99_n0 < 1; ++v99_n0) {
                #pragma unroll
                for (int32_t v100_n1 = 0; v100_n1 < 3; ++v100_n1) {
                  int32_t v101_a = v99_n0 + v100_n1;
                  int32_t v102_a = v99_n0 + v100_n1;
                  float v103_data = ir3[v102_a];
                  int32_t v104_a = v99_n0 + v100_n1;
                  float v106_data = r2[v102_a];
                  r3[v102_a] = (v106_data + v103_data);
                }
              }
              float r4[3]{};
              // r4 = +(r3) + None
              // [(0, 32), (0, 3)] []
              float ir4[3]{};
              float v114_data = r3[0];
              float v115_data = ir4[0];
              ir4[0] = (v115_data + v114_data);
              float v117_data = r3[1];
              float v118_data = ir4[1];
              ir4[1] = (v118_data + v117_data);
              float v120_data = r3[2];
              float v121_data = ir4[2];
              ir4[2] = (v121_data + v120_data);
              #pragma unroll
              for (int32_t v126_n0 = 0; v126_n0 < 1; ++v126_n0) {
                #pragma unroll
                for (int32_t v127_n1 = 0; v127_n1 < 3; ++v127_n1) {
                  int32_t v128_a = v126_n0 + v127_n1;
                  int32_t v129_a = v126_n0 + v127_n1;
                  float v130_data = ir4[v129_a];
                  r4[v129_a] = v130_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v135_i0 = 0; v135_i0 < 1; ++v135_i0) {
                int32_t v144_lead = v6_lead + (v135_i0 * 32);
                #pragma unroll
                for (int32_t v136_i1 = 0; v136_i1 < 3; ++v136_i1) {
                  int32_t v137_a = v135_i0 + v136_i1;
                  float v139_data = r4[(v135_i0 + v136_i1)];
                  glb_m2[(v144_lead + (v136_i1 * 32))] = v139_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

