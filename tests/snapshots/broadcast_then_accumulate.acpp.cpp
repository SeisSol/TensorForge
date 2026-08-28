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
              float v12_data = glb_m0[v6_lead];
              float v13_data = r0[0];
              r0[0] = (v13_data + v12_data);
              float r1[3]{};
              // r1 = +(glb_m1) + None
              // [(0, 32), (0, 3)] []
              float v25_data = glb_m1[v6_lead];
              float v26_data = r1[0];
              r1[0] = (v26_data + v25_data);
              float v34_data = glb_m1[(v6_lead + 32)];
              float v35_data = r1[1];
              r1[1] = (v35_data + v34_data);
              float v43_data = glb_m1[(v6_lead + 64)];
              float v44_data = r1[2];
              r1[2] = (v44_data + v43_data);
              float r2[3]{};
              // r2 = +(r0) + None
              // [(0, 32), (0, 3)] []
              float v50_data = r0[0];
              float v51_data = r2[0];
              r2[0] = (v51_data + v50_data);
              float v54_data = r2[1];
              r2[1] = (v54_data + v50_data);
              float v57_data = r2[2];
              r2[2] = (v57_data + v50_data);
              float r3[3]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir3[3]{};
              float v64_data = r1[0];
              float v65_data = ir3[0];
              ir3[0] = (v65_data + v64_data);
              float v67_data = r1[1];
              float v68_data = ir3[1];
              ir3[1] = (v68_data + v67_data);
              float v70_data = r1[2];
              float v71_data = ir3[2];
              ir3[2] = (v71_data + v70_data);
              #pragma unroll
              for (int32_t v76_n0 = 0; v76_n0 < 1; ++v76_n0) {
                #pragma unroll
                for (int32_t v77_n1 = 0; v77_n1 < 3; ++v77_n1) {
                  int32_t v78_a = v76_n0 + v77_n1;
                  float v79_data = ir3[v78_a];
                  float v81_data = r2[v78_a];
                  r3[v78_a] = (v81_data + v79_data);
                }
              }
              float r4[3]{};
              // r4 = +(r3) + None
              // [(0, 32), (0, 3)] []
              float ir4[3]{};
              float v89_data = r3[0];
              float v90_data = ir4[0];
              ir4[0] = (v90_data + v89_data);
              float v92_data = r3[1];
              float v93_data = ir4[1];
              ir4[1] = (v93_data + v92_data);
              float v95_data = r3[2];
              float v96_data = ir4[2];
              ir4[2] = (v96_data + v95_data);
              #pragma unroll
              for (int32_t v101_n0 = 0; v101_n0 < 1; ++v101_n0) {
                #pragma unroll
                for (int32_t v102_n1 = 0; v102_n1 < 3; ++v102_n1) {
                  int32_t v103_a = v101_n0 + v102_n1;
                  float v104_data = ir4[v103_a];
                  r4[v103_a] = v104_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v109_i0 = 0; v109_i0 < 1; ++v109_i0) {
                int32_t v117_lead = v6_lead + (v109_i0 * 32);
                #pragma unroll
                for (int32_t v110_i1 = 0; v110_i1 < 3; ++v110_i1) {
                  float v112_data = r4[(v109_i0 + v110_i1)];
                  glb_m2[(v117_lead + (v110_i1 * 32))] = v112_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

