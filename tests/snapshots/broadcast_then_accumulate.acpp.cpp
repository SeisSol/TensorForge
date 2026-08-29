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
              // r0 = load{g>r}(glb_m0);
              int32_t v6_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
                float v13_data = glb_m0[(v6_lead + (v7_i0 * 32))];
                r0[v7_i0] = v13_data;
              }
              float r2[3]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v24_lead = v6_lead + (v18_i0 * 32);
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 3; ++v19_i1) {
                  float v27_data = glb_m1[(v24_lead + (v19_i1 * 32))];
                  r2[(v18_i0 + v19_i1)] = v27_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 32)] []
              float v33_data = r0[0];
              float v34_data = r1[0];
              r1[0] = (v34_data + v33_data);
              // wait(r2 = load{g>r}(glb_m1););
              float r3[3]{};
              // r3 = +(r2) + None
              // [(0, 32), (0, 3)] []
              float v40_data = r2[0];
              float v41_data = r3[0];
              r3[0] = (v41_data + v40_data);
              float v43_data = r2[1];
              float v44_data = r3[1];
              r3[1] = (v44_data + v43_data);
              float v46_data = r2[2];
              float v47_data = r3[2];
              r3[2] = (v47_data + v46_data);
              float r4[3]{};
              // r4 = +(r1) + None
              // [(0, 32), (0, 3)] []
              float v53_data = r1[0];
              float v54_data = r4[0];
              r4[0] = (v54_data + v53_data);
              float v57_data = r4[1];
              r4[1] = (v57_data + v53_data);
              float v60_data = r4[2];
              r4[2] = (v60_data + v53_data);
              float r5[3]{};
              // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir5[3]{};
              float v67_data = r3[0];
              float v68_data = ir5[0];
              ir5[0] = (v68_data + v67_data);
              float v70_data = r3[1];
              float v71_data = ir5[1];
              ir5[1] = (v71_data + v70_data);
              float v73_data = r3[2];
              float v74_data = ir5[2];
              ir5[2] = (v74_data + v73_data);
              #pragma unroll
              for (int32_t v79_n0 = 0; v79_n0 < 1; ++v79_n0) {
                #pragma unroll
                for (int32_t v80_n1 = 0; v80_n1 < 3; ++v80_n1) {
                  int32_t v81_a = v79_n0 + v80_n1;
                  float v82_data = ir5[v81_a];
                  float v84_data = r4[v81_a];
                  r5[v81_a] = (v84_data + v82_data);
                }
              }
              float r6[3]{};
              // r6 = +(r5) + None
              // [(0, 32), (0, 3)] []
              float ir6[3]{};
              float v92_data = r5[0];
              float v93_data = ir6[0];
              ir6[0] = (v93_data + v92_data);
              float v95_data = r5[1];
              float v96_data = ir6[1];
              ir6[1] = (v96_data + v95_data);
              float v98_data = r5[2];
              float v99_data = ir6[2];
              ir6[2] = (v99_data + v98_data);
              #pragma unroll
              for (int32_t v104_n0 = 0; v104_n0 < 1; ++v104_n0) {
                #pragma unroll
                for (int32_t v105_n1 = 0; v105_n1 < 3; ++v105_n1) {
                  int32_t v106_a = v104_n0 + v105_n1;
                  float v107_data = ir6[v106_a];
                  r6[v106_a] = v107_data;
                }
              }
              // glb_m2 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v112_i0 = 0; v112_i0 < 1; ++v112_i0) {
                int32_t v120_lead = v6_lead + (v112_i0 * 32);
                #pragma unroll
                for (int32_t v113_i1 = 0; v113_i1 < 3; ++v113_i1) {
                  float v115_data = r6[(v112_i0 + v113_i1)];
                  glb_m2[(v120_lead + (v113_i1 * 32))] = v115_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

