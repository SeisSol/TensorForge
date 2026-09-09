// === base name ===
kernel_5ff8d918db2683af

// === header ===
void launcher_kernel_5ff8d918db2683af(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5ff8d918db2683af(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5ff8d918db2683af(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5ff8d918db2683af(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v10_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                float v17_data = glb_m0[(v10_lead + (v11_i0 * 32))];
                r0[v11_i0] = v17_data;
              }
              float r2[3]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v28_lead = v10_lead + (v22_i0 * 32);
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 3; ++v23_i1) {
                  float v31_data = glb_m1[(v28_lead + (v23_i1 * 32))];
                  r2[(v22_i0 + v23_i1)] = v31_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 32)] []
              float v37_data = r0[0];
              float v38_data = r1[0];
              r1[0] = (v38_data + v37_data);
              // wait(r2 = load{g>r}(glb_m1););
              float r3[3]{};
              // r3 = +(r2) + None
              // [(0, 32), (0, 3)] []
              float v44_data = r2[0];
              float v45_data = r3[0];
              r3[0] = (v45_data + v44_data);
              float v47_data = r2[1];
              float v48_data = r3[1];
              r3[1] = (v48_data + v47_data);
              float v50_data = r2[2];
              float v51_data = r3[2];
              r3[2] = (v51_data + v50_data);
              float r4[3]{};
              // r4 = +(r1) + None
              // [(0, 32), (0, 3)] []
              float v57_data = r1[0];
              float v58_data = r4[0];
              r4[0] = (v58_data + v57_data);
              float v61_data = r4[1];
              r4[1] = (v61_data + v57_data);
              float v64_data = r4[2];
              r4[2] = (v64_data + v57_data);
              float r5[3]{};
              // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir5[3]{};
              float v71_data = r3[0];
              float v72_data = ir5[0];
              ir5[0] = (v72_data + v71_data);
              float v74_data = r3[1];
              float v75_data = ir5[1];
              ir5[1] = (v75_data + v74_data);
              float v77_data = r3[2];
              float v78_data = ir5[2];
              ir5[2] = (v78_data + v77_data);
              #pragma unroll
              for (int32_t v83_n0 = 0; v83_n0 < 1; ++v83_n0) {
                #pragma unroll
                for (int32_t v84_n1 = 0; v84_n1 < 3; ++v84_n1) {
                  int32_t v85_a = v83_n0 + v84_n1;
                  float v86_data = ir5[v85_a];
                  float v88_data = r4[v85_a];
                  r5[v85_a] = (v88_data + v86_data);
                }
              }
              float r6[3]{};
              // r6 = +(r5) + None
              // [(0, 32), (0, 3)] []
              float ir6[3]{};
              float v96_data = r5[0];
              float v97_data = ir6[0];
              ir6[0] = (v97_data + v96_data);
              float v99_data = r5[1];
              float v100_data = ir6[1];
              ir6[1] = (v100_data + v99_data);
              float v102_data = r5[2];
              float v103_data = ir6[2];
              ir6[2] = (v103_data + v102_data);
              #pragma unroll
              for (int32_t v108_n0 = 0; v108_n0 < 1; ++v108_n0) {
                #pragma unroll
                for (int32_t v109_n1 = 0; v109_n1 < 3; ++v109_n1) {
                  int32_t v110_a = v108_n0 + v109_n1;
                  float v111_data = ir6[v110_a];
                  r6[v110_a] = v111_data;
                }
              }
              // glb_m2 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v116_i0 = 0; v116_i0 < 1; ++v116_i0) {
                int32_t v124_lead = v10_lead + (v116_i0 * 32);
                #pragma unroll
                for (int32_t v117_i1 = 0; v117_i1 < 3; ++v117_i1) {
                  float v119_data = r6[(v116_i0 + v117_i1)];
                  glb_m2[(v124_lead + (v117_i1 * 32))] = v119_data;
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

