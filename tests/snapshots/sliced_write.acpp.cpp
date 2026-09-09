// === base name ===
kernel_59e23527beb7a2b7

// === header ===
void launcher_kernel_59e23527beb7a2b7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_59e23527beb7a2b7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_59e23527beb7a2b7(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_59e23527beb7a2b7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×13(32×13) {0..32}×{0..13} strided
        // m1 32×13(32×13) {0..32}×{0..13} strided
        // m2 13×13(13×13) {0..13}×{0..13} strided
        // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{6..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{6..13})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v10_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v17_lead = v10_lead + (v11_i0 * 32);
                #pragma unroll
                for (int32_t v12_i1 = 10; v12_i1 < 13; ++v12_i1) {
                  float v20_data = glb_m1[(v17_lead + (v12_i1 * 32))];
                  r0[(v11_i0 + (v12_i1 - 10))] = v20_data;
                }
              }
              float r1[7]{};
              // r1 = load{g>r}(glb_m2);
              if ((v10_lead >= 10) && (v10_lead < 13)) {
                #pragma unroll
                for (int32_t v30_i1 = 6; v30_i1 < 13; ++v30_i1) {
                  float v38_data = glb_m2[(v10_lead + (v30_i1 * 13))];
                  r1[(v30_i1 - 6)] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[7]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir2[7]{};
              float v46_data = r0[0];
              float v47_data = r1[0];
              float v50_data = ir2[0];
              ir2[0] = (v50_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 10))));
              float v53_data = r1[1];
              float v56_data = ir2[1];
              ir2[1] = (v56_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 10))));
              float v59_data = r1[2];
              float v62_data = ir2[2];
              ir2[2] = (v62_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 10))));
              float v65_data = r1[3];
              float v68_data = ir2[3];
              ir2[3] = (v68_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 10))));
              float v71_data = r1[4];
              float v74_data = ir2[4];
              ir2[4] = (v74_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 10))));
              float v77_data = r1[5];
              float v80_data = ir2[5];
              ir2[5] = (v80_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 10))));
              float v83_data = r1[6];
              float v86_data = ir2[6];
              ir2[6] = (v86_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 10))));
              float v91_data = r0[1];
              float v95_data = ir2[0];
              ir2[0] = (v95_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 11))));
              float v101_data = ir2[1];
              ir2[1] = (v101_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 11))));
              float v107_data = ir2[2];
              ir2[2] = (v107_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 11))));
              float v113_data = ir2[3];
              ir2[3] = (v113_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 11))));
              float v119_data = ir2[4];
              ir2[4] = (v119_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 11))));
              float v125_data = ir2[5];
              ir2[5] = (v125_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 11))));
              float v131_data = ir2[6];
              ir2[6] = (v131_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 11))));
              float v136_data = r0[2];
              float v140_data = ir2[0];
              ir2[0] = (v140_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 12))));
              float v146_data = ir2[1];
              ir2[1] = (v146_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 12))));
              float v152_data = ir2[2];
              ir2[2] = (v152_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 12))));
              float v158_data = ir2[3];
              ir2[3] = (v158_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 12))));
              float v164_data = ir2[4];
              ir2[4] = (v164_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 12))));
              float v170_data = ir2[5];
              ir2[5] = (v170_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 12))));
              float v176_data = ir2[6];
              ir2[6] = (v176_data + (v136_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 12))));
              #pragma unroll
              for (int32_t v181_n0 = 0; v181_n0 < 1; ++v181_n0) {
                #pragma unroll
                for (int32_t v182_n1 = 6; v182_n1 < 13; ++v182_n1) {
                  int32_t v184_a = v181_n0 + (v182_n1 - 6);
                  float v185_data = ir2[v184_a];
                  r2[v184_a] = v185_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v191_i0 = 0; v191_i0 < 1; ++v191_i0) {
                int32_t v196_lead = v191_i0 * 32;
                glb_m0[(v10_lead + v196_lead)] = 0.0f;
                glb_m0[((v10_lead + v196_lead) + 32)] = 0.0f;
                glb_m0[((v10_lead + v196_lead) + 64)] = 0.0f;
                glb_m0[((v10_lead + v196_lead) + 96)] = 0.0f;
                glb_m0[((v10_lead + v196_lead) + 128)] = 0.0f;
                glb_m0[((v10_lead + v196_lead) + 160)] = 0.0f;
                float v235_data = r2[v191_i0];
                glb_m0[((v10_lead + v196_lead) + 192)] = v235_data;
                float v243_data = r2[(v191_i0 + 1)];
                glb_m0[((v10_lead + v196_lead) + 224)] = v243_data;
                float v251_data = r2[(v191_i0 + 2)];
                glb_m0[((v10_lead + v196_lead) + 256)] = v251_data;
                float v259_data = r2[(v191_i0 + 3)];
                glb_m0[((v10_lead + v196_lead) + 288)] = v259_data;
                float v267_data = r2[(v191_i0 + 4)];
                glb_m0[((v10_lead + v196_lead) + 320)] = v267_data;
                float v275_data = r2[(v191_i0 + 5)];
                glb_m0[((v10_lead + v196_lead) + 352)] = v275_data;
                float v283_data = r2[(v191_i0 + 6)];
                glb_m0[((v10_lead + v196_lead) + 384)] = v283_data;
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

