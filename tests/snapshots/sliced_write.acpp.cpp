// === base name ===
kernel_49acf988a6

// === header ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_49acf988a6(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_49acf988a6(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_49acf988a6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×13(32×13) {0..32}×{0..13} strided
        // m1 32×13(32×13) {0..32}×{0..13} strided
        // m2 13×13(13×13) {0..13}×{0..13} strided
        // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{6..13})[0, 1] = m1 32×13(32×13) {0..32}×{0..13} strided({0..32}×{10..13})[0, -1]×m2 13×13(13×13) {0..13}×{0..13} strided({10..13}×{6..13})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v6_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v7_i0 = 0; v7_i0 < 1; ++v7_i0) {
                int32_t v13_lead = v6_lead + (v7_i0 * 32);
                #pragma unroll
                for (int32_t v8_i1 = 10; v8_i1 < 13; ++v8_i1) {
                  float v16_data = glb_m1[(v13_lead + (v8_i1 * 32))];
                  r0[(v7_i0 + (v8_i1 - 10))] = v16_data;
                }
              }
              float r1[13]{};
              // r1 = load{g>r}(glb_m2);
              float v20_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v20_lin;
              float v21_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[1] = v21_lin;
              float v22_lin = glb_m2[64 + item.get_local_id(0) * 1];
              r1[2] = v22_lin;
              float v23_lin = glb_m2[96 + item.get_local_id(0) * 1];
              r1[3] = v23_lin;
              float v24_lin = glb_m2[128 + item.get_local_id(0) * 1];
              r1[4] = v24_lin;
              float v25_lin = glb_m2[160 + item.get_local_id(0) * 1];
              r1[5] = v25_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[7]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir2[7]{};
              float v31_data = r0[0];
              float v32_data = r1[6];
              float v35_data = ir2[0];
              ir2[0] = (v35_data + (v31_data * (sycl::group_broadcast(item.get_sub_group(), v32_data, 10))));
              float v38_data = r1[7];
              float v41_data = ir2[1];
              ir2[1] = (v41_data + (v31_data * (sycl::group_broadcast(item.get_sub_group(), v38_data, 10))));
              float v44_data = r1[8];
              float v47_data = ir2[2];
              ir2[2] = (v47_data + (v31_data * (sycl::group_broadcast(item.get_sub_group(), v44_data, 10))));
              float v50_data = r1[9];
              float v53_data = ir2[3];
              ir2[3] = (v53_data + (v31_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 10))));
              float v56_data = r1[10];
              float v59_data = ir2[4];
              ir2[4] = (v59_data + (v31_data * (sycl::group_broadcast(item.get_sub_group(), v56_data, 10))));
              float v62_data = r1[11];
              float v65_data = ir2[5];
              ir2[5] = (v65_data + (v31_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 10))));
              float v68_data = r1[12];
              float v71_data = ir2[6];
              ir2[6] = (v71_data + (v31_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 10))));
              float v76_data = r0[1];
              float v80_data = ir2[0];
              ir2[0] = (v80_data + (v76_data * (sycl::group_broadcast(item.get_sub_group(), v32_data, 11))));
              float v86_data = ir2[1];
              ir2[1] = (v86_data + (v76_data * (sycl::group_broadcast(item.get_sub_group(), v38_data, 11))));
              float v92_data = ir2[2];
              ir2[2] = (v92_data + (v76_data * (sycl::group_broadcast(item.get_sub_group(), v44_data, 11))));
              float v98_data = ir2[3];
              ir2[3] = (v98_data + (v76_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 11))));
              float v104_data = ir2[4];
              ir2[4] = (v104_data + (v76_data * (sycl::group_broadcast(item.get_sub_group(), v56_data, 11))));
              float v110_data = ir2[5];
              ir2[5] = (v110_data + (v76_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 11))));
              float v116_data = ir2[6];
              ir2[6] = (v116_data + (v76_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 11))));
              float v121_data = r0[2];
              float v125_data = ir2[0];
              ir2[0] = (v125_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v32_data, 12))));
              float v131_data = ir2[1];
              ir2[1] = (v131_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v38_data, 12))));
              float v137_data = ir2[2];
              ir2[2] = (v137_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v44_data, 12))));
              float v143_data = ir2[3];
              ir2[3] = (v143_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 12))));
              float v149_data = ir2[4];
              ir2[4] = (v149_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v56_data, 12))));
              float v155_data = ir2[5];
              ir2[5] = (v155_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 12))));
              float v161_data = ir2[6];
              ir2[6] = (v161_data + (v121_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 12))));
              #pragma unroll
              for (int32_t v166_n0 = 0; v166_n0 < 1; ++v166_n0) {
                #pragma unroll
                for (int32_t v167_n1 = 6; v167_n1 < 13; ++v167_n1) {
                  int32_t v169_a = v166_n0 + (v167_n1 - 6);
                  float v170_data = ir2[v169_a];
                  r2[v169_a] = v170_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v176_i0 = 0; v176_i0 < 1; ++v176_i0) {
                int32_t v181_lead = v176_i0 * 32;
                glb_m0[(v6_lead + v181_lead)] = 0.0f;
                glb_m0[((v6_lead + v181_lead) + 32)] = 0.0f;
                glb_m0[((v6_lead + v181_lead) + 64)] = 0.0f;
                glb_m0[((v6_lead + v181_lead) + 96)] = 0.0f;
                glb_m0[((v6_lead + v181_lead) + 128)] = 0.0f;
                glb_m0[((v6_lead + v181_lead) + 160)] = 0.0f;
                float v220_data = r2[v176_i0];
                glb_m0[((v6_lead + v181_lead) + 192)] = v220_data;
                float v228_data = r2[(v176_i0 + 1)];
                glb_m0[((v6_lead + v181_lead) + 224)] = v228_data;
                float v236_data = r2[(v176_i0 + 2)];
                glb_m0[((v6_lead + v181_lead) + 256)] = v236_data;
                float v244_data = r2[(v176_i0 + 3)];
                glb_m0[((v6_lead + v181_lead) + 288)] = v244_data;
                float v252_data = r2[(v176_i0 + 4)];
                glb_m0[((v6_lead + v181_lead) + 320)] = v252_data;
                float v260_data = r2[(v176_i0 + 5)];
                glb_m0[((v6_lead + v181_lead) + 352)] = v260_data;
                float v268_data = r2[(v176_i0 + 6)];
                glb_m0[((v6_lead + v181_lead) + 384)] = v268_data;
              }
            }
          }
        }
      });
    }
  });
}

