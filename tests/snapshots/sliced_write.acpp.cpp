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
              float r1[7]{};
              // r1 = load{g>r}(glb_m2);
              if ((v6_lead >= 10) && (v6_lead < 13)) {
                #pragma unroll
                for (int32_t v26_i1 = 6; v26_i1 < 13; ++v26_i1) {
                  float v34_data = glb_m2[(v6_lead + (v26_i1 * 13))];
                  r1[(v26_i1 - 6)] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[7]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir2[7]{};
              float v42_data = r0[0];
              float v43_data = r1[0];
              float v46_data = ir2[0];
              ir2[0] = (v46_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 10))));
              float v49_data = r1[1];
              float v52_data = ir2[1];
              ir2[1] = (v52_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 10))));
              float v55_data = r1[2];
              float v58_data = ir2[2];
              ir2[2] = (v58_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 10))));
              float v61_data = r1[3];
              float v64_data = ir2[3];
              ir2[3] = (v64_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 10))));
              float v67_data = r1[4];
              float v70_data = ir2[4];
              ir2[4] = (v70_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 10))));
              float v73_data = r1[5];
              float v76_data = ir2[5];
              ir2[5] = (v76_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 10))));
              float v79_data = r1[6];
              float v82_data = ir2[6];
              ir2[6] = (v82_data + (v42_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 10))));
              float v87_data = r0[1];
              float v91_data = ir2[0];
              ir2[0] = (v91_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 11))));
              float v97_data = ir2[1];
              ir2[1] = (v97_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 11))));
              float v103_data = ir2[2];
              ir2[2] = (v103_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 11))));
              float v109_data = ir2[3];
              ir2[3] = (v109_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 11))));
              float v115_data = ir2[4];
              ir2[4] = (v115_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 11))));
              float v121_data = ir2[5];
              ir2[5] = (v121_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 11))));
              float v127_data = ir2[6];
              ir2[6] = (v127_data + (v87_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 11))));
              float v132_data = r0[2];
              float v136_data = ir2[0];
              ir2[0] = (v136_data + (v132_data * (sycl::group_broadcast(item.get_sub_group(), v43_data, 12))));
              float v142_data = ir2[1];
              ir2[1] = (v142_data + (v132_data * (sycl::group_broadcast(item.get_sub_group(), v49_data, 12))));
              float v148_data = ir2[2];
              ir2[2] = (v148_data + (v132_data * (sycl::group_broadcast(item.get_sub_group(), v55_data, 12))));
              float v154_data = ir2[3];
              ir2[3] = (v154_data + (v132_data * (sycl::group_broadcast(item.get_sub_group(), v61_data, 12))));
              float v160_data = ir2[4];
              ir2[4] = (v160_data + (v132_data * (sycl::group_broadcast(item.get_sub_group(), v67_data, 12))));
              float v166_data = ir2[5];
              ir2[5] = (v166_data + (v132_data * (sycl::group_broadcast(item.get_sub_group(), v73_data, 12))));
              float v172_data = ir2[6];
              ir2[6] = (v172_data + (v132_data * (sycl::group_broadcast(item.get_sub_group(), v79_data, 12))));
              #pragma unroll
              for (int32_t v177_n0 = 0; v177_n0 < 1; ++v177_n0) {
                #pragma unroll
                for (int32_t v178_n1 = 6; v178_n1 < 13; ++v178_n1) {
                  int32_t v180_a = v177_n0 + (v178_n1 - 6);
                  float v181_data = ir2[v180_a];
                  r2[v180_a] = v181_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v187_i0 = 0; v187_i0 < 1; ++v187_i0) {
                int32_t v192_lead = v187_i0 * 32;
                glb_m0[(v6_lead + v192_lead)] = 0.0f;
                glb_m0[((v6_lead + v192_lead) + 32)] = 0.0f;
                glb_m0[((v6_lead + v192_lead) + 64)] = 0.0f;
                glb_m0[((v6_lead + v192_lead) + 96)] = 0.0f;
                glb_m0[((v6_lead + v192_lead) + 128)] = 0.0f;
                glb_m0[((v6_lead + v192_lead) + 160)] = 0.0f;
                float v231_data = r2[v187_i0];
                glb_m0[((v6_lead + v192_lead) + 192)] = v231_data;
                float v239_data = r2[(v187_i0 + 1)];
                glb_m0[((v6_lead + v192_lead) + 224)] = v239_data;
                float v247_data = r2[(v187_i0 + 2)];
                glb_m0[((v6_lead + v192_lead) + 256)] = v247_data;
                float v255_data = r2[(v187_i0 + 3)];
                glb_m0[((v6_lead + v192_lead) + 288)] = v255_data;
                float v263_data = r2[(v187_i0 + 4)];
                glb_m0[((v6_lead + v192_lead) + 320)] = v263_data;
                float v271_data = r2[(v187_i0 + 5)];
                glb_m0[((v6_lead + v192_lead) + 352)] = v271_data;
                float v279_data = r2[(v187_i0 + 6)];
                glb_m0[((v6_lead + v192_lead) + 384)] = v279_data;
              }
            }
          }
        }
      });
    }
  });
}

