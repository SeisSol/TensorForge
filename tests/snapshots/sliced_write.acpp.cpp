// === base name ===
kernel_446bbff08a5289d0

// === header ===
void launcher_kernel_446bbff08a5289d0(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_446bbff08a5289d0(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_446bbff08a5289d0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_446bbff08a5289d0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 169 + 0 + m2_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v14_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v21_lead = v14_lead + (v15_i0 * 32);
                #pragma unroll
                for (int32_t v16_i1 = 10; v16_i1 < 13; ++v16_i1) {
                  float v24_data = glb_m1[(v21_lead + (v16_i1 * 32))];
                  r0[(v15_i0 + (v16_i1 - 10))] = v24_data;
                }
              }
              float r1[7]{};
              // r1 = load{g>r}(glb_m2);
              if ((v14_lead >= 10) && (v14_lead < 13)) {
                #pragma unroll
                for (int32_t v34_i1 = 6; v34_i1 < 13; ++v34_i1) {
                  float v42_data = glb_m2[(v14_lead + (v34_i1 * 13))];
                  r1[(v34_i1 - 6)] = v42_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[7]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir2[7]{};
              float v50_data = r0[0];
              float v51_data = r1[0];
              float v54_data = ir2[0];
              ir2[0] = (v54_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 10))));
              float v57_data = r1[1];
              float v60_data = ir2[1];
              ir2[1] = (v60_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 10))));
              float v63_data = r1[2];
              float v66_data = ir2[2];
              ir2[2] = (v66_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 10))));
              float v69_data = r1[3];
              float v72_data = ir2[3];
              ir2[3] = (v72_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 10))));
              float v75_data = r1[4];
              float v78_data = ir2[4];
              ir2[4] = (v78_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 10))));
              float v81_data = r1[5];
              float v84_data = ir2[5];
              ir2[5] = (v84_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 10))));
              float v87_data = r1[6];
              float v90_data = ir2[6];
              ir2[6] = (v90_data + (v50_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 10))));
              float v95_data = r0[1];
              float v99_data = ir2[0];
              ir2[0] = (v99_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 11))));
              float v105_data = ir2[1];
              ir2[1] = (v105_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 11))));
              float v111_data = ir2[2];
              ir2[2] = (v111_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 11))));
              float v117_data = ir2[3];
              ir2[3] = (v117_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 11))));
              float v123_data = ir2[4];
              ir2[4] = (v123_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 11))));
              float v129_data = ir2[5];
              ir2[5] = (v129_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 11))));
              float v135_data = ir2[6];
              ir2[6] = (v135_data + (v95_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 11))));
              float v140_data = r0[2];
              float v144_data = ir2[0];
              ir2[0] = (v144_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v51_data, 12))));
              float v150_data = ir2[1];
              ir2[1] = (v150_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v57_data, 12))));
              float v156_data = ir2[2];
              ir2[2] = (v156_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v63_data, 12))));
              float v162_data = ir2[3];
              ir2[3] = (v162_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v69_data, 12))));
              float v168_data = ir2[4];
              ir2[4] = (v168_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v75_data, 12))));
              float v174_data = ir2[5];
              ir2[5] = (v174_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v81_data, 12))));
              float v180_data = ir2[6];
              ir2[6] = (v180_data + (v140_data * (sycl::group_broadcast(item.get_sub_group(), v87_data, 12))));
              #pragma unroll
              for (int32_t v185_n0 = 0; v185_n0 < 1; ++v185_n0) {
                #pragma unroll
                for (int32_t v186_n1 = 6; v186_n1 < 13; ++v186_n1) {
                  int32_t v188_a = v185_n0 + (v186_n1 - 6);
                  float v189_data = ir2[v188_a];
                  r2[v188_a] = v189_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v195_i0 = 0; v195_i0 < 1; ++v195_i0) {
                int32_t v200_lead = v195_i0 * 32;
                glb_m0[(v14_lead + v200_lead)] = 0.0f;
                glb_m0[((v14_lead + v200_lead) + 32)] = 0.0f;
                glb_m0[((v14_lead + v200_lead) + 64)] = 0.0f;
                glb_m0[((v14_lead + v200_lead) + 96)] = 0.0f;
                glb_m0[((v14_lead + v200_lead) + 128)] = 0.0f;
                glb_m0[((v14_lead + v200_lead) + 160)] = 0.0f;
                float v239_data = r2[v195_i0];
                glb_m0[((v14_lead + v200_lead) + 192)] = v239_data;
                float v247_data = r2[(v195_i0 + 1)];
                glb_m0[((v14_lead + v200_lead) + 224)] = v247_data;
                float v255_data = r2[(v195_i0 + 2)];
                glb_m0[((v14_lead + v200_lead) + 256)] = v255_data;
                float v263_data = r2[(v195_i0 + 3)];
                glb_m0[((v14_lead + v200_lead) + 288)] = v263_data;
                float v271_data = r2[(v195_i0 + 4)];
                glb_m0[((v14_lead + v200_lead) + 320)] = v271_data;
                float v279_data = r2[(v195_i0 + 5)];
                glb_m0[((v14_lead + v200_lead) + 352)] = v279_data;
                float v287_data = r2[(v195_i0 + 6)];
                glb_m0[((v14_lead + v200_lead) + 384)] = v287_data;
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

