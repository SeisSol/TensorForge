// === base name ===
kernel_5bfbf577fdc960dd

// === header ===
void launcher_kernel_5bfbf577fdc960dd(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5bfbf577fdc960dd(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_5bfbf577fdc960dd(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5bfbf577fdc960dd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16(16) {0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 256 + 0 + m1_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v15_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
                int32_t v22_lead = v15_lead + (v16_i0 * 16);
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 16; ++v17_i1) {
                  float v25_data = glb_m1[(v22_lead + (v17_i1 * 16))];
                  r0[(v16_i0 + v17_i1)] = v25_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 16)] [(0, 16)]
              float ir1[1]{};
              float v32_data = r0[0];
              float v33_data = ir1[0];
              ir1[0] = (v33_data + v32_data);
              float v38_data = r0[1];
              float v39_data = ir1[0];
              ir1[0] = (v39_data + v38_data);
              float v44_data = r0[2];
              float v45_data = ir1[0];
              ir1[0] = (v45_data + v44_data);
              float v50_data = r0[3];
              float v51_data = ir1[0];
              ir1[0] = (v51_data + v50_data);
              float v56_data = r0[4];
              float v57_data = ir1[0];
              ir1[0] = (v57_data + v56_data);
              float v62_data = r0[5];
              float v63_data = ir1[0];
              ir1[0] = (v63_data + v62_data);
              float v68_data = r0[6];
              float v69_data = ir1[0];
              ir1[0] = (v69_data + v68_data);
              float v74_data = r0[7];
              float v75_data = ir1[0];
              ir1[0] = (v75_data + v74_data);
              float v80_data = r0[8];
              float v81_data = ir1[0];
              ir1[0] = (v81_data + v80_data);
              float v86_data = r0[9];
              float v87_data = ir1[0];
              ir1[0] = (v87_data + v86_data);
              float v92_data = r0[10];
              float v93_data = ir1[0];
              ir1[0] = (v93_data + v92_data);
              float v98_data = r0[11];
              float v99_data = ir1[0];
              ir1[0] = (v99_data + v98_data);
              float v104_data = r0[12];
              float v105_data = ir1[0];
              ir1[0] = (v105_data + v104_data);
              float v110_data = r0[13];
              float v111_data = ir1[0];
              ir1[0] = (v111_data + v110_data);
              float v116_data = r0[14];
              float v117_data = ir1[0];
              ir1[0] = (v117_data + v116_data);
              float v122_data = r0[15];
              float v123_data = ir1[0];
              ir1[0] = (v123_data + v122_data);
              #pragma unroll
              for (int32_t v128_n0 = 0; v128_n0 < 1; ++v128_n0) {
                float v129_data = ir1[v128_n0];
                r1[v128_n0] = v129_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v133_i0 = 0; v133_i0 < 1; ++v133_i0) {
                float v134_data = r1[v133_i0];
                glb_m0[(v15_lead + (v133_i0 * 16))] = v134_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

