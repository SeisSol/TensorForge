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
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              float r0[1]{};
              // r0 = +(glb_m1) + None
              // [(0, 16)] [(0, 16)]
              float ir0[1]{};
              int32_t v6_lead = item.get_local_id(0) % 16;
              int32_t v12_a = v6_lead + 0;
              float v19_data = glb_m1[v6_lead];
              float v20_data = ir0[0];
              ir0[0] = (v20_data + v19_data);
              int32_t v30_a = v6_lead + 16;
              float v37_data = glb_m1[(v6_lead + 16)];
              float v38_data = ir0[0];
              ir0[0] = (v38_data + v37_data);
              int32_t v48_a = v6_lead + 32;
              float v55_data = glb_m1[(v6_lead + 32)];
              float v56_data = ir0[0];
              ir0[0] = (v56_data + v55_data);
              int32_t v66_a = v6_lead + 48;
              float v73_data = glb_m1[(v6_lead + 48)];
              float v74_data = ir0[0];
              ir0[0] = (v74_data + v73_data);
              int32_t v84_a = v6_lead + 64;
              float v91_data = glb_m1[(v6_lead + 64)];
              float v92_data = ir0[0];
              ir0[0] = (v92_data + v91_data);
              int32_t v102_a = v6_lead + 80;
              float v109_data = glb_m1[(v6_lead + 80)];
              float v110_data = ir0[0];
              ir0[0] = (v110_data + v109_data);
              int32_t v120_a = v6_lead + 96;
              float v127_data = glb_m1[(v6_lead + 96)];
              float v128_data = ir0[0];
              ir0[0] = (v128_data + v127_data);
              int32_t v138_a = v6_lead + 112;
              float v145_data = glb_m1[(v6_lead + 112)];
              float v146_data = ir0[0];
              ir0[0] = (v146_data + v145_data);
              int32_t v156_a = v6_lead + 128;
              float v163_data = glb_m1[(v6_lead + 128)];
              float v164_data = ir0[0];
              ir0[0] = (v164_data + v163_data);
              int32_t v174_a = v6_lead + 144;
              float v181_data = glb_m1[(v6_lead + 144)];
              float v182_data = ir0[0];
              ir0[0] = (v182_data + v181_data);
              int32_t v192_a = v6_lead + 160;
              float v199_data = glb_m1[(v6_lead + 160)];
              float v200_data = ir0[0];
              ir0[0] = (v200_data + v199_data);
              int32_t v210_a = v6_lead + 176;
              float v217_data = glb_m1[(v6_lead + 176)];
              float v218_data = ir0[0];
              ir0[0] = (v218_data + v217_data);
              int32_t v228_a = v6_lead + 192;
              float v235_data = glb_m1[(v6_lead + 192)];
              float v236_data = ir0[0];
              ir0[0] = (v236_data + v235_data);
              int32_t v246_a = v6_lead + 208;
              float v253_data = glb_m1[(v6_lead + 208)];
              float v254_data = ir0[0];
              ir0[0] = (v254_data + v253_data);
              int32_t v264_a = v6_lead + 224;
              float v271_data = glb_m1[(v6_lead + 224)];
              float v272_data = ir0[0];
              ir0[0] = (v272_data + v271_data);
              int32_t v282_a = v6_lead + 240;
              float v289_data = glb_m1[(v6_lead + 240)];
              float v290_data = ir0[0];
              ir0[0] = (v290_data + v289_data);
              #pragma unroll
              for (int32_t v295_n0 = 0; v295_n0 < 1; ++v295_n0) {
                float v296_data = ir0[v295_n0];
                r0[v295_n0] = v296_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v300_i0 = 0; v300_i0 < 1; ++v300_i0) {
                float v301_data = r0[v300_i0];
                glb_m0[(v6_lead + (v300_i0 * 16))] = v301_data;
              }
            }
          }
        }
      });
    }
  });
}

