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
    sycl::local_accessor<float, 1> totalShrMem (1408, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
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
          float* localShrMem0 = &totalShrMem[176 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[176];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
              float r0[96]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 32;
                #pragma unroll
                for (int32_t v7_i1 = 10; v7_i1 < 13; ++v7_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v12_data;
                  v12_data.copy_from(glb_m1 + ((v8_lead + (v7_i1 * 32))));
                  v12_data.copy_to(r0 + ((v8_lead + ((v7_i1 - 10) * 32))));
                }
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v18_ld;
              v18_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v18_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 128));
              v19_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 128));
              if (item.get_local_id(0) < 9) {
                tensorforge::intel_esimd::simd<float, 32> v20_ld;
                v20_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 160));
                v20_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 160));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[224]{};
              // r1 = +(r0 * s0) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir1[224]{};
              tensorforge::intel_esimd::simd<float, 32> v23_data;
              v23_data.copy_from(r0 + (0));
              float v24_data = s0[88];
              tensorforge::intel_esimd::simd<float, 32> v26_data;
              v26_data.copy_from(ir1 + (0));
              (v26_data + (v23_data * v24_data)).copy_to(ir1 + (0));
              float v29_data = s0[101];
              tensorforge::intel_esimd::simd<float, 32> v31_data;
              v31_data.copy_from(ir1 + (32));
              (v31_data + (v23_data * v29_data)).copy_to(ir1 + (32));
              float v34_data = s0[114];
              tensorforge::intel_esimd::simd<float, 32> v36_data;
              v36_data.copy_from(ir1 + (64));
              (v36_data + (v23_data * v34_data)).copy_to(ir1 + (64));
              float v39_data = s0[127];
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(ir1 + (96));
              (v41_data + (v23_data * v39_data)).copy_to(ir1 + (96));
              float v44_data = s0[140];
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(ir1 + (128));
              (v46_data + (v23_data * v44_data)).copy_to(ir1 + (128));
              float v49_data = s0[153];
              tensorforge::intel_esimd::simd<float, 32> v51_data;
              v51_data.copy_from(ir1 + (160));
              (v51_data + (v23_data * v49_data)).copy_to(ir1 + (160));
              float v54_data = s0[166];
              tensorforge::intel_esimd::simd<float, 32> v56_data;
              v56_data.copy_from(ir1 + (192));
              (v56_data + (v23_data * v54_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(r0 + (32));
              float v59_data = s0[89];
              tensorforge::intel_esimd::simd<float, 32> v61_data;
              v61_data.copy_from(ir1 + (0));
              (v61_data + (v58_data * v59_data)).copy_to(ir1 + (0));
              float v64_data = s0[102];
              tensorforge::intel_esimd::simd<float, 32> v66_data;
              v66_data.copy_from(ir1 + (32));
              (v66_data + (v58_data * v64_data)).copy_to(ir1 + (32));
              float v69_data = s0[115];
              tensorforge::intel_esimd::simd<float, 32> v71_data;
              v71_data.copy_from(ir1 + (64));
              (v71_data + (v58_data * v69_data)).copy_to(ir1 + (64));
              float v74_data = s0[128];
              tensorforge::intel_esimd::simd<float, 32> v76_data;
              v76_data.copy_from(ir1 + (96));
              (v76_data + (v58_data * v74_data)).copy_to(ir1 + (96));
              float v79_data = s0[141];
              tensorforge::intel_esimd::simd<float, 32> v81_data;
              v81_data.copy_from(ir1 + (128));
              (v81_data + (v58_data * v79_data)).copy_to(ir1 + (128));
              float v84_data = s0[154];
              tensorforge::intel_esimd::simd<float, 32> v86_data;
              v86_data.copy_from(ir1 + (160));
              (v86_data + (v58_data * v84_data)).copy_to(ir1 + (160));
              float v89_data = s0[167];
              tensorforge::intel_esimd::simd<float, 32> v91_data;
              v91_data.copy_from(ir1 + (192));
              (v91_data + (v58_data * v89_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v93_data;
              v93_data.copy_from(r0 + (64));
              float v94_data = s0[90];
              tensorforge::intel_esimd::simd<float, 32> v96_data;
              v96_data.copy_from(ir1 + (0));
              (v96_data + (v93_data * v94_data)).copy_to(ir1 + (0));
              float v99_data = s0[103];
              tensorforge::intel_esimd::simd<float, 32> v101_data;
              v101_data.copy_from(ir1 + (32));
              (v101_data + (v93_data * v99_data)).copy_to(ir1 + (32));
              float v104_data = s0[116];
              tensorforge::intel_esimd::simd<float, 32> v106_data;
              v106_data.copy_from(ir1 + (64));
              (v106_data + (v93_data * v104_data)).copy_to(ir1 + (64));
              float v109_data = s0[129];
              tensorforge::intel_esimd::simd<float, 32> v111_data;
              v111_data.copy_from(ir1 + (96));
              (v111_data + (v93_data * v109_data)).copy_to(ir1 + (96));
              float v114_data = s0[142];
              tensorforge::intel_esimd::simd<float, 32> v116_data;
              v116_data.copy_from(ir1 + (128));
              (v116_data + (v93_data * v114_data)).copy_to(ir1 + (128));
              float v119_data = s0[155];
              tensorforge::intel_esimd::simd<float, 32> v121_data;
              v121_data.copy_from(ir1 + (160));
              (v121_data + (v93_data * v119_data)).copy_to(ir1 + (160));
              float v124_data = s0[168];
              tensorforge::intel_esimd::simd<float, 32> v126_data;
              v126_data.copy_from(ir1 + (192));
              (v126_data + (v93_data * v124_data)).copy_to(ir1 + (192));
              #pragma unroll
              for (int32_t v128_n0 = 0; v128_n0 < 1; ++v128_n0) {
                int32_t v130_a = v128_n0 * 32;
                #pragma unroll
                for (int32_t v129_n1 = 6; v129_n1 < 13; ++v129_n1) {
                  int32_t v132_a = (v129_n1 - 6) * 32;
                  tensorforge::intel_esimd::simd<float, 32> v134_data;
                  v134_data.copy_from(ir1 + ((v130_a + v132_a)));
                  v134_data.copy_to(r1 + ((v130_a + v132_a)));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v139_i0 = 0; v139_i0 < 1; ++v139_i0) {
                int32_t v141_lead = v139_i0 * 32;
                glb_m0[v141_lead] = 0.0f;
                int32_t v147_a = v141_lead + 32;
                glb_m0[v147_a] = 0.0f;
                int32_t v151_a = v141_lead + 64;
                glb_m0[v151_a] = 0.0f;
                int32_t v155_a = v141_lead + 96;
                glb_m0[v155_a] = 0.0f;
                int32_t v159_a = v141_lead + 128;
                glb_m0[v159_a] = 0.0f;
                int32_t v163_a = v141_lead + 160;
                glb_m0[v163_a] = 0.0f;
                tensorforge::intel_esimd::simd<float, 32> v166_data;
                v166_data.copy_from(r1 + (v141_lead));
                int32_t v169_a = v141_lead + 192;
                v166_data.copy_to(glb_m0 + (v169_a));
                tensorforge::intel_esimd::simd<float, 32> v172_data;
                v172_data.copy_from(r1 + (v147_a));
                v172_data.copy_to(glb_m0 + ((v141_lead + 224)));
                tensorforge::intel_esimd::simd<float, 32> v178_data;
                v178_data.copy_from(r1 + (v151_a));
                v178_data.copy_to(glb_m0 + ((v141_lead + 256)));
                tensorforge::intel_esimd::simd<float, 32> v184_data;
                v184_data.copy_from(r1 + (v155_a));
                v184_data.copy_to(glb_m0 + ((v141_lead + 288)));
                tensorforge::intel_esimd::simd<float, 32> v190_data;
                v190_data.copy_from(r1 + (v159_a));
                v190_data.copy_to(glb_m0 + ((v141_lead + 320)));
                tensorforge::intel_esimd::simd<float, 32> v196_data;
                v196_data.copy_from(r1 + (v163_a));
                v196_data.copy_to(glb_m0 + ((v141_lead + 352)));
                tensorforge::intel_esimd::simd<float, 32> v202_data;
                v202_data.copy_from(r1 + (v169_a));
                v202_data.copy_to(glb_m0 + ((v141_lead + 384)));
              }
            }
          }
        }
      });
    }
  });
}

