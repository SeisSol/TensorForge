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
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 128] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 128];
              if (item.get_local_id(0) < 9) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 160] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 160];
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[224]{};
              // r1 = +(r0 * s0) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir1[224]{};
              tensorforge::intel_esimd::simd<float, 32> v20_data;
              v20_data.copy_from(r0 + (0));
              float v21_data = s0[88];
              tensorforge::intel_esimd::simd<float, 32> v23_data;
              v23_data.copy_from(ir1 + (0));
              (v23_data + (v20_data * v21_data)).copy_to(ir1 + (0));
              float v26_data = s0[101];
              tensorforge::intel_esimd::simd<float, 32> v28_data;
              v28_data.copy_from(ir1 + (32));
              (v28_data + (v20_data * v26_data)).copy_to(ir1 + (32));
              float v31_data = s0[114];
              tensorforge::intel_esimd::simd<float, 32> v33_data;
              v33_data.copy_from(ir1 + (64));
              (v33_data + (v20_data * v31_data)).copy_to(ir1 + (64));
              float v36_data = s0[127];
              tensorforge::intel_esimd::simd<float, 32> v38_data;
              v38_data.copy_from(ir1 + (96));
              (v38_data + (v20_data * v36_data)).copy_to(ir1 + (96));
              float v41_data = s0[140];
              tensorforge::intel_esimd::simd<float, 32> v43_data;
              v43_data.copy_from(ir1 + (128));
              (v43_data + (v20_data * v41_data)).copy_to(ir1 + (128));
              float v46_data = s0[153];
              tensorforge::intel_esimd::simd<float, 32> v48_data;
              v48_data.copy_from(ir1 + (160));
              (v48_data + (v20_data * v46_data)).copy_to(ir1 + (160));
              float v51_data = s0[166];
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(ir1 + (192));
              (v53_data + (v20_data * v51_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v55_data;
              v55_data.copy_from(r0 + (32));
              float v56_data = s0[89];
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(ir1 + (0));
              (v58_data + (v55_data * v56_data)).copy_to(ir1 + (0));
              float v61_data = s0[102];
              tensorforge::intel_esimd::simd<float, 32> v63_data;
              v63_data.copy_from(ir1 + (32));
              (v63_data + (v55_data * v61_data)).copy_to(ir1 + (32));
              float v66_data = s0[115];
              tensorforge::intel_esimd::simd<float, 32> v68_data;
              v68_data.copy_from(ir1 + (64));
              (v68_data + (v55_data * v66_data)).copy_to(ir1 + (64));
              float v71_data = s0[128];
              tensorforge::intel_esimd::simd<float, 32> v73_data;
              v73_data.copy_from(ir1 + (96));
              (v73_data + (v55_data * v71_data)).copy_to(ir1 + (96));
              float v76_data = s0[141];
              tensorforge::intel_esimd::simd<float, 32> v78_data;
              v78_data.copy_from(ir1 + (128));
              (v78_data + (v55_data * v76_data)).copy_to(ir1 + (128));
              float v81_data = s0[154];
              tensorforge::intel_esimd::simd<float, 32> v83_data;
              v83_data.copy_from(ir1 + (160));
              (v83_data + (v55_data * v81_data)).copy_to(ir1 + (160));
              float v86_data = s0[167];
              tensorforge::intel_esimd::simd<float, 32> v88_data;
              v88_data.copy_from(ir1 + (192));
              (v88_data + (v55_data * v86_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v90_data;
              v90_data.copy_from(r0 + (64));
              float v91_data = s0[90];
              tensorforge::intel_esimd::simd<float, 32> v93_data;
              v93_data.copy_from(ir1 + (0));
              (v93_data + (v90_data * v91_data)).copy_to(ir1 + (0));
              float v96_data = s0[103];
              tensorforge::intel_esimd::simd<float, 32> v98_data;
              v98_data.copy_from(ir1 + (32));
              (v98_data + (v90_data * v96_data)).copy_to(ir1 + (32));
              float v101_data = s0[116];
              tensorforge::intel_esimd::simd<float, 32> v103_data;
              v103_data.copy_from(ir1 + (64));
              (v103_data + (v90_data * v101_data)).copy_to(ir1 + (64));
              float v106_data = s0[129];
              tensorforge::intel_esimd::simd<float, 32> v108_data;
              v108_data.copy_from(ir1 + (96));
              (v108_data + (v90_data * v106_data)).copy_to(ir1 + (96));
              float v111_data = s0[142];
              tensorforge::intel_esimd::simd<float, 32> v113_data;
              v113_data.copy_from(ir1 + (128));
              (v113_data + (v90_data * v111_data)).copy_to(ir1 + (128));
              float v116_data = s0[155];
              tensorforge::intel_esimd::simd<float, 32> v118_data;
              v118_data.copy_from(ir1 + (160));
              (v118_data + (v90_data * v116_data)).copy_to(ir1 + (160));
              float v121_data = s0[168];
              tensorforge::intel_esimd::simd<float, 32> v123_data;
              v123_data.copy_from(ir1 + (192));
              (v123_data + (v90_data * v121_data)).copy_to(ir1 + (192));
              #pragma unroll
              for (int32_t v125_n0 = 0; v125_n0 < 1; ++v125_n0) {
                int32_t v127_a = v125_n0 * 32;
                #pragma unroll
                for (int32_t v126_n1 = 6; v126_n1 < 13; ++v126_n1) {
                  int32_t v129_a = (v126_n1 - 6) * 32;
                  tensorforge::intel_esimd::simd<float, 32> v131_data;
                  v131_data.copy_from(ir1 + ((v127_a + v129_a)));
                  v131_data.copy_to(r1 + ((v127_a + v129_a)));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v136_i0 = 0; v136_i0 < 1; ++v136_i0) {
                int32_t v138_lead = v136_i0 * 32;
                glb_m0[v138_lead] = 0.0f;
                int32_t v144_a = v138_lead + 32;
                glb_m0[v144_a] = 0.0f;
                int32_t v148_a = v138_lead + 64;
                glb_m0[v148_a] = 0.0f;
                int32_t v152_a = v138_lead + 96;
                glb_m0[v152_a] = 0.0f;
                int32_t v156_a = v138_lead + 128;
                glb_m0[v156_a] = 0.0f;
                int32_t v160_a = v138_lead + 160;
                glb_m0[v160_a] = 0.0f;
                tensorforge::intel_esimd::simd<float, 32> v163_data;
                v163_data.copy_from(r1 + (v138_lead));
                int32_t v166_a = v138_lead + 192;
                v163_data.copy_to(glb_m0 + (v166_a));
                tensorforge::intel_esimd::simd<float, 32> v169_data;
                v169_data.copy_from(r1 + (v144_a));
                v169_data.copy_to(glb_m0 + ((v138_lead + 224)));
                tensorforge::intel_esimd::simd<float, 32> v175_data;
                v175_data.copy_from(r1 + (v148_a));
                v175_data.copy_to(glb_m0 + ((v138_lead + 256)));
                tensorforge::intel_esimd::simd<float, 32> v181_data;
                v181_data.copy_from(r1 + (v152_a));
                v181_data.copy_to(glb_m0 + ((v138_lead + 288)));
                tensorforge::intel_esimd::simd<float, 32> v187_data;
                v187_data.copy_from(r1 + (v156_a));
                v187_data.copy_to(glb_m0 + ((v138_lead + 320)));
                tensorforge::intel_esimd::simd<float, 32> v193_data;
                v193_data.copy_from(r1 + (v160_a));
                v193_data.copy_to(glb_m0 + ((v138_lead + 352)));
                tensorforge::intel_esimd::simd<float, 32> v199_data;
                v199_data.copy_from(r1 + (v166_a));
                v199_data.copy_to(glb_m0 + ((v138_lead + 384)));
              }
            }
          }
        }
      });
    }
  });
}

