// === base name ===
kernel_e072362310177832

// === header ===
void launcher_kernel_e072362310177832(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e072362310177832(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e072362310177832(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e072362310177832(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1408, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
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
          float* localShrMem0 = &totalShrMem[176 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[176];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 169 + 0 + m2_extraOffset];
              float r0[96]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v17_lead = v15_i0 * 32;
                #pragma unroll
                for (int32_t v16_i1 = 10; v16_i1 < 13; ++v16_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v21_data;
                  v21_data.copy_from(glb_m1 + ((v17_lead + (v16_i1 * 32))));
                  v21_data.copy_to(r0 + ((v17_lead + ((v16_i1 - 10) * 32))));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v26_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 128));
              v27_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 128));
              if (item.get_local_id(0) < 9) {
                tensorforge::intel_esimd::simd<float, 32> v28_ld;
                v28_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 160));
                v28_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 160));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[224]{};
              // r1 = +(r0 * s0) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir1[224]{};
              tensorforge::intel_esimd::simd<float, 32> v31_data;
              v31_data.copy_from(r0 + (0));
              float v32_data = s0[88];
              tensorforge::intel_esimd::simd<float, 32> v34_data;
              v34_data.copy_from(ir1 + (0));
              (v34_data + (v31_data * v32_data)).copy_to(ir1 + (0));
              float v37_data = s0[101];
              tensorforge::intel_esimd::simd<float, 32> v39_data;
              v39_data.copy_from(ir1 + (32));
              (v39_data + (v31_data * v37_data)).copy_to(ir1 + (32));
              float v42_data = s0[114];
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(ir1 + (64));
              (v44_data + (v31_data * v42_data)).copy_to(ir1 + (64));
              float v47_data = s0[127];
              tensorforge::intel_esimd::simd<float, 32> v49_data;
              v49_data.copy_from(ir1 + (96));
              (v49_data + (v31_data * v47_data)).copy_to(ir1 + (96));
              float v52_data = s0[140];
              tensorforge::intel_esimd::simd<float, 32> v54_data;
              v54_data.copy_from(ir1 + (128));
              (v54_data + (v31_data * v52_data)).copy_to(ir1 + (128));
              float v57_data = s0[153];
              tensorforge::intel_esimd::simd<float, 32> v59_data;
              v59_data.copy_from(ir1 + (160));
              (v59_data + (v31_data * v57_data)).copy_to(ir1 + (160));
              float v62_data = s0[166];
              tensorforge::intel_esimd::simd<float, 32> v64_data;
              v64_data.copy_from(ir1 + (192));
              (v64_data + (v31_data * v62_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v66_data;
              v66_data.copy_from(r0 + (32));
              float v67_data = s0[89];
              tensorforge::intel_esimd::simd<float, 32> v69_data;
              v69_data.copy_from(ir1 + (0));
              (v69_data + (v66_data * v67_data)).copy_to(ir1 + (0));
              float v72_data = s0[102];
              tensorforge::intel_esimd::simd<float, 32> v74_data;
              v74_data.copy_from(ir1 + (32));
              (v74_data + (v66_data * v72_data)).copy_to(ir1 + (32));
              float v77_data = s0[115];
              tensorforge::intel_esimd::simd<float, 32> v79_data;
              v79_data.copy_from(ir1 + (64));
              (v79_data + (v66_data * v77_data)).copy_to(ir1 + (64));
              float v82_data = s0[128];
              tensorforge::intel_esimd::simd<float, 32> v84_data;
              v84_data.copy_from(ir1 + (96));
              (v84_data + (v66_data * v82_data)).copy_to(ir1 + (96));
              float v87_data = s0[141];
              tensorforge::intel_esimd::simd<float, 32> v89_data;
              v89_data.copy_from(ir1 + (128));
              (v89_data + (v66_data * v87_data)).copy_to(ir1 + (128));
              float v92_data = s0[154];
              tensorforge::intel_esimd::simd<float, 32> v94_data;
              v94_data.copy_from(ir1 + (160));
              (v94_data + (v66_data * v92_data)).copy_to(ir1 + (160));
              float v97_data = s0[167];
              tensorforge::intel_esimd::simd<float, 32> v99_data;
              v99_data.copy_from(ir1 + (192));
              (v99_data + (v66_data * v97_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v101_data;
              v101_data.copy_from(r0 + (64));
              float v102_data = s0[90];
              tensorforge::intel_esimd::simd<float, 32> v104_data;
              v104_data.copy_from(ir1 + (0));
              (v104_data + (v101_data * v102_data)).copy_to(ir1 + (0));
              float v107_data = s0[103];
              tensorforge::intel_esimd::simd<float, 32> v109_data;
              v109_data.copy_from(ir1 + (32));
              (v109_data + (v101_data * v107_data)).copy_to(ir1 + (32));
              float v112_data = s0[116];
              tensorforge::intel_esimd::simd<float, 32> v114_data;
              v114_data.copy_from(ir1 + (64));
              (v114_data + (v101_data * v112_data)).copy_to(ir1 + (64));
              float v117_data = s0[129];
              tensorforge::intel_esimd::simd<float, 32> v119_data;
              v119_data.copy_from(ir1 + (96));
              (v119_data + (v101_data * v117_data)).copy_to(ir1 + (96));
              float v122_data = s0[142];
              tensorforge::intel_esimd::simd<float, 32> v124_data;
              v124_data.copy_from(ir1 + (128));
              (v124_data + (v101_data * v122_data)).copy_to(ir1 + (128));
              float v127_data = s0[155];
              tensorforge::intel_esimd::simd<float, 32> v129_data;
              v129_data.copy_from(ir1 + (160));
              (v129_data + (v101_data * v127_data)).copy_to(ir1 + (160));
              float v132_data = s0[168];
              tensorforge::intel_esimd::simd<float, 32> v134_data;
              v134_data.copy_from(ir1 + (192));
              (v134_data + (v101_data * v132_data)).copy_to(ir1 + (192));
              #pragma unroll
              for (int32_t v136_n0 = 0; v136_n0 < 1; ++v136_n0) {
                int32_t v138_a = v136_n0 * 32;
                #pragma unroll
                for (int32_t v137_n1 = 6; v137_n1 < 13; ++v137_n1) {
                  int32_t v140_a = (v137_n1 - 6) * 32;
                  tensorforge::intel_esimd::simd<float, 32> v142_data;
                  v142_data.copy_from(ir1 + ((v138_a + v140_a)));
                  v142_data.copy_to(r1 + ((v138_a + v140_a)));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v147_i0 = 0; v147_i0 < 1; ++v147_i0) {
                int32_t v149_lead = v147_i0 * 32;
                glb_m0[v149_lead] = 0.0f;
                int32_t v155_a = v149_lead + 32;
                glb_m0[v155_a] = 0.0f;
                int32_t v159_a = v149_lead + 64;
                glb_m0[v159_a] = 0.0f;
                int32_t v163_a = v149_lead + 96;
                glb_m0[v163_a] = 0.0f;
                int32_t v167_a = v149_lead + 128;
                glb_m0[v167_a] = 0.0f;
                int32_t v171_a = v149_lead + 160;
                glb_m0[v171_a] = 0.0f;
                tensorforge::intel_esimd::simd<float, 32> v174_data;
                v174_data.copy_from(r1 + (v149_lead));
                int32_t v177_a = v149_lead + 192;
                v174_data.copy_to(glb_m0 + (v177_a));
                tensorforge::intel_esimd::simd<float, 32> v180_data;
                v180_data.copy_from(r1 + (v155_a));
                v180_data.copy_to(glb_m0 + ((v149_lead + 224)));
                tensorforge::intel_esimd::simd<float, 32> v186_data;
                v186_data.copy_from(r1 + (v159_a));
                v186_data.copy_to(glb_m0 + ((v149_lead + 256)));
                tensorforge::intel_esimd::simd<float, 32> v192_data;
                v192_data.copy_from(r1 + (v163_a));
                v192_data.copy_to(glb_m0 + ((v149_lead + 288)));
                tensorforge::intel_esimd::simd<float, 32> v198_data;
                v198_data.copy_from(r1 + (v167_a));
                v198_data.copy_to(glb_m0 + ((v149_lead + 320)));
                tensorforge::intel_esimd::simd<float, 32> v204_data;
                v204_data.copy_from(r1 + (v171_a));
                v204_data.copy_to(glb_m0 + ((v149_lead + 352)));
                tensorforge::intel_esimd::simd<float, 32> v210_data;
                v210_data.copy_from(r1 + (v177_a));
                v210_data.copy_to(glb_m0 + ((v149_lead + 384)));
              }
            }
          }
        }
      });
    }
  });
}

