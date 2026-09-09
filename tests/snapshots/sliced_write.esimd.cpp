// === base name ===
kernel_4827d931e4a754fa

// === header ===
void launcher_kernel_4827d931e4a754fa(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4827d931e4a754fa(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4827d931e4a754fa(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4827d931e4a754fa(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1408, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
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
          float* localShrMem0 = &totalShrMem[176 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[176];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 169 + 0 + m2_extraOffset];
              float r0[96]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v13_lead = v11_i0 * 32;
                #pragma unroll
                for (int32_t v12_i1 = 10; v12_i1 < 13; ++v12_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v17_data;
                  v17_data.copy_from(glb_m1 + ((v13_lead + (v12_i1 * 32))));
                  v17_data.copy_to(r0 + ((v13_lead + ((v12_i1 - 10) * 32))));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 128> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v22_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v23_ld;
              v23_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 128));
              v23_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 128));
              if (item.get_local_id(0) < 9) {
                tensorforge::intel_esimd::simd<float, 32> v24_ld;
                v24_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 160));
                v24_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 160));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[224]{};
              // r1 = +(r0 * s0) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir1[224]{};
              tensorforge::intel_esimd::simd<float, 32> v27_data;
              v27_data.copy_from(r0 + (0));
              float v28_data = s0[88];
              tensorforge::intel_esimd::simd<float, 32> v30_data;
              v30_data.copy_from(ir1 + (0));
              (v30_data + (v27_data * v28_data)).copy_to(ir1 + (0));
              float v33_data = s0[101];
              tensorforge::intel_esimd::simd<float, 32> v35_data;
              v35_data.copy_from(ir1 + (32));
              (v35_data + (v27_data * v33_data)).copy_to(ir1 + (32));
              float v38_data = s0[114];
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(ir1 + (64));
              (v40_data + (v27_data * v38_data)).copy_to(ir1 + (64));
              float v43_data = s0[127];
              tensorforge::intel_esimd::simd<float, 32> v45_data;
              v45_data.copy_from(ir1 + (96));
              (v45_data + (v27_data * v43_data)).copy_to(ir1 + (96));
              float v48_data = s0[140];
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(ir1 + (128));
              (v50_data + (v27_data * v48_data)).copy_to(ir1 + (128));
              float v53_data = s0[153];
              tensorforge::intel_esimd::simd<float, 32> v55_data;
              v55_data.copy_from(ir1 + (160));
              (v55_data + (v27_data * v53_data)).copy_to(ir1 + (160));
              float v58_data = s0[166];
              tensorforge::intel_esimd::simd<float, 32> v60_data;
              v60_data.copy_from(ir1 + (192));
              (v60_data + (v27_data * v58_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v62_data;
              v62_data.copy_from(r0 + (32));
              float v63_data = s0[89];
              tensorforge::intel_esimd::simd<float, 32> v65_data;
              v65_data.copy_from(ir1 + (0));
              (v65_data + (v62_data * v63_data)).copy_to(ir1 + (0));
              float v68_data = s0[102];
              tensorforge::intel_esimd::simd<float, 32> v70_data;
              v70_data.copy_from(ir1 + (32));
              (v70_data + (v62_data * v68_data)).copy_to(ir1 + (32));
              float v73_data = s0[115];
              tensorforge::intel_esimd::simd<float, 32> v75_data;
              v75_data.copy_from(ir1 + (64));
              (v75_data + (v62_data * v73_data)).copy_to(ir1 + (64));
              float v78_data = s0[128];
              tensorforge::intel_esimd::simd<float, 32> v80_data;
              v80_data.copy_from(ir1 + (96));
              (v80_data + (v62_data * v78_data)).copy_to(ir1 + (96));
              float v83_data = s0[141];
              tensorforge::intel_esimd::simd<float, 32> v85_data;
              v85_data.copy_from(ir1 + (128));
              (v85_data + (v62_data * v83_data)).copy_to(ir1 + (128));
              float v88_data = s0[154];
              tensorforge::intel_esimd::simd<float, 32> v90_data;
              v90_data.copy_from(ir1 + (160));
              (v90_data + (v62_data * v88_data)).copy_to(ir1 + (160));
              float v93_data = s0[167];
              tensorforge::intel_esimd::simd<float, 32> v95_data;
              v95_data.copy_from(ir1 + (192));
              (v95_data + (v62_data * v93_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v97_data;
              v97_data.copy_from(r0 + (64));
              float v98_data = s0[90];
              tensorforge::intel_esimd::simd<float, 32> v100_data;
              v100_data.copy_from(ir1 + (0));
              (v100_data + (v97_data * v98_data)).copy_to(ir1 + (0));
              float v103_data = s0[103];
              tensorforge::intel_esimd::simd<float, 32> v105_data;
              v105_data.copy_from(ir1 + (32));
              (v105_data + (v97_data * v103_data)).copy_to(ir1 + (32));
              float v108_data = s0[116];
              tensorforge::intel_esimd::simd<float, 32> v110_data;
              v110_data.copy_from(ir1 + (64));
              (v110_data + (v97_data * v108_data)).copy_to(ir1 + (64));
              float v113_data = s0[129];
              tensorforge::intel_esimd::simd<float, 32> v115_data;
              v115_data.copy_from(ir1 + (96));
              (v115_data + (v97_data * v113_data)).copy_to(ir1 + (96));
              float v118_data = s0[142];
              tensorforge::intel_esimd::simd<float, 32> v120_data;
              v120_data.copy_from(ir1 + (128));
              (v120_data + (v97_data * v118_data)).copy_to(ir1 + (128));
              float v123_data = s0[155];
              tensorforge::intel_esimd::simd<float, 32> v125_data;
              v125_data.copy_from(ir1 + (160));
              (v125_data + (v97_data * v123_data)).copy_to(ir1 + (160));
              float v128_data = s0[168];
              tensorforge::intel_esimd::simd<float, 32> v130_data;
              v130_data.copy_from(ir1 + (192));
              (v130_data + (v97_data * v128_data)).copy_to(ir1 + (192));
              #pragma unroll
              for (int32_t v132_n0 = 0; v132_n0 < 1; ++v132_n0) {
                int32_t v134_a = v132_n0 * 32;
                #pragma unroll
                for (int32_t v133_n1 = 6; v133_n1 < 13; ++v133_n1) {
                  int32_t v136_a = (v133_n1 - 6) * 32;
                  tensorforge::intel_esimd::simd<float, 32> v138_data;
                  v138_data.copy_from(ir1 + ((v134_a + v136_a)));
                  v138_data.copy_to(r1 + ((v134_a + v136_a)));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v143_i0 = 0; v143_i0 < 1; ++v143_i0) {
                int32_t v145_lead = v143_i0 * 32;
                glb_m0[v145_lead] = 0.0f;
                int32_t v151_a = v145_lead + 32;
                glb_m0[v151_a] = 0.0f;
                int32_t v155_a = v145_lead + 64;
                glb_m0[v155_a] = 0.0f;
                int32_t v159_a = v145_lead + 96;
                glb_m0[v159_a] = 0.0f;
                int32_t v163_a = v145_lead + 128;
                glb_m0[v163_a] = 0.0f;
                int32_t v167_a = v145_lead + 160;
                glb_m0[v167_a] = 0.0f;
                tensorforge::intel_esimd::simd<float, 32> v170_data;
                v170_data.copy_from(r1 + (v145_lead));
                int32_t v173_a = v145_lead + 192;
                v170_data.copy_to(glb_m0 + (v173_a));
                tensorforge::intel_esimd::simd<float, 32> v176_data;
                v176_data.copy_from(r1 + (v151_a));
                v176_data.copy_to(glb_m0 + ((v145_lead + 224)));
                tensorforge::intel_esimd::simd<float, 32> v182_data;
                v182_data.copy_from(r1 + (v155_a));
                v182_data.copy_to(glb_m0 + ((v145_lead + 256)));
                tensorforge::intel_esimd::simd<float, 32> v188_data;
                v188_data.copy_from(r1 + (v159_a));
                v188_data.copy_to(glb_m0 + ((v145_lead + 288)));
                tensorforge::intel_esimd::simd<float, 32> v194_data;
                v194_data.copy_from(r1 + (v163_a));
                v194_data.copy_to(glb_m0 + ((v145_lead + 320)));
                tensorforge::intel_esimd::simd<float, 32> v200_data;
                v200_data.copy_from(r1 + (v167_a));
                v200_data.copy_to(glb_m0 + ((v145_lead + 352)));
                tensorforge::intel_esimd::simd<float, 32> v206_data;
                v206_data.copy_from(r1 + (v173_a));
                v206_data.copy_to(glb_m0 + ((v145_lead + 384)));
              }
            }
          }
        }
      });
    }
  });
}

