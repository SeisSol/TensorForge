// === base name ===
kernel_db1e9c7521d6ded9

// === header ===
void launcher_kernel_db1e9c7521d6ded9(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_db1e9c7521d6ded9(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_db1e9c7521d6ded9(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_db1e9c7521d6ded9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1536, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[96 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[80];
          float* __restrict__ s0 = &localShrMem0[0];
          float* __restrict__ s1 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float r0[96]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v13_i1 = 0; v13_i1 < 6; ++v13_i1) {
                tensorforge::intel_esimd::simd<float, 12> v18_data;
                v18_data.copy_from(glb_m0 + ((v13_i1 * 12)));
                v18_data.copy_to(r0 + ((v13_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v21_ld;
              v21_ld.copy_from(glb_m1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v21_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 4) {
                tensorforge::intel_esimd::simd<float, 16> v22_ld;
                v22_ld.copy_from(glb_m1 + (0 + 0 + 1 * item.get_local_id(0) + 32));
                v22_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 32));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 12> v29_data;
                v29_data.copy_from(glb_m3 + ((v24_i1 * 12)));
                v29_data.copy_to(r2 + ((v24_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[96]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v39_acc{};
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(s0 + (0_i32));
              v39_acc += ((v43_data[0]) * v33_data);
              v39_acc += ((v43_data[1]) * v34_data);
              v39_acc += ((v43_data[2]) * v35_data);
              v39_acc += ((v43_data[3]) * v36_data);
              v39_acc += ((v43_data[4]) * v37_data);
              v39_acc += ((v43_data[5]) * v38_data);
              v39_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v56_acc{};
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(s0 + (6_i32));
              v56_acc += ((v60_data[0]) * v33_data);
              v56_acc += ((v60_data[1]) * v34_data);
              v56_acc += ((v60_data[2]) * v35_data);
              v56_acc += ((v60_data[3]) * v36_data);
              v56_acc += ((v60_data[4]) * v37_data);
              v56_acc += ((v60_data[5]) * v38_data);
              v56_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v73_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(s0 + (12_i32));
              v73_acc += ((v77_data[0]) * v33_data);
              v73_acc += ((v77_data[1]) * v34_data);
              v73_acc += ((v77_data[2]) * v35_data);
              v73_acc += ((v77_data[3]) * v36_data);
              v73_acc += ((v77_data[4]) * v37_data);
              v73_acc += ((v77_data[5]) * v38_data);
              v73_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v90_acc{};
              tensorforge::intel_esimd::simd<float, 16> v94_data;
              v94_data.copy_from(s0 + (18_i32));
              v90_acc += ((v94_data[0]) * v33_data);
              v90_acc += ((v94_data[1]) * v34_data);
              v90_acc += ((v94_data[2]) * v35_data);
              v90_acc += ((v94_data[3]) * v36_data);
              v90_acc += ((v94_data[4]) * v37_data);
              v90_acc += ((v94_data[5]) * v38_data);
              v90_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v107_acc{};
              tensorforge::intel_esimd::simd<float, 16> v111_data;
              v111_data.copy_from(s0 + (24_i32));
              v107_acc += ((v111_data[0]) * v33_data);
              v107_acc += ((v111_data[1]) * v34_data);
              v107_acc += ((v111_data[2]) * v35_data);
              v107_acc += ((v111_data[3]) * v36_data);
              v107_acc += ((v111_data[4]) * v37_data);
              v107_acc += ((v111_data[5]) * v38_data);
              v107_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v124_acc{};
              tensorforge::intel_esimd::simd<float, 16> v128_data;
              v128_data.copy_from(s0 + (30_i32));
              v124_acc += ((v128_data[0]) * v33_data);
              v124_acc += ((v128_data[1]) * v34_data);
              v124_acc += ((v128_data[2]) * v35_data);
              v124_acc += ((v128_data[3]) * v36_data);
              v124_acc += ((v128_data[4]) * v37_data);
              v124_acc += ((v128_data[5]) * v38_data);
              v124_acc.copy_to(r1 + (80));
              // wait(r2 = load{g>r}(glb_m3););
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v141_i1 = 0; v141_i1 < 6; ++v141_i1) {
                tensorforge::intel_esimd::simd<float, 12> v144_data;
                v144_data.copy_from(r1 + ((v141_i1 * 16)));
                v144_data.copy_to(s1 + ((v141_i1 * 12)));
              }
              float r3[96]{};
              // r3 = +(r2 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir3[96]{};
              tensorforge::intel_esimd::simd<float, 16> v151_data;
              v151_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v152_data;
              v152_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v153_data;
              v153_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v154_data;
              v154_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v158_data;
              v158_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v159_data;
              v159_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v160_data;
              v160_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v161_data;
              v161_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v162_data;
              v162_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v163_acc{};
              tensorforge::intel_esimd::simd<float, 16> v167_data;
              v167_data.copy_from(s1 + (0_i32));
              v163_acc += ((v167_data[0]) * v151_data);
              v163_acc += ((v167_data[1]) * v152_data);
              v163_acc += ((v167_data[2]) * v153_data);
              v163_acc += ((v167_data[3]) * v154_data);
              v163_acc += ((v167_data[4]) * v155_data);
              v163_acc += ((v167_data[5]) * v156_data);
              v163_acc += ((v167_data[6]) * v157_data);
              v163_acc += ((v167_data[7]) * v158_data);
              v163_acc += ((v167_data[8]) * v159_data);
              v163_acc += ((v167_data[9]) * v160_data);
              v163_acc += ((v167_data[10]) * v161_data);
              v163_acc += ((v167_data[11]) * v162_data);
              v163_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v192_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data;
              v196_data.copy_from(s1 + (12_i32));
              v192_acc += ((v196_data[0]) * v151_data);
              v192_acc += ((v196_data[1]) * v152_data);
              v192_acc += ((v196_data[2]) * v153_data);
              v192_acc += ((v196_data[3]) * v154_data);
              v192_acc += ((v196_data[4]) * v155_data);
              v192_acc += ((v196_data[5]) * v156_data);
              v192_acc += ((v196_data[6]) * v157_data);
              v192_acc += ((v196_data[7]) * v158_data);
              v192_acc += ((v196_data[8]) * v159_data);
              v192_acc += ((v196_data[9]) * v160_data);
              v192_acc += ((v196_data[10]) * v161_data);
              v192_acc += ((v196_data[11]) * v162_data);
              v192_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v221_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data;
              v225_data.copy_from(s1 + (24_i32));
              v221_acc += ((v225_data[0]) * v151_data);
              v221_acc += ((v225_data[1]) * v152_data);
              v221_acc += ((v225_data[2]) * v153_data);
              v221_acc += ((v225_data[3]) * v154_data);
              v221_acc += ((v225_data[4]) * v155_data);
              v221_acc += ((v225_data[5]) * v156_data);
              v221_acc += ((v225_data[6]) * v157_data);
              v221_acc += ((v225_data[7]) * v158_data);
              v221_acc += ((v225_data[8]) * v159_data);
              v221_acc += ((v225_data[9]) * v160_data);
              v221_acc += ((v225_data[10]) * v161_data);
              v221_acc += ((v225_data[11]) * v162_data);
              v221_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v250_acc{};
              tensorforge::intel_esimd::simd<float, 16> v254_data;
              v254_data.copy_from(s1 + (36_i32));
              v250_acc += ((v254_data[0]) * v151_data);
              v250_acc += ((v254_data[1]) * v152_data);
              v250_acc += ((v254_data[2]) * v153_data);
              v250_acc += ((v254_data[3]) * v154_data);
              v250_acc += ((v254_data[4]) * v155_data);
              v250_acc += ((v254_data[5]) * v156_data);
              v250_acc += ((v254_data[6]) * v157_data);
              v250_acc += ((v254_data[7]) * v158_data);
              v250_acc += ((v254_data[8]) * v159_data);
              v250_acc += ((v254_data[9]) * v160_data);
              v250_acc += ((v254_data[10]) * v161_data);
              v250_acc += ((v254_data[11]) * v162_data);
              v250_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v279_acc{};
              tensorforge::intel_esimd::simd<float, 16> v283_data;
              v283_data.copy_from(s1 + (48_i32));
              v279_acc += ((v283_data[0]) * v151_data);
              v279_acc += ((v283_data[1]) * v152_data);
              v279_acc += ((v283_data[2]) * v153_data);
              v279_acc += ((v283_data[3]) * v154_data);
              v279_acc += ((v283_data[4]) * v155_data);
              v279_acc += ((v283_data[5]) * v156_data);
              v279_acc += ((v283_data[6]) * v157_data);
              v279_acc += ((v283_data[7]) * v158_data);
              v279_acc += ((v283_data[8]) * v159_data);
              v279_acc += ((v283_data[9]) * v160_data);
              v279_acc += ((v283_data[10]) * v161_data);
              v279_acc += ((v283_data[11]) * v162_data);
              v279_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v308_acc{};
              tensorforge::intel_esimd::simd<float, 16> v312_data;
              v312_data.copy_from(s1 + (60_i32));
              v308_acc += ((v312_data[0]) * v151_data);
              v308_acc += ((v312_data[1]) * v152_data);
              v308_acc += ((v312_data[2]) * v153_data);
              v308_acc += ((v312_data[3]) * v154_data);
              v308_acc += ((v312_data[4]) * v155_data);
              v308_acc += ((v312_data[5]) * v156_data);
              v308_acc += ((v312_data[6]) * v157_data);
              v308_acc += ((v312_data[7]) * v158_data);
              v308_acc += ((v312_data[8]) * v159_data);
              v308_acc += ((v312_data[9]) * v160_data);
              v308_acc += ((v312_data[10]) * v161_data);
              v308_acc += ((v312_data[11]) * v162_data);
              v308_acc.copy_to(ir3 + (80));
              #pragma unroll
              for (int32_t v337_n1 = 0; v337_n1 < 6; ++v337_n1) {
                int32_t v338_a = v337_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v340_data;
                v340_data.copy_from(ir3 + (v338_a));
                v340_data.copy_to(r3 + (v338_a));
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v343_i1 = 0; v343_i1 < 6; ++v343_i1) {
                tensorforge::intel_esimd::simd<float, 12> v346_data;
                v346_data.copy_from(r3 + ((v343_i1 * 16)));
                v346_data.copy_to(glb_m2 + ((v343_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

