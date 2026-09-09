// === base name ===
kernel_609dd06e89

// === header ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_609dd06e89(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_609dd06e89(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_609dd06e89(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, float* m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m4 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[batchId0 * 64 + 0 + m4_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v8_i1 = 0; v8_i1 < 8; ++v8_i1) {
                tensorforge::intel_esimd::simd<float, 8> v13_data;
                v13_data.copy_from(glb_m0 + ((v8_i1 * 8)));
                v13_data.copy_to(r0 + ((v8_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v17_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 8> v24_data;
                v24_data.copy_from(glb_m2 + ((v19_i1 * 8)));
                v24_data.copy_to(r2 + ((v19_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v36_acc{};
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(s0 + (0_i32));
              v36_acc += ((v40_data[0]) * v28_data);
              v36_acc += ((v40_data[1]) * v29_data);
              v36_acc += ((v40_data[2]) * v30_data);
              v36_acc += ((v40_data[3]) * v31_data);
              v36_acc += ((v40_data[4]) * v32_data);
              v36_acc += ((v40_data[5]) * v33_data);
              v36_acc += ((v40_data[6]) * v34_data);
              v36_acc += ((v40_data[7]) * v35_data);
              v36_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v57_acc{};
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(s0 + (8_i32));
              v57_acc += ((v61_data[0]) * v28_data);
              v57_acc += ((v61_data[1]) * v29_data);
              v57_acc += ((v61_data[2]) * v30_data);
              v57_acc += ((v61_data[3]) * v31_data);
              v57_acc += ((v61_data[4]) * v32_data);
              v57_acc += ((v61_data[5]) * v33_data);
              v57_acc += ((v61_data[6]) * v34_data);
              v57_acc += ((v61_data[7]) * v35_data);
              v57_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v78_acc{};
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(s0 + (16_i32));
              v78_acc += ((v82_data[0]) * v28_data);
              v78_acc += ((v82_data[1]) * v29_data);
              v78_acc += ((v82_data[2]) * v30_data);
              v78_acc += ((v82_data[3]) * v31_data);
              v78_acc += ((v82_data[4]) * v32_data);
              v78_acc += ((v82_data[5]) * v33_data);
              v78_acc += ((v82_data[6]) * v34_data);
              v78_acc += ((v82_data[7]) * v35_data);
              v78_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v99_acc{};
              tensorforge::intel_esimd::simd<float, 16> v103_data;
              v103_data.copy_from(s0 + (24_i32));
              v99_acc += ((v103_data[0]) * v28_data);
              v99_acc += ((v103_data[1]) * v29_data);
              v99_acc += ((v103_data[2]) * v30_data);
              v99_acc += ((v103_data[3]) * v31_data);
              v99_acc += ((v103_data[4]) * v32_data);
              v99_acc += ((v103_data[5]) * v33_data);
              v99_acc += ((v103_data[6]) * v34_data);
              v99_acc += ((v103_data[7]) * v35_data);
              v99_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v120_acc{};
              tensorforge::intel_esimd::simd<float, 16> v124_data;
              v124_data.copy_from(s0 + (32_i32));
              v120_acc += ((v124_data[0]) * v28_data);
              v120_acc += ((v124_data[1]) * v29_data);
              v120_acc += ((v124_data[2]) * v30_data);
              v120_acc += ((v124_data[3]) * v31_data);
              v120_acc += ((v124_data[4]) * v32_data);
              v120_acc += ((v124_data[5]) * v33_data);
              v120_acc += ((v124_data[6]) * v34_data);
              v120_acc += ((v124_data[7]) * v35_data);
              v120_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v141_acc{};
              tensorforge::intel_esimd::simd<float, 16> v145_data;
              v145_data.copy_from(s0 + (40_i32));
              v141_acc += ((v145_data[0]) * v28_data);
              v141_acc += ((v145_data[1]) * v29_data);
              v141_acc += ((v145_data[2]) * v30_data);
              v141_acc += ((v145_data[3]) * v31_data);
              v141_acc += ((v145_data[4]) * v32_data);
              v141_acc += ((v145_data[5]) * v33_data);
              v141_acc += ((v145_data[6]) * v34_data);
              v141_acc += ((v145_data[7]) * v35_data);
              v141_acc.copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v162_acc{};
              tensorforge::intel_esimd::simd<float, 16> v166_data;
              v166_data.copy_from(s0 + (48_i32));
              v162_acc += ((v166_data[0]) * v28_data);
              v162_acc += ((v166_data[1]) * v29_data);
              v162_acc += ((v166_data[2]) * v30_data);
              v162_acc += ((v166_data[3]) * v31_data);
              v162_acc += ((v166_data[4]) * v32_data);
              v162_acc += ((v166_data[5]) * v33_data);
              v162_acc += ((v166_data[6]) * v34_data);
              v162_acc += ((v166_data[7]) * v35_data);
              v162_acc.copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v183_acc{};
              tensorforge::intel_esimd::simd<float, 16> v187_data;
              v187_data.copy_from(s0 + (56_i32));
              v183_acc += ((v187_data[0]) * v28_data);
              v183_acc += ((v187_data[1]) * v29_data);
              v183_acc += ((v187_data[2]) * v30_data);
              v183_acc += ((v187_data[3]) * v31_data);
              v183_acc += ((v187_data[4]) * v32_data);
              v183_acc += ((v187_data[5]) * v33_data);
              v183_acc += ((v187_data[6]) * v34_data);
              v183_acc += ((v187_data[7]) * v35_data);
              v183_acc.copy_to(r1 + (112));
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v205_ld;
              v205_ld.copy_from(glb_m3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v205_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v208_data;
              v208_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v209_data;
              v209_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v210_data;
              v210_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v211_data;
              v211_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v212_data;
              v212_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v213_data;
              v213_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v214_data;
              v214_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v215_data;
              v215_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v216_acc{};
              tensorforge::intel_esimd::simd<float, 16> v220_data;
              v220_data.copy_from(s2 + (0_i32));
              v216_acc += ((v220_data[0]) * v208_data);
              v216_acc += ((v220_data[1]) * v209_data);
              v216_acc += ((v220_data[2]) * v210_data);
              v216_acc += ((v220_data[3]) * v211_data);
              v216_acc += ((v220_data[4]) * v212_data);
              v216_acc += ((v220_data[5]) * v213_data);
              v216_acc += ((v220_data[6]) * v214_data);
              v216_acc += ((v220_data[7]) * v215_data);
              v216_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v237_acc{};
              tensorforge::intel_esimd::simd<float, 16> v241_data;
              v241_data.copy_from(s2 + (8_i32));
              v237_acc += ((v241_data[0]) * v208_data);
              v237_acc += ((v241_data[1]) * v209_data);
              v237_acc += ((v241_data[2]) * v210_data);
              v237_acc += ((v241_data[3]) * v211_data);
              v237_acc += ((v241_data[4]) * v212_data);
              v237_acc += ((v241_data[5]) * v213_data);
              v237_acc += ((v241_data[6]) * v214_data);
              v237_acc += ((v241_data[7]) * v215_data);
              v237_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v258_acc{};
              tensorforge::intel_esimd::simd<float, 16> v262_data;
              v262_data.copy_from(s2 + (16_i32));
              v258_acc += ((v262_data[0]) * v208_data);
              v258_acc += ((v262_data[1]) * v209_data);
              v258_acc += ((v262_data[2]) * v210_data);
              v258_acc += ((v262_data[3]) * v211_data);
              v258_acc += ((v262_data[4]) * v212_data);
              v258_acc += ((v262_data[5]) * v213_data);
              v258_acc += ((v262_data[6]) * v214_data);
              v258_acc += ((v262_data[7]) * v215_data);
              v258_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v279_acc{};
              tensorforge::intel_esimd::simd<float, 16> v283_data;
              v283_data.copy_from(s2 + (24_i32));
              v279_acc += ((v283_data[0]) * v208_data);
              v279_acc += ((v283_data[1]) * v209_data);
              v279_acc += ((v283_data[2]) * v210_data);
              v279_acc += ((v283_data[3]) * v211_data);
              v279_acc += ((v283_data[4]) * v212_data);
              v279_acc += ((v283_data[5]) * v213_data);
              v279_acc += ((v283_data[6]) * v214_data);
              v279_acc += ((v283_data[7]) * v215_data);
              v279_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              tensorforge::intel_esimd::simd<float, 16> v304_data;
              v304_data.copy_from(s2 + (32_i32));
              v300_acc += ((v304_data[0]) * v208_data);
              v300_acc += ((v304_data[1]) * v209_data);
              v300_acc += ((v304_data[2]) * v210_data);
              v300_acc += ((v304_data[3]) * v211_data);
              v300_acc += ((v304_data[4]) * v212_data);
              v300_acc += ((v304_data[5]) * v213_data);
              v300_acc += ((v304_data[6]) * v214_data);
              v300_acc += ((v304_data[7]) * v215_data);
              v300_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v321_acc{};
              tensorforge::intel_esimd::simd<float, 16> v325_data;
              v325_data.copy_from(s2 + (40_i32));
              v321_acc += ((v325_data[0]) * v208_data);
              v321_acc += ((v325_data[1]) * v209_data);
              v321_acc += ((v325_data[2]) * v210_data);
              v321_acc += ((v325_data[3]) * v211_data);
              v321_acc += ((v325_data[4]) * v212_data);
              v321_acc += ((v325_data[5]) * v213_data);
              v321_acc += ((v325_data[6]) * v214_data);
              v321_acc += ((v325_data[7]) * v215_data);
              v321_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v342_acc{};
              tensorforge::intel_esimd::simd<float, 16> v346_data;
              v346_data.copy_from(s2 + (48_i32));
              v342_acc += ((v346_data[0]) * v208_data);
              v342_acc += ((v346_data[1]) * v209_data);
              v342_acc += ((v346_data[2]) * v210_data);
              v342_acc += ((v346_data[3]) * v211_data);
              v342_acc += ((v346_data[4]) * v212_data);
              v342_acc += ((v346_data[5]) * v213_data);
              v342_acc += ((v346_data[6]) * v214_data);
              v342_acc += ((v346_data[7]) * v215_data);
              v342_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v363_acc{};
              tensorforge::intel_esimd::simd<float, 16> v367_data;
              v367_data.copy_from(s2 + (56_i32));
              v363_acc += ((v367_data[0]) * v208_data);
              v363_acc += ((v367_data[1]) * v209_data);
              v363_acc += ((v367_data[2]) * v210_data);
              v363_acc += ((v367_data[3]) * v211_data);
              v363_acc += ((v367_data[4]) * v212_data);
              v363_acc += ((v367_data[5]) * v213_data);
              v363_acc += ((v367_data[6]) * v214_data);
              v363_acc += ((v367_data[7]) * v215_data);
              v363_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v384_n1 = 0; v384_n1 < 8; ++v384_n1) {
                int32_t v385_a = v384_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v387_data;
                v387_data.copy_from(ir3 + (v385_a));
                tensorforge::intel_esimd::simd<float, 8> v390_data;
                v390_data.copy_from(r1 + (v385_a));
                (v390_data + v387_data).copy_to(r3 + (v385_a));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v395_i1 = 0; v395_i1 < 8; ++v395_i1) {
                tensorforge::intel_esimd::simd<float, 8> v398_data;
                v398_data.copy_from(r3 + ((v395_i1 * 16)));
                v398_data.copy_to(s1 + ((v395_i1 * 8)));
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v403_k1 = 0; v403_k1 < 8; ++v403_k1) {
                int32_t v406_a = v403_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v408_data;
                v408_data.copy_from(s1 + (v406_a));
                (tensorforge::intel_esimd::abs(v408_data)).copy_to(glb_m4 + (v406_a));
              }
            }
          }
        }
      });
    }
  });
}

