// === base name ===
kernel_c954a8a4c98c34a3

// === header ===
void launcher_kernel_c954a8a4c98c34a3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c954a8a4c98c34a3(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c954a8a4c98c34a3(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c954a8a4c98c34a3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          float* __restrict__ s0 = &localShrMem0[0];
          float* __restrict__ s2 = &localShrMem0[0];
          float* __restrict__ s1 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
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
              for (int32_t v15_i1 = 0; v15_i1 < 8; ++v15_i1) {
                tensorforge::intel_esimd::simd<float, 8> v20_data;
                v20_data.copy_from(glb_m0 + ((v15_i1 * 8)));
                v20_data.copy_to(r0 + ((v15_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v23_ld;
              v23_ld.copy_from(glb_m1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v23_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                tensorforge::intel_esimd::simd<float, 8> v30_data;
                v30_data.copy_from(glb_m2 + ((v25_i1 * 8)));
                v30_data.copy_to(r2 + ((v25_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v42_acc{};
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(s0 + (0_i32));
              v42_acc += ((v46_data[0]) * v34_data);
              v42_acc += ((v46_data[1]) * v35_data);
              v42_acc += ((v46_data[2]) * v36_data);
              v42_acc += ((v46_data[3]) * v37_data);
              v42_acc += ((v46_data[4]) * v38_data);
              v42_acc += ((v46_data[5]) * v39_data);
              v42_acc += ((v46_data[6]) * v40_data);
              v42_acc += ((v46_data[7]) * v41_data);
              v42_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v63_acc{};
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(s0 + (8_i32));
              v63_acc += ((v67_data[0]) * v34_data);
              v63_acc += ((v67_data[1]) * v35_data);
              v63_acc += ((v67_data[2]) * v36_data);
              v63_acc += ((v67_data[3]) * v37_data);
              v63_acc += ((v67_data[4]) * v38_data);
              v63_acc += ((v67_data[5]) * v39_data);
              v63_acc += ((v67_data[6]) * v40_data);
              v63_acc += ((v67_data[7]) * v41_data);
              v63_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v84_acc{};
              tensorforge::intel_esimd::simd<float, 16> v88_data;
              v88_data.copy_from(s0 + (16_i32));
              v84_acc += ((v88_data[0]) * v34_data);
              v84_acc += ((v88_data[1]) * v35_data);
              v84_acc += ((v88_data[2]) * v36_data);
              v84_acc += ((v88_data[3]) * v37_data);
              v84_acc += ((v88_data[4]) * v38_data);
              v84_acc += ((v88_data[5]) * v39_data);
              v84_acc += ((v88_data[6]) * v40_data);
              v84_acc += ((v88_data[7]) * v41_data);
              v84_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v105_acc{};
              tensorforge::intel_esimd::simd<float, 16> v109_data;
              v109_data.copy_from(s0 + (24_i32));
              v105_acc += ((v109_data[0]) * v34_data);
              v105_acc += ((v109_data[1]) * v35_data);
              v105_acc += ((v109_data[2]) * v36_data);
              v105_acc += ((v109_data[3]) * v37_data);
              v105_acc += ((v109_data[4]) * v38_data);
              v105_acc += ((v109_data[5]) * v39_data);
              v105_acc += ((v109_data[6]) * v40_data);
              v105_acc += ((v109_data[7]) * v41_data);
              v105_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v126_acc{};
              tensorforge::intel_esimd::simd<float, 16> v130_data;
              v130_data.copy_from(s0 + (32_i32));
              v126_acc += ((v130_data[0]) * v34_data);
              v126_acc += ((v130_data[1]) * v35_data);
              v126_acc += ((v130_data[2]) * v36_data);
              v126_acc += ((v130_data[3]) * v37_data);
              v126_acc += ((v130_data[4]) * v38_data);
              v126_acc += ((v130_data[5]) * v39_data);
              v126_acc += ((v130_data[6]) * v40_data);
              v126_acc += ((v130_data[7]) * v41_data);
              v126_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v147_acc{};
              tensorforge::intel_esimd::simd<float, 16> v151_data;
              v151_data.copy_from(s0 + (40_i32));
              v147_acc += ((v151_data[0]) * v34_data);
              v147_acc += ((v151_data[1]) * v35_data);
              v147_acc += ((v151_data[2]) * v36_data);
              v147_acc += ((v151_data[3]) * v37_data);
              v147_acc += ((v151_data[4]) * v38_data);
              v147_acc += ((v151_data[5]) * v39_data);
              v147_acc += ((v151_data[6]) * v40_data);
              v147_acc += ((v151_data[7]) * v41_data);
              v147_acc.copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v168_acc{};
              tensorforge::intel_esimd::simd<float, 16> v172_data;
              v172_data.copy_from(s0 + (48_i32));
              v168_acc += ((v172_data[0]) * v34_data);
              v168_acc += ((v172_data[1]) * v35_data);
              v168_acc += ((v172_data[2]) * v36_data);
              v168_acc += ((v172_data[3]) * v37_data);
              v168_acc += ((v172_data[4]) * v38_data);
              v168_acc += ((v172_data[5]) * v39_data);
              v168_acc += ((v172_data[6]) * v40_data);
              v168_acc += ((v172_data[7]) * v41_data);
              v168_acc.copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v189_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data;
              v193_data.copy_from(s0 + (56_i32));
              v189_acc += ((v193_data[0]) * v34_data);
              v189_acc += ((v193_data[1]) * v35_data);
              v189_acc += ((v193_data[2]) * v36_data);
              v189_acc += ((v193_data[3]) * v37_data);
              v189_acc += ((v193_data[4]) * v38_data);
              v189_acc += ((v193_data[5]) * v39_data);
              v189_acc += ((v193_data[6]) * v40_data);
              v189_acc += ((v193_data[7]) * v41_data);
              v189_acc.copy_to(r1 + (112));
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v210_ld;
              v210_ld.copy_from(glb_m3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v210_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v213_data;
              v213_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v214_data;
              v214_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v215_data;
              v215_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v216_data;
              v216_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v217_data;
              v217_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v218_data;
              v218_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v219_data;
              v219_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v220_data;
              v220_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v221_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data;
              v225_data.copy_from(s2 + (0_i32));
              v221_acc += ((v225_data[0]) * v213_data);
              v221_acc += ((v225_data[1]) * v214_data);
              v221_acc += ((v225_data[2]) * v215_data);
              v221_acc += ((v225_data[3]) * v216_data);
              v221_acc += ((v225_data[4]) * v217_data);
              v221_acc += ((v225_data[5]) * v218_data);
              v221_acc += ((v225_data[6]) * v219_data);
              v221_acc += ((v225_data[7]) * v220_data);
              v221_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v242_acc{};
              tensorforge::intel_esimd::simd<float, 16> v246_data;
              v246_data.copy_from(s2 + (8_i32));
              v242_acc += ((v246_data[0]) * v213_data);
              v242_acc += ((v246_data[1]) * v214_data);
              v242_acc += ((v246_data[2]) * v215_data);
              v242_acc += ((v246_data[3]) * v216_data);
              v242_acc += ((v246_data[4]) * v217_data);
              v242_acc += ((v246_data[5]) * v218_data);
              v242_acc += ((v246_data[6]) * v219_data);
              v242_acc += ((v246_data[7]) * v220_data);
              v242_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v267_data;
              v267_data.copy_from(s2 + (16_i32));
              v263_acc += ((v267_data[0]) * v213_data);
              v263_acc += ((v267_data[1]) * v214_data);
              v263_acc += ((v267_data[2]) * v215_data);
              v263_acc += ((v267_data[3]) * v216_data);
              v263_acc += ((v267_data[4]) * v217_data);
              v263_acc += ((v267_data[5]) * v218_data);
              v263_acc += ((v267_data[6]) * v219_data);
              v263_acc += ((v267_data[7]) * v220_data);
              v263_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v284_acc{};
              tensorforge::intel_esimd::simd<float, 16> v288_data;
              v288_data.copy_from(s2 + (24_i32));
              v284_acc += ((v288_data[0]) * v213_data);
              v284_acc += ((v288_data[1]) * v214_data);
              v284_acc += ((v288_data[2]) * v215_data);
              v284_acc += ((v288_data[3]) * v216_data);
              v284_acc += ((v288_data[4]) * v217_data);
              v284_acc += ((v288_data[5]) * v218_data);
              v284_acc += ((v288_data[6]) * v219_data);
              v284_acc += ((v288_data[7]) * v220_data);
              v284_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v305_acc{};
              tensorforge::intel_esimd::simd<float, 16> v309_data;
              v309_data.copy_from(s2 + (32_i32));
              v305_acc += ((v309_data[0]) * v213_data);
              v305_acc += ((v309_data[1]) * v214_data);
              v305_acc += ((v309_data[2]) * v215_data);
              v305_acc += ((v309_data[3]) * v216_data);
              v305_acc += ((v309_data[4]) * v217_data);
              v305_acc += ((v309_data[5]) * v218_data);
              v305_acc += ((v309_data[6]) * v219_data);
              v305_acc += ((v309_data[7]) * v220_data);
              v305_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v326_acc{};
              tensorforge::intel_esimd::simd<float, 16> v330_data;
              v330_data.copy_from(s2 + (40_i32));
              v326_acc += ((v330_data[0]) * v213_data);
              v326_acc += ((v330_data[1]) * v214_data);
              v326_acc += ((v330_data[2]) * v215_data);
              v326_acc += ((v330_data[3]) * v216_data);
              v326_acc += ((v330_data[4]) * v217_data);
              v326_acc += ((v330_data[5]) * v218_data);
              v326_acc += ((v330_data[6]) * v219_data);
              v326_acc += ((v330_data[7]) * v220_data);
              v326_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v347_acc{};
              tensorforge::intel_esimd::simd<float, 16> v351_data;
              v351_data.copy_from(s2 + (48_i32));
              v347_acc += ((v351_data[0]) * v213_data);
              v347_acc += ((v351_data[1]) * v214_data);
              v347_acc += ((v351_data[2]) * v215_data);
              v347_acc += ((v351_data[3]) * v216_data);
              v347_acc += ((v351_data[4]) * v217_data);
              v347_acc += ((v351_data[5]) * v218_data);
              v347_acc += ((v351_data[6]) * v219_data);
              v347_acc += ((v351_data[7]) * v220_data);
              v347_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v368_acc{};
              tensorforge::intel_esimd::simd<float, 16> v372_data;
              v372_data.copy_from(s2 + (56_i32));
              v368_acc += ((v372_data[0]) * v213_data);
              v368_acc += ((v372_data[1]) * v214_data);
              v368_acc += ((v372_data[2]) * v215_data);
              v368_acc += ((v372_data[3]) * v216_data);
              v368_acc += ((v372_data[4]) * v217_data);
              v368_acc += ((v372_data[5]) * v218_data);
              v368_acc += ((v372_data[6]) * v219_data);
              v368_acc += ((v372_data[7]) * v220_data);
              v368_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v389_n1 = 0; v389_n1 < 8; ++v389_n1) {
                int32_t v390_a = v389_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v392_data;
                v392_data.copy_from(ir3 + (v390_a));
                tensorforge::intel_esimd::simd<float, 8> v395_data;
                v395_data.copy_from(r1 + (v390_a));
                (v395_data + v392_data).copy_to(r3 + (v390_a));
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v399_i1 = 0; v399_i1 < 8; ++v399_i1) {
                tensorforge::intel_esimd::simd<float, 8> v402_data;
                v402_data.copy_from(r3 + ((v399_i1 * 16)));
                v402_data.copy_to(s1 + ((v399_i1 * 8)));
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v407_k1 = 0; v407_k1 < 8; ++v407_k1) {
                int32_t v410_a = v407_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v412_data;
                v412_data.copy_from(s1 + (v410_a));
                (tensorforge::intel_esimd::abs(v412_data)).copy_to(glb_m4 + (v410_a));
              }
            }
          }
        }
      });
    }
  });
}

