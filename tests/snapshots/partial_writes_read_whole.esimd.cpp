// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7ab185b978(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7ab185b978(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (3072, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×9(32×9) {0..32}×{0..9} pointer_based
        // m1 16×9(16×9) {0..16}×{0..9} pointer_based
        // m2 16×9(16×9) {0..16}×{0..9} pointer_based
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based
        // m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[384 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[384];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
              float r0[288]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v8_i0 = 0; v8_i0 < 1; ++v8_i0) {
                int32_t v10_lead = v8_i0 * 32;
                #pragma unroll
                for (int32_t v9_i1 = 0; v9_i1 < 9; ++v9_i1) {
                  int32_t v13_a = v10_lead + (v9_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v14_data;
                  v14_data.copy_from(glb_m0 + (v13_a));
                  v14_data.copy_to(r0 + (v13_a));
                }
              }
              float r2[288]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 16> v24_data;
                v24_data.copy_from(glb_m1 + ((v19_i1 * 16)));
                v24_data.copy_to(r2 + ((v19_i1 * 32)));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[288]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v28_data;
              v28_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v29_data;
              v29_data.copy_from(r1 + (0));
              (v29_data + v28_data).copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v31_data;
              v31_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 32> v32_data;
              v32_data.copy_from(r1 + (32));
              (v32_data + v31_data).copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v34_data;
              v34_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 32> v35_data;
              v35_data.copy_from(r1 + (64));
              (v35_data + v34_data).copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v37_data;
              v37_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 32> v38_data;
              v38_data.copy_from(r1 + (96));
              (v38_data + v37_data).copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(r1 + (128));
              (v41_data + v40_data).copy_to(r1 + (128));
              tensorforge::intel_esimd::simd<float, 32> v43_data;
              v43_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(r1 + (160));
              (v44_data + v43_data).copy_to(r1 + (160));
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r1 + (192));
              (v47_data + v46_data).copy_to(r1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v49_data;
              v49_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r1 + (224));
              (v50_data + v49_data).copy_to(r1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v52_data;
              v52_data.copy_from(r0 + (256));
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(r1 + (256));
              (v53_data + v52_data).copy_to(r1 + (256));
              float* __restrict__ s0 = &localShrMem0[96];
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v56_i0 = 0; v56_i0 < 1; ++v56_i0) {
                int32_t v58_a = v56_i0 * 32;
                #pragma unroll
                for (int32_t v57_i1 = 0; v57_i1 < 9; ++v57_i1) {
                  int32_t v60_a = v58_a + (v57_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v61_data;
                  v61_data.copy_from(r1 + (v60_a));
                  v61_data.copy_to(s0 + (v60_a));
                }
              }
              float r4[288]{};
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v67_i1 = 0; v67_i1 < 9; ++v67_i1) {
                tensorforge::intel_esimd::simd<float, 16> v72_data;
                v72_data.copy_from(glb_m2 + ((v67_i1 * 16)));
                v72_data.copy_to(r4 + ((v67_i1 * 32)));
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[288]{};
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir3[288]{};
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v78_data;
              v78_data.copy_from(ir3 + (0));
              (v78_data + v77_data).copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(ir3 + (32));
              (v81_data + v80_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v84_data;
              v84_data.copy_from(ir3 + (64));
              (v84_data + v83_data).copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v87_data;
              v87_data.copy_from(ir3 + (96));
              (v87_data + v86_data).copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v90_data;
              v90_data.copy_from(ir3 + (128));
              (v90_data + v89_data).copy_to(ir3 + (128));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(ir3 + (160));
              (v93_data + v92_data).copy_to(ir3 + (160));
              tensorforge::intel_esimd::simd<float, 16> v95_data;
              v95_data.copy_from(r2 + (192));
              tensorforge::intel_esimd::simd<float, 16> v96_data;
              v96_data.copy_from(ir3 + (192));
              (v96_data + v95_data).copy_to(ir3 + (192));
              tensorforge::intel_esimd::simd<float, 16> v98_data;
              v98_data.copy_from(r2 + (224));
              tensorforge::intel_esimd::simd<float, 16> v99_data;
              v99_data.copy_from(ir3 + (224));
              (v99_data + v98_data).copy_to(ir3 + (224));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(r2 + (256));
              tensorforge::intel_esimd::simd<float, 16> v102_data;
              v102_data.copy_from(ir3 + (256));
              (v102_data + v101_data).copy_to(ir3 + (256));
              #pragma unroll
              for (int32_t v104_n1 = 0; v104_n1 < 9; ++v104_n1) {
                int32_t v105_a = v104_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v107_data;
                v107_data.copy_from(ir3 + (v105_a));
                tensorforge::intel_esimd::simd<float, 16> v112_data;
                v112_data.copy_from(s0 + (v105_a));
                (v112_data + v107_data).copy_to(r3 + (v105_a));
              }
              // s0 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v116_i1 = 0; v116_i1 < 9; ++v116_i1) {
                int32_t v117_a = v116_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v119_data;
                v119_data.copy_from(r3 + (v117_a));
                v119_data.copy_to(s0 + (v117_a));
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[288]{};
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir5[288]{};
              tensorforge::intel_esimd::simd<float, 16> v126_data;
              v126_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v127_data;
              v127_data.copy_from(ir5 + (0));
              (v127_data + v126_data).copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v129_data;
              v129_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v130_data;
              v130_data.copy_from(ir5 + (32));
              (v130_data + v129_data).copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v132_data;
              v132_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v133_data;
              v133_data.copy_from(ir5 + (64));
              (v133_data + v132_data).copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v135_data;
              v135_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v136_data;
              v136_data.copy_from(ir5 + (96));
              (v136_data + v135_data).copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v138_data;
              v138_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v139_data;
              v139_data.copy_from(ir5 + (128));
              (v139_data + v138_data).copy_to(ir5 + (128));
              tensorforge::intel_esimd::simd<float, 16> v141_data;
              v141_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v142_data;
              v142_data.copy_from(ir5 + (160));
              (v142_data + v141_data).copy_to(ir5 + (160));
              tensorforge::intel_esimd::simd<float, 16> v144_data;
              v144_data.copy_from(r4 + (192));
              tensorforge::intel_esimd::simd<float, 16> v145_data;
              v145_data.copy_from(ir5 + (192));
              (v145_data + v144_data).copy_to(ir5 + (192));
              tensorforge::intel_esimd::simd<float, 16> v147_data;
              v147_data.copy_from(r4 + (224));
              tensorforge::intel_esimd::simd<float, 16> v148_data;
              v148_data.copy_from(ir5 + (224));
              (v148_data + v147_data).copy_to(ir5 + (224));
              tensorforge::intel_esimd::simd<float, 16> v150_data;
              v150_data.copy_from(r4 + (256));
              tensorforge::intel_esimd::simd<float, 16> v151_data;
              v151_data.copy_from(ir5 + (256));
              (v151_data + v150_data).copy_to(ir5 + (256));
              #pragma unroll
              for (int32_t v153_n1 = 0; v153_n1 < 9; ++v153_n1) {
                int32_t v154_a = v153_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v156_data;
                v156_data.copy_from(ir5 + (v154_a));
                tensorforge::intel_esimd::simd<float, 16> v161_data;
                v161_data.copy_from(s0 + (v154_a));
                (v161_data + v156_data).copy_to(r5 + (v154_a));
              }
              // s0 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v165_i1 = 0; v165_i1 < 9; ++v165_i1) {
                int32_t v166_a = v165_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v168_data;
                v168_data.copy_from(r5 + (v166_a));
                v168_data.copy_to(s0 + (v166_a));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v174_ld;
              v174_ld.copy_from(glb_m4 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v174_ld.copy_to(s1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 17) {
                tensorforge::intel_esimd::simd<float, 32> v175_ld;
                v175_ld.copy_from(glb_m4 + (0 + 0 + 1 * item.get_local_id(0) + 64));
                v175_ld.copy_to(s1 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r6[288]{};
              // r6 = +(s0 * s1) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir6[288]{};
              tensorforge::intel_esimd::simd<float, 32> v181_data;
              v181_data.copy_from(s0 + (0_i32));
              float v182_data = s1[0];
              tensorforge::intel_esimd::simd<float, 32> v184_data;
              v184_data.copy_from(ir6 + (0));
              (v184_data + (v181_data * v182_data)).copy_to(ir6 + (0));
              float v190_data = s1[9];
              tensorforge::intel_esimd::simd<float, 32> v192_data;
              v192_data.copy_from(ir6 + (32));
              (v192_data + (v181_data * v190_data)).copy_to(ir6 + (32));
              float v198_data = s1[18];
              tensorforge::intel_esimd::simd<float, 32> v200_data;
              v200_data.copy_from(ir6 + (64));
              (v200_data + (v181_data * v198_data)).copy_to(ir6 + (64));
              float v206_data = s1[27];
              tensorforge::intel_esimd::simd<float, 32> v208_data;
              v208_data.copy_from(ir6 + (96));
              (v208_data + (v181_data * v206_data)).copy_to(ir6 + (96));
              float v214_data = s1[36];
              tensorforge::intel_esimd::simd<float, 32> v216_data;
              v216_data.copy_from(ir6 + (128));
              (v216_data + (v181_data * v214_data)).copy_to(ir6 + (128));
              float v222_data = s1[45];
              tensorforge::intel_esimd::simd<float, 32> v224_data;
              v224_data.copy_from(ir6 + (160));
              (v224_data + (v181_data * v222_data)).copy_to(ir6 + (160));
              float v230_data = s1[54];
              tensorforge::intel_esimd::simd<float, 32> v232_data;
              v232_data.copy_from(ir6 + (192));
              (v232_data + (v181_data * v230_data)).copy_to(ir6 + (192));
              float v238_data = s1[63];
              tensorforge::intel_esimd::simd<float, 32> v240_data;
              v240_data.copy_from(ir6 + (224));
              (v240_data + (v181_data * v238_data)).copy_to(ir6 + (224));
              float v246_data = s1[72];
              tensorforge::intel_esimd::simd<float, 32> v248_data;
              v248_data.copy_from(ir6 + (256));
              (v248_data + (v181_data * v246_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v253_data;
              v253_data.copy_from(s0 + (32_i32));
              float v254_data = s1[1];
              tensorforge::intel_esimd::simd<float, 32> v256_data;
              v256_data.copy_from(ir6 + (0));
              (v256_data + (v253_data * v254_data)).copy_to(ir6 + (0));
              float v262_data = s1[10];
              tensorforge::intel_esimd::simd<float, 32> v264_data;
              v264_data.copy_from(ir6 + (32));
              (v264_data + (v253_data * v262_data)).copy_to(ir6 + (32));
              float v270_data = s1[19];
              tensorforge::intel_esimd::simd<float, 32> v272_data;
              v272_data.copy_from(ir6 + (64));
              (v272_data + (v253_data * v270_data)).copy_to(ir6 + (64));
              float v278_data = s1[28];
              tensorforge::intel_esimd::simd<float, 32> v280_data;
              v280_data.copy_from(ir6 + (96));
              (v280_data + (v253_data * v278_data)).copy_to(ir6 + (96));
              float v286_data = s1[37];
              tensorforge::intel_esimd::simd<float, 32> v288_data;
              v288_data.copy_from(ir6 + (128));
              (v288_data + (v253_data * v286_data)).copy_to(ir6 + (128));
              float v294_data = s1[46];
              tensorforge::intel_esimd::simd<float, 32> v296_data;
              v296_data.copy_from(ir6 + (160));
              (v296_data + (v253_data * v294_data)).copy_to(ir6 + (160));
              float v302_data = s1[55];
              tensorforge::intel_esimd::simd<float, 32> v304_data;
              v304_data.copy_from(ir6 + (192));
              (v304_data + (v253_data * v302_data)).copy_to(ir6 + (192));
              float v310_data = s1[64];
              tensorforge::intel_esimd::simd<float, 32> v312_data;
              v312_data.copy_from(ir6 + (224));
              (v312_data + (v253_data * v310_data)).copy_to(ir6 + (224));
              float v318_data = s1[73];
              tensorforge::intel_esimd::simd<float, 32> v320_data;
              v320_data.copy_from(ir6 + (256));
              (v320_data + (v253_data * v318_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v325_data;
              v325_data.copy_from(s0 + (64_i32));
              float v326_data = s1[2];
              tensorforge::intel_esimd::simd<float, 32> v328_data;
              v328_data.copy_from(ir6 + (0));
              (v328_data + (v325_data * v326_data)).copy_to(ir6 + (0));
              float v334_data = s1[11];
              tensorforge::intel_esimd::simd<float, 32> v336_data;
              v336_data.copy_from(ir6 + (32));
              (v336_data + (v325_data * v334_data)).copy_to(ir6 + (32));
              float v342_data = s1[20];
              tensorforge::intel_esimd::simd<float, 32> v344_data;
              v344_data.copy_from(ir6 + (64));
              (v344_data + (v325_data * v342_data)).copy_to(ir6 + (64));
              float v350_data = s1[29];
              tensorforge::intel_esimd::simd<float, 32> v352_data;
              v352_data.copy_from(ir6 + (96));
              (v352_data + (v325_data * v350_data)).copy_to(ir6 + (96));
              float v358_data = s1[38];
              tensorforge::intel_esimd::simd<float, 32> v360_data;
              v360_data.copy_from(ir6 + (128));
              (v360_data + (v325_data * v358_data)).copy_to(ir6 + (128));
              float v366_data = s1[47];
              tensorforge::intel_esimd::simd<float, 32> v368_data;
              v368_data.copy_from(ir6 + (160));
              (v368_data + (v325_data * v366_data)).copy_to(ir6 + (160));
              float v374_data = s1[56];
              tensorforge::intel_esimd::simd<float, 32> v376_data;
              v376_data.copy_from(ir6 + (192));
              (v376_data + (v325_data * v374_data)).copy_to(ir6 + (192));
              float v382_data = s1[65];
              tensorforge::intel_esimd::simd<float, 32> v384_data;
              v384_data.copy_from(ir6 + (224));
              (v384_data + (v325_data * v382_data)).copy_to(ir6 + (224));
              float v390_data = s1[74];
              tensorforge::intel_esimd::simd<float, 32> v392_data;
              v392_data.copy_from(ir6 + (256));
              (v392_data + (v325_data * v390_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v397_data;
              v397_data.copy_from(s0 + (96_i32));
              float v398_data = s1[3];
              tensorforge::intel_esimd::simd<float, 32> v400_data;
              v400_data.copy_from(ir6 + (0));
              (v400_data + (v397_data * v398_data)).copy_to(ir6 + (0));
              float v406_data = s1[12];
              tensorforge::intel_esimd::simd<float, 32> v408_data;
              v408_data.copy_from(ir6 + (32));
              (v408_data + (v397_data * v406_data)).copy_to(ir6 + (32));
              float v414_data = s1[21];
              tensorforge::intel_esimd::simd<float, 32> v416_data;
              v416_data.copy_from(ir6 + (64));
              (v416_data + (v397_data * v414_data)).copy_to(ir6 + (64));
              float v422_data = s1[30];
              tensorforge::intel_esimd::simd<float, 32> v424_data;
              v424_data.copy_from(ir6 + (96));
              (v424_data + (v397_data * v422_data)).copy_to(ir6 + (96));
              float v430_data = s1[39];
              tensorforge::intel_esimd::simd<float, 32> v432_data;
              v432_data.copy_from(ir6 + (128));
              (v432_data + (v397_data * v430_data)).copy_to(ir6 + (128));
              float v438_data = s1[48];
              tensorforge::intel_esimd::simd<float, 32> v440_data;
              v440_data.copy_from(ir6 + (160));
              (v440_data + (v397_data * v438_data)).copy_to(ir6 + (160));
              float v446_data = s1[57];
              tensorforge::intel_esimd::simd<float, 32> v448_data;
              v448_data.copy_from(ir6 + (192));
              (v448_data + (v397_data * v446_data)).copy_to(ir6 + (192));
              float v454_data = s1[66];
              tensorforge::intel_esimd::simd<float, 32> v456_data;
              v456_data.copy_from(ir6 + (224));
              (v456_data + (v397_data * v454_data)).copy_to(ir6 + (224));
              float v462_data = s1[75];
              tensorforge::intel_esimd::simd<float, 32> v464_data;
              v464_data.copy_from(ir6 + (256));
              (v464_data + (v397_data * v462_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v469_data;
              v469_data.copy_from(s0 + (128_i32));
              float v470_data = s1[4];
              tensorforge::intel_esimd::simd<float, 32> v472_data;
              v472_data.copy_from(ir6 + (0));
              (v472_data + (v469_data * v470_data)).copy_to(ir6 + (0));
              float v478_data = s1[13];
              tensorforge::intel_esimd::simd<float, 32> v480_data;
              v480_data.copy_from(ir6 + (32));
              (v480_data + (v469_data * v478_data)).copy_to(ir6 + (32));
              float v486_data = s1[22];
              tensorforge::intel_esimd::simd<float, 32> v488_data;
              v488_data.copy_from(ir6 + (64));
              (v488_data + (v469_data * v486_data)).copy_to(ir6 + (64));
              float v494_data = s1[31];
              tensorforge::intel_esimd::simd<float, 32> v496_data;
              v496_data.copy_from(ir6 + (96));
              (v496_data + (v469_data * v494_data)).copy_to(ir6 + (96));
              float v502_data = s1[40];
              tensorforge::intel_esimd::simd<float, 32> v504_data;
              v504_data.copy_from(ir6 + (128));
              (v504_data + (v469_data * v502_data)).copy_to(ir6 + (128));
              float v510_data = s1[49];
              tensorforge::intel_esimd::simd<float, 32> v512_data;
              v512_data.copy_from(ir6 + (160));
              (v512_data + (v469_data * v510_data)).copy_to(ir6 + (160));
              float v518_data = s1[58];
              tensorforge::intel_esimd::simd<float, 32> v520_data;
              v520_data.copy_from(ir6 + (192));
              (v520_data + (v469_data * v518_data)).copy_to(ir6 + (192));
              float v526_data = s1[67];
              tensorforge::intel_esimd::simd<float, 32> v528_data;
              v528_data.copy_from(ir6 + (224));
              (v528_data + (v469_data * v526_data)).copy_to(ir6 + (224));
              float v534_data = s1[76];
              tensorforge::intel_esimd::simd<float, 32> v536_data;
              v536_data.copy_from(ir6 + (256));
              (v536_data + (v469_data * v534_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v541_data;
              v541_data.copy_from(s0 + (160_i32));
              float v542_data = s1[5];
              tensorforge::intel_esimd::simd<float, 32> v544_data;
              v544_data.copy_from(ir6 + (0));
              (v544_data + (v541_data * v542_data)).copy_to(ir6 + (0));
              float v550_data = s1[14];
              tensorforge::intel_esimd::simd<float, 32> v552_data;
              v552_data.copy_from(ir6 + (32));
              (v552_data + (v541_data * v550_data)).copy_to(ir6 + (32));
              float v558_data = s1[23];
              tensorforge::intel_esimd::simd<float, 32> v560_data;
              v560_data.copy_from(ir6 + (64));
              (v560_data + (v541_data * v558_data)).copy_to(ir6 + (64));
              float v566_data = s1[32];
              tensorforge::intel_esimd::simd<float, 32> v568_data;
              v568_data.copy_from(ir6 + (96));
              (v568_data + (v541_data * v566_data)).copy_to(ir6 + (96));
              float v574_data = s1[41];
              tensorforge::intel_esimd::simd<float, 32> v576_data;
              v576_data.copy_from(ir6 + (128));
              (v576_data + (v541_data * v574_data)).copy_to(ir6 + (128));
              float v582_data = s1[50];
              tensorforge::intel_esimd::simd<float, 32> v584_data;
              v584_data.copy_from(ir6 + (160));
              (v584_data + (v541_data * v582_data)).copy_to(ir6 + (160));
              float v590_data = s1[59];
              tensorforge::intel_esimd::simd<float, 32> v592_data;
              v592_data.copy_from(ir6 + (192));
              (v592_data + (v541_data * v590_data)).copy_to(ir6 + (192));
              float v598_data = s1[68];
              tensorforge::intel_esimd::simd<float, 32> v600_data;
              v600_data.copy_from(ir6 + (224));
              (v600_data + (v541_data * v598_data)).copy_to(ir6 + (224));
              float v606_data = s1[77];
              tensorforge::intel_esimd::simd<float, 32> v608_data;
              v608_data.copy_from(ir6 + (256));
              (v608_data + (v541_data * v606_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v613_data;
              v613_data.copy_from(s0 + (192_i32));
              float v614_data = s1[6];
              tensorforge::intel_esimd::simd<float, 32> v616_data;
              v616_data.copy_from(ir6 + (0));
              (v616_data + (v613_data * v614_data)).copy_to(ir6 + (0));
              float v622_data = s1[15];
              tensorforge::intel_esimd::simd<float, 32> v624_data;
              v624_data.copy_from(ir6 + (32));
              (v624_data + (v613_data * v622_data)).copy_to(ir6 + (32));
              float v630_data = s1[24];
              tensorforge::intel_esimd::simd<float, 32> v632_data;
              v632_data.copy_from(ir6 + (64));
              (v632_data + (v613_data * v630_data)).copy_to(ir6 + (64));
              float v638_data = s1[33];
              tensorforge::intel_esimd::simd<float, 32> v640_data;
              v640_data.copy_from(ir6 + (96));
              (v640_data + (v613_data * v638_data)).copy_to(ir6 + (96));
              float v646_data = s1[42];
              tensorforge::intel_esimd::simd<float, 32> v648_data;
              v648_data.copy_from(ir6 + (128));
              (v648_data + (v613_data * v646_data)).copy_to(ir6 + (128));
              float v654_data = s1[51];
              tensorforge::intel_esimd::simd<float, 32> v656_data;
              v656_data.copy_from(ir6 + (160));
              (v656_data + (v613_data * v654_data)).copy_to(ir6 + (160));
              float v662_data = s1[60];
              tensorforge::intel_esimd::simd<float, 32> v664_data;
              v664_data.copy_from(ir6 + (192));
              (v664_data + (v613_data * v662_data)).copy_to(ir6 + (192));
              float v670_data = s1[69];
              tensorforge::intel_esimd::simd<float, 32> v672_data;
              v672_data.copy_from(ir6 + (224));
              (v672_data + (v613_data * v670_data)).copy_to(ir6 + (224));
              float v678_data = s1[78];
              tensorforge::intel_esimd::simd<float, 32> v680_data;
              v680_data.copy_from(ir6 + (256));
              (v680_data + (v613_data * v678_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v685_data;
              v685_data.copy_from(s0 + (224_i32));
              float v686_data = s1[7];
              tensorforge::intel_esimd::simd<float, 32> v688_data;
              v688_data.copy_from(ir6 + (0));
              (v688_data + (v685_data * v686_data)).copy_to(ir6 + (0));
              float v694_data = s1[16];
              tensorforge::intel_esimd::simd<float, 32> v696_data;
              v696_data.copy_from(ir6 + (32));
              (v696_data + (v685_data * v694_data)).copy_to(ir6 + (32));
              float v702_data = s1[25];
              tensorforge::intel_esimd::simd<float, 32> v704_data;
              v704_data.copy_from(ir6 + (64));
              (v704_data + (v685_data * v702_data)).copy_to(ir6 + (64));
              float v710_data = s1[34];
              tensorforge::intel_esimd::simd<float, 32> v712_data;
              v712_data.copy_from(ir6 + (96));
              (v712_data + (v685_data * v710_data)).copy_to(ir6 + (96));
              float v718_data = s1[43];
              tensorforge::intel_esimd::simd<float, 32> v720_data;
              v720_data.copy_from(ir6 + (128));
              (v720_data + (v685_data * v718_data)).copy_to(ir6 + (128));
              float v726_data = s1[52];
              tensorforge::intel_esimd::simd<float, 32> v728_data;
              v728_data.copy_from(ir6 + (160));
              (v728_data + (v685_data * v726_data)).copy_to(ir6 + (160));
              float v734_data = s1[61];
              tensorforge::intel_esimd::simd<float, 32> v736_data;
              v736_data.copy_from(ir6 + (192));
              (v736_data + (v685_data * v734_data)).copy_to(ir6 + (192));
              float v742_data = s1[70];
              tensorforge::intel_esimd::simd<float, 32> v744_data;
              v744_data.copy_from(ir6 + (224));
              (v744_data + (v685_data * v742_data)).copy_to(ir6 + (224));
              float v750_data = s1[79];
              tensorforge::intel_esimd::simd<float, 32> v752_data;
              v752_data.copy_from(ir6 + (256));
              (v752_data + (v685_data * v750_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v757_data;
              v757_data.copy_from(s0 + (256_i32));
              float v758_data = s1[8];
              tensorforge::intel_esimd::simd<float, 32> v760_data;
              v760_data.copy_from(ir6 + (0));
              (v760_data + (v757_data * v758_data)).copy_to(ir6 + (0));
              float v766_data = s1[17];
              tensorforge::intel_esimd::simd<float, 32> v768_data;
              v768_data.copy_from(ir6 + (32));
              (v768_data + (v757_data * v766_data)).copy_to(ir6 + (32));
              float v774_data = s1[26];
              tensorforge::intel_esimd::simd<float, 32> v776_data;
              v776_data.copy_from(ir6 + (64));
              (v776_data + (v757_data * v774_data)).copy_to(ir6 + (64));
              float v782_data = s1[35];
              tensorforge::intel_esimd::simd<float, 32> v784_data;
              v784_data.copy_from(ir6 + (96));
              (v784_data + (v757_data * v782_data)).copy_to(ir6 + (96));
              float v790_data = s1[44];
              tensorforge::intel_esimd::simd<float, 32> v792_data;
              v792_data.copy_from(ir6 + (128));
              (v792_data + (v757_data * v790_data)).copy_to(ir6 + (128));
              float v798_data = s1[53];
              tensorforge::intel_esimd::simd<float, 32> v800_data;
              v800_data.copy_from(ir6 + (160));
              (v800_data + (v757_data * v798_data)).copy_to(ir6 + (160));
              float v806_data = s1[62];
              tensorforge::intel_esimd::simd<float, 32> v808_data;
              v808_data.copy_from(ir6 + (192));
              (v808_data + (v757_data * v806_data)).copy_to(ir6 + (192));
              float v814_data = s1[71];
              tensorforge::intel_esimd::simd<float, 32> v816_data;
              v816_data.copy_from(ir6 + (224));
              (v816_data + (v757_data * v814_data)).copy_to(ir6 + (224));
              float v822_data = s1[80];
              tensorforge::intel_esimd::simd<float, 32> v824_data;
              v824_data.copy_from(ir6 + (256));
              (v824_data + (v757_data * v822_data)).copy_to(ir6 + (256));
              #pragma unroll
              for (int32_t v826_n0 = 0; v826_n0 < 1; ++v826_n0) {
                int32_t v828_a = v826_n0 * 32;
                #pragma unroll
                for (int32_t v827_n1 = 0; v827_n1 < 9; ++v827_n1) {
                  int32_t v830_a = v828_a + (v827_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v831_data;
                  v831_data.copy_from(ir6 + (v830_a));
                  v831_data.copy_to(r6 + (v830_a));
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v835_i0 = 0; v835_i0 < 1; ++v835_i0) {
                int32_t v837_a = v835_i0 * 32;
                #pragma unroll
                for (int32_t v836_i1 = 0; v836_i1 < 9; ++v836_i1) {
                  int32_t v839_a = v837_a + (v836_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v840_data;
                  v840_data.copy_from(r6 + (v839_a));
                  v840_data.copy_to(glb_m3 + (v839_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

