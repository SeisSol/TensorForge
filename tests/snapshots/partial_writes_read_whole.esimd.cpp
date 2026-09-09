// === base name ===
kernel_5f371cff4a19d8b7

// === header ===
void launcher_kernel_5f371cff4a19d8b7(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5f371cff4a19d8b7(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5f371cff4a19d8b7(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5f371cff4a19d8b7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (3072, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[96];
          float* __restrict__ s1 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
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
              for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
                int32_t v16_lead = v14_i0 * 32;
                #pragma unroll
                for (int32_t v15_i1 = 0; v15_i1 < 9; ++v15_i1) {
                  int32_t v19_a = v16_lead + (v15_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v20_data;
                  v20_data.copy_from(glb_m0 + (v19_a));
                  v20_data.copy_to(r0 + (v19_a));
                }
              }
              float r2[288]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
                tensorforge::intel_esimd::simd<float, 16> v30_data;
                v30_data.copy_from(glb_m1 + ((v25_i1 * 16)));
                v30_data.copy_to(r2 + ((v25_i1 * 32)));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[288]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v34_data;
              v34_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v35_data;
              v35_data.copy_from(r1 + (0));
              (v35_data + v34_data).copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v37_data;
              v37_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 32> v38_data;
              v38_data.copy_from(r1 + (32));
              (v38_data + v37_data).copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(r1 + (64));
              (v41_data + v40_data).copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v43_data;
              v43_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(r1 + (96));
              (v44_data + v43_data).copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 32> v46_data;
              v46_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r1 + (128));
              (v47_data + v46_data).copy_to(r1 + (128));
              tensorforge::intel_esimd::simd<float, 32> v49_data;
              v49_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r1 + (160));
              (v50_data + v49_data).copy_to(r1 + (160));
              tensorforge::intel_esimd::simd<float, 32> v52_data;
              v52_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(r1 + (192));
              (v53_data + v52_data).copy_to(r1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v55_data;
              v55_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 32> v56_data;
              v56_data.copy_from(r1 + (224));
              (v56_data + v55_data).copy_to(r1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v58_data;
              v58_data.copy_from(r0 + (256));
              tensorforge::intel_esimd::simd<float, 32> v59_data;
              v59_data.copy_from(r1 + (256));
              (v59_data + v58_data).copy_to(r1 + (256));
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v61_i0 = 0; v61_i0 < 1; ++v61_i0) {
                int32_t v63_a = v61_i0 * 32;
                #pragma unroll
                for (int32_t v62_i1 = 0; v62_i1 < 9; ++v62_i1) {
                  int32_t v65_a = v63_a + (v62_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v66_data;
                  v66_data.copy_from(r1 + (v65_a));
                  v66_data.copy_to(s0 + (v65_a));
                }
              }
              float r4[288]{};
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v72_i1 = 0; v72_i1 < 9; ++v72_i1) {
                tensorforge::intel_esimd::simd<float, 16> v77_data;
                v77_data.copy_from(glb_m2 + ((v72_i1 * 16)));
                v77_data.copy_to(r4 + ((v72_i1 * 32)));
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[288]{};
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir3[288]{};
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(ir3 + (0));
              (v83_data + v82_data).copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v85_data;
              v85_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(ir3 + (32));
              (v86_data + v85_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v88_data;
              v88_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(ir3 + (64));
              (v89_data + v88_data).copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v91_data;
              v91_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(ir3 + (96));
              (v92_data + v91_data).copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v94_data;
              v94_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v95_data;
              v95_data.copy_from(ir3 + (128));
              (v95_data + v94_data).copy_to(ir3 + (128));
              tensorforge::intel_esimd::simd<float, 16> v97_data;
              v97_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v98_data;
              v98_data.copy_from(ir3 + (160));
              (v98_data + v97_data).copy_to(ir3 + (160));
              tensorforge::intel_esimd::simd<float, 16> v100_data;
              v100_data.copy_from(r2 + (192));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(ir3 + (192));
              (v101_data + v100_data).copy_to(ir3 + (192));
              tensorforge::intel_esimd::simd<float, 16> v103_data;
              v103_data.copy_from(r2 + (224));
              tensorforge::intel_esimd::simd<float, 16> v104_data;
              v104_data.copy_from(ir3 + (224));
              (v104_data + v103_data).copy_to(ir3 + (224));
              tensorforge::intel_esimd::simd<float, 16> v106_data;
              v106_data.copy_from(r2 + (256));
              tensorforge::intel_esimd::simd<float, 16> v107_data;
              v107_data.copy_from(ir3 + (256));
              (v107_data + v106_data).copy_to(ir3 + (256));
              #pragma unroll
              for (int32_t v109_n1 = 0; v109_n1 < 9; ++v109_n1) {
                int32_t v110_a = v109_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v112_data;
                v112_data.copy_from(ir3 + (v110_a));
                tensorforge::intel_esimd::simd<float, 16> v117_data;
                v117_data.copy_from(s0 + (v110_a));
                (v117_data + v112_data).copy_to(r3 + (v110_a));
              }
              // s0 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v121_i1 = 0; v121_i1 < 9; ++v121_i1) {
                int32_t v122_a = v121_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v124_data;
                v124_data.copy_from(r3 + (v122_a));
                v124_data.copy_to(s0 + (v122_a));
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[288]{};
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir5[288]{};
              tensorforge::intel_esimd::simd<float, 16> v131_data;
              v131_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v132_data;
              v132_data.copy_from(ir5 + (0));
              (v132_data + v131_data).copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v134_data;
              v134_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v135_data;
              v135_data.copy_from(ir5 + (32));
              (v135_data + v134_data).copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v137_data;
              v137_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v138_data;
              v138_data.copy_from(ir5 + (64));
              (v138_data + v137_data).copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v140_data;
              v140_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v141_data;
              v141_data.copy_from(ir5 + (96));
              (v141_data + v140_data).copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v143_data;
              v143_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v144_data;
              v144_data.copy_from(ir5 + (128));
              (v144_data + v143_data).copy_to(ir5 + (128));
              tensorforge::intel_esimd::simd<float, 16> v146_data;
              v146_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v147_data;
              v147_data.copy_from(ir5 + (160));
              (v147_data + v146_data).copy_to(ir5 + (160));
              tensorforge::intel_esimd::simd<float, 16> v149_data;
              v149_data.copy_from(r4 + (192));
              tensorforge::intel_esimd::simd<float, 16> v150_data;
              v150_data.copy_from(ir5 + (192));
              (v150_data + v149_data).copy_to(ir5 + (192));
              tensorforge::intel_esimd::simd<float, 16> v152_data;
              v152_data.copy_from(r4 + (224));
              tensorforge::intel_esimd::simd<float, 16> v153_data;
              v153_data.copy_from(ir5 + (224));
              (v153_data + v152_data).copy_to(ir5 + (224));
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(r4 + (256));
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(ir5 + (256));
              (v156_data + v155_data).copy_to(ir5 + (256));
              #pragma unroll
              for (int32_t v158_n1 = 0; v158_n1 < 9; ++v158_n1) {
                int32_t v159_a = v158_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v161_data;
                v161_data.copy_from(ir5 + (v159_a));
                tensorforge::intel_esimd::simd<float, 16> v166_data;
                v166_data.copy_from(s0 + (v159_a));
                (v166_data + v161_data).copy_to(r5 + (v159_a));
              }
              // s0 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v170_i1 = 0; v170_i1 < 9; ++v170_i1) {
                int32_t v171_a = v170_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v173_data;
                v173_data.copy_from(r5 + (v171_a));
                v173_data.copy_to(s0 + (v171_a));
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v178_ld;
              v178_ld.copy_from(glb_m4 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v178_ld.copy_to(s1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 17) {
                tensorforge::intel_esimd::simd<float, 32> v179_ld;
                v179_ld.copy_from(glb_m4 + (0 + 0 + 1 * item.get_local_id(0) + 64));
                v179_ld.copy_to(s1 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r6[288]{};
              // r6 = +(s0 * s1) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir6[288]{};
              tensorforge::intel_esimd::simd<float, 32> v185_data;
              v185_data.copy_from(s0 + (0_i32));
              float v186_data = s1[0];
              tensorforge::intel_esimd::simd<float, 32> v188_data;
              v188_data.copy_from(ir6 + (0));
              (v188_data + (v185_data * v186_data)).copy_to(ir6 + (0));
              float v194_data = s1[9];
              tensorforge::intel_esimd::simd<float, 32> v196_data;
              v196_data.copy_from(ir6 + (32));
              (v196_data + (v185_data * v194_data)).copy_to(ir6 + (32));
              float v202_data = s1[18];
              tensorforge::intel_esimd::simd<float, 32> v204_data;
              v204_data.copy_from(ir6 + (64));
              (v204_data + (v185_data * v202_data)).copy_to(ir6 + (64));
              float v210_data = s1[27];
              tensorforge::intel_esimd::simd<float, 32> v212_data;
              v212_data.copy_from(ir6 + (96));
              (v212_data + (v185_data * v210_data)).copy_to(ir6 + (96));
              float v218_data = s1[36];
              tensorforge::intel_esimd::simd<float, 32> v220_data;
              v220_data.copy_from(ir6 + (128));
              (v220_data + (v185_data * v218_data)).copy_to(ir6 + (128));
              float v226_data = s1[45];
              tensorforge::intel_esimd::simd<float, 32> v228_data;
              v228_data.copy_from(ir6 + (160));
              (v228_data + (v185_data * v226_data)).copy_to(ir6 + (160));
              float v234_data = s1[54];
              tensorforge::intel_esimd::simd<float, 32> v236_data;
              v236_data.copy_from(ir6 + (192));
              (v236_data + (v185_data * v234_data)).copy_to(ir6 + (192));
              float v242_data = s1[63];
              tensorforge::intel_esimd::simd<float, 32> v244_data;
              v244_data.copy_from(ir6 + (224));
              (v244_data + (v185_data * v242_data)).copy_to(ir6 + (224));
              float v250_data = s1[72];
              tensorforge::intel_esimd::simd<float, 32> v252_data;
              v252_data.copy_from(ir6 + (256));
              (v252_data + (v185_data * v250_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v257_data;
              v257_data.copy_from(s0 + (32_i32));
              float v258_data = s1[1];
              tensorforge::intel_esimd::simd<float, 32> v260_data;
              v260_data.copy_from(ir6 + (0));
              (v260_data + (v257_data * v258_data)).copy_to(ir6 + (0));
              float v266_data = s1[10];
              tensorforge::intel_esimd::simd<float, 32> v268_data;
              v268_data.copy_from(ir6 + (32));
              (v268_data + (v257_data * v266_data)).copy_to(ir6 + (32));
              float v274_data = s1[19];
              tensorforge::intel_esimd::simd<float, 32> v276_data;
              v276_data.copy_from(ir6 + (64));
              (v276_data + (v257_data * v274_data)).copy_to(ir6 + (64));
              float v282_data = s1[28];
              tensorforge::intel_esimd::simd<float, 32> v284_data;
              v284_data.copy_from(ir6 + (96));
              (v284_data + (v257_data * v282_data)).copy_to(ir6 + (96));
              float v290_data = s1[37];
              tensorforge::intel_esimd::simd<float, 32> v292_data;
              v292_data.copy_from(ir6 + (128));
              (v292_data + (v257_data * v290_data)).copy_to(ir6 + (128));
              float v298_data = s1[46];
              tensorforge::intel_esimd::simd<float, 32> v300_data;
              v300_data.copy_from(ir6 + (160));
              (v300_data + (v257_data * v298_data)).copy_to(ir6 + (160));
              float v306_data = s1[55];
              tensorforge::intel_esimd::simd<float, 32> v308_data;
              v308_data.copy_from(ir6 + (192));
              (v308_data + (v257_data * v306_data)).copy_to(ir6 + (192));
              float v314_data = s1[64];
              tensorforge::intel_esimd::simd<float, 32> v316_data;
              v316_data.copy_from(ir6 + (224));
              (v316_data + (v257_data * v314_data)).copy_to(ir6 + (224));
              float v322_data = s1[73];
              tensorforge::intel_esimd::simd<float, 32> v324_data;
              v324_data.copy_from(ir6 + (256));
              (v324_data + (v257_data * v322_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v329_data;
              v329_data.copy_from(s0 + (64_i32));
              float v330_data = s1[2];
              tensorforge::intel_esimd::simd<float, 32> v332_data;
              v332_data.copy_from(ir6 + (0));
              (v332_data + (v329_data * v330_data)).copy_to(ir6 + (0));
              float v338_data = s1[11];
              tensorforge::intel_esimd::simd<float, 32> v340_data;
              v340_data.copy_from(ir6 + (32));
              (v340_data + (v329_data * v338_data)).copy_to(ir6 + (32));
              float v346_data = s1[20];
              tensorforge::intel_esimd::simd<float, 32> v348_data;
              v348_data.copy_from(ir6 + (64));
              (v348_data + (v329_data * v346_data)).copy_to(ir6 + (64));
              float v354_data = s1[29];
              tensorforge::intel_esimd::simd<float, 32> v356_data;
              v356_data.copy_from(ir6 + (96));
              (v356_data + (v329_data * v354_data)).copy_to(ir6 + (96));
              float v362_data = s1[38];
              tensorforge::intel_esimd::simd<float, 32> v364_data;
              v364_data.copy_from(ir6 + (128));
              (v364_data + (v329_data * v362_data)).copy_to(ir6 + (128));
              float v370_data = s1[47];
              tensorforge::intel_esimd::simd<float, 32> v372_data;
              v372_data.copy_from(ir6 + (160));
              (v372_data + (v329_data * v370_data)).copy_to(ir6 + (160));
              float v378_data = s1[56];
              tensorforge::intel_esimd::simd<float, 32> v380_data;
              v380_data.copy_from(ir6 + (192));
              (v380_data + (v329_data * v378_data)).copy_to(ir6 + (192));
              float v386_data = s1[65];
              tensorforge::intel_esimd::simd<float, 32> v388_data;
              v388_data.copy_from(ir6 + (224));
              (v388_data + (v329_data * v386_data)).copy_to(ir6 + (224));
              float v394_data = s1[74];
              tensorforge::intel_esimd::simd<float, 32> v396_data;
              v396_data.copy_from(ir6 + (256));
              (v396_data + (v329_data * v394_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v401_data;
              v401_data.copy_from(s0 + (96_i32));
              float v402_data = s1[3];
              tensorforge::intel_esimd::simd<float, 32> v404_data;
              v404_data.copy_from(ir6 + (0));
              (v404_data + (v401_data * v402_data)).copy_to(ir6 + (0));
              float v410_data = s1[12];
              tensorforge::intel_esimd::simd<float, 32> v412_data;
              v412_data.copy_from(ir6 + (32));
              (v412_data + (v401_data * v410_data)).copy_to(ir6 + (32));
              float v418_data = s1[21];
              tensorforge::intel_esimd::simd<float, 32> v420_data;
              v420_data.copy_from(ir6 + (64));
              (v420_data + (v401_data * v418_data)).copy_to(ir6 + (64));
              float v426_data = s1[30];
              tensorforge::intel_esimd::simd<float, 32> v428_data;
              v428_data.copy_from(ir6 + (96));
              (v428_data + (v401_data * v426_data)).copy_to(ir6 + (96));
              float v434_data = s1[39];
              tensorforge::intel_esimd::simd<float, 32> v436_data;
              v436_data.copy_from(ir6 + (128));
              (v436_data + (v401_data * v434_data)).copy_to(ir6 + (128));
              float v442_data = s1[48];
              tensorforge::intel_esimd::simd<float, 32> v444_data;
              v444_data.copy_from(ir6 + (160));
              (v444_data + (v401_data * v442_data)).copy_to(ir6 + (160));
              float v450_data = s1[57];
              tensorforge::intel_esimd::simd<float, 32> v452_data;
              v452_data.copy_from(ir6 + (192));
              (v452_data + (v401_data * v450_data)).copy_to(ir6 + (192));
              float v458_data = s1[66];
              tensorforge::intel_esimd::simd<float, 32> v460_data;
              v460_data.copy_from(ir6 + (224));
              (v460_data + (v401_data * v458_data)).copy_to(ir6 + (224));
              float v466_data = s1[75];
              tensorforge::intel_esimd::simd<float, 32> v468_data;
              v468_data.copy_from(ir6 + (256));
              (v468_data + (v401_data * v466_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v473_data;
              v473_data.copy_from(s0 + (128_i32));
              float v474_data = s1[4];
              tensorforge::intel_esimd::simd<float, 32> v476_data;
              v476_data.copy_from(ir6 + (0));
              (v476_data + (v473_data * v474_data)).copy_to(ir6 + (0));
              float v482_data = s1[13];
              tensorforge::intel_esimd::simd<float, 32> v484_data;
              v484_data.copy_from(ir6 + (32));
              (v484_data + (v473_data * v482_data)).copy_to(ir6 + (32));
              float v490_data = s1[22];
              tensorforge::intel_esimd::simd<float, 32> v492_data;
              v492_data.copy_from(ir6 + (64));
              (v492_data + (v473_data * v490_data)).copy_to(ir6 + (64));
              float v498_data = s1[31];
              tensorforge::intel_esimd::simd<float, 32> v500_data;
              v500_data.copy_from(ir6 + (96));
              (v500_data + (v473_data * v498_data)).copy_to(ir6 + (96));
              float v506_data = s1[40];
              tensorforge::intel_esimd::simd<float, 32> v508_data;
              v508_data.copy_from(ir6 + (128));
              (v508_data + (v473_data * v506_data)).copy_to(ir6 + (128));
              float v514_data = s1[49];
              tensorforge::intel_esimd::simd<float, 32> v516_data;
              v516_data.copy_from(ir6 + (160));
              (v516_data + (v473_data * v514_data)).copy_to(ir6 + (160));
              float v522_data = s1[58];
              tensorforge::intel_esimd::simd<float, 32> v524_data;
              v524_data.copy_from(ir6 + (192));
              (v524_data + (v473_data * v522_data)).copy_to(ir6 + (192));
              float v530_data = s1[67];
              tensorforge::intel_esimd::simd<float, 32> v532_data;
              v532_data.copy_from(ir6 + (224));
              (v532_data + (v473_data * v530_data)).copy_to(ir6 + (224));
              float v538_data = s1[76];
              tensorforge::intel_esimd::simd<float, 32> v540_data;
              v540_data.copy_from(ir6 + (256));
              (v540_data + (v473_data * v538_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v545_data;
              v545_data.copy_from(s0 + (160_i32));
              float v546_data = s1[5];
              tensorforge::intel_esimd::simd<float, 32> v548_data;
              v548_data.copy_from(ir6 + (0));
              (v548_data + (v545_data * v546_data)).copy_to(ir6 + (0));
              float v554_data = s1[14];
              tensorforge::intel_esimd::simd<float, 32> v556_data;
              v556_data.copy_from(ir6 + (32));
              (v556_data + (v545_data * v554_data)).copy_to(ir6 + (32));
              float v562_data = s1[23];
              tensorforge::intel_esimd::simd<float, 32> v564_data;
              v564_data.copy_from(ir6 + (64));
              (v564_data + (v545_data * v562_data)).copy_to(ir6 + (64));
              float v570_data = s1[32];
              tensorforge::intel_esimd::simd<float, 32> v572_data;
              v572_data.copy_from(ir6 + (96));
              (v572_data + (v545_data * v570_data)).copy_to(ir6 + (96));
              float v578_data = s1[41];
              tensorforge::intel_esimd::simd<float, 32> v580_data;
              v580_data.copy_from(ir6 + (128));
              (v580_data + (v545_data * v578_data)).copy_to(ir6 + (128));
              float v586_data = s1[50];
              tensorforge::intel_esimd::simd<float, 32> v588_data;
              v588_data.copy_from(ir6 + (160));
              (v588_data + (v545_data * v586_data)).copy_to(ir6 + (160));
              float v594_data = s1[59];
              tensorforge::intel_esimd::simd<float, 32> v596_data;
              v596_data.copy_from(ir6 + (192));
              (v596_data + (v545_data * v594_data)).copy_to(ir6 + (192));
              float v602_data = s1[68];
              tensorforge::intel_esimd::simd<float, 32> v604_data;
              v604_data.copy_from(ir6 + (224));
              (v604_data + (v545_data * v602_data)).copy_to(ir6 + (224));
              float v610_data = s1[77];
              tensorforge::intel_esimd::simd<float, 32> v612_data;
              v612_data.copy_from(ir6 + (256));
              (v612_data + (v545_data * v610_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v617_data;
              v617_data.copy_from(s0 + (192_i32));
              float v618_data = s1[6];
              tensorforge::intel_esimd::simd<float, 32> v620_data;
              v620_data.copy_from(ir6 + (0));
              (v620_data + (v617_data * v618_data)).copy_to(ir6 + (0));
              float v626_data = s1[15];
              tensorforge::intel_esimd::simd<float, 32> v628_data;
              v628_data.copy_from(ir6 + (32));
              (v628_data + (v617_data * v626_data)).copy_to(ir6 + (32));
              float v634_data = s1[24];
              tensorforge::intel_esimd::simd<float, 32> v636_data;
              v636_data.copy_from(ir6 + (64));
              (v636_data + (v617_data * v634_data)).copy_to(ir6 + (64));
              float v642_data = s1[33];
              tensorforge::intel_esimd::simd<float, 32> v644_data;
              v644_data.copy_from(ir6 + (96));
              (v644_data + (v617_data * v642_data)).copy_to(ir6 + (96));
              float v650_data = s1[42];
              tensorforge::intel_esimd::simd<float, 32> v652_data;
              v652_data.copy_from(ir6 + (128));
              (v652_data + (v617_data * v650_data)).copy_to(ir6 + (128));
              float v658_data = s1[51];
              tensorforge::intel_esimd::simd<float, 32> v660_data;
              v660_data.copy_from(ir6 + (160));
              (v660_data + (v617_data * v658_data)).copy_to(ir6 + (160));
              float v666_data = s1[60];
              tensorforge::intel_esimd::simd<float, 32> v668_data;
              v668_data.copy_from(ir6 + (192));
              (v668_data + (v617_data * v666_data)).copy_to(ir6 + (192));
              float v674_data = s1[69];
              tensorforge::intel_esimd::simd<float, 32> v676_data;
              v676_data.copy_from(ir6 + (224));
              (v676_data + (v617_data * v674_data)).copy_to(ir6 + (224));
              float v682_data = s1[78];
              tensorforge::intel_esimd::simd<float, 32> v684_data;
              v684_data.copy_from(ir6 + (256));
              (v684_data + (v617_data * v682_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v689_data;
              v689_data.copy_from(s0 + (224_i32));
              float v690_data = s1[7];
              tensorforge::intel_esimd::simd<float, 32> v692_data;
              v692_data.copy_from(ir6 + (0));
              (v692_data + (v689_data * v690_data)).copy_to(ir6 + (0));
              float v698_data = s1[16];
              tensorforge::intel_esimd::simd<float, 32> v700_data;
              v700_data.copy_from(ir6 + (32));
              (v700_data + (v689_data * v698_data)).copy_to(ir6 + (32));
              float v706_data = s1[25];
              tensorforge::intel_esimd::simd<float, 32> v708_data;
              v708_data.copy_from(ir6 + (64));
              (v708_data + (v689_data * v706_data)).copy_to(ir6 + (64));
              float v714_data = s1[34];
              tensorforge::intel_esimd::simd<float, 32> v716_data;
              v716_data.copy_from(ir6 + (96));
              (v716_data + (v689_data * v714_data)).copy_to(ir6 + (96));
              float v722_data = s1[43];
              tensorforge::intel_esimd::simd<float, 32> v724_data;
              v724_data.copy_from(ir6 + (128));
              (v724_data + (v689_data * v722_data)).copy_to(ir6 + (128));
              float v730_data = s1[52];
              tensorforge::intel_esimd::simd<float, 32> v732_data;
              v732_data.copy_from(ir6 + (160));
              (v732_data + (v689_data * v730_data)).copy_to(ir6 + (160));
              float v738_data = s1[61];
              tensorforge::intel_esimd::simd<float, 32> v740_data;
              v740_data.copy_from(ir6 + (192));
              (v740_data + (v689_data * v738_data)).copy_to(ir6 + (192));
              float v746_data = s1[70];
              tensorforge::intel_esimd::simd<float, 32> v748_data;
              v748_data.copy_from(ir6 + (224));
              (v748_data + (v689_data * v746_data)).copy_to(ir6 + (224));
              float v754_data = s1[79];
              tensorforge::intel_esimd::simd<float, 32> v756_data;
              v756_data.copy_from(ir6 + (256));
              (v756_data + (v689_data * v754_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v761_data;
              v761_data.copy_from(s0 + (256_i32));
              float v762_data = s1[8];
              tensorforge::intel_esimd::simd<float, 32> v764_data;
              v764_data.copy_from(ir6 + (0));
              (v764_data + (v761_data * v762_data)).copy_to(ir6 + (0));
              float v770_data = s1[17];
              tensorforge::intel_esimd::simd<float, 32> v772_data;
              v772_data.copy_from(ir6 + (32));
              (v772_data + (v761_data * v770_data)).copy_to(ir6 + (32));
              float v778_data = s1[26];
              tensorforge::intel_esimd::simd<float, 32> v780_data;
              v780_data.copy_from(ir6 + (64));
              (v780_data + (v761_data * v778_data)).copy_to(ir6 + (64));
              float v786_data = s1[35];
              tensorforge::intel_esimd::simd<float, 32> v788_data;
              v788_data.copy_from(ir6 + (96));
              (v788_data + (v761_data * v786_data)).copy_to(ir6 + (96));
              float v794_data = s1[44];
              tensorforge::intel_esimd::simd<float, 32> v796_data;
              v796_data.copy_from(ir6 + (128));
              (v796_data + (v761_data * v794_data)).copy_to(ir6 + (128));
              float v802_data = s1[53];
              tensorforge::intel_esimd::simd<float, 32> v804_data;
              v804_data.copy_from(ir6 + (160));
              (v804_data + (v761_data * v802_data)).copy_to(ir6 + (160));
              float v810_data = s1[62];
              tensorforge::intel_esimd::simd<float, 32> v812_data;
              v812_data.copy_from(ir6 + (192));
              (v812_data + (v761_data * v810_data)).copy_to(ir6 + (192));
              float v818_data = s1[71];
              tensorforge::intel_esimd::simd<float, 32> v820_data;
              v820_data.copy_from(ir6 + (224));
              (v820_data + (v761_data * v818_data)).copy_to(ir6 + (224));
              float v826_data = s1[80];
              tensorforge::intel_esimd::simd<float, 32> v828_data;
              v828_data.copy_from(ir6 + (256));
              (v828_data + (v761_data * v826_data)).copy_to(ir6 + (256));
              #pragma unroll
              for (int32_t v830_n0 = 0; v830_n0 < 1; ++v830_n0) {
                int32_t v832_a = v830_n0 * 32;
                #pragma unroll
                for (int32_t v831_n1 = 0; v831_n1 < 9; ++v831_n1) {
                  int32_t v834_a = v832_a + (v831_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v835_data;
                  v835_data.copy_from(ir6 + (v834_a));
                  v835_data.copy_to(r6 + (v834_a));
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v839_i0 = 0; v839_i0 < 1; ++v839_i0) {
                int32_t v841_a = v839_i0 * 32;
                #pragma unroll
                for (int32_t v840_i1 = 0; v840_i1 < 9; ++v840_i1) {
                  int32_t v843_a = v841_a + (v840_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v844_data;
                  v844_data.copy_from(r6 + (v843_a));
                  v844_data.copy_to(glb_m3 + (v843_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

