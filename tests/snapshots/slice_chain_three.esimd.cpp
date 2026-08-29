// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08703cce1d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08703cce1d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1536, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float r0[96]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v7_i1 = 0; v7_i1 < 6; ++v7_i1) {
                tensorforge::intel_esimd::simd<float, 12> v12_data;
                v12_data.copy_from(glb_m0 + ((v7_i1 * 12)));
                v12_data.copy_to(r0 + ((v7_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m1[0 + 0 + 2 * item.get_local_id(0) + 0];
              if (item.get_local_id(0) < 4) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 32] = glb_m1[0 + 0 + 1 * item.get_local_id(0) + 32];
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 12; ++v17_i1) {
                tensorforge::intel_esimd::simd<float, 12> v22_data;
                v22_data.copy_from(glb_m3 + ((v17_i1 * 12)));
                v22_data.copy_to(r2 + ((v17_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[96]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd<float, 12> v26_data;
              v26_data.copy_from(r0 + (0));
              float v27_data = s0[0];
              tensorforge::intel_esimd::simd<float, 12> v29_data;
              v29_data.copy_from(r1 + (0));
              (v29_data + (v26_data * v27_data)).copy_to(r1 + (0));
              float v32_data = s0[7];
              tensorforge::intel_esimd::simd<float, 12> v34_data;
              v34_data.copy_from(r1 + (16));
              (v34_data + (v26_data * v32_data)).copy_to(r1 + (16));
              float v37_data = s0[15];
              tensorforge::intel_esimd::simd<float, 12> v39_data;
              v39_data.copy_from(r1 + (32));
              (v39_data + (v26_data * v37_data)).copy_to(r1 + (32));
              float v42_data = s0[18];
              tensorforge::intel_esimd::simd<float, 12> v44_data;
              v44_data.copy_from(r1 + (48));
              (v44_data + (v26_data * v42_data)).copy_to(r1 + (48));
              float v47_data = s0[26];
              tensorforge::intel_esimd::simd<float, 12> v49_data;
              v49_data.copy_from(r1 + (64));
              (v49_data + (v26_data * v47_data)).copy_to(r1 + (64));
              float v52_data = s0[29];
              tensorforge::intel_esimd::simd<float, 12> v54_data;
              v54_data.copy_from(r1 + (80));
              (v54_data + (v26_data * v52_data)).copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 12> v56_data;
              v56_data.copy_from(r0 + (16));
              float v57_data = s0[1];
              tensorforge::intel_esimd::simd<float, 12> v59_data;
              v59_data.copy_from(r1 + (0));
              (v59_data + (v56_data * v57_data)).copy_to(r1 + (0));
              float v62_data = s0[6];
              tensorforge::intel_esimd::simd<float, 12> v64_data;
              v64_data.copy_from(r1 + (16));
              (v64_data + (v56_data * v62_data)).copy_to(r1 + (16));
              float v67_data = s0[14];
              tensorforge::intel_esimd::simd<float, 12> v69_data;
              v69_data.copy_from(r1 + (32));
              (v69_data + (v56_data * v67_data)).copy_to(r1 + (32));
              float v72_data = s0[19];
              tensorforge::intel_esimd::simd<float, 12> v74_data;
              v74_data.copy_from(r1 + (48));
              (v74_data + (v56_data * v72_data)).copy_to(r1 + (48));
              float v77_data = s0[27];
              tensorforge::intel_esimd::simd<float, 12> v79_data;
              v79_data.copy_from(r1 + (64));
              (v79_data + (v56_data * v77_data)).copy_to(r1 + (64));
              float v82_data = s0[28];
              tensorforge::intel_esimd::simd<float, 12> v84_data;
              v84_data.copy_from(r1 + (80));
              (v84_data + (v56_data * v82_data)).copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 12> v86_data;
              v86_data.copy_from(r0 + (32));
              float v87_data = s0[2];
              tensorforge::intel_esimd::simd<float, 12> v89_data;
              v89_data.copy_from(r1 + (0));
              (v89_data + (v86_data * v87_data)).copy_to(r1 + (0));
              float v92_data = s0[10];
              tensorforge::intel_esimd::simd<float, 12> v94_data;
              v94_data.copy_from(r1 + (16));
              (v94_data + (v86_data * v92_data)).copy_to(r1 + (16));
              float v97_data = s0[13];
              tensorforge::intel_esimd::simd<float, 12> v99_data;
              v99_data.copy_from(r1 + (32));
              (v99_data + (v86_data * v97_data)).copy_to(r1 + (32));
              float v102_data = s0[21];
              tensorforge::intel_esimd::simd<float, 12> v104_data;
              v104_data.copy_from(r1 + (48));
              (v104_data + (v86_data * v102_data)).copy_to(r1 + (48));
              float v107_data = s0[24];
              tensorforge::intel_esimd::simd<float, 12> v109_data;
              v109_data.copy_from(r1 + (64));
              (v109_data + (v86_data * v107_data)).copy_to(r1 + (64));
              float v112_data = s0[32];
              tensorforge::intel_esimd::simd<float, 12> v114_data;
              v114_data.copy_from(r1 + (80));
              (v114_data + (v86_data * v112_data)).copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 12> v116_data;
              v116_data.copy_from(r0 + (48));
              float v117_data = s0[3];
              tensorforge::intel_esimd::simd<float, 12> v119_data;
              v119_data.copy_from(r1 + (0));
              (v119_data + (v116_data * v117_data)).copy_to(r1 + (0));
              float v122_data = s0[11];
              tensorforge::intel_esimd::simd<float, 12> v124_data;
              v124_data.copy_from(r1 + (16));
              (v124_data + (v116_data * v122_data)).copy_to(r1 + (16));
              float v127_data = s0[12];
              tensorforge::intel_esimd::simd<float, 12> v129_data;
              v129_data.copy_from(r1 + (32));
              (v129_data + (v116_data * v127_data)).copy_to(r1 + (32));
              float v132_data = s0[20];
              tensorforge::intel_esimd::simd<float, 12> v134_data;
              v134_data.copy_from(r1 + (48));
              (v134_data + (v116_data * v132_data)).copy_to(r1 + (48));
              float v137_data = s0[25];
              tensorforge::intel_esimd::simd<float, 12> v139_data;
              v139_data.copy_from(r1 + (64));
              (v139_data + (v116_data * v137_data)).copy_to(r1 + (64));
              float v142_data = s0[33];
              tensorforge::intel_esimd::simd<float, 12> v144_data;
              v144_data.copy_from(r1 + (80));
              (v144_data + (v116_data * v142_data)).copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 12> v146_data;
              v146_data.copy_from(r0 + (64));
              float v147_data = s0[5];
              tensorforge::intel_esimd::simd<float, 12> v149_data;
              v149_data.copy_from(r1 + (0));
              (v149_data + (v146_data * v147_data)).copy_to(r1 + (0));
              float v152_data = s0[8];
              tensorforge::intel_esimd::simd<float, 12> v154_data;
              v154_data.copy_from(r1 + (16));
              (v154_data + (v146_data * v152_data)).copy_to(r1 + (16));
              float v157_data = s0[16];
              tensorforge::intel_esimd::simd<float, 12> v159_data;
              v159_data.copy_from(r1 + (32));
              (v159_data + (v146_data * v157_data)).copy_to(r1 + (32));
              float v162_data = s0[23];
              tensorforge::intel_esimd::simd<float, 12> v164_data;
              v164_data.copy_from(r1 + (48));
              (v164_data + (v146_data * v162_data)).copy_to(r1 + (48));
              float v167_data = s0[31];
              tensorforge::intel_esimd::simd<float, 12> v169_data;
              v169_data.copy_from(r1 + (64));
              (v169_data + (v146_data * v167_data)).copy_to(r1 + (64));
              float v172_data = s0[34];
              tensorforge::intel_esimd::simd<float, 12> v174_data;
              v174_data.copy_from(r1 + (80));
              (v174_data + (v146_data * v172_data)).copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 12> v176_data;
              v176_data.copy_from(r0 + (80));
              float v177_data = s0[4];
              tensorforge::intel_esimd::simd<float, 12> v179_data;
              v179_data.copy_from(r1 + (0));
              (v179_data + (v176_data * v177_data)).copy_to(r1 + (0));
              float v182_data = s0[9];
              tensorforge::intel_esimd::simd<float, 12> v184_data;
              v184_data.copy_from(r1 + (16));
              (v184_data + (v176_data * v182_data)).copy_to(r1 + (16));
              float v187_data = s0[17];
              tensorforge::intel_esimd::simd<float, 12> v189_data;
              v189_data.copy_from(r1 + (32));
              (v189_data + (v176_data * v187_data)).copy_to(r1 + (32));
              float v192_data = s0[22];
              tensorforge::intel_esimd::simd<float, 12> v194_data;
              v194_data.copy_from(r1 + (48));
              (v194_data + (v176_data * v192_data)).copy_to(r1 + (48));
              float v197_data = s0[30];
              tensorforge::intel_esimd::simd<float, 12> v199_data;
              v199_data.copy_from(r1 + (64));
              (v199_data + (v176_data * v197_data)).copy_to(r1 + (64));
              float v202_data = s0[35];
              tensorforge::intel_esimd::simd<float, 12> v204_data;
              v204_data.copy_from(r1 + (80));
              (v204_data + (v176_data * v202_data)).copy_to(r1 + (80));
              // wait(r2 = load{g>r}(glb_m3););
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v207_i1 = 0; v207_i1 < 6; ++v207_i1) {
                tensorforge::intel_esimd::simd<float, 12> v210_data;
                v210_data.copy_from(r1 + ((v207_i1 * 16)));
                int32_t v213_a = v207_i1 * 12;
                v210_data.copy_to(s1 + ((v213_a ^ ((v213_a >> 3) & 7))));
              }
              float r3[96]{};
              // r3 = +(r2 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir3[96]{};
              tensorforge::intel_esimd::simd<float, 12> v220_data;
              v220_data.copy_from(r2 + (0));
              float v221_data = s1[0];
              tensorforge::intel_esimd::simd<float, 12> v223_data;
              v223_data.copy_from(ir3 + (0));
              (v223_data + (v220_data * v221_data)).copy_to(ir3 + (0));
              float v226_data = s1[13];
              tensorforge::intel_esimd::simd<float, 12> v228_data;
              v228_data.copy_from(ir3 + (16));
              (v228_data + (v220_data * v226_data)).copy_to(ir3 + (16));
              float v231_data = s1[27];
              tensorforge::intel_esimd::simd<float, 12> v233_data;
              v233_data.copy_from(ir3 + (32));
              (v233_data + (v220_data * v231_data)).copy_to(ir3 + (32));
              float v236_data = s1[32];
              tensorforge::intel_esimd::simd<float, 12> v238_data;
              v238_data.copy_from(ir3 + (48));
              (v238_data + (v220_data * v236_data)).copy_to(ir3 + (48));
              float v241_data = s1[54];
              tensorforge::intel_esimd::simd<float, 12> v243_data;
              v243_data.copy_from(ir3 + (64));
              (v243_data + (v220_data * v241_data)).copy_to(ir3 + (64));
              float v246_data = s1[59];
              tensorforge::intel_esimd::simd<float, 12> v248_data;
              v248_data.copy_from(ir3 + (80));
              (v248_data + (v220_data * v246_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v250_data;
              v250_data.copy_from(r2 + (16));
              float v251_data = s1[1];
              tensorforge::intel_esimd::simd<float, 12> v253_data;
              v253_data.copy_from(ir3 + (0));
              (v253_data + (v250_data * v251_data)).copy_to(ir3 + (0));
              float v256_data = s1[12];
              tensorforge::intel_esimd::simd<float, 12> v258_data;
              v258_data.copy_from(ir3 + (16));
              (v258_data + (v250_data * v256_data)).copy_to(ir3 + (16));
              float v261_data = s1[26];
              tensorforge::intel_esimd::simd<float, 12> v263_data;
              v263_data.copy_from(ir3 + (32));
              (v263_data + (v250_data * v261_data)).copy_to(ir3 + (32));
              float v266_data = s1[33];
              tensorforge::intel_esimd::simd<float, 12> v268_data;
              v268_data.copy_from(ir3 + (48));
              (v268_data + (v250_data * v266_data)).copy_to(ir3 + (48));
              float v271_data = s1[55];
              tensorforge::intel_esimd::simd<float, 12> v273_data;
              v273_data.copy_from(ir3 + (64));
              (v273_data + (v250_data * v271_data)).copy_to(ir3 + (64));
              float v276_data = s1[58];
              tensorforge::intel_esimd::simd<float, 12> v278_data;
              v278_data.copy_from(ir3 + (80));
              (v278_data + (v250_data * v276_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v280_data;
              v280_data.copy_from(r2 + (32));
              float v281_data = s1[2];
              tensorforge::intel_esimd::simd<float, 12> v283_data;
              v283_data.copy_from(ir3 + (0));
              (v283_data + (v280_data * v281_data)).copy_to(ir3 + (0));
              float v286_data = s1[15];
              tensorforge::intel_esimd::simd<float, 12> v288_data;
              v288_data.copy_from(ir3 + (16));
              (v288_data + (v280_data * v286_data)).copy_to(ir3 + (16));
              float v291_data = s1[25];
              tensorforge::intel_esimd::simd<float, 12> v293_data;
              v293_data.copy_from(ir3 + (32));
              (v293_data + (v280_data * v291_data)).copy_to(ir3 + (32));
              float v296_data = s1[34];
              tensorforge::intel_esimd::simd<float, 12> v298_data;
              v298_data.copy_from(ir3 + (48));
              (v298_data + (v280_data * v296_data)).copy_to(ir3 + (48));
              float v301_data = s1[52];
              tensorforge::intel_esimd::simd<float, 12> v303_data;
              v303_data.copy_from(ir3 + (64));
              (v303_data + (v280_data * v301_data)).copy_to(ir3 + (64));
              float v306_data = s1[57];
              tensorforge::intel_esimd::simd<float, 12> v308_data;
              v308_data.copy_from(ir3 + (80));
              (v308_data + (v280_data * v306_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v310_data;
              v310_data.copy_from(r2 + (48));
              float v311_data = s1[3];
              tensorforge::intel_esimd::simd<float, 12> v313_data;
              v313_data.copy_from(ir3 + (0));
              (v313_data + (v310_data * v311_data)).copy_to(ir3 + (0));
              float v316_data = s1[14];
              tensorforge::intel_esimd::simd<float, 12> v318_data;
              v318_data.copy_from(ir3 + (16));
              (v318_data + (v310_data * v316_data)).copy_to(ir3 + (16));
              float v321_data = s1[24];
              tensorforge::intel_esimd::simd<float, 12> v323_data;
              v323_data.copy_from(ir3 + (32));
              (v323_data + (v310_data * v321_data)).copy_to(ir3 + (32));
              float v326_data = s1[35];
              tensorforge::intel_esimd::simd<float, 12> v328_data;
              v328_data.copy_from(ir3 + (48));
              (v328_data + (v310_data * v326_data)).copy_to(ir3 + (48));
              float v331_data = s1[53];
              tensorforge::intel_esimd::simd<float, 12> v333_data;
              v333_data.copy_from(ir3 + (64));
              (v333_data + (v310_data * v331_data)).copy_to(ir3 + (64));
              float v336_data = s1[56];
              tensorforge::intel_esimd::simd<float, 12> v338_data;
              v338_data.copy_from(ir3 + (80));
              (v338_data + (v310_data * v336_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v340_data;
              v340_data.copy_from(r2 + (64));
              float v341_data = s1[4];
              tensorforge::intel_esimd::simd<float, 12> v343_data;
              v343_data.copy_from(ir3 + (0));
              (v343_data + (v340_data * v341_data)).copy_to(ir3 + (0));
              float v346_data = s1[18];
              tensorforge::intel_esimd::simd<float, 12> v348_data;
              v348_data.copy_from(ir3 + (16));
              (v348_data + (v340_data * v346_data)).copy_to(ir3 + (16));
              float v351_data = s1[31];
              tensorforge::intel_esimd::simd<float, 12> v353_data;
              v353_data.copy_from(ir3 + (32));
              (v353_data + (v340_data * v351_data)).copy_to(ir3 + (32));
              float v356_data = s1[45];
              tensorforge::intel_esimd::simd<float, 12> v358_data;
              v358_data.copy_from(ir3 + (48));
              (v358_data + (v340_data * v356_data)).copy_to(ir3 + (48));
              float v361_data = s1[50];
              tensorforge::intel_esimd::simd<float, 12> v363_data;
              v363_data.copy_from(ir3 + (64));
              (v363_data + (v340_data * v361_data)).copy_to(ir3 + (64));
              float v366_data = s1[64];
              tensorforge::intel_esimd::simd<float, 12> v368_data;
              v368_data.copy_from(ir3 + (80));
              (v368_data + (v340_data * v366_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v370_data;
              v370_data.copy_from(r2 + (80));
              float v371_data = s1[5];
              tensorforge::intel_esimd::simd<float, 12> v373_data;
              v373_data.copy_from(ir3 + (0));
              (v373_data + (v370_data * v371_data)).copy_to(ir3 + (0));
              float v376_data = s1[19];
              tensorforge::intel_esimd::simd<float, 12> v378_data;
              v378_data.copy_from(ir3 + (16));
              (v378_data + (v370_data * v376_data)).copy_to(ir3 + (16));
              float v381_data = s1[30];
              tensorforge::intel_esimd::simd<float, 12> v383_data;
              v383_data.copy_from(ir3 + (32));
              (v383_data + (v370_data * v381_data)).copy_to(ir3 + (32));
              float v386_data = s1[44];
              tensorforge::intel_esimd::simd<float, 12> v388_data;
              v388_data.copy_from(ir3 + (48));
              (v388_data + (v370_data * v386_data)).copy_to(ir3 + (48));
              float v391_data = s1[51];
              tensorforge::intel_esimd::simd<float, 12> v393_data;
              v393_data.copy_from(ir3 + (64));
              (v393_data + (v370_data * v391_data)).copy_to(ir3 + (64));
              float v396_data = s1[65];
              tensorforge::intel_esimd::simd<float, 12> v398_data;
              v398_data.copy_from(ir3 + (80));
              (v398_data + (v370_data * v396_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v400_data;
              v400_data.copy_from(r2 + (96));
              float v401_data = s1[6];
              tensorforge::intel_esimd::simd<float, 12> v403_data;
              v403_data.copy_from(ir3 + (0));
              (v403_data + (v400_data * v401_data)).copy_to(ir3 + (0));
              float v406_data = s1[16];
              tensorforge::intel_esimd::simd<float, 12> v408_data;
              v408_data.copy_from(ir3 + (16));
              (v408_data + (v400_data * v406_data)).copy_to(ir3 + (16));
              float v411_data = s1[29];
              tensorforge::intel_esimd::simd<float, 12> v413_data;
              v413_data.copy_from(ir3 + (32));
              (v413_data + (v400_data * v411_data)).copy_to(ir3 + (32));
              float v416_data = s1[47];
              tensorforge::intel_esimd::simd<float, 12> v418_data;
              v418_data.copy_from(ir3 + (48));
              (v418_data + (v400_data * v416_data)).copy_to(ir3 + (48));
              float v421_data = s1[48];
              tensorforge::intel_esimd::simd<float, 12> v423_data;
              v423_data.copy_from(ir3 + (64));
              (v423_data + (v400_data * v421_data)).copy_to(ir3 + (64));
              float v426_data = s1[66];
              tensorforge::intel_esimd::simd<float, 12> v428_data;
              v428_data.copy_from(ir3 + (80));
              (v428_data + (v400_data * v426_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v430_data;
              v430_data.copy_from(r2 + (112));
              float v431_data = s1[7];
              tensorforge::intel_esimd::simd<float, 12> v433_data;
              v433_data.copy_from(ir3 + (0));
              (v433_data + (v430_data * v431_data)).copy_to(ir3 + (0));
              float v436_data = s1[17];
              tensorforge::intel_esimd::simd<float, 12> v438_data;
              v438_data.copy_from(ir3 + (16));
              (v438_data + (v430_data * v436_data)).copy_to(ir3 + (16));
              float v441_data = s1[28];
              tensorforge::intel_esimd::simd<float, 12> v443_data;
              v443_data.copy_from(ir3 + (32));
              (v443_data + (v430_data * v441_data)).copy_to(ir3 + (32));
              float v446_data = s1[46];
              tensorforge::intel_esimd::simd<float, 12> v448_data;
              v448_data.copy_from(ir3 + (48));
              (v448_data + (v430_data * v446_data)).copy_to(ir3 + (48));
              float v451_data = s1[49];
              tensorforge::intel_esimd::simd<float, 12> v453_data;
              v453_data.copy_from(ir3 + (64));
              (v453_data + (v430_data * v451_data)).copy_to(ir3 + (64));
              float v456_data = s1[67];
              tensorforge::intel_esimd::simd<float, 12> v458_data;
              v458_data.copy_from(ir3 + (80));
              (v458_data + (v430_data * v456_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v460_data;
              v460_data.copy_from(r2 + (128));
              float v461_data = s1[9];
              tensorforge::intel_esimd::simd<float, 12> v463_data;
              v463_data.copy_from(ir3 + (0));
              (v463_data + (v460_data * v461_data)).copy_to(ir3 + (0));
              float v466_data = s1[22];
              tensorforge::intel_esimd::simd<float, 12> v468_data;
              v468_data.copy_from(ir3 + (16));
              (v468_data + (v460_data * v466_data)).copy_to(ir3 + (16));
              float v471_data = s1[36];
              tensorforge::intel_esimd::simd<float, 12> v473_data;
              v473_data.copy_from(ir3 + (32));
              (v473_data + (v460_data * v471_data)).copy_to(ir3 + (32));
              float v476_data = s1[41];
              tensorforge::intel_esimd::simd<float, 12> v478_data;
              v478_data.copy_from(ir3 + (48));
              (v478_data + (v460_data * v476_data)).copy_to(ir3 + (48));
              float v481_data = s1[63];
              tensorforge::intel_esimd::simd<float, 12> v483_data;
              v483_data.copy_from(ir3 + (64));
              (v483_data + (v460_data * v481_data)).copy_to(ir3 + (64));
              float v486_data = s1[68];
              tensorforge::intel_esimd::simd<float, 12> v488_data;
              v488_data.copy_from(ir3 + (80));
              (v488_data + (v460_data * v486_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v490_data;
              v490_data.copy_from(r2 + (144));
              float v491_data = s1[8];
              tensorforge::intel_esimd::simd<float, 12> v493_data;
              v493_data.copy_from(ir3 + (0));
              (v493_data + (v490_data * v491_data)).copy_to(ir3 + (0));
              float v496_data = s1[23];
              tensorforge::intel_esimd::simd<float, 12> v498_data;
              v498_data.copy_from(ir3 + (16));
              (v498_data + (v490_data * v496_data)).copy_to(ir3 + (16));
              float v501_data = s1[37];
              tensorforge::intel_esimd::simd<float, 12> v503_data;
              v503_data.copy_from(ir3 + (32));
              (v503_data + (v490_data * v501_data)).copy_to(ir3 + (32));
              float v506_data = s1[40];
              tensorforge::intel_esimd::simd<float, 12> v508_data;
              v508_data.copy_from(ir3 + (48));
              (v508_data + (v490_data * v506_data)).copy_to(ir3 + (48));
              float v511_data = s1[62];
              tensorforge::intel_esimd::simd<float, 12> v513_data;
              v513_data.copy_from(ir3 + (64));
              (v513_data + (v490_data * v511_data)).copy_to(ir3 + (64));
              float v516_data = s1[69];
              tensorforge::intel_esimd::simd<float, 12> v518_data;
              v518_data.copy_from(ir3 + (80));
              (v518_data + (v490_data * v516_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v520_data;
              v520_data.copy_from(r2 + (160));
              float v521_data = s1[11];
              tensorforge::intel_esimd::simd<float, 12> v523_data;
              v523_data.copy_from(ir3 + (0));
              (v523_data + (v520_data * v521_data)).copy_to(ir3 + (0));
              float v526_data = s1[20];
              tensorforge::intel_esimd::simd<float, 12> v528_data;
              v528_data.copy_from(ir3 + (16));
              (v528_data + (v520_data * v526_data)).copy_to(ir3 + (16));
              float v531_data = s1[38];
              tensorforge::intel_esimd::simd<float, 12> v533_data;
              v533_data.copy_from(ir3 + (32));
              (v533_data + (v520_data * v531_data)).copy_to(ir3 + (32));
              float v536_data = s1[43];
              tensorforge::intel_esimd::simd<float, 12> v538_data;
              v538_data.copy_from(ir3 + (48));
              (v538_data + (v520_data * v536_data)).copy_to(ir3 + (48));
              float v541_data = s1[61];
              tensorforge::intel_esimd::simd<float, 12> v543_data;
              v543_data.copy_from(ir3 + (64));
              (v543_data + (v520_data * v541_data)).copy_to(ir3 + (64));
              float v546_data = s1[70];
              tensorforge::intel_esimd::simd<float, 12> v548_data;
              v548_data.copy_from(ir3 + (80));
              (v548_data + (v520_data * v546_data)).copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 12> v550_data;
              v550_data.copy_from(r2 + (176));
              float v551_data = s1[10];
              tensorforge::intel_esimd::simd<float, 12> v553_data;
              v553_data.copy_from(ir3 + (0));
              (v553_data + (v550_data * v551_data)).copy_to(ir3 + (0));
              float v556_data = s1[21];
              tensorforge::intel_esimd::simd<float, 12> v558_data;
              v558_data.copy_from(ir3 + (16));
              (v558_data + (v550_data * v556_data)).copy_to(ir3 + (16));
              float v561_data = s1[39];
              tensorforge::intel_esimd::simd<float, 12> v563_data;
              v563_data.copy_from(ir3 + (32));
              (v563_data + (v550_data * v561_data)).copy_to(ir3 + (32));
              float v566_data = s1[42];
              tensorforge::intel_esimd::simd<float, 12> v568_data;
              v568_data.copy_from(ir3 + (48));
              (v568_data + (v550_data * v566_data)).copy_to(ir3 + (48));
              float v571_data = s1[60];
              tensorforge::intel_esimd::simd<float, 12> v573_data;
              v573_data.copy_from(ir3 + (64));
              (v573_data + (v550_data * v571_data)).copy_to(ir3 + (64));
              float v576_data = s1[71];
              tensorforge::intel_esimd::simd<float, 12> v578_data;
              v578_data.copy_from(ir3 + (80));
              (v578_data + (v550_data * v576_data)).copy_to(ir3 + (80));
              #pragma unroll
              for (int32_t v580_n1 = 0; v580_n1 < 6; ++v580_n1) {
                int32_t v581_a = v580_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v583_data;
                v583_data.copy_from(ir3 + (v581_a));
                v583_data.copy_to(r3 + (v581_a));
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v586_i1 = 0; v586_i1 < 6; ++v586_i1) {
                tensorforge::intel_esimd::simd<float, 12> v589_data;
                v589_data.copy_from(r3 + ((v586_i1 * 16)));
                v589_data.copy_to(glb_m2 + ((v586_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

