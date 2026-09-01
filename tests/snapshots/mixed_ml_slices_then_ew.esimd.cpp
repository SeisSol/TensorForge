// === base name ===
kernel_924fd3d329

// === header ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_924fd3d329(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_924fd3d329(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_924fd3d329(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×4(8×4) {0..8}×{0..4} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..4})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 64 + 0 + m3_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v7_i1 = 0; v7_i1 < 8; ++v7_i1) {
                tensorforge::intel_esimd::simd<float, 8> v12_data;
                v12_data.copy_from(glb_m0 + ((v7_i1 * 8)));
                v12_data.copy_to(r0 + ((v7_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v16_ld;
              v16_ld.copy_from(glb_m1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v16_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[64]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v18_data;
              v18_data.copy_from(r0 + (0));
              float v19_data = s0[0];
              tensorforge::intel_esimd::simd<float, 8> v21_data;
              v21_data.copy_from(r1 + (0));
              (v21_data + (v18_data * v19_data)).copy_to(r1 + (0));
              float v24_data = s0[8];
              tensorforge::intel_esimd::simd<float, 8> v26_data;
              v26_data.copy_from(r1 + (16));
              (v26_data + (v18_data * v24_data)).copy_to(r1 + (16));
              float v29_data = s0[16];
              tensorforge::intel_esimd::simd<float, 8> v31_data;
              v31_data.copy_from(r1 + (32));
              (v31_data + (v18_data * v29_data)).copy_to(r1 + (32));
              float v34_data = s0[24];
              tensorforge::intel_esimd::simd<float, 8> v36_data;
              v36_data.copy_from(r1 + (48));
              (v36_data + (v18_data * v34_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v38_data;
              v38_data.copy_from(r0 + (16));
              float v39_data = s0[1];
              tensorforge::intel_esimd::simd<float, 8> v41_data;
              v41_data.copy_from(r1 + (0));
              (v41_data + (v38_data * v39_data)).copy_to(r1 + (0));
              float v44_data = s0[9];
              tensorforge::intel_esimd::simd<float, 8> v46_data;
              v46_data.copy_from(r1 + (16));
              (v46_data + (v38_data * v44_data)).copy_to(r1 + (16));
              float v49_data = s0[17];
              tensorforge::intel_esimd::simd<float, 8> v51_data;
              v51_data.copy_from(r1 + (32));
              (v51_data + (v38_data * v49_data)).copy_to(r1 + (32));
              float v54_data = s0[25];
              tensorforge::intel_esimd::simd<float, 8> v56_data;
              v56_data.copy_from(r1 + (48));
              (v56_data + (v38_data * v54_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v58_data;
              v58_data.copy_from(r0 + (32));
              float v59_data = s0[2];
              tensorforge::intel_esimd::simd<float, 8> v61_data;
              v61_data.copy_from(r1 + (0));
              (v61_data + (v58_data * v59_data)).copy_to(r1 + (0));
              float v64_data = s0[10];
              tensorforge::intel_esimd::simd<float, 8> v66_data;
              v66_data.copy_from(r1 + (16));
              (v66_data + (v58_data * v64_data)).copy_to(r1 + (16));
              float v69_data = s0[18];
              tensorforge::intel_esimd::simd<float, 8> v71_data;
              v71_data.copy_from(r1 + (32));
              (v71_data + (v58_data * v69_data)).copy_to(r1 + (32));
              float v74_data = s0[26];
              tensorforge::intel_esimd::simd<float, 8> v76_data;
              v76_data.copy_from(r1 + (48));
              (v76_data + (v58_data * v74_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v78_data;
              v78_data.copy_from(r0 + (48));
              float v79_data = s0[3];
              tensorforge::intel_esimd::simd<float, 8> v81_data;
              v81_data.copy_from(r1 + (0));
              (v81_data + (v78_data * v79_data)).copy_to(r1 + (0));
              float v84_data = s0[11];
              tensorforge::intel_esimd::simd<float, 8> v86_data;
              v86_data.copy_from(r1 + (16));
              (v86_data + (v78_data * v84_data)).copy_to(r1 + (16));
              float v89_data = s0[19];
              tensorforge::intel_esimd::simd<float, 8> v91_data;
              v91_data.copy_from(r1 + (32));
              (v91_data + (v78_data * v89_data)).copy_to(r1 + (32));
              float v94_data = s0[27];
              tensorforge::intel_esimd::simd<float, 8> v96_data;
              v96_data.copy_from(r1 + (48));
              (v96_data + (v78_data * v94_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v98_data;
              v98_data.copy_from(r0 + (64));
              float v99_data = s0[4];
              tensorforge::intel_esimd::simd<float, 8> v101_data;
              v101_data.copy_from(r1 + (0));
              (v101_data + (v98_data * v99_data)).copy_to(r1 + (0));
              float v104_data = s0[12];
              tensorforge::intel_esimd::simd<float, 8> v106_data;
              v106_data.copy_from(r1 + (16));
              (v106_data + (v98_data * v104_data)).copy_to(r1 + (16));
              float v109_data = s0[20];
              tensorforge::intel_esimd::simd<float, 8> v111_data;
              v111_data.copy_from(r1 + (32));
              (v111_data + (v98_data * v109_data)).copy_to(r1 + (32));
              float v114_data = s0[28];
              tensorforge::intel_esimd::simd<float, 8> v116_data;
              v116_data.copy_from(r1 + (48));
              (v116_data + (v98_data * v114_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v118_data;
              v118_data.copy_from(r0 + (80));
              float v119_data = s0[5];
              tensorforge::intel_esimd::simd<float, 8> v121_data;
              v121_data.copy_from(r1 + (0));
              (v121_data + (v118_data * v119_data)).copy_to(r1 + (0));
              float v124_data = s0[13];
              tensorforge::intel_esimd::simd<float, 8> v126_data;
              v126_data.copy_from(r1 + (16));
              (v126_data + (v118_data * v124_data)).copy_to(r1 + (16));
              float v129_data = s0[21];
              tensorforge::intel_esimd::simd<float, 8> v131_data;
              v131_data.copy_from(r1 + (32));
              (v131_data + (v118_data * v129_data)).copy_to(r1 + (32));
              float v134_data = s0[29];
              tensorforge::intel_esimd::simd<float, 8> v136_data;
              v136_data.copy_from(r1 + (48));
              (v136_data + (v118_data * v134_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v138_data;
              v138_data.copy_from(r0 + (96));
              float v139_data = s0[6];
              tensorforge::intel_esimd::simd<float, 8> v141_data;
              v141_data.copy_from(r1 + (0));
              (v141_data + (v138_data * v139_data)).copy_to(r1 + (0));
              float v144_data = s0[14];
              tensorforge::intel_esimd::simd<float, 8> v146_data;
              v146_data.copy_from(r1 + (16));
              (v146_data + (v138_data * v144_data)).copy_to(r1 + (16));
              float v149_data = s0[22];
              tensorforge::intel_esimd::simd<float, 8> v151_data;
              v151_data.copy_from(r1 + (32));
              (v151_data + (v138_data * v149_data)).copy_to(r1 + (32));
              float v154_data = s0[30];
              tensorforge::intel_esimd::simd<float, 8> v156_data;
              v156_data.copy_from(r1 + (48));
              (v156_data + (v138_data * v154_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v158_data;
              v158_data.copy_from(r0 + (112));
              float v159_data = s0[7];
              tensorforge::intel_esimd::simd<float, 8> v161_data;
              v161_data.copy_from(r1 + (0));
              (v161_data + (v158_data * v159_data)).copy_to(r1 + (0));
              float v164_data = s0[15];
              tensorforge::intel_esimd::simd<float, 8> v166_data;
              v166_data.copy_from(r1 + (16));
              (v166_data + (v158_data * v164_data)).copy_to(r1 + (16));
              float v169_data = s0[23];
              tensorforge::intel_esimd::simd<float, 8> v171_data;
              v171_data.copy_from(r1 + (32));
              (v171_data + (v158_data * v169_data)).copy_to(r1 + (32));
              float v174_data = s0[31];
              tensorforge::intel_esimd::simd<float, 8> v176_data;
              v176_data.copy_from(r1 + (48));
              (v176_data + (v158_data * v174_data)).copy_to(r1 + (48));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v179_i1 = 0; v179_i1 < 4; ++v179_i1) {
                tensorforge::intel_esimd::simd<float, 8> v182_data;
                v182_data.copy_from(r1 + ((v179_i1 * 16)));
                v182_data.copy_to(s1 + ((v179_i1 * 8)));
              }
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v188_ld;
              v188_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v188_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              float r2[64]{};
              // r2 = +(r0 * s2) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir2[64]{};
              float v192_data = s2[0];
              tensorforge::intel_esimd::simd<float, 8> v194_data;
              v194_data.copy_from(ir2 + (0));
              (v194_data + (v18_data * v192_data)).copy_to(ir2 + (0));
              float v197_data = s2[8];
              tensorforge::intel_esimd::simd<float, 8> v199_data;
              v199_data.copy_from(ir2 + (16));
              (v199_data + (v18_data * v197_data)).copy_to(ir2 + (16));
              float v202_data = s2[16];
              tensorforge::intel_esimd::simd<float, 8> v204_data;
              v204_data.copy_from(ir2 + (32));
              (v204_data + (v18_data * v202_data)).copy_to(ir2 + (32));
              float v207_data = s2[24];
              tensorforge::intel_esimd::simd<float, 8> v209_data;
              v209_data.copy_from(ir2 + (48));
              (v209_data + (v18_data * v207_data)).copy_to(ir2 + (48));
              float v212_data = s2[1];
              tensorforge::intel_esimd::simd<float, 8> v214_data;
              v214_data.copy_from(ir2 + (0));
              (v214_data + (v38_data * v212_data)).copy_to(ir2 + (0));
              float v217_data = s2[9];
              tensorforge::intel_esimd::simd<float, 8> v219_data;
              v219_data.copy_from(ir2 + (16));
              (v219_data + (v38_data * v217_data)).copy_to(ir2 + (16));
              float v222_data = s2[17];
              tensorforge::intel_esimd::simd<float, 8> v224_data;
              v224_data.copy_from(ir2 + (32));
              (v224_data + (v38_data * v222_data)).copy_to(ir2 + (32));
              float v227_data = s2[25];
              tensorforge::intel_esimd::simd<float, 8> v229_data;
              v229_data.copy_from(ir2 + (48));
              (v229_data + (v38_data * v227_data)).copy_to(ir2 + (48));
              float v232_data = s2[2];
              tensorforge::intel_esimd::simd<float, 8> v234_data;
              v234_data.copy_from(ir2 + (0));
              (v234_data + (v58_data * v232_data)).copy_to(ir2 + (0));
              float v237_data = s2[10];
              tensorforge::intel_esimd::simd<float, 8> v239_data;
              v239_data.copy_from(ir2 + (16));
              (v239_data + (v58_data * v237_data)).copy_to(ir2 + (16));
              float v242_data = s2[18];
              tensorforge::intel_esimd::simd<float, 8> v244_data;
              v244_data.copy_from(ir2 + (32));
              (v244_data + (v58_data * v242_data)).copy_to(ir2 + (32));
              float v247_data = s2[26];
              tensorforge::intel_esimd::simd<float, 8> v249_data;
              v249_data.copy_from(ir2 + (48));
              (v249_data + (v58_data * v247_data)).copy_to(ir2 + (48));
              float v252_data = s2[3];
              tensorforge::intel_esimd::simd<float, 8> v254_data;
              v254_data.copy_from(ir2 + (0));
              (v254_data + (v78_data * v252_data)).copy_to(ir2 + (0));
              float v257_data = s2[11];
              tensorforge::intel_esimd::simd<float, 8> v259_data;
              v259_data.copy_from(ir2 + (16));
              (v259_data + (v78_data * v257_data)).copy_to(ir2 + (16));
              float v262_data = s2[19];
              tensorforge::intel_esimd::simd<float, 8> v264_data;
              v264_data.copy_from(ir2 + (32));
              (v264_data + (v78_data * v262_data)).copy_to(ir2 + (32));
              float v267_data = s2[27];
              tensorforge::intel_esimd::simd<float, 8> v269_data;
              v269_data.copy_from(ir2 + (48));
              (v269_data + (v78_data * v267_data)).copy_to(ir2 + (48));
              float v272_data = s2[4];
              tensorforge::intel_esimd::simd<float, 8> v274_data;
              v274_data.copy_from(ir2 + (0));
              (v274_data + (v98_data * v272_data)).copy_to(ir2 + (0));
              float v277_data = s2[12];
              tensorforge::intel_esimd::simd<float, 8> v279_data;
              v279_data.copy_from(ir2 + (16));
              (v279_data + (v98_data * v277_data)).copy_to(ir2 + (16));
              float v282_data = s2[20];
              tensorforge::intel_esimd::simd<float, 8> v284_data;
              v284_data.copy_from(ir2 + (32));
              (v284_data + (v98_data * v282_data)).copy_to(ir2 + (32));
              float v287_data = s2[28];
              tensorforge::intel_esimd::simd<float, 8> v289_data;
              v289_data.copy_from(ir2 + (48));
              (v289_data + (v98_data * v287_data)).copy_to(ir2 + (48));
              float v292_data = s2[5];
              tensorforge::intel_esimd::simd<float, 8> v294_data;
              v294_data.copy_from(ir2 + (0));
              (v294_data + (v118_data * v292_data)).copy_to(ir2 + (0));
              float v297_data = s2[13];
              tensorforge::intel_esimd::simd<float, 8> v299_data;
              v299_data.copy_from(ir2 + (16));
              (v299_data + (v118_data * v297_data)).copy_to(ir2 + (16));
              float v302_data = s2[21];
              tensorforge::intel_esimd::simd<float, 8> v304_data;
              v304_data.copy_from(ir2 + (32));
              (v304_data + (v118_data * v302_data)).copy_to(ir2 + (32));
              float v307_data = s2[29];
              tensorforge::intel_esimd::simd<float, 8> v309_data;
              v309_data.copy_from(ir2 + (48));
              (v309_data + (v118_data * v307_data)).copy_to(ir2 + (48));
              float v312_data = s2[6];
              tensorforge::intel_esimd::simd<float, 8> v314_data;
              v314_data.copy_from(ir2 + (0));
              (v314_data + (v138_data * v312_data)).copy_to(ir2 + (0));
              float v317_data = s2[14];
              tensorforge::intel_esimd::simd<float, 8> v319_data;
              v319_data.copy_from(ir2 + (16));
              (v319_data + (v138_data * v317_data)).copy_to(ir2 + (16));
              float v322_data = s2[22];
              tensorforge::intel_esimd::simd<float, 8> v324_data;
              v324_data.copy_from(ir2 + (32));
              (v324_data + (v138_data * v322_data)).copy_to(ir2 + (32));
              float v327_data = s2[30];
              tensorforge::intel_esimd::simd<float, 8> v329_data;
              v329_data.copy_from(ir2 + (48));
              (v329_data + (v138_data * v327_data)).copy_to(ir2 + (48));
              float v332_data = s2[7];
              tensorforge::intel_esimd::simd<float, 8> v334_data;
              v334_data.copy_from(ir2 + (0));
              (v334_data + (v158_data * v332_data)).copy_to(ir2 + (0));
              float v337_data = s2[15];
              tensorforge::intel_esimd::simd<float, 8> v339_data;
              v339_data.copy_from(ir2 + (16));
              (v339_data + (v158_data * v337_data)).copy_to(ir2 + (16));
              float v342_data = s2[23];
              tensorforge::intel_esimd::simd<float, 8> v344_data;
              v344_data.copy_from(ir2 + (32));
              (v344_data + (v158_data * v342_data)).copy_to(ir2 + (32));
              float v347_data = s2[31];
              tensorforge::intel_esimd::simd<float, 8> v349_data;
              v349_data.copy_from(ir2 + (48));
              (v349_data + (v158_data * v347_data)).copy_to(ir2 + (48));
              #pragma unroll
              for (int32_t v351_n1 = 0; v351_n1 < 4; ++v351_n1) {
                int32_t v352_a = v351_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v354_data;
                v354_data.copy_from(ir2 + (v352_a));
                v354_data.copy_to(r2 + (v352_a));
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v357_i1 = 0; v357_i1 < 4; ++v357_i1) {
                tensorforge::intel_esimd::simd<float, 8> v360_data;
                v360_data.copy_from(r2 + ((v357_i1 * 16)));
                v360_data.copy_to(s1 + (((v357_i1 + 4) * 8)));
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v366_k1 = 0; v366_k1 < 8; ++v366_k1) {
                int32_t v369_a = v366_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v371_data;
                v371_data.copy_from(s1 + (v369_a));
                (tensorforge::intel_esimd::abs(v371_data)).copy_to(glb_m3 + (v369_a));
              }
            }
          }
        }
      });
    }
  });
}

