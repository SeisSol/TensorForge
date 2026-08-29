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
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m1[0 + 0 + 2 * item.get_local_id(0) + 0];
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[64]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 8> v17_data;
              v17_data.copy_from(r0 + (0));
              float v18_data = s0[0];
              tensorforge::intel_esimd::simd<float, 8> v20_data;
              v20_data.copy_from(r1 + (0));
              (v20_data + (v17_data * v18_data)).copy_to(r1 + (0));
              float v23_data = s0[8];
              tensorforge::intel_esimd::simd<float, 8> v25_data;
              v25_data.copy_from(r1 + (16));
              (v25_data + (v17_data * v23_data)).copy_to(r1 + (16));
              float v28_data = s0[16];
              tensorforge::intel_esimd::simd<float, 8> v30_data;
              v30_data.copy_from(r1 + (32));
              (v30_data + (v17_data * v28_data)).copy_to(r1 + (32));
              float v33_data = s0[24];
              tensorforge::intel_esimd::simd<float, 8> v35_data;
              v35_data.copy_from(r1 + (48));
              (v35_data + (v17_data * v33_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v37_data;
              v37_data.copy_from(r0 + (16));
              float v38_data = s0[1];
              tensorforge::intel_esimd::simd<float, 8> v40_data;
              v40_data.copy_from(r1 + (0));
              (v40_data + (v37_data * v38_data)).copy_to(r1 + (0));
              float v43_data = s0[9];
              tensorforge::intel_esimd::simd<float, 8> v45_data;
              v45_data.copy_from(r1 + (16));
              (v45_data + (v37_data * v43_data)).copy_to(r1 + (16));
              float v48_data = s0[17];
              tensorforge::intel_esimd::simd<float, 8> v50_data;
              v50_data.copy_from(r1 + (32));
              (v50_data + (v37_data * v48_data)).copy_to(r1 + (32));
              float v53_data = s0[25];
              tensorforge::intel_esimd::simd<float, 8> v55_data;
              v55_data.copy_from(r1 + (48));
              (v55_data + (v37_data * v53_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v57_data;
              v57_data.copy_from(r0 + (32));
              float v58_data = s0[2];
              tensorforge::intel_esimd::simd<float, 8> v60_data;
              v60_data.copy_from(r1 + (0));
              (v60_data + (v57_data * v58_data)).copy_to(r1 + (0));
              float v63_data = s0[10];
              tensorforge::intel_esimd::simd<float, 8> v65_data;
              v65_data.copy_from(r1 + (16));
              (v65_data + (v57_data * v63_data)).copy_to(r1 + (16));
              float v68_data = s0[18];
              tensorforge::intel_esimd::simd<float, 8> v70_data;
              v70_data.copy_from(r1 + (32));
              (v70_data + (v57_data * v68_data)).copy_to(r1 + (32));
              float v73_data = s0[26];
              tensorforge::intel_esimd::simd<float, 8> v75_data;
              v75_data.copy_from(r1 + (48));
              (v75_data + (v57_data * v73_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v77_data;
              v77_data.copy_from(r0 + (48));
              float v78_data = s0[3];
              tensorforge::intel_esimd::simd<float, 8> v80_data;
              v80_data.copy_from(r1 + (0));
              (v80_data + (v77_data * v78_data)).copy_to(r1 + (0));
              float v83_data = s0[11];
              tensorforge::intel_esimd::simd<float, 8> v85_data;
              v85_data.copy_from(r1 + (16));
              (v85_data + (v77_data * v83_data)).copy_to(r1 + (16));
              float v88_data = s0[19];
              tensorforge::intel_esimd::simd<float, 8> v90_data;
              v90_data.copy_from(r1 + (32));
              (v90_data + (v77_data * v88_data)).copy_to(r1 + (32));
              float v93_data = s0[27];
              tensorforge::intel_esimd::simd<float, 8> v95_data;
              v95_data.copy_from(r1 + (48));
              (v95_data + (v77_data * v93_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v97_data;
              v97_data.copy_from(r0 + (64));
              float v98_data = s0[4];
              tensorforge::intel_esimd::simd<float, 8> v100_data;
              v100_data.copy_from(r1 + (0));
              (v100_data + (v97_data * v98_data)).copy_to(r1 + (0));
              float v103_data = s0[12];
              tensorforge::intel_esimd::simd<float, 8> v105_data;
              v105_data.copy_from(r1 + (16));
              (v105_data + (v97_data * v103_data)).copy_to(r1 + (16));
              float v108_data = s0[20];
              tensorforge::intel_esimd::simd<float, 8> v110_data;
              v110_data.copy_from(r1 + (32));
              (v110_data + (v97_data * v108_data)).copy_to(r1 + (32));
              float v113_data = s0[28];
              tensorforge::intel_esimd::simd<float, 8> v115_data;
              v115_data.copy_from(r1 + (48));
              (v115_data + (v97_data * v113_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v117_data;
              v117_data.copy_from(r0 + (80));
              float v118_data = s0[5];
              tensorforge::intel_esimd::simd<float, 8> v120_data;
              v120_data.copy_from(r1 + (0));
              (v120_data + (v117_data * v118_data)).copy_to(r1 + (0));
              float v123_data = s0[13];
              tensorforge::intel_esimd::simd<float, 8> v125_data;
              v125_data.copy_from(r1 + (16));
              (v125_data + (v117_data * v123_data)).copy_to(r1 + (16));
              float v128_data = s0[21];
              tensorforge::intel_esimd::simd<float, 8> v130_data;
              v130_data.copy_from(r1 + (32));
              (v130_data + (v117_data * v128_data)).copy_to(r1 + (32));
              float v133_data = s0[29];
              tensorforge::intel_esimd::simd<float, 8> v135_data;
              v135_data.copy_from(r1 + (48));
              (v135_data + (v117_data * v133_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v137_data;
              v137_data.copy_from(r0 + (96));
              float v138_data = s0[6];
              tensorforge::intel_esimd::simd<float, 8> v140_data;
              v140_data.copy_from(r1 + (0));
              (v140_data + (v137_data * v138_data)).copy_to(r1 + (0));
              float v143_data = s0[14];
              tensorforge::intel_esimd::simd<float, 8> v145_data;
              v145_data.copy_from(r1 + (16));
              (v145_data + (v137_data * v143_data)).copy_to(r1 + (16));
              float v148_data = s0[22];
              tensorforge::intel_esimd::simd<float, 8> v150_data;
              v150_data.copy_from(r1 + (32));
              (v150_data + (v137_data * v148_data)).copy_to(r1 + (32));
              float v153_data = s0[30];
              tensorforge::intel_esimd::simd<float, 8> v155_data;
              v155_data.copy_from(r1 + (48));
              (v155_data + (v137_data * v153_data)).copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 8> v157_data;
              v157_data.copy_from(r0 + (112));
              float v158_data = s0[7];
              tensorforge::intel_esimd::simd<float, 8> v160_data;
              v160_data.copy_from(r1 + (0));
              (v160_data + (v157_data * v158_data)).copy_to(r1 + (0));
              float v163_data = s0[15];
              tensorforge::intel_esimd::simd<float, 8> v165_data;
              v165_data.copy_from(r1 + (16));
              (v165_data + (v157_data * v163_data)).copy_to(r1 + (16));
              float v168_data = s0[23];
              tensorforge::intel_esimd::simd<float, 8> v170_data;
              v170_data.copy_from(r1 + (32));
              (v170_data + (v157_data * v168_data)).copy_to(r1 + (32));
              float v173_data = s0[31];
              tensorforge::intel_esimd::simd<float, 8> v175_data;
              v175_data.copy_from(r1 + (48));
              (v175_data + (v157_data * v173_data)).copy_to(r1 + (48));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v178_i1 = 0; v178_i1 < 4; ++v178_i1) {
                tensorforge::intel_esimd::simd<float, 8> v181_data;
                v181_data.copy_from(r1 + ((v178_i1 * 16)));
                int32_t v184_a = v178_i1 * 8;
                v181_data.copy_to(s1 + ((v184_a ^ ((v184_a >> 5) & 31))));
              }
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 2>*)&s2[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 0];
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              float r2[64]{};
              // r2 = +(r0 * s2) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir2[64]{};
              tensorforge::intel_esimd::simd<float, 8> v192_data;
              v192_data.copy_from(r0 + (0));
              float v193_data = s2[0];
              tensorforge::intel_esimd::simd<float, 8> v195_data;
              v195_data.copy_from(ir2 + (0));
              (v195_data + (v192_data * v193_data)).copy_to(ir2 + (0));
              float v198_data = s2[8];
              tensorforge::intel_esimd::simd<float, 8> v200_data;
              v200_data.copy_from(ir2 + (16));
              (v200_data + (v192_data * v198_data)).copy_to(ir2 + (16));
              float v203_data = s2[16];
              tensorforge::intel_esimd::simd<float, 8> v205_data;
              v205_data.copy_from(ir2 + (32));
              (v205_data + (v192_data * v203_data)).copy_to(ir2 + (32));
              float v208_data = s2[24];
              tensorforge::intel_esimd::simd<float, 8> v210_data;
              v210_data.copy_from(ir2 + (48));
              (v210_data + (v192_data * v208_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v212_data;
              v212_data.copy_from(r0 + (16));
              float v213_data = s2[1];
              tensorforge::intel_esimd::simd<float, 8> v215_data;
              v215_data.copy_from(ir2 + (0));
              (v215_data + (v212_data * v213_data)).copy_to(ir2 + (0));
              float v218_data = s2[9];
              tensorforge::intel_esimd::simd<float, 8> v220_data;
              v220_data.copy_from(ir2 + (16));
              (v220_data + (v212_data * v218_data)).copy_to(ir2 + (16));
              float v223_data = s2[17];
              tensorforge::intel_esimd::simd<float, 8> v225_data;
              v225_data.copy_from(ir2 + (32));
              (v225_data + (v212_data * v223_data)).copy_to(ir2 + (32));
              float v228_data = s2[25];
              tensorforge::intel_esimd::simd<float, 8> v230_data;
              v230_data.copy_from(ir2 + (48));
              (v230_data + (v212_data * v228_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v232_data;
              v232_data.copy_from(r0 + (32));
              float v233_data = s2[2];
              tensorforge::intel_esimd::simd<float, 8> v235_data;
              v235_data.copy_from(ir2 + (0));
              (v235_data + (v232_data * v233_data)).copy_to(ir2 + (0));
              float v238_data = s2[10];
              tensorforge::intel_esimd::simd<float, 8> v240_data;
              v240_data.copy_from(ir2 + (16));
              (v240_data + (v232_data * v238_data)).copy_to(ir2 + (16));
              float v243_data = s2[18];
              tensorforge::intel_esimd::simd<float, 8> v245_data;
              v245_data.copy_from(ir2 + (32));
              (v245_data + (v232_data * v243_data)).copy_to(ir2 + (32));
              float v248_data = s2[26];
              tensorforge::intel_esimd::simd<float, 8> v250_data;
              v250_data.copy_from(ir2 + (48));
              (v250_data + (v232_data * v248_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v252_data;
              v252_data.copy_from(r0 + (48));
              float v253_data = s2[3];
              tensorforge::intel_esimd::simd<float, 8> v255_data;
              v255_data.copy_from(ir2 + (0));
              (v255_data + (v252_data * v253_data)).copy_to(ir2 + (0));
              float v258_data = s2[11];
              tensorforge::intel_esimd::simd<float, 8> v260_data;
              v260_data.copy_from(ir2 + (16));
              (v260_data + (v252_data * v258_data)).copy_to(ir2 + (16));
              float v263_data = s2[19];
              tensorforge::intel_esimd::simd<float, 8> v265_data;
              v265_data.copy_from(ir2 + (32));
              (v265_data + (v252_data * v263_data)).copy_to(ir2 + (32));
              float v268_data = s2[27];
              tensorforge::intel_esimd::simd<float, 8> v270_data;
              v270_data.copy_from(ir2 + (48));
              (v270_data + (v252_data * v268_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v272_data;
              v272_data.copy_from(r0 + (64));
              float v273_data = s2[4];
              tensorforge::intel_esimd::simd<float, 8> v275_data;
              v275_data.copy_from(ir2 + (0));
              (v275_data + (v272_data * v273_data)).copy_to(ir2 + (0));
              float v278_data = s2[12];
              tensorforge::intel_esimd::simd<float, 8> v280_data;
              v280_data.copy_from(ir2 + (16));
              (v280_data + (v272_data * v278_data)).copy_to(ir2 + (16));
              float v283_data = s2[20];
              tensorforge::intel_esimd::simd<float, 8> v285_data;
              v285_data.copy_from(ir2 + (32));
              (v285_data + (v272_data * v283_data)).copy_to(ir2 + (32));
              float v288_data = s2[28];
              tensorforge::intel_esimd::simd<float, 8> v290_data;
              v290_data.copy_from(ir2 + (48));
              (v290_data + (v272_data * v288_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v292_data;
              v292_data.copy_from(r0 + (80));
              float v293_data = s2[5];
              tensorforge::intel_esimd::simd<float, 8> v295_data;
              v295_data.copy_from(ir2 + (0));
              (v295_data + (v292_data * v293_data)).copy_to(ir2 + (0));
              float v298_data = s2[13];
              tensorforge::intel_esimd::simd<float, 8> v300_data;
              v300_data.copy_from(ir2 + (16));
              (v300_data + (v292_data * v298_data)).copy_to(ir2 + (16));
              float v303_data = s2[21];
              tensorforge::intel_esimd::simd<float, 8> v305_data;
              v305_data.copy_from(ir2 + (32));
              (v305_data + (v292_data * v303_data)).copy_to(ir2 + (32));
              float v308_data = s2[29];
              tensorforge::intel_esimd::simd<float, 8> v310_data;
              v310_data.copy_from(ir2 + (48));
              (v310_data + (v292_data * v308_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v312_data;
              v312_data.copy_from(r0 + (96));
              float v313_data = s2[6];
              tensorforge::intel_esimd::simd<float, 8> v315_data;
              v315_data.copy_from(ir2 + (0));
              (v315_data + (v312_data * v313_data)).copy_to(ir2 + (0));
              float v318_data = s2[14];
              tensorforge::intel_esimd::simd<float, 8> v320_data;
              v320_data.copy_from(ir2 + (16));
              (v320_data + (v312_data * v318_data)).copy_to(ir2 + (16));
              float v323_data = s2[22];
              tensorforge::intel_esimd::simd<float, 8> v325_data;
              v325_data.copy_from(ir2 + (32));
              (v325_data + (v312_data * v323_data)).copy_to(ir2 + (32));
              float v328_data = s2[30];
              tensorforge::intel_esimd::simd<float, 8> v330_data;
              v330_data.copy_from(ir2 + (48));
              (v330_data + (v312_data * v328_data)).copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 8> v332_data;
              v332_data.copy_from(r0 + (112));
              float v333_data = s2[7];
              tensorforge::intel_esimd::simd<float, 8> v335_data;
              v335_data.copy_from(ir2 + (0));
              (v335_data + (v332_data * v333_data)).copy_to(ir2 + (0));
              float v338_data = s2[15];
              tensorforge::intel_esimd::simd<float, 8> v340_data;
              v340_data.copy_from(ir2 + (16));
              (v340_data + (v332_data * v338_data)).copy_to(ir2 + (16));
              float v343_data = s2[23];
              tensorforge::intel_esimd::simd<float, 8> v345_data;
              v345_data.copy_from(ir2 + (32));
              (v345_data + (v332_data * v343_data)).copy_to(ir2 + (32));
              float v348_data = s2[31];
              tensorforge::intel_esimd::simd<float, 8> v350_data;
              v350_data.copy_from(ir2 + (48));
              (v350_data + (v332_data * v348_data)).copy_to(ir2 + (48));
              #pragma unroll
              for (int32_t v352_n1 = 0; v352_n1 < 4; ++v352_n1) {
                int32_t v353_a = v352_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v355_data;
                v355_data.copy_from(ir2 + (v353_a));
                v355_data.copy_to(r2 + (v353_a));
              }
              // s1 = store{r>s}(localShrMem0, r2);
              #pragma unroll
              for (int32_t v358_i1 = 0; v358_i1 < 4; ++v358_i1) {
                tensorforge::intel_esimd::simd<float, 8> v361_data;
                v361_data.copy_from(r2 + ((v358_i1 * 16)));
                int32_t v365_a = (v358_i1 + 4) * 8;
                v361_data.copy_to(s1 + ((v365_a ^ ((v365_a >> 5) & 31))));
              }
              // glb_m3 = abs(s1)
              #pragma unroll
              for (int32_t v370_k1 = 0; v370_k1 < 8; ++v370_k1) {
                int32_t v373_a = v370_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v378_data;
                v378_data.copy_from(s1 + ((v373_a ^ ((v373_a >> 5) & 31))));
                (tensorforge::intel_esimd::abs(v378_data)).copy_to(glb_m3 + (v373_a));
              }
            }
          }
        }
      });
    }
  });
}

