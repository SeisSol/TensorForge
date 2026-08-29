// === base name ===
kernel_08a27dccde

// === header ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08a27dccde(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08a27dccde(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08a27dccde(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float r0[144]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 9; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 9> v11_data;
                v11_data.copy_from(glb_m1 + ((v6_i1 * 9)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 64] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 64];
              if (item.get_local_id(0) < 1) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 80] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 80];
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[144]{};
              // r1 = +(r0 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir1[144]{};
              tensorforge::intel_esimd::simd<float, 9> v17_data;
              v17_data.copy_from(r0 + (0));
              float v18_data = s0[0];
              tensorforge::intel_esimd::simd<float, 9> v20_data;
              v20_data.copy_from(ir1 + (0));
              (v20_data + (v17_data * v18_data)).copy_to(ir1 + (0));
              float v23_data = s0[9];
              tensorforge::intel_esimd::simd<float, 9> v25_data;
              v25_data.copy_from(ir1 + (16));
              (v25_data + (v17_data * v23_data)).copy_to(ir1 + (16));
              float v28_data = s0[18];
              tensorforge::intel_esimd::simd<float, 9> v30_data;
              v30_data.copy_from(ir1 + (32));
              (v30_data + (v17_data * v28_data)).copy_to(ir1 + (32));
              float v33_data = s0[27];
              tensorforge::intel_esimd::simd<float, 9> v35_data;
              v35_data.copy_from(ir1 + (48));
              (v35_data + (v17_data * v33_data)).copy_to(ir1 + (48));
              float v38_data = s0[36];
              tensorforge::intel_esimd::simd<float, 9> v40_data;
              v40_data.copy_from(ir1 + (64));
              (v40_data + (v17_data * v38_data)).copy_to(ir1 + (64));
              float v43_data = s0[45];
              tensorforge::intel_esimd::simd<float, 9> v45_data;
              v45_data.copy_from(ir1 + (80));
              (v45_data + (v17_data * v43_data)).copy_to(ir1 + (80));
              float v48_data = s0[54];
              tensorforge::intel_esimd::simd<float, 9> v50_data;
              v50_data.copy_from(ir1 + (96));
              (v50_data + (v17_data * v48_data)).copy_to(ir1 + (96));
              float v53_data = s0[63];
              tensorforge::intel_esimd::simd<float, 9> v55_data;
              v55_data.copy_from(ir1 + (112));
              (v55_data + (v17_data * v53_data)).copy_to(ir1 + (112));
              float v58_data = s0[72];
              tensorforge::intel_esimd::simd<float, 9> v60_data;
              v60_data.copy_from(ir1 + (128));
              (v60_data + (v17_data * v58_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v62_data;
              v62_data.copy_from(r0 + (16));
              float v63_data = s0[1];
              tensorforge::intel_esimd::simd<float, 9> v65_data;
              v65_data.copy_from(ir1 + (0));
              (v65_data + (v62_data * v63_data)).copy_to(ir1 + (0));
              float v68_data = s0[10];
              tensorforge::intel_esimd::simd<float, 9> v70_data;
              v70_data.copy_from(ir1 + (16));
              (v70_data + (v62_data * v68_data)).copy_to(ir1 + (16));
              float v73_data = s0[19];
              tensorforge::intel_esimd::simd<float, 9> v75_data;
              v75_data.copy_from(ir1 + (32));
              (v75_data + (v62_data * v73_data)).copy_to(ir1 + (32));
              float v78_data = s0[28];
              tensorforge::intel_esimd::simd<float, 9> v80_data;
              v80_data.copy_from(ir1 + (48));
              (v80_data + (v62_data * v78_data)).copy_to(ir1 + (48));
              float v83_data = s0[37];
              tensorforge::intel_esimd::simd<float, 9> v85_data;
              v85_data.copy_from(ir1 + (64));
              (v85_data + (v62_data * v83_data)).copy_to(ir1 + (64));
              float v88_data = s0[46];
              tensorforge::intel_esimd::simd<float, 9> v90_data;
              v90_data.copy_from(ir1 + (80));
              (v90_data + (v62_data * v88_data)).copy_to(ir1 + (80));
              float v93_data = s0[55];
              tensorforge::intel_esimd::simd<float, 9> v95_data;
              v95_data.copy_from(ir1 + (96));
              (v95_data + (v62_data * v93_data)).copy_to(ir1 + (96));
              float v98_data = s0[64];
              tensorforge::intel_esimd::simd<float, 9> v100_data;
              v100_data.copy_from(ir1 + (112));
              (v100_data + (v62_data * v98_data)).copy_to(ir1 + (112));
              float v103_data = s0[73];
              tensorforge::intel_esimd::simd<float, 9> v105_data;
              v105_data.copy_from(ir1 + (128));
              (v105_data + (v62_data * v103_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v107_data;
              v107_data.copy_from(r0 + (32));
              float v108_data = s0[2];
              tensorforge::intel_esimd::simd<float, 9> v110_data;
              v110_data.copy_from(ir1 + (0));
              (v110_data + (v107_data * v108_data)).copy_to(ir1 + (0));
              float v113_data = s0[11];
              tensorforge::intel_esimd::simd<float, 9> v115_data;
              v115_data.copy_from(ir1 + (16));
              (v115_data + (v107_data * v113_data)).copy_to(ir1 + (16));
              float v118_data = s0[20];
              tensorforge::intel_esimd::simd<float, 9> v120_data;
              v120_data.copy_from(ir1 + (32));
              (v120_data + (v107_data * v118_data)).copy_to(ir1 + (32));
              float v123_data = s0[29];
              tensorforge::intel_esimd::simd<float, 9> v125_data;
              v125_data.copy_from(ir1 + (48));
              (v125_data + (v107_data * v123_data)).copy_to(ir1 + (48));
              float v128_data = s0[38];
              tensorforge::intel_esimd::simd<float, 9> v130_data;
              v130_data.copy_from(ir1 + (64));
              (v130_data + (v107_data * v128_data)).copy_to(ir1 + (64));
              float v133_data = s0[47];
              tensorforge::intel_esimd::simd<float, 9> v135_data;
              v135_data.copy_from(ir1 + (80));
              (v135_data + (v107_data * v133_data)).copy_to(ir1 + (80));
              float v138_data = s0[56];
              tensorforge::intel_esimd::simd<float, 9> v140_data;
              v140_data.copy_from(ir1 + (96));
              (v140_data + (v107_data * v138_data)).copy_to(ir1 + (96));
              float v143_data = s0[65];
              tensorforge::intel_esimd::simd<float, 9> v145_data;
              v145_data.copy_from(ir1 + (112));
              (v145_data + (v107_data * v143_data)).copy_to(ir1 + (112));
              float v148_data = s0[74];
              tensorforge::intel_esimd::simd<float, 9> v150_data;
              v150_data.copy_from(ir1 + (128));
              (v150_data + (v107_data * v148_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v152_data;
              v152_data.copy_from(r0 + (48));
              float v153_data = s0[3];
              tensorforge::intel_esimd::simd<float, 9> v155_data;
              v155_data.copy_from(ir1 + (0));
              (v155_data + (v152_data * v153_data)).copy_to(ir1 + (0));
              float v158_data = s0[12];
              tensorforge::intel_esimd::simd<float, 9> v160_data;
              v160_data.copy_from(ir1 + (16));
              (v160_data + (v152_data * v158_data)).copy_to(ir1 + (16));
              float v163_data = s0[21];
              tensorforge::intel_esimd::simd<float, 9> v165_data;
              v165_data.copy_from(ir1 + (32));
              (v165_data + (v152_data * v163_data)).copy_to(ir1 + (32));
              float v168_data = s0[30];
              tensorforge::intel_esimd::simd<float, 9> v170_data;
              v170_data.copy_from(ir1 + (48));
              (v170_data + (v152_data * v168_data)).copy_to(ir1 + (48));
              float v173_data = s0[39];
              tensorforge::intel_esimd::simd<float, 9> v175_data;
              v175_data.copy_from(ir1 + (64));
              (v175_data + (v152_data * v173_data)).copy_to(ir1 + (64));
              float v178_data = s0[48];
              tensorforge::intel_esimd::simd<float, 9> v180_data;
              v180_data.copy_from(ir1 + (80));
              (v180_data + (v152_data * v178_data)).copy_to(ir1 + (80));
              float v183_data = s0[57];
              tensorforge::intel_esimd::simd<float, 9> v185_data;
              v185_data.copy_from(ir1 + (96));
              (v185_data + (v152_data * v183_data)).copy_to(ir1 + (96));
              float v188_data = s0[66];
              tensorforge::intel_esimd::simd<float, 9> v190_data;
              v190_data.copy_from(ir1 + (112));
              (v190_data + (v152_data * v188_data)).copy_to(ir1 + (112));
              float v193_data = s0[75];
              tensorforge::intel_esimd::simd<float, 9> v195_data;
              v195_data.copy_from(ir1 + (128));
              (v195_data + (v152_data * v193_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v197_data;
              v197_data.copy_from(r0 + (64));
              float v198_data = s0[4];
              tensorforge::intel_esimd::simd<float, 9> v200_data;
              v200_data.copy_from(ir1 + (0));
              (v200_data + (v197_data * v198_data)).copy_to(ir1 + (0));
              float v203_data = s0[13];
              tensorforge::intel_esimd::simd<float, 9> v205_data;
              v205_data.copy_from(ir1 + (16));
              (v205_data + (v197_data * v203_data)).copy_to(ir1 + (16));
              float v208_data = s0[22];
              tensorforge::intel_esimd::simd<float, 9> v210_data;
              v210_data.copy_from(ir1 + (32));
              (v210_data + (v197_data * v208_data)).copy_to(ir1 + (32));
              float v213_data = s0[31];
              tensorforge::intel_esimd::simd<float, 9> v215_data;
              v215_data.copy_from(ir1 + (48));
              (v215_data + (v197_data * v213_data)).copy_to(ir1 + (48));
              float v218_data = s0[40];
              tensorforge::intel_esimd::simd<float, 9> v220_data;
              v220_data.copy_from(ir1 + (64));
              (v220_data + (v197_data * v218_data)).copy_to(ir1 + (64));
              float v223_data = s0[49];
              tensorforge::intel_esimd::simd<float, 9> v225_data;
              v225_data.copy_from(ir1 + (80));
              (v225_data + (v197_data * v223_data)).copy_to(ir1 + (80));
              float v228_data = s0[58];
              tensorforge::intel_esimd::simd<float, 9> v230_data;
              v230_data.copy_from(ir1 + (96));
              (v230_data + (v197_data * v228_data)).copy_to(ir1 + (96));
              float v233_data = s0[67];
              tensorforge::intel_esimd::simd<float, 9> v235_data;
              v235_data.copy_from(ir1 + (112));
              (v235_data + (v197_data * v233_data)).copy_to(ir1 + (112));
              float v238_data = s0[76];
              tensorforge::intel_esimd::simd<float, 9> v240_data;
              v240_data.copy_from(ir1 + (128));
              (v240_data + (v197_data * v238_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v242_data;
              v242_data.copy_from(r0 + (80));
              float v243_data = s0[5];
              tensorforge::intel_esimd::simd<float, 9> v245_data;
              v245_data.copy_from(ir1 + (0));
              (v245_data + (v242_data * v243_data)).copy_to(ir1 + (0));
              float v248_data = s0[14];
              tensorforge::intel_esimd::simd<float, 9> v250_data;
              v250_data.copy_from(ir1 + (16));
              (v250_data + (v242_data * v248_data)).copy_to(ir1 + (16));
              float v253_data = s0[23];
              tensorforge::intel_esimd::simd<float, 9> v255_data;
              v255_data.copy_from(ir1 + (32));
              (v255_data + (v242_data * v253_data)).copy_to(ir1 + (32));
              float v258_data = s0[32];
              tensorforge::intel_esimd::simd<float, 9> v260_data;
              v260_data.copy_from(ir1 + (48));
              (v260_data + (v242_data * v258_data)).copy_to(ir1 + (48));
              float v263_data = s0[41];
              tensorforge::intel_esimd::simd<float, 9> v265_data;
              v265_data.copy_from(ir1 + (64));
              (v265_data + (v242_data * v263_data)).copy_to(ir1 + (64));
              float v268_data = s0[50];
              tensorforge::intel_esimd::simd<float, 9> v270_data;
              v270_data.copy_from(ir1 + (80));
              (v270_data + (v242_data * v268_data)).copy_to(ir1 + (80));
              float v273_data = s0[59];
              tensorforge::intel_esimd::simd<float, 9> v275_data;
              v275_data.copy_from(ir1 + (96));
              (v275_data + (v242_data * v273_data)).copy_to(ir1 + (96));
              float v278_data = s0[68];
              tensorforge::intel_esimd::simd<float, 9> v280_data;
              v280_data.copy_from(ir1 + (112));
              (v280_data + (v242_data * v278_data)).copy_to(ir1 + (112));
              float v283_data = s0[77];
              tensorforge::intel_esimd::simd<float, 9> v285_data;
              v285_data.copy_from(ir1 + (128));
              (v285_data + (v242_data * v283_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v287_data;
              v287_data.copy_from(r0 + (96));
              float v288_data = s0[6];
              tensorforge::intel_esimd::simd<float, 9> v290_data;
              v290_data.copy_from(ir1 + (0));
              (v290_data + (v287_data * v288_data)).copy_to(ir1 + (0));
              float v293_data = s0[15];
              tensorforge::intel_esimd::simd<float, 9> v295_data;
              v295_data.copy_from(ir1 + (16));
              (v295_data + (v287_data * v293_data)).copy_to(ir1 + (16));
              float v298_data = s0[24];
              tensorforge::intel_esimd::simd<float, 9> v300_data;
              v300_data.copy_from(ir1 + (32));
              (v300_data + (v287_data * v298_data)).copy_to(ir1 + (32));
              float v303_data = s0[33];
              tensorforge::intel_esimd::simd<float, 9> v305_data;
              v305_data.copy_from(ir1 + (48));
              (v305_data + (v287_data * v303_data)).copy_to(ir1 + (48));
              float v308_data = s0[42];
              tensorforge::intel_esimd::simd<float, 9> v310_data;
              v310_data.copy_from(ir1 + (64));
              (v310_data + (v287_data * v308_data)).copy_to(ir1 + (64));
              float v313_data = s0[51];
              tensorforge::intel_esimd::simd<float, 9> v315_data;
              v315_data.copy_from(ir1 + (80));
              (v315_data + (v287_data * v313_data)).copy_to(ir1 + (80));
              float v318_data = s0[60];
              tensorforge::intel_esimd::simd<float, 9> v320_data;
              v320_data.copy_from(ir1 + (96));
              (v320_data + (v287_data * v318_data)).copy_to(ir1 + (96));
              float v323_data = s0[69];
              tensorforge::intel_esimd::simd<float, 9> v325_data;
              v325_data.copy_from(ir1 + (112));
              (v325_data + (v287_data * v323_data)).copy_to(ir1 + (112));
              float v328_data = s0[78];
              tensorforge::intel_esimd::simd<float, 9> v330_data;
              v330_data.copy_from(ir1 + (128));
              (v330_data + (v287_data * v328_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v332_data;
              v332_data.copy_from(r0 + (112));
              float v333_data = s0[7];
              tensorforge::intel_esimd::simd<float, 9> v335_data;
              v335_data.copy_from(ir1 + (0));
              (v335_data + (v332_data * v333_data)).copy_to(ir1 + (0));
              float v338_data = s0[16];
              tensorforge::intel_esimd::simd<float, 9> v340_data;
              v340_data.copy_from(ir1 + (16));
              (v340_data + (v332_data * v338_data)).copy_to(ir1 + (16));
              float v343_data = s0[25];
              tensorforge::intel_esimd::simd<float, 9> v345_data;
              v345_data.copy_from(ir1 + (32));
              (v345_data + (v332_data * v343_data)).copy_to(ir1 + (32));
              float v348_data = s0[34];
              tensorforge::intel_esimd::simd<float, 9> v350_data;
              v350_data.copy_from(ir1 + (48));
              (v350_data + (v332_data * v348_data)).copy_to(ir1 + (48));
              float v353_data = s0[43];
              tensorforge::intel_esimd::simd<float, 9> v355_data;
              v355_data.copy_from(ir1 + (64));
              (v355_data + (v332_data * v353_data)).copy_to(ir1 + (64));
              float v358_data = s0[52];
              tensorforge::intel_esimd::simd<float, 9> v360_data;
              v360_data.copy_from(ir1 + (80));
              (v360_data + (v332_data * v358_data)).copy_to(ir1 + (80));
              float v363_data = s0[61];
              tensorforge::intel_esimd::simd<float, 9> v365_data;
              v365_data.copy_from(ir1 + (96));
              (v365_data + (v332_data * v363_data)).copy_to(ir1 + (96));
              float v368_data = s0[70];
              tensorforge::intel_esimd::simd<float, 9> v370_data;
              v370_data.copy_from(ir1 + (112));
              (v370_data + (v332_data * v368_data)).copy_to(ir1 + (112));
              float v373_data = s0[79];
              tensorforge::intel_esimd::simd<float, 9> v375_data;
              v375_data.copy_from(ir1 + (128));
              (v375_data + (v332_data * v373_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v377_data;
              v377_data.copy_from(r0 + (128));
              float v378_data = s0[8];
              tensorforge::intel_esimd::simd<float, 9> v380_data;
              v380_data.copy_from(ir1 + (0));
              (v380_data + (v377_data * v378_data)).copy_to(ir1 + (0));
              float v383_data = s0[17];
              tensorforge::intel_esimd::simd<float, 9> v385_data;
              v385_data.copy_from(ir1 + (16));
              (v385_data + (v377_data * v383_data)).copy_to(ir1 + (16));
              float v388_data = s0[26];
              tensorforge::intel_esimd::simd<float, 9> v390_data;
              v390_data.copy_from(ir1 + (32));
              (v390_data + (v377_data * v388_data)).copy_to(ir1 + (32));
              float v393_data = s0[35];
              tensorforge::intel_esimd::simd<float, 9> v395_data;
              v395_data.copy_from(ir1 + (48));
              (v395_data + (v377_data * v393_data)).copy_to(ir1 + (48));
              float v398_data = s0[44];
              tensorforge::intel_esimd::simd<float, 9> v400_data;
              v400_data.copy_from(ir1 + (64));
              (v400_data + (v377_data * v398_data)).copy_to(ir1 + (64));
              float v403_data = s0[53];
              tensorforge::intel_esimd::simd<float, 9> v405_data;
              v405_data.copy_from(ir1 + (80));
              (v405_data + (v377_data * v403_data)).copy_to(ir1 + (80));
              float v408_data = s0[62];
              tensorforge::intel_esimd::simd<float, 9> v410_data;
              v410_data.copy_from(ir1 + (96));
              (v410_data + (v377_data * v408_data)).copy_to(ir1 + (96));
              float v413_data = s0[71];
              tensorforge::intel_esimd::simd<float, 9> v415_data;
              v415_data.copy_from(ir1 + (112));
              (v415_data + (v377_data * v413_data)).copy_to(ir1 + (112));
              float v418_data = s0[80];
              tensorforge::intel_esimd::simd<float, 9> v420_data;
              v420_data.copy_from(ir1 + (128));
              (v420_data + (v377_data * v418_data)).copy_to(ir1 + (128));
              #pragma unroll
              for (int32_t v423_n1 = 0; v423_n1 < 9; ++v423_n1) {
                int32_t v424_a = v423_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v426_data;
                v426_data.copy_from(ir1 + (v424_a));
                (v426_data * 13.0f).copy_to(r1 + (v424_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v430_i1 = 0; v430_i1 < 9; ++v430_i1) {
                tensorforge::intel_esimd::simd<float, 9> v433_data;
                v433_data.copy_from(r1 + ((v430_i1 * 16)));
                v433_data.copy_to(glb_m0 + ((v430_i1 * 9)));
              }
            }
          }
        }
      });
    }
  });
}

