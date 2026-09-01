// === base name ===
kernel_a587425bdd

// === header ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a587425bdd(const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a587425bdd(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a587425bdd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // TMP = abs(A)
        // m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v6_ld;
              v6_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v6_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              float r0[128]{};
              // r0 = abs(glb_m0)
              #pragma unroll
              for (int32_t v8_k1 = 0; v8_k1 < 8; ++v8_k1) {
                tensorforge::intel_esimd::simd<float, 8> v13_data;
                v13_data.copy_from(glb_m0 + ((v8_k1 * 8)));
                (tensorforge::intel_esimd::abs(v13_data)).copy_to(r0 + ((v8_k1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 8> v19_data;
              v19_data.copy_from(r0 + (0));
              float v20_data = s1[0];
              tensorforge::intel_esimd::simd<float, 8> v22_data;
              v22_data.copy_from(ir1 + (0));
              (v22_data + (v19_data * v20_data)).copy_to(ir1 + (0));
              float v25_data = s1[8];
              tensorforge::intel_esimd::simd<float, 8> v27_data;
              v27_data.copy_from(ir1 + (16));
              (v27_data + (v19_data * v25_data)).copy_to(ir1 + (16));
              float v30_data = s1[16];
              tensorforge::intel_esimd::simd<float, 8> v32_data;
              v32_data.copy_from(ir1 + (32));
              (v32_data + (v19_data * v30_data)).copy_to(ir1 + (32));
              float v35_data = s1[24];
              tensorforge::intel_esimd::simd<float, 8> v37_data;
              v37_data.copy_from(ir1 + (48));
              (v37_data + (v19_data * v35_data)).copy_to(ir1 + (48));
              float v40_data = s1[32];
              tensorforge::intel_esimd::simd<float, 8> v42_data;
              v42_data.copy_from(ir1 + (64));
              (v42_data + (v19_data * v40_data)).copy_to(ir1 + (64));
              float v45_data = s1[40];
              tensorforge::intel_esimd::simd<float, 8> v47_data;
              v47_data.copy_from(ir1 + (80));
              (v47_data + (v19_data * v45_data)).copy_to(ir1 + (80));
              float v50_data = s1[48];
              tensorforge::intel_esimd::simd<float, 8> v52_data;
              v52_data.copy_from(ir1 + (96));
              (v52_data + (v19_data * v50_data)).copy_to(ir1 + (96));
              float v55_data = s1[56];
              tensorforge::intel_esimd::simd<float, 8> v57_data;
              v57_data.copy_from(ir1 + (112));
              (v57_data + (v19_data * v55_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v59_data;
              v59_data.copy_from(r0 + (16));
              float v60_data = s1[1];
              tensorforge::intel_esimd::simd<float, 8> v62_data;
              v62_data.copy_from(ir1 + (0));
              (v62_data + (v59_data * v60_data)).copy_to(ir1 + (0));
              float v65_data = s1[9];
              tensorforge::intel_esimd::simd<float, 8> v67_data;
              v67_data.copy_from(ir1 + (16));
              (v67_data + (v59_data * v65_data)).copy_to(ir1 + (16));
              float v70_data = s1[17];
              tensorforge::intel_esimd::simd<float, 8> v72_data;
              v72_data.copy_from(ir1 + (32));
              (v72_data + (v59_data * v70_data)).copy_to(ir1 + (32));
              float v75_data = s1[25];
              tensorforge::intel_esimd::simd<float, 8> v77_data;
              v77_data.copy_from(ir1 + (48));
              (v77_data + (v59_data * v75_data)).copy_to(ir1 + (48));
              float v80_data = s1[33];
              tensorforge::intel_esimd::simd<float, 8> v82_data;
              v82_data.copy_from(ir1 + (64));
              (v82_data + (v59_data * v80_data)).copy_to(ir1 + (64));
              float v85_data = s1[41];
              tensorforge::intel_esimd::simd<float, 8> v87_data;
              v87_data.copy_from(ir1 + (80));
              (v87_data + (v59_data * v85_data)).copy_to(ir1 + (80));
              float v90_data = s1[49];
              tensorforge::intel_esimd::simd<float, 8> v92_data;
              v92_data.copy_from(ir1 + (96));
              (v92_data + (v59_data * v90_data)).copy_to(ir1 + (96));
              float v95_data = s1[57];
              tensorforge::intel_esimd::simd<float, 8> v97_data;
              v97_data.copy_from(ir1 + (112));
              (v97_data + (v59_data * v95_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v99_data;
              v99_data.copy_from(r0 + (32));
              float v100_data = s1[2];
              tensorforge::intel_esimd::simd<float, 8> v102_data;
              v102_data.copy_from(ir1 + (0));
              (v102_data + (v99_data * v100_data)).copy_to(ir1 + (0));
              float v105_data = s1[10];
              tensorforge::intel_esimd::simd<float, 8> v107_data;
              v107_data.copy_from(ir1 + (16));
              (v107_data + (v99_data * v105_data)).copy_to(ir1 + (16));
              float v110_data = s1[18];
              tensorforge::intel_esimd::simd<float, 8> v112_data;
              v112_data.copy_from(ir1 + (32));
              (v112_data + (v99_data * v110_data)).copy_to(ir1 + (32));
              float v115_data = s1[26];
              tensorforge::intel_esimd::simd<float, 8> v117_data;
              v117_data.copy_from(ir1 + (48));
              (v117_data + (v99_data * v115_data)).copy_to(ir1 + (48));
              float v120_data = s1[34];
              tensorforge::intel_esimd::simd<float, 8> v122_data;
              v122_data.copy_from(ir1 + (64));
              (v122_data + (v99_data * v120_data)).copy_to(ir1 + (64));
              float v125_data = s1[42];
              tensorforge::intel_esimd::simd<float, 8> v127_data;
              v127_data.copy_from(ir1 + (80));
              (v127_data + (v99_data * v125_data)).copy_to(ir1 + (80));
              float v130_data = s1[50];
              tensorforge::intel_esimd::simd<float, 8> v132_data;
              v132_data.copy_from(ir1 + (96));
              (v132_data + (v99_data * v130_data)).copy_to(ir1 + (96));
              float v135_data = s1[58];
              tensorforge::intel_esimd::simd<float, 8> v137_data;
              v137_data.copy_from(ir1 + (112));
              (v137_data + (v99_data * v135_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v139_data;
              v139_data.copy_from(r0 + (48));
              float v140_data = s1[3];
              tensorforge::intel_esimd::simd<float, 8> v142_data;
              v142_data.copy_from(ir1 + (0));
              (v142_data + (v139_data * v140_data)).copy_to(ir1 + (0));
              float v145_data = s1[11];
              tensorforge::intel_esimd::simd<float, 8> v147_data;
              v147_data.copy_from(ir1 + (16));
              (v147_data + (v139_data * v145_data)).copy_to(ir1 + (16));
              float v150_data = s1[19];
              tensorforge::intel_esimd::simd<float, 8> v152_data;
              v152_data.copy_from(ir1 + (32));
              (v152_data + (v139_data * v150_data)).copy_to(ir1 + (32));
              float v155_data = s1[27];
              tensorforge::intel_esimd::simd<float, 8> v157_data;
              v157_data.copy_from(ir1 + (48));
              (v157_data + (v139_data * v155_data)).copy_to(ir1 + (48));
              float v160_data = s1[35];
              tensorforge::intel_esimd::simd<float, 8> v162_data;
              v162_data.copy_from(ir1 + (64));
              (v162_data + (v139_data * v160_data)).copy_to(ir1 + (64));
              float v165_data = s1[43];
              tensorforge::intel_esimd::simd<float, 8> v167_data;
              v167_data.copy_from(ir1 + (80));
              (v167_data + (v139_data * v165_data)).copy_to(ir1 + (80));
              float v170_data = s1[51];
              tensorforge::intel_esimd::simd<float, 8> v172_data;
              v172_data.copy_from(ir1 + (96));
              (v172_data + (v139_data * v170_data)).copy_to(ir1 + (96));
              float v175_data = s1[59];
              tensorforge::intel_esimd::simd<float, 8> v177_data;
              v177_data.copy_from(ir1 + (112));
              (v177_data + (v139_data * v175_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v179_data;
              v179_data.copy_from(r0 + (64));
              float v180_data = s1[4];
              tensorforge::intel_esimd::simd<float, 8> v182_data;
              v182_data.copy_from(ir1 + (0));
              (v182_data + (v179_data * v180_data)).copy_to(ir1 + (0));
              float v185_data = s1[12];
              tensorforge::intel_esimd::simd<float, 8> v187_data;
              v187_data.copy_from(ir1 + (16));
              (v187_data + (v179_data * v185_data)).copy_to(ir1 + (16));
              float v190_data = s1[20];
              tensorforge::intel_esimd::simd<float, 8> v192_data;
              v192_data.copy_from(ir1 + (32));
              (v192_data + (v179_data * v190_data)).copy_to(ir1 + (32));
              float v195_data = s1[28];
              tensorforge::intel_esimd::simd<float, 8> v197_data;
              v197_data.copy_from(ir1 + (48));
              (v197_data + (v179_data * v195_data)).copy_to(ir1 + (48));
              float v200_data = s1[36];
              tensorforge::intel_esimd::simd<float, 8> v202_data;
              v202_data.copy_from(ir1 + (64));
              (v202_data + (v179_data * v200_data)).copy_to(ir1 + (64));
              float v205_data = s1[44];
              tensorforge::intel_esimd::simd<float, 8> v207_data;
              v207_data.copy_from(ir1 + (80));
              (v207_data + (v179_data * v205_data)).copy_to(ir1 + (80));
              float v210_data = s1[52];
              tensorforge::intel_esimd::simd<float, 8> v212_data;
              v212_data.copy_from(ir1 + (96));
              (v212_data + (v179_data * v210_data)).copy_to(ir1 + (96));
              float v215_data = s1[60];
              tensorforge::intel_esimd::simd<float, 8> v217_data;
              v217_data.copy_from(ir1 + (112));
              (v217_data + (v179_data * v215_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v219_data;
              v219_data.copy_from(r0 + (80));
              float v220_data = s1[5];
              tensorforge::intel_esimd::simd<float, 8> v222_data;
              v222_data.copy_from(ir1 + (0));
              (v222_data + (v219_data * v220_data)).copy_to(ir1 + (0));
              float v225_data = s1[13];
              tensorforge::intel_esimd::simd<float, 8> v227_data;
              v227_data.copy_from(ir1 + (16));
              (v227_data + (v219_data * v225_data)).copy_to(ir1 + (16));
              float v230_data = s1[21];
              tensorforge::intel_esimd::simd<float, 8> v232_data;
              v232_data.copy_from(ir1 + (32));
              (v232_data + (v219_data * v230_data)).copy_to(ir1 + (32));
              float v235_data = s1[29];
              tensorforge::intel_esimd::simd<float, 8> v237_data;
              v237_data.copy_from(ir1 + (48));
              (v237_data + (v219_data * v235_data)).copy_to(ir1 + (48));
              float v240_data = s1[37];
              tensorforge::intel_esimd::simd<float, 8> v242_data;
              v242_data.copy_from(ir1 + (64));
              (v242_data + (v219_data * v240_data)).copy_to(ir1 + (64));
              float v245_data = s1[45];
              tensorforge::intel_esimd::simd<float, 8> v247_data;
              v247_data.copy_from(ir1 + (80));
              (v247_data + (v219_data * v245_data)).copy_to(ir1 + (80));
              float v250_data = s1[53];
              tensorforge::intel_esimd::simd<float, 8> v252_data;
              v252_data.copy_from(ir1 + (96));
              (v252_data + (v219_data * v250_data)).copy_to(ir1 + (96));
              float v255_data = s1[61];
              tensorforge::intel_esimd::simd<float, 8> v257_data;
              v257_data.copy_from(ir1 + (112));
              (v257_data + (v219_data * v255_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v259_data;
              v259_data.copy_from(r0 + (96));
              float v260_data = s1[6];
              tensorforge::intel_esimd::simd<float, 8> v262_data;
              v262_data.copy_from(ir1 + (0));
              (v262_data + (v259_data * v260_data)).copy_to(ir1 + (0));
              float v265_data = s1[14];
              tensorforge::intel_esimd::simd<float, 8> v267_data;
              v267_data.copy_from(ir1 + (16));
              (v267_data + (v259_data * v265_data)).copy_to(ir1 + (16));
              float v270_data = s1[22];
              tensorforge::intel_esimd::simd<float, 8> v272_data;
              v272_data.copy_from(ir1 + (32));
              (v272_data + (v259_data * v270_data)).copy_to(ir1 + (32));
              float v275_data = s1[30];
              tensorforge::intel_esimd::simd<float, 8> v277_data;
              v277_data.copy_from(ir1 + (48));
              (v277_data + (v259_data * v275_data)).copy_to(ir1 + (48));
              float v280_data = s1[38];
              tensorforge::intel_esimd::simd<float, 8> v282_data;
              v282_data.copy_from(ir1 + (64));
              (v282_data + (v259_data * v280_data)).copy_to(ir1 + (64));
              float v285_data = s1[46];
              tensorforge::intel_esimd::simd<float, 8> v287_data;
              v287_data.copy_from(ir1 + (80));
              (v287_data + (v259_data * v285_data)).copy_to(ir1 + (80));
              float v290_data = s1[54];
              tensorforge::intel_esimd::simd<float, 8> v292_data;
              v292_data.copy_from(ir1 + (96));
              (v292_data + (v259_data * v290_data)).copy_to(ir1 + (96));
              float v295_data = s1[62];
              tensorforge::intel_esimd::simd<float, 8> v297_data;
              v297_data.copy_from(ir1 + (112));
              (v297_data + (v259_data * v295_data)).copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v299_data;
              v299_data.copy_from(r0 + (112));
              float v300_data = s1[7];
              tensorforge::intel_esimd::simd<float, 8> v302_data;
              v302_data.copy_from(ir1 + (0));
              (v302_data + (v299_data * v300_data)).copy_to(ir1 + (0));
              float v305_data = s1[15];
              tensorforge::intel_esimd::simd<float, 8> v307_data;
              v307_data.copy_from(ir1 + (16));
              (v307_data + (v299_data * v305_data)).copy_to(ir1 + (16));
              float v310_data = s1[23];
              tensorforge::intel_esimd::simd<float, 8> v312_data;
              v312_data.copy_from(ir1 + (32));
              (v312_data + (v299_data * v310_data)).copy_to(ir1 + (32));
              float v315_data = s1[31];
              tensorforge::intel_esimd::simd<float, 8> v317_data;
              v317_data.copy_from(ir1 + (48));
              (v317_data + (v299_data * v315_data)).copy_to(ir1 + (48));
              float v320_data = s1[39];
              tensorforge::intel_esimd::simd<float, 8> v322_data;
              v322_data.copy_from(ir1 + (64));
              (v322_data + (v299_data * v320_data)).copy_to(ir1 + (64));
              float v325_data = s1[47];
              tensorforge::intel_esimd::simd<float, 8> v327_data;
              v327_data.copy_from(ir1 + (80));
              (v327_data + (v299_data * v325_data)).copy_to(ir1 + (80));
              float v330_data = s1[55];
              tensorforge::intel_esimd::simd<float, 8> v332_data;
              v332_data.copy_from(ir1 + (96));
              (v332_data + (v299_data * v330_data)).copy_to(ir1 + (96));
              float v335_data = s1[63];
              tensorforge::intel_esimd::simd<float, 8> v337_data;
              v337_data.copy_from(ir1 + (112));
              (v337_data + (v299_data * v335_data)).copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v339_n1 = 0; v339_n1 < 8; ++v339_n1) {
                int32_t v340_a = v339_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v342_data;
                v342_data.copy_from(ir1 + (v340_a));
                v342_data.copy_to(r1 + (v340_a));
              }
              // glb_m1 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v345_i1 = 0; v345_i1 < 8; ++v345_i1) {
                tensorforge::intel_esimd::simd<float, 8> v348_data;
                v348_data.copy_from(r1 + ((v345_i1 * 16)));
                v348_data.copy_to(glb_m1 + ((v345_i1 * 8)));
              }
            }
          }
        }
      });
    }
  });
}

