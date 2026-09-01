// === base name ===
kernel_4b748443ff

// === header ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b748443ff(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4b748443ff(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4b748443ff(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8(8) {0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // OUT = +(TMP, dims=[1])
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
              float *const __restrict__ glb_m2 = &m2[batchId0 * 8 + 0 + m2_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v6_i1 = 0; v6_i1 < 8; ++v6_i1) {
                tensorforge::intel_esimd::simd<float, 8> v11_data;
                v11_data.copy_from(glb_m0 + ((v6_i1 * 8)));
                v11_data.copy_to(r0 + ((v6_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v15_ld;
              v15_ld.copy_from(glb_m1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v15_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
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
              float v38_data = s0[32];
              tensorforge::intel_esimd::simd<float, 8> v40_data;
              v40_data.copy_from(r1 + (64));
              (v40_data + (v17_data * v38_data)).copy_to(r1 + (64));
              float v43_data = s0[40];
              tensorforge::intel_esimd::simd<float, 8> v45_data;
              v45_data.copy_from(r1 + (80));
              (v45_data + (v17_data * v43_data)).copy_to(r1 + (80));
              float v48_data = s0[48];
              tensorforge::intel_esimd::simd<float, 8> v50_data;
              v50_data.copy_from(r1 + (96));
              (v50_data + (v17_data * v48_data)).copy_to(r1 + (96));
              float v53_data = s0[56];
              tensorforge::intel_esimd::simd<float, 8> v55_data;
              v55_data.copy_from(r1 + (112));
              (v55_data + (v17_data * v53_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v57_data;
              v57_data.copy_from(r0 + (16));
              float v58_data = s0[1];
              tensorforge::intel_esimd::simd<float, 8> v60_data;
              v60_data.copy_from(r1 + (0));
              (v60_data + (v57_data * v58_data)).copy_to(r1 + (0));
              float v63_data = s0[9];
              tensorforge::intel_esimd::simd<float, 8> v65_data;
              v65_data.copy_from(r1 + (16));
              (v65_data + (v57_data * v63_data)).copy_to(r1 + (16));
              float v68_data = s0[17];
              tensorforge::intel_esimd::simd<float, 8> v70_data;
              v70_data.copy_from(r1 + (32));
              (v70_data + (v57_data * v68_data)).copy_to(r1 + (32));
              float v73_data = s0[25];
              tensorforge::intel_esimd::simd<float, 8> v75_data;
              v75_data.copy_from(r1 + (48));
              (v75_data + (v57_data * v73_data)).copy_to(r1 + (48));
              float v78_data = s0[33];
              tensorforge::intel_esimd::simd<float, 8> v80_data;
              v80_data.copy_from(r1 + (64));
              (v80_data + (v57_data * v78_data)).copy_to(r1 + (64));
              float v83_data = s0[41];
              tensorforge::intel_esimd::simd<float, 8> v85_data;
              v85_data.copy_from(r1 + (80));
              (v85_data + (v57_data * v83_data)).copy_to(r1 + (80));
              float v88_data = s0[49];
              tensorforge::intel_esimd::simd<float, 8> v90_data;
              v90_data.copy_from(r1 + (96));
              (v90_data + (v57_data * v88_data)).copy_to(r1 + (96));
              float v93_data = s0[57];
              tensorforge::intel_esimd::simd<float, 8> v95_data;
              v95_data.copy_from(r1 + (112));
              (v95_data + (v57_data * v93_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v97_data;
              v97_data.copy_from(r0 + (32));
              float v98_data = s0[2];
              tensorforge::intel_esimd::simd<float, 8> v100_data;
              v100_data.copy_from(r1 + (0));
              (v100_data + (v97_data * v98_data)).copy_to(r1 + (0));
              float v103_data = s0[10];
              tensorforge::intel_esimd::simd<float, 8> v105_data;
              v105_data.copy_from(r1 + (16));
              (v105_data + (v97_data * v103_data)).copy_to(r1 + (16));
              float v108_data = s0[18];
              tensorforge::intel_esimd::simd<float, 8> v110_data;
              v110_data.copy_from(r1 + (32));
              (v110_data + (v97_data * v108_data)).copy_to(r1 + (32));
              float v113_data = s0[26];
              tensorforge::intel_esimd::simd<float, 8> v115_data;
              v115_data.copy_from(r1 + (48));
              (v115_data + (v97_data * v113_data)).copy_to(r1 + (48));
              float v118_data = s0[34];
              tensorforge::intel_esimd::simd<float, 8> v120_data;
              v120_data.copy_from(r1 + (64));
              (v120_data + (v97_data * v118_data)).copy_to(r1 + (64));
              float v123_data = s0[42];
              tensorforge::intel_esimd::simd<float, 8> v125_data;
              v125_data.copy_from(r1 + (80));
              (v125_data + (v97_data * v123_data)).copy_to(r1 + (80));
              float v128_data = s0[50];
              tensorforge::intel_esimd::simd<float, 8> v130_data;
              v130_data.copy_from(r1 + (96));
              (v130_data + (v97_data * v128_data)).copy_to(r1 + (96));
              float v133_data = s0[58];
              tensorforge::intel_esimd::simd<float, 8> v135_data;
              v135_data.copy_from(r1 + (112));
              (v135_data + (v97_data * v133_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v137_data;
              v137_data.copy_from(r0 + (48));
              float v138_data = s0[3];
              tensorforge::intel_esimd::simd<float, 8> v140_data;
              v140_data.copy_from(r1 + (0));
              (v140_data + (v137_data * v138_data)).copy_to(r1 + (0));
              float v143_data = s0[11];
              tensorforge::intel_esimd::simd<float, 8> v145_data;
              v145_data.copy_from(r1 + (16));
              (v145_data + (v137_data * v143_data)).copy_to(r1 + (16));
              float v148_data = s0[19];
              tensorforge::intel_esimd::simd<float, 8> v150_data;
              v150_data.copy_from(r1 + (32));
              (v150_data + (v137_data * v148_data)).copy_to(r1 + (32));
              float v153_data = s0[27];
              tensorforge::intel_esimd::simd<float, 8> v155_data;
              v155_data.copy_from(r1 + (48));
              (v155_data + (v137_data * v153_data)).copy_to(r1 + (48));
              float v158_data = s0[35];
              tensorforge::intel_esimd::simd<float, 8> v160_data;
              v160_data.copy_from(r1 + (64));
              (v160_data + (v137_data * v158_data)).copy_to(r1 + (64));
              float v163_data = s0[43];
              tensorforge::intel_esimd::simd<float, 8> v165_data;
              v165_data.copy_from(r1 + (80));
              (v165_data + (v137_data * v163_data)).copy_to(r1 + (80));
              float v168_data = s0[51];
              tensorforge::intel_esimd::simd<float, 8> v170_data;
              v170_data.copy_from(r1 + (96));
              (v170_data + (v137_data * v168_data)).copy_to(r1 + (96));
              float v173_data = s0[59];
              tensorforge::intel_esimd::simd<float, 8> v175_data;
              v175_data.copy_from(r1 + (112));
              (v175_data + (v137_data * v173_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v177_data;
              v177_data.copy_from(r0 + (64));
              float v178_data = s0[4];
              tensorforge::intel_esimd::simd<float, 8> v180_data;
              v180_data.copy_from(r1 + (0));
              (v180_data + (v177_data * v178_data)).copy_to(r1 + (0));
              float v183_data = s0[12];
              tensorforge::intel_esimd::simd<float, 8> v185_data;
              v185_data.copy_from(r1 + (16));
              (v185_data + (v177_data * v183_data)).copy_to(r1 + (16));
              float v188_data = s0[20];
              tensorforge::intel_esimd::simd<float, 8> v190_data;
              v190_data.copy_from(r1 + (32));
              (v190_data + (v177_data * v188_data)).copy_to(r1 + (32));
              float v193_data = s0[28];
              tensorforge::intel_esimd::simd<float, 8> v195_data;
              v195_data.copy_from(r1 + (48));
              (v195_data + (v177_data * v193_data)).copy_to(r1 + (48));
              float v198_data = s0[36];
              tensorforge::intel_esimd::simd<float, 8> v200_data;
              v200_data.copy_from(r1 + (64));
              (v200_data + (v177_data * v198_data)).copy_to(r1 + (64));
              float v203_data = s0[44];
              tensorforge::intel_esimd::simd<float, 8> v205_data;
              v205_data.copy_from(r1 + (80));
              (v205_data + (v177_data * v203_data)).copy_to(r1 + (80));
              float v208_data = s0[52];
              tensorforge::intel_esimd::simd<float, 8> v210_data;
              v210_data.copy_from(r1 + (96));
              (v210_data + (v177_data * v208_data)).copy_to(r1 + (96));
              float v213_data = s0[60];
              tensorforge::intel_esimd::simd<float, 8> v215_data;
              v215_data.copy_from(r1 + (112));
              (v215_data + (v177_data * v213_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v217_data;
              v217_data.copy_from(r0 + (80));
              float v218_data = s0[5];
              tensorforge::intel_esimd::simd<float, 8> v220_data;
              v220_data.copy_from(r1 + (0));
              (v220_data + (v217_data * v218_data)).copy_to(r1 + (0));
              float v223_data = s0[13];
              tensorforge::intel_esimd::simd<float, 8> v225_data;
              v225_data.copy_from(r1 + (16));
              (v225_data + (v217_data * v223_data)).copy_to(r1 + (16));
              float v228_data = s0[21];
              tensorforge::intel_esimd::simd<float, 8> v230_data;
              v230_data.copy_from(r1 + (32));
              (v230_data + (v217_data * v228_data)).copy_to(r1 + (32));
              float v233_data = s0[29];
              tensorforge::intel_esimd::simd<float, 8> v235_data;
              v235_data.copy_from(r1 + (48));
              (v235_data + (v217_data * v233_data)).copy_to(r1 + (48));
              float v238_data = s0[37];
              tensorforge::intel_esimd::simd<float, 8> v240_data;
              v240_data.copy_from(r1 + (64));
              (v240_data + (v217_data * v238_data)).copy_to(r1 + (64));
              float v243_data = s0[45];
              tensorforge::intel_esimd::simd<float, 8> v245_data;
              v245_data.copy_from(r1 + (80));
              (v245_data + (v217_data * v243_data)).copy_to(r1 + (80));
              float v248_data = s0[53];
              tensorforge::intel_esimd::simd<float, 8> v250_data;
              v250_data.copy_from(r1 + (96));
              (v250_data + (v217_data * v248_data)).copy_to(r1 + (96));
              float v253_data = s0[61];
              tensorforge::intel_esimd::simd<float, 8> v255_data;
              v255_data.copy_from(r1 + (112));
              (v255_data + (v217_data * v253_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v257_data;
              v257_data.copy_from(r0 + (96));
              float v258_data = s0[6];
              tensorforge::intel_esimd::simd<float, 8> v260_data;
              v260_data.copy_from(r1 + (0));
              (v260_data + (v257_data * v258_data)).copy_to(r1 + (0));
              float v263_data = s0[14];
              tensorforge::intel_esimd::simd<float, 8> v265_data;
              v265_data.copy_from(r1 + (16));
              (v265_data + (v257_data * v263_data)).copy_to(r1 + (16));
              float v268_data = s0[22];
              tensorforge::intel_esimd::simd<float, 8> v270_data;
              v270_data.copy_from(r1 + (32));
              (v270_data + (v257_data * v268_data)).copy_to(r1 + (32));
              float v273_data = s0[30];
              tensorforge::intel_esimd::simd<float, 8> v275_data;
              v275_data.copy_from(r1 + (48));
              (v275_data + (v257_data * v273_data)).copy_to(r1 + (48));
              float v278_data = s0[38];
              tensorforge::intel_esimd::simd<float, 8> v280_data;
              v280_data.copy_from(r1 + (64));
              (v280_data + (v257_data * v278_data)).copy_to(r1 + (64));
              float v283_data = s0[46];
              tensorforge::intel_esimd::simd<float, 8> v285_data;
              v285_data.copy_from(r1 + (80));
              (v285_data + (v257_data * v283_data)).copy_to(r1 + (80));
              float v288_data = s0[54];
              tensorforge::intel_esimd::simd<float, 8> v290_data;
              v290_data.copy_from(r1 + (96));
              (v290_data + (v257_data * v288_data)).copy_to(r1 + (96));
              float v293_data = s0[62];
              tensorforge::intel_esimd::simd<float, 8> v295_data;
              v295_data.copy_from(r1 + (112));
              (v295_data + (v257_data * v293_data)).copy_to(r1 + (112));
              tensorforge::intel_esimd::simd<float, 8> v297_data;
              v297_data.copy_from(r0 + (112));
              float v298_data = s0[7];
              tensorforge::intel_esimd::simd<float, 8> v300_data;
              v300_data.copy_from(r1 + (0));
              (v300_data + (v297_data * v298_data)).copy_to(r1 + (0));
              float v303_data = s0[15];
              tensorforge::intel_esimd::simd<float, 8> v305_data;
              v305_data.copy_from(r1 + (16));
              (v305_data + (v297_data * v303_data)).copy_to(r1 + (16));
              float v308_data = s0[23];
              tensorforge::intel_esimd::simd<float, 8> v310_data;
              v310_data.copy_from(r1 + (32));
              (v310_data + (v297_data * v308_data)).copy_to(r1 + (32));
              float v313_data = s0[31];
              tensorforge::intel_esimd::simd<float, 8> v315_data;
              v315_data.copy_from(r1 + (48));
              (v315_data + (v297_data * v313_data)).copy_to(r1 + (48));
              float v318_data = s0[39];
              tensorforge::intel_esimd::simd<float, 8> v320_data;
              v320_data.copy_from(r1 + (64));
              (v320_data + (v297_data * v318_data)).copy_to(r1 + (64));
              float v323_data = s0[47];
              tensorforge::intel_esimd::simd<float, 8> v325_data;
              v325_data.copy_from(r1 + (80));
              (v325_data + (v297_data * v323_data)).copy_to(r1 + (80));
              float v328_data = s0[55];
              tensorforge::intel_esimd::simd<float, 8> v330_data;
              v330_data.copy_from(r1 + (96));
              (v330_data + (v297_data * v328_data)).copy_to(r1 + (96));
              float v333_data = s0[63];
              tensorforge::intel_esimd::simd<float, 8> v335_data;
              v335_data.copy_from(r1 + (112));
              (v335_data + (v297_data * v333_data)).copy_to(r1 + (112));
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v338_i1 = 0; v338_i1 < 8; ++v338_i1) {
                tensorforge::intel_esimd::simd<float, 8> v341_data;
                v341_data.copy_from(r1 + ((v338_i1 * 16)));
                v341_data.copy_to(s1 + ((v338_i1 * 8)));
              }
              // glb_m2 = +(s1, dims=[1])
              tensorforge::intel_esimd::simd<float, 8> v347_acc0(0.0f);
              #pragma unroll
              for (int32_t v346_r1 = 0; v346_r1 < 8; ++v346_r1) {
                tensorforge::intel_esimd::simd<float, 8> v352_data;
                v352_data.copy_from(s1 + ((v346_r1 * 8)));
                v347_acc0 = (v347_acc0 + v352_data);
              }
              v347_acc0.copy_to(glb_m2 + (0_i32));
            }
          }
        }
      });
    }
  });
}

