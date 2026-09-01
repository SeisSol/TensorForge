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
              tensorforge::intel_esimd::simd<float, 64> v15_ld;
              v15_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v15_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 16> v16_ld;
              v16_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              v16_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              if (item.get_local_id(0) < 1) {
                tensorforge::intel_esimd::simd<float, 16> v17_ld;
                v17_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 80));
                v17_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 80));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[144]{};
              // r1 = +(r0 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir1[144]{};
              tensorforge::intel_esimd::simd<float, 9> v20_data;
              v20_data.copy_from(r0 + (0));
              float v21_data = s0[0];
              tensorforge::intel_esimd::simd<float, 9> v23_data;
              v23_data.copy_from(ir1 + (0));
              (v23_data + (v20_data * v21_data)).copy_to(ir1 + (0));
              float v26_data = s0[9];
              tensorforge::intel_esimd::simd<float, 9> v28_data;
              v28_data.copy_from(ir1 + (16));
              (v28_data + (v20_data * v26_data)).copy_to(ir1 + (16));
              float v31_data = s0[18];
              tensorforge::intel_esimd::simd<float, 9> v33_data;
              v33_data.copy_from(ir1 + (32));
              (v33_data + (v20_data * v31_data)).copy_to(ir1 + (32));
              float v36_data = s0[27];
              tensorforge::intel_esimd::simd<float, 9> v38_data;
              v38_data.copy_from(ir1 + (48));
              (v38_data + (v20_data * v36_data)).copy_to(ir1 + (48));
              float v41_data = s0[36];
              tensorforge::intel_esimd::simd<float, 9> v43_data;
              v43_data.copy_from(ir1 + (64));
              (v43_data + (v20_data * v41_data)).copy_to(ir1 + (64));
              float v46_data = s0[45];
              tensorforge::intel_esimd::simd<float, 9> v48_data;
              v48_data.copy_from(ir1 + (80));
              (v48_data + (v20_data * v46_data)).copy_to(ir1 + (80));
              float v51_data = s0[54];
              tensorforge::intel_esimd::simd<float, 9> v53_data;
              v53_data.copy_from(ir1 + (96));
              (v53_data + (v20_data * v51_data)).copy_to(ir1 + (96));
              float v56_data = s0[63];
              tensorforge::intel_esimd::simd<float, 9> v58_data;
              v58_data.copy_from(ir1 + (112));
              (v58_data + (v20_data * v56_data)).copy_to(ir1 + (112));
              float v61_data = s0[72];
              tensorforge::intel_esimd::simd<float, 9> v63_data;
              v63_data.copy_from(ir1 + (128));
              (v63_data + (v20_data * v61_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v65_data;
              v65_data.copy_from(r0 + (16));
              float v66_data = s0[1];
              tensorforge::intel_esimd::simd<float, 9> v68_data;
              v68_data.copy_from(ir1 + (0));
              (v68_data + (v65_data * v66_data)).copy_to(ir1 + (0));
              float v71_data = s0[10];
              tensorforge::intel_esimd::simd<float, 9> v73_data;
              v73_data.copy_from(ir1 + (16));
              (v73_data + (v65_data * v71_data)).copy_to(ir1 + (16));
              float v76_data = s0[19];
              tensorforge::intel_esimd::simd<float, 9> v78_data;
              v78_data.copy_from(ir1 + (32));
              (v78_data + (v65_data * v76_data)).copy_to(ir1 + (32));
              float v81_data = s0[28];
              tensorforge::intel_esimd::simd<float, 9> v83_data;
              v83_data.copy_from(ir1 + (48));
              (v83_data + (v65_data * v81_data)).copy_to(ir1 + (48));
              float v86_data = s0[37];
              tensorforge::intel_esimd::simd<float, 9> v88_data;
              v88_data.copy_from(ir1 + (64));
              (v88_data + (v65_data * v86_data)).copy_to(ir1 + (64));
              float v91_data = s0[46];
              tensorforge::intel_esimd::simd<float, 9> v93_data;
              v93_data.copy_from(ir1 + (80));
              (v93_data + (v65_data * v91_data)).copy_to(ir1 + (80));
              float v96_data = s0[55];
              tensorforge::intel_esimd::simd<float, 9> v98_data;
              v98_data.copy_from(ir1 + (96));
              (v98_data + (v65_data * v96_data)).copy_to(ir1 + (96));
              float v101_data = s0[64];
              tensorforge::intel_esimd::simd<float, 9> v103_data;
              v103_data.copy_from(ir1 + (112));
              (v103_data + (v65_data * v101_data)).copy_to(ir1 + (112));
              float v106_data = s0[73];
              tensorforge::intel_esimd::simd<float, 9> v108_data;
              v108_data.copy_from(ir1 + (128));
              (v108_data + (v65_data * v106_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v110_data;
              v110_data.copy_from(r0 + (32));
              float v111_data = s0[2];
              tensorforge::intel_esimd::simd<float, 9> v113_data;
              v113_data.copy_from(ir1 + (0));
              (v113_data + (v110_data * v111_data)).copy_to(ir1 + (0));
              float v116_data = s0[11];
              tensorforge::intel_esimd::simd<float, 9> v118_data;
              v118_data.copy_from(ir1 + (16));
              (v118_data + (v110_data * v116_data)).copy_to(ir1 + (16));
              float v121_data = s0[20];
              tensorforge::intel_esimd::simd<float, 9> v123_data;
              v123_data.copy_from(ir1 + (32));
              (v123_data + (v110_data * v121_data)).copy_to(ir1 + (32));
              float v126_data = s0[29];
              tensorforge::intel_esimd::simd<float, 9> v128_data;
              v128_data.copy_from(ir1 + (48));
              (v128_data + (v110_data * v126_data)).copy_to(ir1 + (48));
              float v131_data = s0[38];
              tensorforge::intel_esimd::simd<float, 9> v133_data;
              v133_data.copy_from(ir1 + (64));
              (v133_data + (v110_data * v131_data)).copy_to(ir1 + (64));
              float v136_data = s0[47];
              tensorforge::intel_esimd::simd<float, 9> v138_data;
              v138_data.copy_from(ir1 + (80));
              (v138_data + (v110_data * v136_data)).copy_to(ir1 + (80));
              float v141_data = s0[56];
              tensorforge::intel_esimd::simd<float, 9> v143_data;
              v143_data.copy_from(ir1 + (96));
              (v143_data + (v110_data * v141_data)).copy_to(ir1 + (96));
              float v146_data = s0[65];
              tensorforge::intel_esimd::simd<float, 9> v148_data;
              v148_data.copy_from(ir1 + (112));
              (v148_data + (v110_data * v146_data)).copy_to(ir1 + (112));
              float v151_data = s0[74];
              tensorforge::intel_esimd::simd<float, 9> v153_data;
              v153_data.copy_from(ir1 + (128));
              (v153_data + (v110_data * v151_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v155_data;
              v155_data.copy_from(r0 + (48));
              float v156_data = s0[3];
              tensorforge::intel_esimd::simd<float, 9> v158_data;
              v158_data.copy_from(ir1 + (0));
              (v158_data + (v155_data * v156_data)).copy_to(ir1 + (0));
              float v161_data = s0[12];
              tensorforge::intel_esimd::simd<float, 9> v163_data;
              v163_data.copy_from(ir1 + (16));
              (v163_data + (v155_data * v161_data)).copy_to(ir1 + (16));
              float v166_data = s0[21];
              tensorforge::intel_esimd::simd<float, 9> v168_data;
              v168_data.copy_from(ir1 + (32));
              (v168_data + (v155_data * v166_data)).copy_to(ir1 + (32));
              float v171_data = s0[30];
              tensorforge::intel_esimd::simd<float, 9> v173_data;
              v173_data.copy_from(ir1 + (48));
              (v173_data + (v155_data * v171_data)).copy_to(ir1 + (48));
              float v176_data = s0[39];
              tensorforge::intel_esimd::simd<float, 9> v178_data;
              v178_data.copy_from(ir1 + (64));
              (v178_data + (v155_data * v176_data)).copy_to(ir1 + (64));
              float v181_data = s0[48];
              tensorforge::intel_esimd::simd<float, 9> v183_data;
              v183_data.copy_from(ir1 + (80));
              (v183_data + (v155_data * v181_data)).copy_to(ir1 + (80));
              float v186_data = s0[57];
              tensorforge::intel_esimd::simd<float, 9> v188_data;
              v188_data.copy_from(ir1 + (96));
              (v188_data + (v155_data * v186_data)).copy_to(ir1 + (96));
              float v191_data = s0[66];
              tensorforge::intel_esimd::simd<float, 9> v193_data;
              v193_data.copy_from(ir1 + (112));
              (v193_data + (v155_data * v191_data)).copy_to(ir1 + (112));
              float v196_data = s0[75];
              tensorforge::intel_esimd::simd<float, 9> v198_data;
              v198_data.copy_from(ir1 + (128));
              (v198_data + (v155_data * v196_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v200_data;
              v200_data.copy_from(r0 + (64));
              float v201_data = s0[4];
              tensorforge::intel_esimd::simd<float, 9> v203_data;
              v203_data.copy_from(ir1 + (0));
              (v203_data + (v200_data * v201_data)).copy_to(ir1 + (0));
              float v206_data = s0[13];
              tensorforge::intel_esimd::simd<float, 9> v208_data;
              v208_data.copy_from(ir1 + (16));
              (v208_data + (v200_data * v206_data)).copy_to(ir1 + (16));
              float v211_data = s0[22];
              tensorforge::intel_esimd::simd<float, 9> v213_data;
              v213_data.copy_from(ir1 + (32));
              (v213_data + (v200_data * v211_data)).copy_to(ir1 + (32));
              float v216_data = s0[31];
              tensorforge::intel_esimd::simd<float, 9> v218_data;
              v218_data.copy_from(ir1 + (48));
              (v218_data + (v200_data * v216_data)).copy_to(ir1 + (48));
              float v221_data = s0[40];
              tensorforge::intel_esimd::simd<float, 9> v223_data;
              v223_data.copy_from(ir1 + (64));
              (v223_data + (v200_data * v221_data)).copy_to(ir1 + (64));
              float v226_data = s0[49];
              tensorforge::intel_esimd::simd<float, 9> v228_data;
              v228_data.copy_from(ir1 + (80));
              (v228_data + (v200_data * v226_data)).copy_to(ir1 + (80));
              float v231_data = s0[58];
              tensorforge::intel_esimd::simd<float, 9> v233_data;
              v233_data.copy_from(ir1 + (96));
              (v233_data + (v200_data * v231_data)).copy_to(ir1 + (96));
              float v236_data = s0[67];
              tensorforge::intel_esimd::simd<float, 9> v238_data;
              v238_data.copy_from(ir1 + (112));
              (v238_data + (v200_data * v236_data)).copy_to(ir1 + (112));
              float v241_data = s0[76];
              tensorforge::intel_esimd::simd<float, 9> v243_data;
              v243_data.copy_from(ir1 + (128));
              (v243_data + (v200_data * v241_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v245_data;
              v245_data.copy_from(r0 + (80));
              float v246_data = s0[5];
              tensorforge::intel_esimd::simd<float, 9> v248_data;
              v248_data.copy_from(ir1 + (0));
              (v248_data + (v245_data * v246_data)).copy_to(ir1 + (0));
              float v251_data = s0[14];
              tensorforge::intel_esimd::simd<float, 9> v253_data;
              v253_data.copy_from(ir1 + (16));
              (v253_data + (v245_data * v251_data)).copy_to(ir1 + (16));
              float v256_data = s0[23];
              tensorforge::intel_esimd::simd<float, 9> v258_data;
              v258_data.copy_from(ir1 + (32));
              (v258_data + (v245_data * v256_data)).copy_to(ir1 + (32));
              float v261_data = s0[32];
              tensorforge::intel_esimd::simd<float, 9> v263_data;
              v263_data.copy_from(ir1 + (48));
              (v263_data + (v245_data * v261_data)).copy_to(ir1 + (48));
              float v266_data = s0[41];
              tensorforge::intel_esimd::simd<float, 9> v268_data;
              v268_data.copy_from(ir1 + (64));
              (v268_data + (v245_data * v266_data)).copy_to(ir1 + (64));
              float v271_data = s0[50];
              tensorforge::intel_esimd::simd<float, 9> v273_data;
              v273_data.copy_from(ir1 + (80));
              (v273_data + (v245_data * v271_data)).copy_to(ir1 + (80));
              float v276_data = s0[59];
              tensorforge::intel_esimd::simd<float, 9> v278_data;
              v278_data.copy_from(ir1 + (96));
              (v278_data + (v245_data * v276_data)).copy_to(ir1 + (96));
              float v281_data = s0[68];
              tensorforge::intel_esimd::simd<float, 9> v283_data;
              v283_data.copy_from(ir1 + (112));
              (v283_data + (v245_data * v281_data)).copy_to(ir1 + (112));
              float v286_data = s0[77];
              tensorforge::intel_esimd::simd<float, 9> v288_data;
              v288_data.copy_from(ir1 + (128));
              (v288_data + (v245_data * v286_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v290_data;
              v290_data.copy_from(r0 + (96));
              float v291_data = s0[6];
              tensorforge::intel_esimd::simd<float, 9> v293_data;
              v293_data.copy_from(ir1 + (0));
              (v293_data + (v290_data * v291_data)).copy_to(ir1 + (0));
              float v296_data = s0[15];
              tensorforge::intel_esimd::simd<float, 9> v298_data;
              v298_data.copy_from(ir1 + (16));
              (v298_data + (v290_data * v296_data)).copy_to(ir1 + (16));
              float v301_data = s0[24];
              tensorforge::intel_esimd::simd<float, 9> v303_data;
              v303_data.copy_from(ir1 + (32));
              (v303_data + (v290_data * v301_data)).copy_to(ir1 + (32));
              float v306_data = s0[33];
              tensorforge::intel_esimd::simd<float, 9> v308_data;
              v308_data.copy_from(ir1 + (48));
              (v308_data + (v290_data * v306_data)).copy_to(ir1 + (48));
              float v311_data = s0[42];
              tensorforge::intel_esimd::simd<float, 9> v313_data;
              v313_data.copy_from(ir1 + (64));
              (v313_data + (v290_data * v311_data)).copy_to(ir1 + (64));
              float v316_data = s0[51];
              tensorforge::intel_esimd::simd<float, 9> v318_data;
              v318_data.copy_from(ir1 + (80));
              (v318_data + (v290_data * v316_data)).copy_to(ir1 + (80));
              float v321_data = s0[60];
              tensorforge::intel_esimd::simd<float, 9> v323_data;
              v323_data.copy_from(ir1 + (96));
              (v323_data + (v290_data * v321_data)).copy_to(ir1 + (96));
              float v326_data = s0[69];
              tensorforge::intel_esimd::simd<float, 9> v328_data;
              v328_data.copy_from(ir1 + (112));
              (v328_data + (v290_data * v326_data)).copy_to(ir1 + (112));
              float v331_data = s0[78];
              tensorforge::intel_esimd::simd<float, 9> v333_data;
              v333_data.copy_from(ir1 + (128));
              (v333_data + (v290_data * v331_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v335_data;
              v335_data.copy_from(r0 + (112));
              float v336_data = s0[7];
              tensorforge::intel_esimd::simd<float, 9> v338_data;
              v338_data.copy_from(ir1 + (0));
              (v338_data + (v335_data * v336_data)).copy_to(ir1 + (0));
              float v341_data = s0[16];
              tensorforge::intel_esimd::simd<float, 9> v343_data;
              v343_data.copy_from(ir1 + (16));
              (v343_data + (v335_data * v341_data)).copy_to(ir1 + (16));
              float v346_data = s0[25];
              tensorforge::intel_esimd::simd<float, 9> v348_data;
              v348_data.copy_from(ir1 + (32));
              (v348_data + (v335_data * v346_data)).copy_to(ir1 + (32));
              float v351_data = s0[34];
              tensorforge::intel_esimd::simd<float, 9> v353_data;
              v353_data.copy_from(ir1 + (48));
              (v353_data + (v335_data * v351_data)).copy_to(ir1 + (48));
              float v356_data = s0[43];
              tensorforge::intel_esimd::simd<float, 9> v358_data;
              v358_data.copy_from(ir1 + (64));
              (v358_data + (v335_data * v356_data)).copy_to(ir1 + (64));
              float v361_data = s0[52];
              tensorforge::intel_esimd::simd<float, 9> v363_data;
              v363_data.copy_from(ir1 + (80));
              (v363_data + (v335_data * v361_data)).copy_to(ir1 + (80));
              float v366_data = s0[61];
              tensorforge::intel_esimd::simd<float, 9> v368_data;
              v368_data.copy_from(ir1 + (96));
              (v368_data + (v335_data * v366_data)).copy_to(ir1 + (96));
              float v371_data = s0[70];
              tensorforge::intel_esimd::simd<float, 9> v373_data;
              v373_data.copy_from(ir1 + (112));
              (v373_data + (v335_data * v371_data)).copy_to(ir1 + (112));
              float v376_data = s0[79];
              tensorforge::intel_esimd::simd<float, 9> v378_data;
              v378_data.copy_from(ir1 + (128));
              (v378_data + (v335_data * v376_data)).copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 9> v380_data;
              v380_data.copy_from(r0 + (128));
              float v381_data = s0[8];
              tensorforge::intel_esimd::simd<float, 9> v383_data;
              v383_data.copy_from(ir1 + (0));
              (v383_data + (v380_data * v381_data)).copy_to(ir1 + (0));
              float v386_data = s0[17];
              tensorforge::intel_esimd::simd<float, 9> v388_data;
              v388_data.copy_from(ir1 + (16));
              (v388_data + (v380_data * v386_data)).copy_to(ir1 + (16));
              float v391_data = s0[26];
              tensorforge::intel_esimd::simd<float, 9> v393_data;
              v393_data.copy_from(ir1 + (32));
              (v393_data + (v380_data * v391_data)).copy_to(ir1 + (32));
              float v396_data = s0[35];
              tensorforge::intel_esimd::simd<float, 9> v398_data;
              v398_data.copy_from(ir1 + (48));
              (v398_data + (v380_data * v396_data)).copy_to(ir1 + (48));
              float v401_data = s0[44];
              tensorforge::intel_esimd::simd<float, 9> v403_data;
              v403_data.copy_from(ir1 + (64));
              (v403_data + (v380_data * v401_data)).copy_to(ir1 + (64));
              float v406_data = s0[53];
              tensorforge::intel_esimd::simd<float, 9> v408_data;
              v408_data.copy_from(ir1 + (80));
              (v408_data + (v380_data * v406_data)).copy_to(ir1 + (80));
              float v411_data = s0[62];
              tensorforge::intel_esimd::simd<float, 9> v413_data;
              v413_data.copy_from(ir1 + (96));
              (v413_data + (v380_data * v411_data)).copy_to(ir1 + (96));
              float v416_data = s0[71];
              tensorforge::intel_esimd::simd<float, 9> v418_data;
              v418_data.copy_from(ir1 + (112));
              (v418_data + (v380_data * v416_data)).copy_to(ir1 + (112));
              float v421_data = s0[80];
              tensorforge::intel_esimd::simd<float, 9> v423_data;
              v423_data.copy_from(ir1 + (128));
              (v423_data + (v380_data * v421_data)).copy_to(ir1 + (128));
              #pragma unroll
              for (int32_t v426_n1 = 0; v426_n1 < 9; ++v426_n1) {
                int32_t v427_a = v426_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v429_data;
                v429_data.copy_from(ir1 + (v427_a));
                (v429_data * 13.0f).copy_to(r1 + (v427_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v433_i1 = 0; v433_i1 < 9; ++v433_i1) {
                tensorforge::intel_esimd::simd<float, 9> v436_data;
                v436_data.copy_from(r1 + ((v433_i1 * 16)));
                v436_data.copy_to(glb_m0 + ((v433_i1 * 9)));
              }
            }
          }
        }
      });
    }
  });
}

