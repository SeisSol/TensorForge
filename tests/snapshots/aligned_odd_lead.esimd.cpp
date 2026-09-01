// === base name ===
kernel_69f2bb9311

// === header ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_69f2bb9311(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_69f2bb9311(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 35×4(35×4) {0..35}×{0..4} strided
        // m1 35×8(35×8) {0..35}×{0..8} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[32 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[32];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
              float r0[512]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
                int32_t v8_lead = v6_i0 * 32;
                #pragma unroll
                for (int32_t v7_i1 = 0; v7_i1 < 8; ++v7_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v12_data;
                  v12_data.copy_from(glb_m1 + ((v8_lead + (v7_i1 * 35))));
                  v12_data.copy_to(r0 + ((v8_lead + (v7_i1 * 64))));
                }
              }
              #pragma unroll
              for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                tensorforge::intel_esimd::simd<float, 3> v22_data;
                v22_data.copy_from(glb_m1 + ((32_i32 + (v16_i1 * 35))));
                v22_data.copy_to(r0 + ((32 + (v16_i1 * 64))));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              v26_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 32> v29_data;
              v29_data.copy_from(r0 + (0));
              float v30_data = s0[0];
              tensorforge::intel_esimd::simd<float, 32> v32_data;
              v32_data.copy_from(ir1 + (0));
              (v32_data + (v29_data * v30_data)).copy_to(ir1 + (0));
              float v35_data = s0[8];
              tensorforge::intel_esimd::simd<float, 32> v37_data;
              v37_data.copy_from(ir1 + (64));
              (v37_data + (v29_data * v35_data)).copy_to(ir1 + (64));
              float v40_data = s0[16];
              tensorforge::intel_esimd::simd<float, 32> v42_data;
              v42_data.copy_from(ir1 + (128));
              (v42_data + (v29_data * v40_data)).copy_to(ir1 + (128));
              float v45_data = s0[24];
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(ir1 + (192));
              (v47_data + (v29_data * v45_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v49_data;
              v49_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 3> v52_data;
              v52_data.copy_from(ir1 + (32));
              (v52_data + (v49_data * v30_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v57_data;
              v57_data.copy_from(ir1 + (96));
              (v57_data + (v49_data * v35_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v62_data;
              v62_data.copy_from(ir1 + (160));
              (v62_data + (v49_data * v40_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v67_data;
              v67_data.copy_from(ir1 + (224));
              (v67_data + (v49_data * v45_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v69_data;
              v69_data.copy_from(r0 + (64));
              float v70_data = s0[1];
              tensorforge::intel_esimd::simd<float, 32> v72_data;
              v72_data.copy_from(ir1 + (0));
              (v72_data + (v69_data * v70_data)).copy_to(ir1 + (0));
              float v75_data = s0[9];
              tensorforge::intel_esimd::simd<float, 32> v77_data;
              v77_data.copy_from(ir1 + (64));
              (v77_data + (v69_data * v75_data)).copy_to(ir1 + (64));
              float v80_data = s0[17];
              tensorforge::intel_esimd::simd<float, 32> v82_data;
              v82_data.copy_from(ir1 + (128));
              (v82_data + (v69_data * v80_data)).copy_to(ir1 + (128));
              float v85_data = s0[25];
              tensorforge::intel_esimd::simd<float, 32> v87_data;
              v87_data.copy_from(ir1 + (192));
              (v87_data + (v69_data * v85_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v89_data;
              v89_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 3> v92_data;
              v92_data.copy_from(ir1 + (32));
              (v92_data + (v89_data * v70_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v97_data;
              v97_data.copy_from(ir1 + (96));
              (v97_data + (v89_data * v75_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v102_data;
              v102_data.copy_from(ir1 + (160));
              (v102_data + (v89_data * v80_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v107_data;
              v107_data.copy_from(ir1 + (224));
              (v107_data + (v89_data * v85_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v109_data;
              v109_data.copy_from(r0 + (128));
              float v110_data = s0[2];
              tensorforge::intel_esimd::simd<float, 32> v112_data;
              v112_data.copy_from(ir1 + (0));
              (v112_data + (v109_data * v110_data)).copy_to(ir1 + (0));
              float v115_data = s0[10];
              tensorforge::intel_esimd::simd<float, 32> v117_data;
              v117_data.copy_from(ir1 + (64));
              (v117_data + (v109_data * v115_data)).copy_to(ir1 + (64));
              float v120_data = s0[18];
              tensorforge::intel_esimd::simd<float, 32> v122_data;
              v122_data.copy_from(ir1 + (128));
              (v122_data + (v109_data * v120_data)).copy_to(ir1 + (128));
              float v125_data = s0[26];
              tensorforge::intel_esimd::simd<float, 32> v127_data;
              v127_data.copy_from(ir1 + (192));
              (v127_data + (v109_data * v125_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v129_data;
              v129_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 3> v132_data;
              v132_data.copy_from(ir1 + (32));
              (v132_data + (v129_data * v110_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v137_data;
              v137_data.copy_from(ir1 + (96));
              (v137_data + (v129_data * v115_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v142_data;
              v142_data.copy_from(ir1 + (160));
              (v142_data + (v129_data * v120_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v147_data;
              v147_data.copy_from(ir1 + (224));
              (v147_data + (v129_data * v125_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v149_data;
              v149_data.copy_from(r0 + (192));
              float v150_data = s0[3];
              tensorforge::intel_esimd::simd<float, 32> v152_data;
              v152_data.copy_from(ir1 + (0));
              (v152_data + (v149_data * v150_data)).copy_to(ir1 + (0));
              float v155_data = s0[11];
              tensorforge::intel_esimd::simd<float, 32> v157_data;
              v157_data.copy_from(ir1 + (64));
              (v157_data + (v149_data * v155_data)).copy_to(ir1 + (64));
              float v160_data = s0[19];
              tensorforge::intel_esimd::simd<float, 32> v162_data;
              v162_data.copy_from(ir1 + (128));
              (v162_data + (v149_data * v160_data)).copy_to(ir1 + (128));
              float v165_data = s0[27];
              tensorforge::intel_esimd::simd<float, 32> v167_data;
              v167_data.copy_from(ir1 + (192));
              (v167_data + (v149_data * v165_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v169_data;
              v169_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 3> v172_data;
              v172_data.copy_from(ir1 + (32));
              (v172_data + (v169_data * v150_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v177_data;
              v177_data.copy_from(ir1 + (96));
              (v177_data + (v169_data * v155_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v182_data;
              v182_data.copy_from(ir1 + (160));
              (v182_data + (v169_data * v160_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v187_data;
              v187_data.copy_from(ir1 + (224));
              (v187_data + (v169_data * v165_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v189_data;
              v189_data.copy_from(r0 + (256));
              float v190_data = s0[4];
              tensorforge::intel_esimd::simd<float, 32> v192_data;
              v192_data.copy_from(ir1 + (0));
              (v192_data + (v189_data * v190_data)).copy_to(ir1 + (0));
              float v195_data = s0[12];
              tensorforge::intel_esimd::simd<float, 32> v197_data;
              v197_data.copy_from(ir1 + (64));
              (v197_data + (v189_data * v195_data)).copy_to(ir1 + (64));
              float v200_data = s0[20];
              tensorforge::intel_esimd::simd<float, 32> v202_data;
              v202_data.copy_from(ir1 + (128));
              (v202_data + (v189_data * v200_data)).copy_to(ir1 + (128));
              float v205_data = s0[28];
              tensorforge::intel_esimd::simd<float, 32> v207_data;
              v207_data.copy_from(ir1 + (192));
              (v207_data + (v189_data * v205_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v209_data;
              v209_data.copy_from(r0 + (288));
              tensorforge::intel_esimd::simd<float, 3> v212_data;
              v212_data.copy_from(ir1 + (32));
              (v212_data + (v209_data * v190_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v217_data;
              v217_data.copy_from(ir1 + (96));
              (v217_data + (v209_data * v195_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v222_data;
              v222_data.copy_from(ir1 + (160));
              (v222_data + (v209_data * v200_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v227_data;
              v227_data.copy_from(ir1 + (224));
              (v227_data + (v209_data * v205_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v229_data;
              v229_data.copy_from(r0 + (320));
              float v230_data = s0[5];
              tensorforge::intel_esimd::simd<float, 32> v232_data;
              v232_data.copy_from(ir1 + (0));
              (v232_data + (v229_data * v230_data)).copy_to(ir1 + (0));
              float v235_data = s0[13];
              tensorforge::intel_esimd::simd<float, 32> v237_data;
              v237_data.copy_from(ir1 + (64));
              (v237_data + (v229_data * v235_data)).copy_to(ir1 + (64));
              float v240_data = s0[21];
              tensorforge::intel_esimd::simd<float, 32> v242_data;
              v242_data.copy_from(ir1 + (128));
              (v242_data + (v229_data * v240_data)).copy_to(ir1 + (128));
              float v245_data = s0[29];
              tensorforge::intel_esimd::simd<float, 32> v247_data;
              v247_data.copy_from(ir1 + (192));
              (v247_data + (v229_data * v245_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v249_data;
              v249_data.copy_from(r0 + (352));
              tensorforge::intel_esimd::simd<float, 3> v252_data;
              v252_data.copy_from(ir1 + (32));
              (v252_data + (v249_data * v230_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v257_data;
              v257_data.copy_from(ir1 + (96));
              (v257_data + (v249_data * v235_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v262_data;
              v262_data.copy_from(ir1 + (160));
              (v262_data + (v249_data * v240_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v267_data;
              v267_data.copy_from(ir1 + (224));
              (v267_data + (v249_data * v245_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v269_data;
              v269_data.copy_from(r0 + (384));
              float v270_data = s0[6];
              tensorforge::intel_esimd::simd<float, 32> v272_data;
              v272_data.copy_from(ir1 + (0));
              (v272_data + (v269_data * v270_data)).copy_to(ir1 + (0));
              float v275_data = s0[14];
              tensorforge::intel_esimd::simd<float, 32> v277_data;
              v277_data.copy_from(ir1 + (64));
              (v277_data + (v269_data * v275_data)).copy_to(ir1 + (64));
              float v280_data = s0[22];
              tensorforge::intel_esimd::simd<float, 32> v282_data;
              v282_data.copy_from(ir1 + (128));
              (v282_data + (v269_data * v280_data)).copy_to(ir1 + (128));
              float v285_data = s0[30];
              tensorforge::intel_esimd::simd<float, 32> v287_data;
              v287_data.copy_from(ir1 + (192));
              (v287_data + (v269_data * v285_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v289_data;
              v289_data.copy_from(r0 + (416));
              tensorforge::intel_esimd::simd<float, 3> v292_data;
              v292_data.copy_from(ir1 + (32));
              (v292_data + (v289_data * v270_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v297_data;
              v297_data.copy_from(ir1 + (96));
              (v297_data + (v289_data * v275_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v302_data;
              v302_data.copy_from(ir1 + (160));
              (v302_data + (v289_data * v280_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v307_data;
              v307_data.copy_from(ir1 + (224));
              (v307_data + (v289_data * v285_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v309_data;
              v309_data.copy_from(r0 + (448));
              float v310_data = s0[7];
              tensorforge::intel_esimd::simd<float, 32> v312_data;
              v312_data.copy_from(ir1 + (0));
              (v312_data + (v309_data * v310_data)).copy_to(ir1 + (0));
              float v315_data = s0[15];
              tensorforge::intel_esimd::simd<float, 32> v317_data;
              v317_data.copy_from(ir1 + (64));
              (v317_data + (v309_data * v315_data)).copy_to(ir1 + (64));
              float v320_data = s0[23];
              tensorforge::intel_esimd::simd<float, 32> v322_data;
              v322_data.copy_from(ir1 + (128));
              (v322_data + (v309_data * v320_data)).copy_to(ir1 + (128));
              float v325_data = s0[31];
              tensorforge::intel_esimd::simd<float, 32> v327_data;
              v327_data.copy_from(ir1 + (192));
              (v327_data + (v309_data * v325_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v329_data;
              v329_data.copy_from(r0 + (480));
              tensorforge::intel_esimd::simd<float, 3> v332_data;
              v332_data.copy_from(ir1 + (32));
              (v332_data + (v329_data * v310_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v337_data;
              v337_data.copy_from(ir1 + (96));
              (v337_data + (v329_data * v315_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v342_data;
              v342_data.copy_from(ir1 + (160));
              (v342_data + (v329_data * v320_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v347_data;
              v347_data.copy_from(ir1 + (224));
              (v347_data + (v329_data * v325_data)).copy_to(ir1 + (224));
              #pragma unroll
              for (int32_t v349_n0 = 0; v349_n0 < 1; ++v349_n0) {
                int32_t v351_a = v349_n0 * 32;
                #pragma unroll
                for (int32_t v350_n1 = 0; v350_n1 < 4; ++v350_n1) {
                  int32_t v353_a = v351_a + (v350_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v354_data;
                  v354_data.copy_from(ir1 + (v353_a));
                  v354_data.copy_to(r1 + (v353_a));
                }
              }
              #pragma unroll
              for (int32_t v358_n1 = 0; v358_n1 < 4; ++v358_n1) {
                int32_t v360_a = 32 + (v358_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v361_data;
                v361_data.copy_from(ir1 + (v360_a));
                v361_data.copy_to(r1 + (v360_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v364_i0 = 0; v364_i0 < 1; ++v364_i0) {
                int32_t v366_a = v364_i0 * 32;
                #pragma unroll
                for (int32_t v365_i1 = 0; v365_i1 < 4; ++v365_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v369_data;
                  v369_data.copy_from(r1 + ((v366_a + (v365_i1 * 64))));
                  v369_data.copy_to(glb_m0 + ((v366_a + (v365_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v374_i1 = 0; v374_i1 < 4; ++v374_i1) {
                tensorforge::intel_esimd::simd<float, 3> v377_data;
                v377_data.copy_from(r1 + ((32 + (v374_i1 * 64))));
                v377_data.copy_to(glb_m0 + ((32_i32 + (v374_i1 * 35))));
              }
            }
          }
        }
      });
    }
  });
}

