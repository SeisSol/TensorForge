// === base name ===
kernel_f94e030d8c

// === header ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f94e030d8c(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f94e030d8c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (9472, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 20×12(20×12) {0..20}×{0..12} strided
        // m2 20×16(20×16) {0..20}×{0..16} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[592 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[576];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[320];
              // s0 = load{g>s}(glb_m1[1, 0])
              tensorforge::intel_esimd::simd<int32_t, 16> v4_lead = tensorforge::intel_esimd::simd<int32_t, 16>(0, 1);
              #pragma unroll
              for (int32_t v5_i0 = 0; v5_i0 < 1; ++v5_i0) {
                int32_t v7_lead = v5_i0 * 16;
                #pragma unroll
                for (int32_t v6_i1 = 0; v6_i1 < 12; ++v6_i1) {
                  int32_t v9_a = v6_i1 * 20;
                  int32_t v10_a = v7_lead + v9_a;
                  tensorforge::intel_esimd::simd<float, 16> v15_data;
                  v15_data.copy_from(glb_m1 + ((v7_lead + v9_a)));
                  int32_t v19_a = v7_lead + (v6_i1 * 21);
                  s0[v19_a] = v15_data;
                }
              }
              tensorforge::intel_esimd::simd_mask<16> v20_g = v4_lead < 4;
              #pragma unroll
              for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                int32_t v24_a = v21_i1 * 20;
                int32_t v25_a = 16_i32 + v24_a;
                tensorforge::intel_esimd::simd<float, 16> v30_data(0.0f);
                v30_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[(16_i32 + v24_a)]), v20_g);
                int32_t v34_a = 16_i32 + (v21_i1 * 21);
                if (v20_g) {
                  s0[v34_a] = v30_data;
                }
              }
              float* __restrict__ s1 = &localShrMem0[0];
              {
                // s1 = load{g>s}(glb_m2[0, 1])
                #pragma unroll
                for (int32_t i = 0; i < 20; i += 4) {
                  *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + i * 16] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + i * 16];
                }
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[16]{};
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[16]{};
              tensorforge::intel_esimd::simd_mask<16> v39_g = v4_lead < 12;
              int32_t v43_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v48_data(0.0f);
              v48_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v49_data(0.0f);
              v49_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[0]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v51_data(0.0f);
              v51_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v51_data + (v48_data * v49_data)).copy_to(ir0 + (0));
              }
              int32_t v56_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v62_data(0.0f);
              v62_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[20]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v64_data(0.0f);
              v64_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v64_data + (v48_data * v62_data)).copy_to(ir0 + (1));
              }
              int32_t v69_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v75_data(0.0f);
              v75_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[40]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v77_data(0.0f);
              v77_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v77_data + (v48_data * v75_data)).copy_to(ir0 + (2));
              }
              int32_t v82_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v88_data(0.0f);
              v88_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[60]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v90_data(0.0f);
              v90_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v90_data + (v48_data * v88_data)).copy_to(ir0 + (3));
              }
              int32_t v95_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v101_data(0.0f);
              v101_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[80]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v103_data(0.0f);
              v103_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v103_data + (v48_data * v101_data)).copy_to(ir0 + (4));
              }
              int32_t v108_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v114_data(0.0f);
              v114_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[100]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v116_data(0.0f);
              v116_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v116_data + (v48_data * v114_data)).copy_to(ir0 + (5));
              }
              int32_t v121_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v127_data(0.0f);
              v127_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[120]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v129_data(0.0f);
              v129_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v129_data + (v48_data * v127_data)).copy_to(ir0 + (6));
              }
              int32_t v134_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v140_data(0.0f);
              v140_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[140]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v142_data(0.0f);
              v142_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v142_data + (v48_data * v140_data)).copy_to(ir0 + (7));
              }
              int32_t v147_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v153_data(0.0f);
              v153_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[160]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v155_data(0.0f);
              v155_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v155_data + (v48_data * v153_data)).copy_to(ir0 + (8));
              }
              int32_t v160_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v166_data(0.0f);
              v166_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[180]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v168_data(0.0f);
              v168_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v168_data + (v48_data * v166_data)).copy_to(ir0 + (9));
              }
              int32_t v173_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v179_data(0.0f);
              v179_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[200]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v181_data(0.0f);
              v181_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v181_data + (v48_data * v179_data)).copy_to(ir0 + (10));
              }
              int32_t v186_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v192_data(0.0f);
              v192_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[220]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v194_data(0.0f);
              v194_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v194_data + (v48_data * v192_data)).copy_to(ir0 + (11));
              }
              int32_t v199_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v205_data(0.0f);
              v205_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[240]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v207_data(0.0f);
              v207_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v207_data + (v48_data * v205_data)).copy_to(ir0 + (12));
              }
              int32_t v212_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v218_data(0.0f);
              v218_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[260]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v220_data(0.0f);
              v220_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v220_data + (v48_data * v218_data)).copy_to(ir0 + (13));
              }
              int32_t v225_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v231_data(0.0f);
              v231_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[280]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v233_data(0.0f);
              v233_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v233_data + (v48_data * v231_data)).copy_to(ir0 + (14));
              }
              int32_t v238_a = 0 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v244_data(0.0f);
              v244_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[300]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v246_data(0.0f);
              v246_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v246_data + (v48_data * v244_data)).copy_to(ir0 + (15));
              }
              int32_t v253_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v258_data(0.0f);
              v258_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v259_data(0.0f);
              v259_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[1]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v261_data(0.0f);
              v261_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v261_data + (v258_data * v259_data)).copy_to(ir0 + (0));
              }
              int32_t v266_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v272_data(0.0f);
              v272_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[21]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v274_data(0.0f);
              v274_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v274_data + (v258_data * v272_data)).copy_to(ir0 + (1));
              }
              int32_t v279_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v285_data(0.0f);
              v285_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[41]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v287_data(0.0f);
              v287_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v287_data + (v258_data * v285_data)).copy_to(ir0 + (2));
              }
              int32_t v292_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v298_data(0.0f);
              v298_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[61]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v300_data(0.0f);
              v300_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v300_data + (v258_data * v298_data)).copy_to(ir0 + (3));
              }
              int32_t v305_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v311_data(0.0f);
              v311_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[81]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v313_data(0.0f);
              v313_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v313_data + (v258_data * v311_data)).copy_to(ir0 + (4));
              }
              int32_t v318_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v324_data(0.0f);
              v324_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[101]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v326_data(0.0f);
              v326_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v326_data + (v258_data * v324_data)).copy_to(ir0 + (5));
              }
              int32_t v331_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v337_data(0.0f);
              v337_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[121]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v339_data(0.0f);
              v339_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v339_data + (v258_data * v337_data)).copy_to(ir0 + (6));
              }
              int32_t v344_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v350_data(0.0f);
              v350_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[141]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v352_data(0.0f);
              v352_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v352_data + (v258_data * v350_data)).copy_to(ir0 + (7));
              }
              int32_t v357_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v363_data(0.0f);
              v363_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[161]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v365_data(0.0f);
              v365_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v365_data + (v258_data * v363_data)).copy_to(ir0 + (8));
              }
              int32_t v370_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v376_data(0.0f);
              v376_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[181]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v378_data(0.0f);
              v378_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v378_data + (v258_data * v376_data)).copy_to(ir0 + (9));
              }
              int32_t v383_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v389_data(0.0f);
              v389_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[201]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v391_data(0.0f);
              v391_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v391_data + (v258_data * v389_data)).copy_to(ir0 + (10));
              }
              int32_t v396_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v402_data(0.0f);
              v402_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[221]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v404_data(0.0f);
              v404_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v404_data + (v258_data * v402_data)).copy_to(ir0 + (11));
              }
              int32_t v409_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v415_data(0.0f);
              v415_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[241]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v417_data(0.0f);
              v417_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v417_data + (v258_data * v415_data)).copy_to(ir0 + (12));
              }
              int32_t v422_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v428_data(0.0f);
              v428_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[261]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v430_data(0.0f);
              v430_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v430_data + (v258_data * v428_data)).copy_to(ir0 + (13));
              }
              int32_t v435_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v441_data(0.0f);
              v441_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[281]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v443_data(0.0f);
              v443_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v443_data + (v258_data * v441_data)).copy_to(ir0 + (14));
              }
              int32_t v448_a = 1 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v454_data(0.0f);
              v454_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[301]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v456_data(0.0f);
              v456_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v456_data + (v258_data * v454_data)).copy_to(ir0 + (15));
              }
              int32_t v463_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v468_data(0.0f);
              v468_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v469_data(0.0f);
              v469_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[2]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v471_data(0.0f);
              v471_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v471_data + (v468_data * v469_data)).copy_to(ir0 + (0));
              }
              int32_t v476_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v482_data(0.0f);
              v482_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[22]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v484_data(0.0f);
              v484_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v484_data + (v468_data * v482_data)).copy_to(ir0 + (1));
              }
              int32_t v489_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v495_data(0.0f);
              v495_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[42]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v497_data(0.0f);
              v497_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v497_data + (v468_data * v495_data)).copy_to(ir0 + (2));
              }
              int32_t v502_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v508_data(0.0f);
              v508_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[62]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v510_data(0.0f);
              v510_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v510_data + (v468_data * v508_data)).copy_to(ir0 + (3));
              }
              int32_t v515_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v521_data(0.0f);
              v521_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[82]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v523_data(0.0f);
              v523_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v523_data + (v468_data * v521_data)).copy_to(ir0 + (4));
              }
              int32_t v528_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v534_data(0.0f);
              v534_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[102]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v536_data(0.0f);
              v536_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v536_data + (v468_data * v534_data)).copy_to(ir0 + (5));
              }
              int32_t v541_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v547_data(0.0f);
              v547_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[122]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v549_data(0.0f);
              v549_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v549_data + (v468_data * v547_data)).copy_to(ir0 + (6));
              }
              int32_t v554_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v560_data(0.0f);
              v560_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[142]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v562_data(0.0f);
              v562_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v562_data + (v468_data * v560_data)).copy_to(ir0 + (7));
              }
              int32_t v567_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v573_data(0.0f);
              v573_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[162]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v575_data(0.0f);
              v575_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v575_data + (v468_data * v573_data)).copy_to(ir0 + (8));
              }
              int32_t v580_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v586_data(0.0f);
              v586_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[182]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v588_data(0.0f);
              v588_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v588_data + (v468_data * v586_data)).copy_to(ir0 + (9));
              }
              int32_t v593_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v599_data(0.0f);
              v599_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[202]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v601_data(0.0f);
              v601_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v601_data + (v468_data * v599_data)).copy_to(ir0 + (10));
              }
              int32_t v606_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v612_data(0.0f);
              v612_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[222]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v614_data(0.0f);
              v614_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v614_data + (v468_data * v612_data)).copy_to(ir0 + (11));
              }
              int32_t v619_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v625_data(0.0f);
              v625_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[242]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v627_data(0.0f);
              v627_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v627_data + (v468_data * v625_data)).copy_to(ir0 + (12));
              }
              int32_t v632_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v638_data(0.0f);
              v638_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[262]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v640_data(0.0f);
              v640_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v640_data + (v468_data * v638_data)).copy_to(ir0 + (13));
              }
              int32_t v645_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v651_data(0.0f);
              v651_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[282]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v653_data(0.0f);
              v653_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v653_data + (v468_data * v651_data)).copy_to(ir0 + (14));
              }
              int32_t v658_a = 2 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v664_data(0.0f);
              v664_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[302]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v666_data(0.0f);
              v666_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v666_data + (v468_data * v664_data)).copy_to(ir0 + (15));
              }
              int32_t v673_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v678_data(0.0f);
              v678_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v679_data(0.0f);
              v679_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[3]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v681_data(0.0f);
              v681_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v681_data + (v678_data * v679_data)).copy_to(ir0 + (0));
              }
              int32_t v686_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v692_data(0.0f);
              v692_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[23]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v694_data(0.0f);
              v694_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v694_data + (v678_data * v692_data)).copy_to(ir0 + (1));
              }
              int32_t v699_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v705_data(0.0f);
              v705_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[43]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v707_data(0.0f);
              v707_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v707_data + (v678_data * v705_data)).copy_to(ir0 + (2));
              }
              int32_t v712_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v718_data(0.0f);
              v718_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[63]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v720_data(0.0f);
              v720_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v720_data + (v678_data * v718_data)).copy_to(ir0 + (3));
              }
              int32_t v725_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v731_data(0.0f);
              v731_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[83]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v733_data(0.0f);
              v733_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v733_data + (v678_data * v731_data)).copy_to(ir0 + (4));
              }
              int32_t v738_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v744_data(0.0f);
              v744_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[103]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v746_data(0.0f);
              v746_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v746_data + (v678_data * v744_data)).copy_to(ir0 + (5));
              }
              int32_t v751_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v757_data(0.0f);
              v757_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[123]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v759_data(0.0f);
              v759_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v759_data + (v678_data * v757_data)).copy_to(ir0 + (6));
              }
              int32_t v764_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v770_data(0.0f);
              v770_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[143]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v772_data(0.0f);
              v772_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v772_data + (v678_data * v770_data)).copy_to(ir0 + (7));
              }
              int32_t v777_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v783_data(0.0f);
              v783_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[163]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v785_data(0.0f);
              v785_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v785_data + (v678_data * v783_data)).copy_to(ir0 + (8));
              }
              int32_t v790_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v796_data(0.0f);
              v796_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[183]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v798_data(0.0f);
              v798_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v798_data + (v678_data * v796_data)).copy_to(ir0 + (9));
              }
              int32_t v803_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v809_data(0.0f);
              v809_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[203]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v811_data(0.0f);
              v811_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v811_data + (v678_data * v809_data)).copy_to(ir0 + (10));
              }
              int32_t v816_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v822_data(0.0f);
              v822_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[223]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v824_data(0.0f);
              v824_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v824_data + (v678_data * v822_data)).copy_to(ir0 + (11));
              }
              int32_t v829_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v835_data(0.0f);
              v835_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[243]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v837_data(0.0f);
              v837_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v837_data + (v678_data * v835_data)).copy_to(ir0 + (12));
              }
              int32_t v842_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v848_data(0.0f);
              v848_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[263]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v850_data(0.0f);
              v850_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v850_data + (v678_data * v848_data)).copy_to(ir0 + (13));
              }
              int32_t v855_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v861_data(0.0f);
              v861_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[283]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v863_data(0.0f);
              v863_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v863_data + (v678_data * v861_data)).copy_to(ir0 + (14));
              }
              int32_t v868_a = 3 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v874_data(0.0f);
              v874_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[303]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v876_data(0.0f);
              v876_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v876_data + (v678_data * v874_data)).copy_to(ir0 + (15));
              }
              int32_t v883_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v888_data(0.0f);
              v888_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v889_data(0.0f);
              v889_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[4]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v891_data(0.0f);
              v891_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v891_data + (v888_data * v889_data)).copy_to(ir0 + (0));
              }
              int32_t v896_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v902_data(0.0f);
              v902_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[24]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v904_data(0.0f);
              v904_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v904_data + (v888_data * v902_data)).copy_to(ir0 + (1));
              }
              int32_t v909_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v915_data(0.0f);
              v915_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[44]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v917_data(0.0f);
              v917_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v917_data + (v888_data * v915_data)).copy_to(ir0 + (2));
              }
              int32_t v922_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v928_data(0.0f);
              v928_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[64]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v930_data(0.0f);
              v930_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v930_data + (v888_data * v928_data)).copy_to(ir0 + (3));
              }
              int32_t v935_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v941_data(0.0f);
              v941_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[84]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v943_data(0.0f);
              v943_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v943_data + (v888_data * v941_data)).copy_to(ir0 + (4));
              }
              int32_t v948_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v954_data(0.0f);
              v954_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[104]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v956_data(0.0f);
              v956_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v956_data + (v888_data * v954_data)).copy_to(ir0 + (5));
              }
              int32_t v961_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v967_data(0.0f);
              v967_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[124]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v969_data(0.0f);
              v969_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v969_data + (v888_data * v967_data)).copy_to(ir0 + (6));
              }
              int32_t v974_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v980_data(0.0f);
              v980_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[144]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v982_data(0.0f);
              v982_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v982_data + (v888_data * v980_data)).copy_to(ir0 + (7));
              }
              int32_t v987_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v993_data(0.0f);
              v993_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[164]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v995_data(0.0f);
              v995_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v995_data + (v888_data * v993_data)).copy_to(ir0 + (8));
              }
              int32_t v1000_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1006_data(0.0f);
              v1006_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[184]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1008_data(0.0f);
              v1008_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v1008_data + (v888_data * v1006_data)).copy_to(ir0 + (9));
              }
              int32_t v1013_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1019_data(0.0f);
              v1019_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[204]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1021_data(0.0f);
              v1021_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v1021_data + (v888_data * v1019_data)).copy_to(ir0 + (10));
              }
              int32_t v1026_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1032_data(0.0f);
              v1032_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[224]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1034_data(0.0f);
              v1034_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v1034_data + (v888_data * v1032_data)).copy_to(ir0 + (11));
              }
              int32_t v1039_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1045_data(0.0f);
              v1045_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[244]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1047_data(0.0f);
              v1047_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v1047_data + (v888_data * v1045_data)).copy_to(ir0 + (12));
              }
              int32_t v1052_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1058_data(0.0f);
              v1058_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[264]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1060_data(0.0f);
              v1060_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v1060_data + (v888_data * v1058_data)).copy_to(ir0 + (13));
              }
              int32_t v1065_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1071_data(0.0f);
              v1071_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[284]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1073_data(0.0f);
              v1073_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v1073_data + (v888_data * v1071_data)).copy_to(ir0 + (14));
              }
              int32_t v1078_a = 4 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1084_data(0.0f);
              v1084_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[304]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1086_data(0.0f);
              v1086_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v1086_data + (v888_data * v1084_data)).copy_to(ir0 + (15));
              }
              int32_t v1093_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1098_data(0.0f);
              v1098_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1099_data(0.0f);
              v1099_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[5]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1101_data(0.0f);
              v1101_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v1101_data + (v1098_data * v1099_data)).copy_to(ir0 + (0));
              }
              int32_t v1106_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1112_data(0.0f);
              v1112_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[25]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1114_data(0.0f);
              v1114_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v1114_data + (v1098_data * v1112_data)).copy_to(ir0 + (1));
              }
              int32_t v1119_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1125_data(0.0f);
              v1125_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[45]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1127_data(0.0f);
              v1127_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v1127_data + (v1098_data * v1125_data)).copy_to(ir0 + (2));
              }
              int32_t v1132_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1138_data(0.0f);
              v1138_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[65]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1140_data(0.0f);
              v1140_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v1140_data + (v1098_data * v1138_data)).copy_to(ir0 + (3));
              }
              int32_t v1145_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1151_data(0.0f);
              v1151_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[85]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1153_data(0.0f);
              v1153_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v1153_data + (v1098_data * v1151_data)).copy_to(ir0 + (4));
              }
              int32_t v1158_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1164_data(0.0f);
              v1164_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[105]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1166_data(0.0f);
              v1166_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v1166_data + (v1098_data * v1164_data)).copy_to(ir0 + (5));
              }
              int32_t v1171_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1177_data(0.0f);
              v1177_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[125]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1179_data(0.0f);
              v1179_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v1179_data + (v1098_data * v1177_data)).copy_to(ir0 + (6));
              }
              int32_t v1184_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1190_data(0.0f);
              v1190_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[145]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1192_data(0.0f);
              v1192_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v1192_data + (v1098_data * v1190_data)).copy_to(ir0 + (7));
              }
              int32_t v1197_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1203_data(0.0f);
              v1203_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[165]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1205_data(0.0f);
              v1205_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v1205_data + (v1098_data * v1203_data)).copy_to(ir0 + (8));
              }
              int32_t v1210_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1216_data(0.0f);
              v1216_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[185]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1218_data(0.0f);
              v1218_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v1218_data + (v1098_data * v1216_data)).copy_to(ir0 + (9));
              }
              int32_t v1223_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1229_data(0.0f);
              v1229_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[205]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1231_data(0.0f);
              v1231_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v1231_data + (v1098_data * v1229_data)).copy_to(ir0 + (10));
              }
              int32_t v1236_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1242_data(0.0f);
              v1242_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[225]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1244_data(0.0f);
              v1244_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v1244_data + (v1098_data * v1242_data)).copy_to(ir0 + (11));
              }
              int32_t v1249_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1255_data(0.0f);
              v1255_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[245]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1257_data(0.0f);
              v1257_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v1257_data + (v1098_data * v1255_data)).copy_to(ir0 + (12));
              }
              int32_t v1262_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1268_data(0.0f);
              v1268_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[265]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1270_data(0.0f);
              v1270_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v1270_data + (v1098_data * v1268_data)).copy_to(ir0 + (13));
              }
              int32_t v1275_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1281_data(0.0f);
              v1281_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[285]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1283_data(0.0f);
              v1283_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v1283_data + (v1098_data * v1281_data)).copy_to(ir0 + (14));
              }
              int32_t v1288_a = 5 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1294_data(0.0f);
              v1294_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[305]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1296_data(0.0f);
              v1296_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v1296_data + (v1098_data * v1294_data)).copy_to(ir0 + (15));
              }
              int32_t v1303_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1308_data(0.0f);
              v1308_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1309_data(0.0f);
              v1309_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[6]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1311_data(0.0f);
              v1311_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v1311_data + (v1308_data * v1309_data)).copy_to(ir0 + (0));
              }
              int32_t v1316_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1322_data(0.0f);
              v1322_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[26]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1324_data(0.0f);
              v1324_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v1324_data + (v1308_data * v1322_data)).copy_to(ir0 + (1));
              }
              int32_t v1329_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1335_data(0.0f);
              v1335_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[46]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1337_data(0.0f);
              v1337_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v1337_data + (v1308_data * v1335_data)).copy_to(ir0 + (2));
              }
              int32_t v1342_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1348_data(0.0f);
              v1348_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[66]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1350_data(0.0f);
              v1350_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v1350_data + (v1308_data * v1348_data)).copy_to(ir0 + (3));
              }
              int32_t v1355_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1361_data(0.0f);
              v1361_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[86]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1363_data(0.0f);
              v1363_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v1363_data + (v1308_data * v1361_data)).copy_to(ir0 + (4));
              }
              int32_t v1368_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1374_data(0.0f);
              v1374_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[106]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1376_data(0.0f);
              v1376_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v1376_data + (v1308_data * v1374_data)).copy_to(ir0 + (5));
              }
              int32_t v1381_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1387_data(0.0f);
              v1387_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[126]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1389_data(0.0f);
              v1389_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v1389_data + (v1308_data * v1387_data)).copy_to(ir0 + (6));
              }
              int32_t v1394_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1400_data(0.0f);
              v1400_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[146]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1402_data(0.0f);
              v1402_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v1402_data + (v1308_data * v1400_data)).copy_to(ir0 + (7));
              }
              int32_t v1407_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1413_data(0.0f);
              v1413_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[166]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1415_data(0.0f);
              v1415_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v1415_data + (v1308_data * v1413_data)).copy_to(ir0 + (8));
              }
              int32_t v1420_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1426_data(0.0f);
              v1426_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[186]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1428_data(0.0f);
              v1428_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v1428_data + (v1308_data * v1426_data)).copy_to(ir0 + (9));
              }
              int32_t v1433_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1439_data(0.0f);
              v1439_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[206]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1441_data(0.0f);
              v1441_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v1441_data + (v1308_data * v1439_data)).copy_to(ir0 + (10));
              }
              int32_t v1446_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1452_data(0.0f);
              v1452_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[226]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1454_data(0.0f);
              v1454_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v1454_data + (v1308_data * v1452_data)).copy_to(ir0 + (11));
              }
              int32_t v1459_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1465_data(0.0f);
              v1465_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[246]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1467_data(0.0f);
              v1467_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v1467_data + (v1308_data * v1465_data)).copy_to(ir0 + (12));
              }
              int32_t v1472_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1478_data(0.0f);
              v1478_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[266]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1480_data(0.0f);
              v1480_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v1480_data + (v1308_data * v1478_data)).copy_to(ir0 + (13));
              }
              int32_t v1485_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1491_data(0.0f);
              v1491_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[286]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1493_data(0.0f);
              v1493_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v1493_data + (v1308_data * v1491_data)).copy_to(ir0 + (14));
              }
              int32_t v1498_a = 6 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1504_data(0.0f);
              v1504_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[306]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1506_data(0.0f);
              v1506_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v1506_data + (v1308_data * v1504_data)).copy_to(ir0 + (15));
              }
              int32_t v1513_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1518_data(0.0f);
              v1518_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1519_data(0.0f);
              v1519_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[7]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1521_data(0.0f);
              v1521_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v1521_data + (v1518_data * v1519_data)).copy_to(ir0 + (0));
              }
              int32_t v1526_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1532_data(0.0f);
              v1532_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[27]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1534_data(0.0f);
              v1534_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v1534_data + (v1518_data * v1532_data)).copy_to(ir0 + (1));
              }
              int32_t v1539_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1545_data(0.0f);
              v1545_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[47]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1547_data(0.0f);
              v1547_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v1547_data + (v1518_data * v1545_data)).copy_to(ir0 + (2));
              }
              int32_t v1552_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1558_data(0.0f);
              v1558_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[67]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1560_data(0.0f);
              v1560_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v1560_data + (v1518_data * v1558_data)).copy_to(ir0 + (3));
              }
              int32_t v1565_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1571_data(0.0f);
              v1571_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[87]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1573_data(0.0f);
              v1573_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v1573_data + (v1518_data * v1571_data)).copy_to(ir0 + (4));
              }
              int32_t v1578_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1584_data(0.0f);
              v1584_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[107]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1586_data(0.0f);
              v1586_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v1586_data + (v1518_data * v1584_data)).copy_to(ir0 + (5));
              }
              int32_t v1591_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1597_data(0.0f);
              v1597_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[127]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1599_data(0.0f);
              v1599_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v1599_data + (v1518_data * v1597_data)).copy_to(ir0 + (6));
              }
              int32_t v1604_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1610_data(0.0f);
              v1610_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[147]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1612_data(0.0f);
              v1612_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v1612_data + (v1518_data * v1610_data)).copy_to(ir0 + (7));
              }
              int32_t v1617_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1623_data(0.0f);
              v1623_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[167]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1625_data(0.0f);
              v1625_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v1625_data + (v1518_data * v1623_data)).copy_to(ir0 + (8));
              }
              int32_t v1630_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1636_data(0.0f);
              v1636_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[187]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1638_data(0.0f);
              v1638_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v1638_data + (v1518_data * v1636_data)).copy_to(ir0 + (9));
              }
              int32_t v1643_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1649_data(0.0f);
              v1649_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[207]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1651_data(0.0f);
              v1651_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v1651_data + (v1518_data * v1649_data)).copy_to(ir0 + (10));
              }
              int32_t v1656_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1662_data(0.0f);
              v1662_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[227]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1664_data(0.0f);
              v1664_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v1664_data + (v1518_data * v1662_data)).copy_to(ir0 + (11));
              }
              int32_t v1669_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1675_data(0.0f);
              v1675_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[247]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1677_data(0.0f);
              v1677_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v1677_data + (v1518_data * v1675_data)).copy_to(ir0 + (12));
              }
              int32_t v1682_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1688_data(0.0f);
              v1688_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[267]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1690_data(0.0f);
              v1690_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v1690_data + (v1518_data * v1688_data)).copy_to(ir0 + (13));
              }
              int32_t v1695_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1701_data(0.0f);
              v1701_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[287]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1703_data(0.0f);
              v1703_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v1703_data + (v1518_data * v1701_data)).copy_to(ir0 + (14));
              }
              int32_t v1708_a = 7 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1714_data(0.0f);
              v1714_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[307]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1716_data(0.0f);
              v1716_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v1716_data + (v1518_data * v1714_data)).copy_to(ir0 + (15));
              }
              int32_t v1723_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1728_data(0.0f);
              v1728_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1729_data(0.0f);
              v1729_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[8]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1731_data(0.0f);
              v1731_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v1731_data + (v1728_data * v1729_data)).copy_to(ir0 + (0));
              }
              int32_t v1736_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1742_data(0.0f);
              v1742_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[28]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1744_data(0.0f);
              v1744_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v1744_data + (v1728_data * v1742_data)).copy_to(ir0 + (1));
              }
              int32_t v1749_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1755_data(0.0f);
              v1755_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[48]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1757_data(0.0f);
              v1757_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v1757_data + (v1728_data * v1755_data)).copy_to(ir0 + (2));
              }
              int32_t v1762_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1768_data(0.0f);
              v1768_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[68]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1770_data(0.0f);
              v1770_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v1770_data + (v1728_data * v1768_data)).copy_to(ir0 + (3));
              }
              int32_t v1775_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1781_data(0.0f);
              v1781_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[88]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1783_data(0.0f);
              v1783_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v1783_data + (v1728_data * v1781_data)).copy_to(ir0 + (4));
              }
              int32_t v1788_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1794_data(0.0f);
              v1794_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[108]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1796_data(0.0f);
              v1796_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v1796_data + (v1728_data * v1794_data)).copy_to(ir0 + (5));
              }
              int32_t v1801_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1807_data(0.0f);
              v1807_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[128]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1809_data(0.0f);
              v1809_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v1809_data + (v1728_data * v1807_data)).copy_to(ir0 + (6));
              }
              int32_t v1814_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1820_data(0.0f);
              v1820_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[148]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1822_data(0.0f);
              v1822_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v1822_data + (v1728_data * v1820_data)).copy_to(ir0 + (7));
              }
              int32_t v1827_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1833_data(0.0f);
              v1833_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[168]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1835_data(0.0f);
              v1835_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v1835_data + (v1728_data * v1833_data)).copy_to(ir0 + (8));
              }
              int32_t v1840_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1846_data(0.0f);
              v1846_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[188]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1848_data(0.0f);
              v1848_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v1848_data + (v1728_data * v1846_data)).copy_to(ir0 + (9));
              }
              int32_t v1853_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1859_data(0.0f);
              v1859_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[208]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1861_data(0.0f);
              v1861_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v1861_data + (v1728_data * v1859_data)).copy_to(ir0 + (10));
              }
              int32_t v1866_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1872_data(0.0f);
              v1872_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[228]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1874_data(0.0f);
              v1874_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v1874_data + (v1728_data * v1872_data)).copy_to(ir0 + (11));
              }
              int32_t v1879_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1885_data(0.0f);
              v1885_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[248]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1887_data(0.0f);
              v1887_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v1887_data + (v1728_data * v1885_data)).copy_to(ir0 + (12));
              }
              int32_t v1892_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1898_data(0.0f);
              v1898_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[268]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1900_data(0.0f);
              v1900_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v1900_data + (v1728_data * v1898_data)).copy_to(ir0 + (13));
              }
              int32_t v1905_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1911_data(0.0f);
              v1911_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[288]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1913_data(0.0f);
              v1913_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v1913_data + (v1728_data * v1911_data)).copy_to(ir0 + (14));
              }
              int32_t v1918_a = 8 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1924_data(0.0f);
              v1924_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[308]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1926_data(0.0f);
              v1926_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v1926_data + (v1728_data * v1924_data)).copy_to(ir0 + (15));
              }
              int32_t v1933_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1938_data(0.0f);
              v1938_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1939_data(0.0f);
              v1939_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[9]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1941_data(0.0f);
              v1941_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v1941_data + (v1938_data * v1939_data)).copy_to(ir0 + (0));
              }
              int32_t v1946_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1952_data(0.0f);
              v1952_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[29]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1954_data(0.0f);
              v1954_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v1954_data + (v1938_data * v1952_data)).copy_to(ir0 + (1));
              }
              int32_t v1959_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1965_data(0.0f);
              v1965_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[49]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1967_data(0.0f);
              v1967_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v1967_data + (v1938_data * v1965_data)).copy_to(ir0 + (2));
              }
              int32_t v1972_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1978_data(0.0f);
              v1978_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[69]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1980_data(0.0f);
              v1980_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v1980_data + (v1938_data * v1978_data)).copy_to(ir0 + (3));
              }
              int32_t v1985_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v1991_data(0.0f);
              v1991_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[89]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v1993_data(0.0f);
              v1993_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v1993_data + (v1938_data * v1991_data)).copy_to(ir0 + (4));
              }
              int32_t v1998_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2004_data(0.0f);
              v2004_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[109]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2006_data(0.0f);
              v2006_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v2006_data + (v1938_data * v2004_data)).copy_to(ir0 + (5));
              }
              int32_t v2011_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2017_data(0.0f);
              v2017_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[129]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2019_data(0.0f);
              v2019_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v2019_data + (v1938_data * v2017_data)).copy_to(ir0 + (6));
              }
              int32_t v2024_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2030_data(0.0f);
              v2030_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[149]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2032_data(0.0f);
              v2032_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v2032_data + (v1938_data * v2030_data)).copy_to(ir0 + (7));
              }
              int32_t v2037_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2043_data(0.0f);
              v2043_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[169]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2045_data(0.0f);
              v2045_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v2045_data + (v1938_data * v2043_data)).copy_to(ir0 + (8));
              }
              int32_t v2050_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2056_data(0.0f);
              v2056_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[189]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2058_data(0.0f);
              v2058_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v2058_data + (v1938_data * v2056_data)).copy_to(ir0 + (9));
              }
              int32_t v2063_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2069_data(0.0f);
              v2069_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[209]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2071_data(0.0f);
              v2071_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v2071_data + (v1938_data * v2069_data)).copy_to(ir0 + (10));
              }
              int32_t v2076_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2082_data(0.0f);
              v2082_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[229]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2084_data(0.0f);
              v2084_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v2084_data + (v1938_data * v2082_data)).copy_to(ir0 + (11));
              }
              int32_t v2089_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2095_data(0.0f);
              v2095_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[249]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2097_data(0.0f);
              v2097_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v2097_data + (v1938_data * v2095_data)).copy_to(ir0 + (12));
              }
              int32_t v2102_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2108_data(0.0f);
              v2108_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[269]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2110_data(0.0f);
              v2110_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v2110_data + (v1938_data * v2108_data)).copy_to(ir0 + (13));
              }
              int32_t v2115_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2121_data(0.0f);
              v2121_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[289]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2123_data(0.0f);
              v2123_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v2123_data + (v1938_data * v2121_data)).copy_to(ir0 + (14));
              }
              int32_t v2128_a = 9 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2134_data(0.0f);
              v2134_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[309]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2136_data(0.0f);
              v2136_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v2136_data + (v1938_data * v2134_data)).copy_to(ir0 + (15));
              }
              int32_t v2143_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2148_data(0.0f);
              v2148_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2149_data(0.0f);
              v2149_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[10]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2151_data(0.0f);
              v2151_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v2151_data + (v2148_data * v2149_data)).copy_to(ir0 + (0));
              }
              int32_t v2156_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2162_data(0.0f);
              v2162_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[30]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2164_data(0.0f);
              v2164_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v2164_data + (v2148_data * v2162_data)).copy_to(ir0 + (1));
              }
              int32_t v2169_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2175_data(0.0f);
              v2175_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[50]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2177_data(0.0f);
              v2177_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v2177_data + (v2148_data * v2175_data)).copy_to(ir0 + (2));
              }
              int32_t v2182_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2188_data(0.0f);
              v2188_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[70]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2190_data(0.0f);
              v2190_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v2190_data + (v2148_data * v2188_data)).copy_to(ir0 + (3));
              }
              int32_t v2195_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2201_data(0.0f);
              v2201_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[90]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2203_data(0.0f);
              v2203_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v2203_data + (v2148_data * v2201_data)).copy_to(ir0 + (4));
              }
              int32_t v2208_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2214_data(0.0f);
              v2214_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[110]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2216_data(0.0f);
              v2216_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v2216_data + (v2148_data * v2214_data)).copy_to(ir0 + (5));
              }
              int32_t v2221_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2227_data(0.0f);
              v2227_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[130]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2229_data(0.0f);
              v2229_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v2229_data + (v2148_data * v2227_data)).copy_to(ir0 + (6));
              }
              int32_t v2234_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2240_data(0.0f);
              v2240_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[150]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2242_data(0.0f);
              v2242_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v2242_data + (v2148_data * v2240_data)).copy_to(ir0 + (7));
              }
              int32_t v2247_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2253_data(0.0f);
              v2253_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[170]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2255_data(0.0f);
              v2255_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v2255_data + (v2148_data * v2253_data)).copy_to(ir0 + (8));
              }
              int32_t v2260_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2266_data(0.0f);
              v2266_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[190]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2268_data(0.0f);
              v2268_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v2268_data + (v2148_data * v2266_data)).copy_to(ir0 + (9));
              }
              int32_t v2273_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2279_data(0.0f);
              v2279_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[210]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2281_data(0.0f);
              v2281_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v2281_data + (v2148_data * v2279_data)).copy_to(ir0 + (10));
              }
              int32_t v2286_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2292_data(0.0f);
              v2292_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[230]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2294_data(0.0f);
              v2294_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v2294_data + (v2148_data * v2292_data)).copy_to(ir0 + (11));
              }
              int32_t v2299_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2305_data(0.0f);
              v2305_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[250]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2307_data(0.0f);
              v2307_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v2307_data + (v2148_data * v2305_data)).copy_to(ir0 + (12));
              }
              int32_t v2312_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2318_data(0.0f);
              v2318_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[270]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2320_data(0.0f);
              v2320_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v2320_data + (v2148_data * v2318_data)).copy_to(ir0 + (13));
              }
              int32_t v2325_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2331_data(0.0f);
              v2331_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[290]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2333_data(0.0f);
              v2333_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v2333_data + (v2148_data * v2331_data)).copy_to(ir0 + (14));
              }
              int32_t v2338_a = 10 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2344_data(0.0f);
              v2344_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[310]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2346_data(0.0f);
              v2346_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v2346_data + (v2148_data * v2344_data)).copy_to(ir0 + (15));
              }
              int32_t v2353_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2358_data(0.0f);
              v2358_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2359_data(0.0f);
              v2359_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[11]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2361_data(0.0f);
              v2361_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v2361_data + (v2358_data * v2359_data)).copy_to(ir0 + (0));
              }
              int32_t v2366_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2372_data(0.0f);
              v2372_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[31]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2374_data(0.0f);
              v2374_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v2374_data + (v2358_data * v2372_data)).copy_to(ir0 + (1));
              }
              int32_t v2379_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2385_data(0.0f);
              v2385_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[51]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2387_data(0.0f);
              v2387_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v2387_data + (v2358_data * v2385_data)).copy_to(ir0 + (2));
              }
              int32_t v2392_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2398_data(0.0f);
              v2398_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[71]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2400_data(0.0f);
              v2400_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v2400_data + (v2358_data * v2398_data)).copy_to(ir0 + (3));
              }
              int32_t v2405_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2411_data(0.0f);
              v2411_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[91]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2413_data(0.0f);
              v2413_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v2413_data + (v2358_data * v2411_data)).copy_to(ir0 + (4));
              }
              int32_t v2418_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2424_data(0.0f);
              v2424_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[111]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2426_data(0.0f);
              v2426_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v2426_data + (v2358_data * v2424_data)).copy_to(ir0 + (5));
              }
              int32_t v2431_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2437_data(0.0f);
              v2437_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[131]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2439_data(0.0f);
              v2439_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v2439_data + (v2358_data * v2437_data)).copy_to(ir0 + (6));
              }
              int32_t v2444_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2450_data(0.0f);
              v2450_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[151]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2452_data(0.0f);
              v2452_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v2452_data + (v2358_data * v2450_data)).copy_to(ir0 + (7));
              }
              int32_t v2457_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2463_data(0.0f);
              v2463_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[171]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2465_data(0.0f);
              v2465_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v2465_data + (v2358_data * v2463_data)).copy_to(ir0 + (8));
              }
              int32_t v2470_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2476_data(0.0f);
              v2476_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[191]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2478_data(0.0f);
              v2478_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v2478_data + (v2358_data * v2476_data)).copy_to(ir0 + (9));
              }
              int32_t v2483_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2489_data(0.0f);
              v2489_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[211]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2491_data(0.0f);
              v2491_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v2491_data + (v2358_data * v2489_data)).copy_to(ir0 + (10));
              }
              int32_t v2496_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2502_data(0.0f);
              v2502_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[231]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2504_data(0.0f);
              v2504_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v2504_data + (v2358_data * v2502_data)).copy_to(ir0 + (11));
              }
              int32_t v2509_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2515_data(0.0f);
              v2515_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[251]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2517_data(0.0f);
              v2517_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v2517_data + (v2358_data * v2515_data)).copy_to(ir0 + (12));
              }
              int32_t v2522_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2528_data(0.0f);
              v2528_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[271]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2530_data(0.0f);
              v2530_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v2530_data + (v2358_data * v2528_data)).copy_to(ir0 + (13));
              }
              int32_t v2535_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2541_data(0.0f);
              v2541_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[291]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2543_data(0.0f);
              v2543_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v2543_data + (v2358_data * v2541_data)).copy_to(ir0 + (14));
              }
              int32_t v2548_a = 11 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2554_data(0.0f);
              v2554_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[311]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2556_data(0.0f);
              v2556_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v2556_data + (v2358_data * v2554_data)).copy_to(ir0 + (15));
              }
              int32_t v2563_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2568_data(0.0f);
              v2568_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2569_data(0.0f);
              v2569_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[12]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2571_data(0.0f);
              v2571_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v2571_data + (v2568_data * v2569_data)).copy_to(ir0 + (0));
              }
              int32_t v2576_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2582_data(0.0f);
              v2582_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2584_data(0.0f);
              v2584_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v2584_data + (v2568_data * v2582_data)).copy_to(ir0 + (1));
              }
              int32_t v2589_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2595_data(0.0f);
              v2595_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[52]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2597_data(0.0f);
              v2597_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v2597_data + (v2568_data * v2595_data)).copy_to(ir0 + (2));
              }
              int32_t v2602_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2608_data(0.0f);
              v2608_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[72]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2610_data(0.0f);
              v2610_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v2610_data + (v2568_data * v2608_data)).copy_to(ir0 + (3));
              }
              int32_t v2615_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2621_data(0.0f);
              v2621_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[92]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2623_data(0.0f);
              v2623_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v2623_data + (v2568_data * v2621_data)).copy_to(ir0 + (4));
              }
              int32_t v2628_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2634_data(0.0f);
              v2634_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[112]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2636_data(0.0f);
              v2636_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v2636_data + (v2568_data * v2634_data)).copy_to(ir0 + (5));
              }
              int32_t v2641_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2647_data(0.0f);
              v2647_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[132]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2649_data(0.0f);
              v2649_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v2649_data + (v2568_data * v2647_data)).copy_to(ir0 + (6));
              }
              int32_t v2654_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2660_data(0.0f);
              v2660_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[152]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2662_data(0.0f);
              v2662_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v2662_data + (v2568_data * v2660_data)).copy_to(ir0 + (7));
              }
              int32_t v2667_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2673_data(0.0f);
              v2673_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[172]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2675_data(0.0f);
              v2675_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v2675_data + (v2568_data * v2673_data)).copy_to(ir0 + (8));
              }
              int32_t v2680_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2686_data(0.0f);
              v2686_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[192]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2688_data(0.0f);
              v2688_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v2688_data + (v2568_data * v2686_data)).copy_to(ir0 + (9));
              }
              int32_t v2693_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2699_data(0.0f);
              v2699_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[212]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2701_data(0.0f);
              v2701_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v2701_data + (v2568_data * v2699_data)).copy_to(ir0 + (10));
              }
              int32_t v2706_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2712_data(0.0f);
              v2712_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[232]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2714_data(0.0f);
              v2714_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v2714_data + (v2568_data * v2712_data)).copy_to(ir0 + (11));
              }
              int32_t v2719_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2725_data(0.0f);
              v2725_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[252]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2727_data(0.0f);
              v2727_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v2727_data + (v2568_data * v2725_data)).copy_to(ir0 + (12));
              }
              int32_t v2732_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2738_data(0.0f);
              v2738_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[272]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2740_data(0.0f);
              v2740_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v2740_data + (v2568_data * v2738_data)).copy_to(ir0 + (13));
              }
              int32_t v2745_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2751_data(0.0f);
              v2751_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[292]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2753_data(0.0f);
              v2753_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v2753_data + (v2568_data * v2751_data)).copy_to(ir0 + (14));
              }
              int32_t v2758_a = 12 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2764_data(0.0f);
              v2764_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[312]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2766_data(0.0f);
              v2766_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v2766_data + (v2568_data * v2764_data)).copy_to(ir0 + (15));
              }
              int32_t v2773_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2778_data(0.0f);
              v2778_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2779_data(0.0f);
              v2779_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[13]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2781_data(0.0f);
              v2781_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v2781_data + (v2778_data * v2779_data)).copy_to(ir0 + (0));
              }
              int32_t v2786_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2792_data(0.0f);
              v2792_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[33]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2794_data(0.0f);
              v2794_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v2794_data + (v2778_data * v2792_data)).copy_to(ir0 + (1));
              }
              int32_t v2799_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2805_data(0.0f);
              v2805_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[53]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2807_data(0.0f);
              v2807_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v2807_data + (v2778_data * v2805_data)).copy_to(ir0 + (2));
              }
              int32_t v2812_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2818_data(0.0f);
              v2818_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[73]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2820_data(0.0f);
              v2820_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v2820_data + (v2778_data * v2818_data)).copy_to(ir0 + (3));
              }
              int32_t v2825_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2831_data(0.0f);
              v2831_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[93]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2833_data(0.0f);
              v2833_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v2833_data + (v2778_data * v2831_data)).copy_to(ir0 + (4));
              }
              int32_t v2838_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2844_data(0.0f);
              v2844_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[113]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2846_data(0.0f);
              v2846_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v2846_data + (v2778_data * v2844_data)).copy_to(ir0 + (5));
              }
              int32_t v2851_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2857_data(0.0f);
              v2857_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[133]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2859_data(0.0f);
              v2859_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v2859_data + (v2778_data * v2857_data)).copy_to(ir0 + (6));
              }
              int32_t v2864_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2870_data(0.0f);
              v2870_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[153]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2872_data(0.0f);
              v2872_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v2872_data + (v2778_data * v2870_data)).copy_to(ir0 + (7));
              }
              int32_t v2877_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2883_data(0.0f);
              v2883_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[173]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2885_data(0.0f);
              v2885_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v2885_data + (v2778_data * v2883_data)).copy_to(ir0 + (8));
              }
              int32_t v2890_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2896_data(0.0f);
              v2896_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[193]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2898_data(0.0f);
              v2898_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v2898_data + (v2778_data * v2896_data)).copy_to(ir0 + (9));
              }
              int32_t v2903_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2909_data(0.0f);
              v2909_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[213]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2911_data(0.0f);
              v2911_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v2911_data + (v2778_data * v2909_data)).copy_to(ir0 + (10));
              }
              int32_t v2916_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2922_data(0.0f);
              v2922_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[233]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2924_data(0.0f);
              v2924_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v2924_data + (v2778_data * v2922_data)).copy_to(ir0 + (11));
              }
              int32_t v2929_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2935_data(0.0f);
              v2935_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[253]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2937_data(0.0f);
              v2937_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v2937_data + (v2778_data * v2935_data)).copy_to(ir0 + (12));
              }
              int32_t v2942_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2948_data(0.0f);
              v2948_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[273]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2950_data(0.0f);
              v2950_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v2950_data + (v2778_data * v2948_data)).copy_to(ir0 + (13));
              }
              int32_t v2955_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2961_data(0.0f);
              v2961_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[293]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2963_data(0.0f);
              v2963_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v2963_data + (v2778_data * v2961_data)).copy_to(ir0 + (14));
              }
              int32_t v2968_a = 13 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2974_data(0.0f);
              v2974_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[313]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2976_data(0.0f);
              v2976_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v2976_data + (v2778_data * v2974_data)).copy_to(ir0 + (15));
              }
              int32_t v2983_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v2988_data(0.0f);
              v2988_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2989_data(0.0f);
              v2989_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[14]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v2991_data(0.0f);
              v2991_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v2991_data + (v2988_data * v2989_data)).copy_to(ir0 + (0));
              }
              int32_t v2996_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3002_data(0.0f);
              v3002_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[34]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3004_data(0.0f);
              v3004_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v3004_data + (v2988_data * v3002_data)).copy_to(ir0 + (1));
              }
              int32_t v3009_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3015_data(0.0f);
              v3015_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[54]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3017_data(0.0f);
              v3017_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v3017_data + (v2988_data * v3015_data)).copy_to(ir0 + (2));
              }
              int32_t v3022_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3028_data(0.0f);
              v3028_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[74]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3030_data(0.0f);
              v3030_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v3030_data + (v2988_data * v3028_data)).copy_to(ir0 + (3));
              }
              int32_t v3035_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3041_data(0.0f);
              v3041_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[94]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3043_data(0.0f);
              v3043_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v3043_data + (v2988_data * v3041_data)).copy_to(ir0 + (4));
              }
              int32_t v3048_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3054_data(0.0f);
              v3054_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[114]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3056_data(0.0f);
              v3056_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v3056_data + (v2988_data * v3054_data)).copy_to(ir0 + (5));
              }
              int32_t v3061_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3067_data(0.0f);
              v3067_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[134]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3069_data(0.0f);
              v3069_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v3069_data + (v2988_data * v3067_data)).copy_to(ir0 + (6));
              }
              int32_t v3074_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3080_data(0.0f);
              v3080_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[154]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3082_data(0.0f);
              v3082_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v3082_data + (v2988_data * v3080_data)).copy_to(ir0 + (7));
              }
              int32_t v3087_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3093_data(0.0f);
              v3093_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[174]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3095_data(0.0f);
              v3095_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v3095_data + (v2988_data * v3093_data)).copy_to(ir0 + (8));
              }
              int32_t v3100_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3106_data(0.0f);
              v3106_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[194]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3108_data(0.0f);
              v3108_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v3108_data + (v2988_data * v3106_data)).copy_to(ir0 + (9));
              }
              int32_t v3113_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3119_data(0.0f);
              v3119_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[214]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3121_data(0.0f);
              v3121_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v3121_data + (v2988_data * v3119_data)).copy_to(ir0 + (10));
              }
              int32_t v3126_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3132_data(0.0f);
              v3132_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[234]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3134_data(0.0f);
              v3134_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v3134_data + (v2988_data * v3132_data)).copy_to(ir0 + (11));
              }
              int32_t v3139_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3145_data(0.0f);
              v3145_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[254]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3147_data(0.0f);
              v3147_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v3147_data + (v2988_data * v3145_data)).copy_to(ir0 + (12));
              }
              int32_t v3152_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3158_data(0.0f);
              v3158_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[274]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3160_data(0.0f);
              v3160_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v3160_data + (v2988_data * v3158_data)).copy_to(ir0 + (13));
              }
              int32_t v3165_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3171_data(0.0f);
              v3171_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[294]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3173_data(0.0f);
              v3173_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v3173_data + (v2988_data * v3171_data)).copy_to(ir0 + (14));
              }
              int32_t v3178_a = 14 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3184_data(0.0f);
              v3184_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[314]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3186_data(0.0f);
              v3186_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v3186_data + (v2988_data * v3184_data)).copy_to(ir0 + (15));
              }
              int32_t v3193_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3198_data(0.0f);
              v3198_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3199_data(0.0f);
              v3199_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[15]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3201_data(0.0f);
              v3201_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v3201_data + (v3198_data * v3199_data)).copy_to(ir0 + (0));
              }
              int32_t v3206_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3212_data(0.0f);
              v3212_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[35]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3214_data(0.0f);
              v3214_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v3214_data + (v3198_data * v3212_data)).copy_to(ir0 + (1));
              }
              int32_t v3219_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3225_data(0.0f);
              v3225_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[55]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3227_data(0.0f);
              v3227_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v3227_data + (v3198_data * v3225_data)).copy_to(ir0 + (2));
              }
              int32_t v3232_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3238_data(0.0f);
              v3238_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[75]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3240_data(0.0f);
              v3240_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v3240_data + (v3198_data * v3238_data)).copy_to(ir0 + (3));
              }
              int32_t v3245_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3251_data(0.0f);
              v3251_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[95]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3253_data(0.0f);
              v3253_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v3253_data + (v3198_data * v3251_data)).copy_to(ir0 + (4));
              }
              int32_t v3258_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3264_data(0.0f);
              v3264_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[115]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3266_data(0.0f);
              v3266_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v3266_data + (v3198_data * v3264_data)).copy_to(ir0 + (5));
              }
              int32_t v3271_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3277_data(0.0f);
              v3277_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[135]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3279_data(0.0f);
              v3279_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v3279_data + (v3198_data * v3277_data)).copy_to(ir0 + (6));
              }
              int32_t v3284_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3290_data(0.0f);
              v3290_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[155]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3292_data(0.0f);
              v3292_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v3292_data + (v3198_data * v3290_data)).copy_to(ir0 + (7));
              }
              int32_t v3297_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3303_data(0.0f);
              v3303_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[175]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3305_data(0.0f);
              v3305_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v3305_data + (v3198_data * v3303_data)).copy_to(ir0 + (8));
              }
              int32_t v3310_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3316_data(0.0f);
              v3316_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[195]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3318_data(0.0f);
              v3318_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v3318_data + (v3198_data * v3316_data)).copy_to(ir0 + (9));
              }
              int32_t v3323_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3329_data(0.0f);
              v3329_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[215]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3331_data(0.0f);
              v3331_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v3331_data + (v3198_data * v3329_data)).copy_to(ir0 + (10));
              }
              int32_t v3336_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3342_data(0.0f);
              v3342_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[235]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3344_data(0.0f);
              v3344_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v3344_data + (v3198_data * v3342_data)).copy_to(ir0 + (11));
              }
              int32_t v3349_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3355_data(0.0f);
              v3355_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[255]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3357_data(0.0f);
              v3357_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v3357_data + (v3198_data * v3355_data)).copy_to(ir0 + (12));
              }
              int32_t v3362_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3368_data(0.0f);
              v3368_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[275]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3370_data(0.0f);
              v3370_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v3370_data + (v3198_data * v3368_data)).copy_to(ir0 + (13));
              }
              int32_t v3375_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3381_data(0.0f);
              v3381_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[295]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3383_data(0.0f);
              v3383_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v3383_data + (v3198_data * v3381_data)).copy_to(ir0 + (14));
              }
              int32_t v3388_a = 15 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3394_data(0.0f);
              v3394_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[315]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3396_data(0.0f);
              v3396_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v3396_data + (v3198_data * v3394_data)).copy_to(ir0 + (15));
              }
              int32_t v3403_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3408_data(0.0f);
              v3408_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3409_data(0.0f);
              v3409_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[16]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3411_data(0.0f);
              v3411_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v3411_data + (v3408_data * v3409_data)).copy_to(ir0 + (0));
              }
              int32_t v3416_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3422_data(0.0f);
              v3422_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[36]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3424_data(0.0f);
              v3424_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v3424_data + (v3408_data * v3422_data)).copy_to(ir0 + (1));
              }
              int32_t v3429_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3435_data(0.0f);
              v3435_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[56]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3437_data(0.0f);
              v3437_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v3437_data + (v3408_data * v3435_data)).copy_to(ir0 + (2));
              }
              int32_t v3442_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3448_data(0.0f);
              v3448_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[76]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3450_data(0.0f);
              v3450_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v3450_data + (v3408_data * v3448_data)).copy_to(ir0 + (3));
              }
              int32_t v3455_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3461_data(0.0f);
              v3461_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[96]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3463_data(0.0f);
              v3463_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v3463_data + (v3408_data * v3461_data)).copy_to(ir0 + (4));
              }
              int32_t v3468_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3474_data(0.0f);
              v3474_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[116]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3476_data(0.0f);
              v3476_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v3476_data + (v3408_data * v3474_data)).copy_to(ir0 + (5));
              }
              int32_t v3481_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3487_data(0.0f);
              v3487_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[136]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3489_data(0.0f);
              v3489_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v3489_data + (v3408_data * v3487_data)).copy_to(ir0 + (6));
              }
              int32_t v3494_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3500_data(0.0f);
              v3500_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[156]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3502_data(0.0f);
              v3502_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v3502_data + (v3408_data * v3500_data)).copy_to(ir0 + (7));
              }
              int32_t v3507_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3513_data(0.0f);
              v3513_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[176]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3515_data(0.0f);
              v3515_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v3515_data + (v3408_data * v3513_data)).copy_to(ir0 + (8));
              }
              int32_t v3520_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3526_data(0.0f);
              v3526_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[196]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3528_data(0.0f);
              v3528_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v3528_data + (v3408_data * v3526_data)).copy_to(ir0 + (9));
              }
              int32_t v3533_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3539_data(0.0f);
              v3539_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[216]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3541_data(0.0f);
              v3541_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v3541_data + (v3408_data * v3539_data)).copy_to(ir0 + (10));
              }
              int32_t v3546_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3552_data(0.0f);
              v3552_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[236]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3554_data(0.0f);
              v3554_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v3554_data + (v3408_data * v3552_data)).copy_to(ir0 + (11));
              }
              int32_t v3559_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3565_data(0.0f);
              v3565_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[256]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3567_data(0.0f);
              v3567_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v3567_data + (v3408_data * v3565_data)).copy_to(ir0 + (12));
              }
              int32_t v3572_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3578_data(0.0f);
              v3578_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[276]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3580_data(0.0f);
              v3580_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v3580_data + (v3408_data * v3578_data)).copy_to(ir0 + (13));
              }
              int32_t v3585_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3591_data(0.0f);
              v3591_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[296]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3593_data(0.0f);
              v3593_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v3593_data + (v3408_data * v3591_data)).copy_to(ir0 + (14));
              }
              int32_t v3598_a = 16 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3604_data(0.0f);
              v3604_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[316]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3606_data(0.0f);
              v3606_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v3606_data + (v3408_data * v3604_data)).copy_to(ir0 + (15));
              }
              int32_t v3613_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3618_data(0.0f);
              v3618_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3619_data(0.0f);
              v3619_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[17]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3621_data(0.0f);
              v3621_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v3621_data + (v3618_data * v3619_data)).copy_to(ir0 + (0));
              }
              int32_t v3626_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3632_data(0.0f);
              v3632_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[37]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3634_data(0.0f);
              v3634_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v3634_data + (v3618_data * v3632_data)).copy_to(ir0 + (1));
              }
              int32_t v3639_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3645_data(0.0f);
              v3645_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[57]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3647_data(0.0f);
              v3647_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v3647_data + (v3618_data * v3645_data)).copy_to(ir0 + (2));
              }
              int32_t v3652_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3658_data(0.0f);
              v3658_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[77]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3660_data(0.0f);
              v3660_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v3660_data + (v3618_data * v3658_data)).copy_to(ir0 + (3));
              }
              int32_t v3665_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3671_data(0.0f);
              v3671_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[97]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3673_data(0.0f);
              v3673_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v3673_data + (v3618_data * v3671_data)).copy_to(ir0 + (4));
              }
              int32_t v3678_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3684_data(0.0f);
              v3684_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[117]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3686_data(0.0f);
              v3686_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v3686_data + (v3618_data * v3684_data)).copy_to(ir0 + (5));
              }
              int32_t v3691_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3697_data(0.0f);
              v3697_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[137]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3699_data(0.0f);
              v3699_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v3699_data + (v3618_data * v3697_data)).copy_to(ir0 + (6));
              }
              int32_t v3704_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3710_data(0.0f);
              v3710_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[157]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3712_data(0.0f);
              v3712_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v3712_data + (v3618_data * v3710_data)).copy_to(ir0 + (7));
              }
              int32_t v3717_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3723_data(0.0f);
              v3723_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[177]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3725_data(0.0f);
              v3725_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v3725_data + (v3618_data * v3723_data)).copy_to(ir0 + (8));
              }
              int32_t v3730_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3736_data(0.0f);
              v3736_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[197]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3738_data(0.0f);
              v3738_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v3738_data + (v3618_data * v3736_data)).copy_to(ir0 + (9));
              }
              int32_t v3743_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3749_data(0.0f);
              v3749_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[217]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3751_data(0.0f);
              v3751_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v3751_data + (v3618_data * v3749_data)).copy_to(ir0 + (10));
              }
              int32_t v3756_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3762_data(0.0f);
              v3762_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[237]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3764_data(0.0f);
              v3764_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v3764_data + (v3618_data * v3762_data)).copy_to(ir0 + (11));
              }
              int32_t v3769_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3775_data(0.0f);
              v3775_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[257]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3777_data(0.0f);
              v3777_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v3777_data + (v3618_data * v3775_data)).copy_to(ir0 + (12));
              }
              int32_t v3782_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3788_data(0.0f);
              v3788_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[277]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3790_data(0.0f);
              v3790_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v3790_data + (v3618_data * v3788_data)).copy_to(ir0 + (13));
              }
              int32_t v3795_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3801_data(0.0f);
              v3801_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[297]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3803_data(0.0f);
              v3803_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v3803_data + (v3618_data * v3801_data)).copy_to(ir0 + (14));
              }
              int32_t v3808_a = 17 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3814_data(0.0f);
              v3814_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[317]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3816_data(0.0f);
              v3816_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v3816_data + (v3618_data * v3814_data)).copy_to(ir0 + (15));
              }
              int32_t v3823_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3828_data(0.0f);
              v3828_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3829_data(0.0f);
              v3829_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[18]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3831_data(0.0f);
              v3831_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v3831_data + (v3828_data * v3829_data)).copy_to(ir0 + (0));
              }
              int32_t v3836_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3842_data(0.0f);
              v3842_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[38]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3844_data(0.0f);
              v3844_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v3844_data + (v3828_data * v3842_data)).copy_to(ir0 + (1));
              }
              int32_t v3849_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3855_data(0.0f);
              v3855_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[58]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3857_data(0.0f);
              v3857_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v3857_data + (v3828_data * v3855_data)).copy_to(ir0 + (2));
              }
              int32_t v3862_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3868_data(0.0f);
              v3868_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[78]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3870_data(0.0f);
              v3870_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v3870_data + (v3828_data * v3868_data)).copy_to(ir0 + (3));
              }
              int32_t v3875_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3881_data(0.0f);
              v3881_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[98]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3883_data(0.0f);
              v3883_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v3883_data + (v3828_data * v3881_data)).copy_to(ir0 + (4));
              }
              int32_t v3888_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3894_data(0.0f);
              v3894_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[118]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3896_data(0.0f);
              v3896_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v3896_data + (v3828_data * v3894_data)).copy_to(ir0 + (5));
              }
              int32_t v3901_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3907_data(0.0f);
              v3907_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[138]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3909_data(0.0f);
              v3909_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v3909_data + (v3828_data * v3907_data)).copy_to(ir0 + (6));
              }
              int32_t v3914_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3920_data(0.0f);
              v3920_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[158]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3922_data(0.0f);
              v3922_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v3922_data + (v3828_data * v3920_data)).copy_to(ir0 + (7));
              }
              int32_t v3927_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3933_data(0.0f);
              v3933_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[178]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3935_data(0.0f);
              v3935_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v3935_data + (v3828_data * v3933_data)).copy_to(ir0 + (8));
              }
              int32_t v3940_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3946_data(0.0f);
              v3946_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[198]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3948_data(0.0f);
              v3948_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v3948_data + (v3828_data * v3946_data)).copy_to(ir0 + (9));
              }
              int32_t v3953_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3959_data(0.0f);
              v3959_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[218]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3961_data(0.0f);
              v3961_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v3961_data + (v3828_data * v3959_data)).copy_to(ir0 + (10));
              }
              int32_t v3966_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3972_data(0.0f);
              v3972_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[238]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3974_data(0.0f);
              v3974_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v3974_data + (v3828_data * v3972_data)).copy_to(ir0 + (11));
              }
              int32_t v3979_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3985_data(0.0f);
              v3985_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[258]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v3987_data(0.0f);
              v3987_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v3987_data + (v3828_data * v3985_data)).copy_to(ir0 + (12));
              }
              int32_t v3992_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v3998_data(0.0f);
              v3998_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[278]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4000_data(0.0f);
              v4000_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v4000_data + (v3828_data * v3998_data)).copy_to(ir0 + (13));
              }
              int32_t v4005_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4011_data(0.0f);
              v4011_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[298]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4013_data(0.0f);
              v4013_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v4013_data + (v3828_data * v4011_data)).copy_to(ir0 + (14));
              }
              int32_t v4018_a = 18 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4024_data(0.0f);
              v4024_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[318]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4026_data(0.0f);
              v4026_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v4026_data + (v3828_data * v4024_data)).copy_to(ir0 + (15));
              }
              int32_t v4033_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4038_data(0.0f);
              v4038_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19_i32]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4039_data(0.0f);
              v4039_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[19]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4041_data(0.0f);
              v4041_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v39_g);
              if (v39_g) {
                (v4041_data + (v4038_data * v4039_data)).copy_to(ir0 + (0));
              }
              int32_t v4046_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4052_data(0.0f);
              v4052_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[39]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4054_data(0.0f);
              v4054_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v39_g);
              if (v39_g) {
                (v4054_data + (v4038_data * v4052_data)).copy_to(ir0 + (1));
              }
              int32_t v4059_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4065_data(0.0f);
              v4065_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[59]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4067_data(0.0f);
              v4067_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v39_g);
              if (v39_g) {
                (v4067_data + (v4038_data * v4065_data)).copy_to(ir0 + (2));
              }
              int32_t v4072_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4078_data(0.0f);
              v4078_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[79]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4080_data(0.0f);
              v4080_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v39_g);
              if (v39_g) {
                (v4080_data + (v4038_data * v4078_data)).copy_to(ir0 + (3));
              }
              int32_t v4085_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4091_data(0.0f);
              v4091_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[99]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4093_data(0.0f);
              v4093_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v39_g);
              if (v39_g) {
                (v4093_data + (v4038_data * v4091_data)).copy_to(ir0 + (4));
              }
              int32_t v4098_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4104_data(0.0f);
              v4104_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[119]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4106_data(0.0f);
              v4106_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v39_g);
              if (v39_g) {
                (v4106_data + (v4038_data * v4104_data)).copy_to(ir0 + (5));
              }
              int32_t v4111_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4117_data(0.0f);
              v4117_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[139]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4119_data(0.0f);
              v4119_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v39_g);
              if (v39_g) {
                (v4119_data + (v4038_data * v4117_data)).copy_to(ir0 + (6));
              }
              int32_t v4124_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4130_data(0.0f);
              v4130_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[159]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4132_data(0.0f);
              v4132_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v39_g);
              if (v39_g) {
                (v4132_data + (v4038_data * v4130_data)).copy_to(ir0 + (7));
              }
              int32_t v4137_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4143_data(0.0f);
              v4143_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[179]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4145_data(0.0f);
              v4145_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v39_g);
              if (v39_g) {
                (v4145_data + (v4038_data * v4143_data)).copy_to(ir0 + (8));
              }
              int32_t v4150_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4156_data(0.0f);
              v4156_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[199]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4158_data(0.0f);
              v4158_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[9]), v39_g);
              if (v39_g) {
                (v4158_data + (v4038_data * v4156_data)).copy_to(ir0 + (9));
              }
              int32_t v4163_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4169_data(0.0f);
              v4169_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[219]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4171_data(0.0f);
              v4171_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[10]), v39_g);
              if (v39_g) {
                (v4171_data + (v4038_data * v4169_data)).copy_to(ir0 + (10));
              }
              int32_t v4176_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4182_data(0.0f);
              v4182_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[239]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4184_data(0.0f);
              v4184_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[11]), v39_g);
              if (v39_g) {
                (v4184_data + (v4038_data * v4182_data)).copy_to(ir0 + (11));
              }
              int32_t v4189_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4195_data(0.0f);
              v4195_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[259]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4197_data(0.0f);
              v4197_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[12]), v39_g);
              if (v39_g) {
                (v4197_data + (v4038_data * v4195_data)).copy_to(ir0 + (12));
              }
              int32_t v4202_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4208_data(0.0f);
              v4208_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[279]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4210_data(0.0f);
              v4210_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[13]), v39_g);
              if (v39_g) {
                (v4210_data + (v4038_data * v4208_data)).copy_to(ir0 + (13));
              }
              int32_t v4215_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4221_data(0.0f);
              v4221_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[299]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4223_data(0.0f);
              v4223_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[14]), v39_g);
              if (v39_g) {
                (v4223_data + (v4038_data * v4221_data)).copy_to(ir0 + (14));
              }
              int32_t v4228_a = 19 + 0_i32;
              tensorforge::intel_esimd::simd<float, 16> v4234_data(0.0f);
              v4234_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[319]), v39_g);
              tensorforge::intel_esimd::simd<float, 16> v4236_data(0.0f);
              v4236_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[15]), v39_g);
              if (v39_g) {
                (v4236_data + (v4038_data * v4234_data)).copy_to(ir0 + (15));
              }
              #pragma unroll
              for (int32_t v4240_n1 = 0; v4240_n1 < 16; ++v4240_n1) {
                int32_t v4241_a = 0 + v4240_n1;
                tensorforge::intel_esimd::simd<float, 16> v4243_data(0.0f);
                v4243_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[v4240_n1]), v39_g);
                if (v39_g) {
                  v4243_data.copy_to(r0 + (v4240_n1));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v4247_i1 = 0; v4247_i1 < 16; ++v4247_i1) {
                int32_t v4248_a = 0 + v4247_i1;
                tensorforge::intel_esimd::simd<float, 16> v4250_data(0.0f);
                v4250_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v4247_i1]), v39_g);
                if (v39_g) {
                  v4250_data.copy_to(glb_m0 + ((v4247_i1 * 12)));
                }
              }
            }
          }
        }
      });
    }
  });
}

