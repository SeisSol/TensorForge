// === base name ===
kernel_4b59b6f027

// === header ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4b59b6f027(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4b59b6f027(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×8(12×8) {4..16}×{0..8} strided
        // m1 16×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              // r0 = +(glb_m1 * s0) + None
              // [(4, 16), (0, 8)] [(0, 16)]
              float ir0[8]{};
              tensorforge::intel_esimd::simd_mask<16> v7_g = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) >= 4;
              int32_t v11_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v16_data(0.0f);
              v16_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[-4_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v17_data(0.0f);
              v17_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v19_data(0.0f);
              v19_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v19_data + (v16_data * v17_data)).copy_to(ir0 + (0));
              }
              int32_t v24_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v30_data(0.0f);
              v30_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v32_data(0.0f);
              v32_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v32_data + (v16_data * v30_data)).copy_to(ir0 + (1));
              }
              int32_t v37_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v43_data(0.0f);
              v43_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v45_data(0.0f);
              v45_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v45_data + (v16_data * v43_data)).copy_to(ir0 + (2));
              }
              int32_t v50_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v56_data(0.0f);
              v56_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v58_data(0.0f);
              v58_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v58_data + (v16_data * v56_data)).copy_to(ir0 + (3));
              }
              int32_t v63_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v69_data(0.0f);
              v69_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v71_data(0.0f);
              v71_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v71_data + (v16_data * v69_data)).copy_to(ir0 + (4));
              }
              int32_t v76_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v82_data(0.0f);
              v82_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v84_data(0.0f);
              v84_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v84_data + (v16_data * v82_data)).copy_to(ir0 + (5));
              }
              int32_t v89_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v95_data(0.0f);
              v95_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[96]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v97_data(0.0f);
              v97_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v97_data + (v16_data * v95_data)).copy_to(ir0 + (6));
              }
              int32_t v102_a = -4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v108_data(0.0f);
              v108_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[112]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v110_data(0.0f);
              v110_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v110_data + (v16_data * v108_data)).copy_to(ir0 + (7));
              }
              int32_t v117_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v122_data(0.0f);
              v122_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[8_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v123_data(0.0f);
              v123_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v125_data(0.0f);
              v125_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v125_data + (v122_data * v123_data)).copy_to(ir0 + (0));
              }
              int32_t v130_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v136_data(0.0f);
              v136_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v138_data(0.0f);
              v138_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v138_data + (v122_data * v136_data)).copy_to(ir0 + (1));
              }
              int32_t v143_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v149_data(0.0f);
              v149_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v151_data(0.0f);
              v151_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v151_data + (v122_data * v149_data)).copy_to(ir0 + (2));
              }
              int32_t v156_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v162_data(0.0f);
              v162_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v164_data(0.0f);
              v164_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v164_data + (v122_data * v162_data)).copy_to(ir0 + (3));
              }
              int32_t v169_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v175_data(0.0f);
              v175_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v177_data(0.0f);
              v177_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v177_data + (v122_data * v175_data)).copy_to(ir0 + (4));
              }
              int32_t v182_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v188_data(0.0f);
              v188_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[81]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v190_data(0.0f);
              v190_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v190_data + (v122_data * v188_data)).copy_to(ir0 + (5));
              }
              int32_t v195_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v201_data(0.0f);
              v201_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[97]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v203_data(0.0f);
              v203_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v203_data + (v122_data * v201_data)).copy_to(ir0 + (6));
              }
              int32_t v208_a = -4_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v214_data(0.0f);
              v214_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[113]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v216_data(0.0f);
              v216_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v216_data + (v122_data * v214_data)).copy_to(ir0 + (7));
              }
              int32_t v223_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v228_data(0.0f);
              v228_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[20_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v229_data(0.0f);
              v229_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v231_data(0.0f);
              v231_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v231_data + (v228_data * v229_data)).copy_to(ir0 + (0));
              }
              int32_t v236_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v242_data(0.0f);
              v242_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v244_data(0.0f);
              v244_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v244_data + (v228_data * v242_data)).copy_to(ir0 + (1));
              }
              int32_t v249_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v255_data(0.0f);
              v255_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v257_data(0.0f);
              v257_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v257_data + (v228_data * v255_data)).copy_to(ir0 + (2));
              }
              int32_t v262_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v268_data(0.0f);
              v268_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[50]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v270_data(0.0f);
              v270_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v270_data + (v228_data * v268_data)).copy_to(ir0 + (3));
              }
              int32_t v275_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v281_data(0.0f);
              v281_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v283_data(0.0f);
              v283_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v283_data + (v228_data * v281_data)).copy_to(ir0 + (4));
              }
              int32_t v288_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v294_data(0.0f);
              v294_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[82]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v296_data(0.0f);
              v296_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v296_data + (v228_data * v294_data)).copy_to(ir0 + (5));
              }
              int32_t v301_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v307_data(0.0f);
              v307_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[98]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v309_data(0.0f);
              v309_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v309_data + (v228_data * v307_data)).copy_to(ir0 + (6));
              }
              int32_t v314_a = -4_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v320_data(0.0f);
              v320_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[114]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v322_data(0.0f);
              v322_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v322_data + (v228_data * v320_data)).copy_to(ir0 + (7));
              }
              int32_t v329_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v334_data(0.0f);
              v334_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[32_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v335_data(0.0f);
              v335_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v337_data(0.0f);
              v337_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v337_data + (v334_data * v335_data)).copy_to(ir0 + (0));
              }
              int32_t v342_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v348_data(0.0f);
              v348_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v350_data(0.0f);
              v350_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v350_data + (v334_data * v348_data)).copy_to(ir0 + (1));
              }
              int32_t v355_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v361_data(0.0f);
              v361_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v363_data(0.0f);
              v363_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v363_data + (v334_data * v361_data)).copy_to(ir0 + (2));
              }
              int32_t v368_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v374_data(0.0f);
              v374_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v376_data(0.0f);
              v376_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v376_data + (v334_data * v374_data)).copy_to(ir0 + (3));
              }
              int32_t v381_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v387_data(0.0f);
              v387_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[67]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v389_data(0.0f);
              v389_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v389_data + (v334_data * v387_data)).copy_to(ir0 + (4));
              }
              int32_t v394_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v400_data(0.0f);
              v400_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[83]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v402_data(0.0f);
              v402_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v402_data + (v334_data * v400_data)).copy_to(ir0 + (5));
              }
              int32_t v407_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v413_data(0.0f);
              v413_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[99]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v415_data(0.0f);
              v415_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v415_data + (v334_data * v413_data)).copy_to(ir0 + (6));
              }
              int32_t v420_a = -4_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v426_data(0.0f);
              v426_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[115]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v428_data(0.0f);
              v428_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v428_data + (v334_data * v426_data)).copy_to(ir0 + (7));
              }
              int32_t v435_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v440_data(0.0f);
              v440_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[44_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v441_data(0.0f);
              v441_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v443_data(0.0f);
              v443_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v443_data + (v440_data * v441_data)).copy_to(ir0 + (0));
              }
              int32_t v448_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v454_data(0.0f);
              v454_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v456_data(0.0f);
              v456_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v456_data + (v440_data * v454_data)).copy_to(ir0 + (1));
              }
              int32_t v461_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v467_data(0.0f);
              v467_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v469_data(0.0f);
              v469_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v469_data + (v440_data * v467_data)).copy_to(ir0 + (2));
              }
              int32_t v474_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v480_data(0.0f);
              v480_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v482_data(0.0f);
              v482_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v482_data + (v440_data * v480_data)).copy_to(ir0 + (3));
              }
              int32_t v487_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v493_data(0.0f);
              v493_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v495_data(0.0f);
              v495_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v495_data + (v440_data * v493_data)).copy_to(ir0 + (4));
              }
              int32_t v500_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v506_data(0.0f);
              v506_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[84]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v508_data(0.0f);
              v508_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v508_data + (v440_data * v506_data)).copy_to(ir0 + (5));
              }
              int32_t v513_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v519_data(0.0f);
              v519_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[100]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v521_data(0.0f);
              v521_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v521_data + (v440_data * v519_data)).copy_to(ir0 + (6));
              }
              int32_t v526_a = -4_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v532_data(0.0f);
              v532_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[116]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v534_data(0.0f);
              v534_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v534_data + (v440_data * v532_data)).copy_to(ir0 + (7));
              }
              int32_t v541_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v546_data(0.0f);
              v546_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[56_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v547_data(0.0f);
              v547_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v549_data(0.0f);
              v549_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v549_data + (v546_data * v547_data)).copy_to(ir0 + (0));
              }
              int32_t v554_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v560_data(0.0f);
              v560_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v562_data(0.0f);
              v562_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v562_data + (v546_data * v560_data)).copy_to(ir0 + (1));
              }
              int32_t v567_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v573_data(0.0f);
              v573_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v575_data(0.0f);
              v575_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v575_data + (v546_data * v573_data)).copy_to(ir0 + (2));
              }
              int32_t v580_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v586_data(0.0f);
              v586_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v588_data(0.0f);
              v588_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v588_data + (v546_data * v586_data)).copy_to(ir0 + (3));
              }
              int32_t v593_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v599_data(0.0f);
              v599_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v601_data(0.0f);
              v601_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v601_data + (v546_data * v599_data)).copy_to(ir0 + (4));
              }
              int32_t v606_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v612_data(0.0f);
              v612_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[85]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v614_data(0.0f);
              v614_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v614_data + (v546_data * v612_data)).copy_to(ir0 + (5));
              }
              int32_t v619_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v625_data(0.0f);
              v625_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[101]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v627_data(0.0f);
              v627_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v627_data + (v546_data * v625_data)).copy_to(ir0 + (6));
              }
              int32_t v632_a = -4_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v638_data(0.0f);
              v638_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[117]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v640_data(0.0f);
              v640_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v640_data + (v546_data * v638_data)).copy_to(ir0 + (7));
              }
              int32_t v647_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v652_data(0.0f);
              v652_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[68_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v653_data(0.0f);
              v653_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v655_data(0.0f);
              v655_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v655_data + (v652_data * v653_data)).copy_to(ir0 + (0));
              }
              int32_t v660_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v666_data(0.0f);
              v666_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v668_data(0.0f);
              v668_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v668_data + (v652_data * v666_data)).copy_to(ir0 + (1));
              }
              int32_t v673_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v679_data(0.0f);
              v679_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v681_data(0.0f);
              v681_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v681_data + (v652_data * v679_data)).copy_to(ir0 + (2));
              }
              int32_t v686_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v692_data(0.0f);
              v692_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v694_data(0.0f);
              v694_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v694_data + (v652_data * v692_data)).copy_to(ir0 + (3));
              }
              int32_t v699_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v705_data(0.0f);
              v705_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v707_data(0.0f);
              v707_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v707_data + (v652_data * v705_data)).copy_to(ir0 + (4));
              }
              int32_t v712_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v718_data(0.0f);
              v718_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[86]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v720_data(0.0f);
              v720_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v720_data + (v652_data * v718_data)).copy_to(ir0 + (5));
              }
              int32_t v725_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v731_data(0.0f);
              v731_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[102]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v733_data(0.0f);
              v733_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v733_data + (v652_data * v731_data)).copy_to(ir0 + (6));
              }
              int32_t v738_a = -4_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v744_data(0.0f);
              v744_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[118]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v746_data(0.0f);
              v746_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v746_data + (v652_data * v744_data)).copy_to(ir0 + (7));
              }
              int32_t v753_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v758_data(0.0f);
              v758_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[80_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v759_data(0.0f);
              v759_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v761_data(0.0f);
              v761_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v761_data + (v758_data * v759_data)).copy_to(ir0 + (0));
              }
              int32_t v766_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v772_data(0.0f);
              v772_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v774_data(0.0f);
              v774_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v774_data + (v758_data * v772_data)).copy_to(ir0 + (1));
              }
              int32_t v779_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v785_data(0.0f);
              v785_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v787_data(0.0f);
              v787_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v787_data + (v758_data * v785_data)).copy_to(ir0 + (2));
              }
              int32_t v792_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v798_data(0.0f);
              v798_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v800_data(0.0f);
              v800_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v800_data + (v758_data * v798_data)).copy_to(ir0 + (3));
              }
              int32_t v805_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v811_data(0.0f);
              v811_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v813_data(0.0f);
              v813_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v813_data + (v758_data * v811_data)).copy_to(ir0 + (4));
              }
              int32_t v818_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v824_data(0.0f);
              v824_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[87]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v826_data(0.0f);
              v826_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v826_data + (v758_data * v824_data)).copy_to(ir0 + (5));
              }
              int32_t v831_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v837_data(0.0f);
              v837_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[103]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v839_data(0.0f);
              v839_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v839_data + (v758_data * v837_data)).copy_to(ir0 + (6));
              }
              int32_t v844_a = -4_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v850_data(0.0f);
              v850_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[119]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v852_data(0.0f);
              v852_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v852_data + (v758_data * v850_data)).copy_to(ir0 + (7));
              }
              int32_t v859_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v864_data(0.0f);
              v864_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[92_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v865_data(0.0f);
              v865_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v867_data(0.0f);
              v867_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v867_data + (v864_data * v865_data)).copy_to(ir0 + (0));
              }
              int32_t v872_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v878_data(0.0f);
              v878_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v880_data(0.0f);
              v880_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v880_data + (v864_data * v878_data)).copy_to(ir0 + (1));
              }
              int32_t v885_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v891_data(0.0f);
              v891_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v893_data(0.0f);
              v893_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v893_data + (v864_data * v891_data)).copy_to(ir0 + (2));
              }
              int32_t v898_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v904_data(0.0f);
              v904_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v906_data(0.0f);
              v906_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v906_data + (v864_data * v904_data)).copy_to(ir0 + (3));
              }
              int32_t v911_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v917_data(0.0f);
              v917_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v919_data(0.0f);
              v919_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v919_data + (v864_data * v917_data)).copy_to(ir0 + (4));
              }
              int32_t v924_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v930_data(0.0f);
              v930_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[88]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v932_data(0.0f);
              v932_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v932_data + (v864_data * v930_data)).copy_to(ir0 + (5));
              }
              int32_t v937_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v943_data(0.0f);
              v943_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[104]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v945_data(0.0f);
              v945_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v945_data + (v864_data * v943_data)).copy_to(ir0 + (6));
              }
              int32_t v950_a = -4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v956_data(0.0f);
              v956_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[120]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v958_data(0.0f);
              v958_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v958_data + (v864_data * v956_data)).copy_to(ir0 + (7));
              }
              int32_t v965_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v970_data(0.0f);
              v970_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[104_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v971_data(0.0f);
              v971_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v973_data(0.0f);
              v973_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v973_data + (v970_data * v971_data)).copy_to(ir0 + (0));
              }
              int32_t v978_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v984_data(0.0f);
              v984_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v986_data(0.0f);
              v986_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v986_data + (v970_data * v984_data)).copy_to(ir0 + (1));
              }
              int32_t v991_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v997_data(0.0f);
              v997_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v999_data(0.0f);
              v999_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v999_data + (v970_data * v997_data)).copy_to(ir0 + (2));
              }
              int32_t v1004_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1010_data(0.0f);
              v1010_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1012_data(0.0f);
              v1012_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1012_data + (v970_data * v1010_data)).copy_to(ir0 + (3));
              }
              int32_t v1017_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1023_data(0.0f);
              v1023_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1025_data(0.0f);
              v1025_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1025_data + (v970_data * v1023_data)).copy_to(ir0 + (4));
              }
              int32_t v1030_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1036_data(0.0f);
              v1036_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[89]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1038_data(0.0f);
              v1038_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1038_data + (v970_data * v1036_data)).copy_to(ir0 + (5));
              }
              int32_t v1043_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1049_data(0.0f);
              v1049_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[105]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1051_data(0.0f);
              v1051_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1051_data + (v970_data * v1049_data)).copy_to(ir0 + (6));
              }
              int32_t v1056_a = -4_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1062_data(0.0f);
              v1062_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[121]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1064_data(0.0f);
              v1064_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1064_data + (v970_data * v1062_data)).copy_to(ir0 + (7));
              }
              int32_t v1071_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1076_data(0.0f);
              v1076_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[116_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1077_data(0.0f);
              v1077_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1079_data(0.0f);
              v1079_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1079_data + (v1076_data * v1077_data)).copy_to(ir0 + (0));
              }
              int32_t v1084_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1090_data(0.0f);
              v1090_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1092_data(0.0f);
              v1092_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1092_data + (v1076_data * v1090_data)).copy_to(ir0 + (1));
              }
              int32_t v1097_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1103_data(0.0f);
              v1103_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1105_data(0.0f);
              v1105_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1105_data + (v1076_data * v1103_data)).copy_to(ir0 + (2));
              }
              int32_t v1110_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1116_data(0.0f);
              v1116_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1118_data(0.0f);
              v1118_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1118_data + (v1076_data * v1116_data)).copy_to(ir0 + (3));
              }
              int32_t v1123_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1129_data(0.0f);
              v1129_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1131_data(0.0f);
              v1131_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1131_data + (v1076_data * v1129_data)).copy_to(ir0 + (4));
              }
              int32_t v1136_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1142_data(0.0f);
              v1142_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[90]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1144_data(0.0f);
              v1144_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1144_data + (v1076_data * v1142_data)).copy_to(ir0 + (5));
              }
              int32_t v1149_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1155_data(0.0f);
              v1155_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[106]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1157_data(0.0f);
              v1157_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1157_data + (v1076_data * v1155_data)).copy_to(ir0 + (6));
              }
              int32_t v1162_a = -4_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1168_data(0.0f);
              v1168_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[122]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1170_data(0.0f);
              v1170_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1170_data + (v1076_data * v1168_data)).copy_to(ir0 + (7));
              }
              int32_t v1177_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1182_data(0.0f);
              v1182_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[128_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1183_data(0.0f);
              v1183_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1185_data(0.0f);
              v1185_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1185_data + (v1182_data * v1183_data)).copy_to(ir0 + (0));
              }
              int32_t v1190_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1196_data(0.0f);
              v1196_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1198_data(0.0f);
              v1198_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1198_data + (v1182_data * v1196_data)).copy_to(ir0 + (1));
              }
              int32_t v1203_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1209_data(0.0f);
              v1209_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1211_data(0.0f);
              v1211_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1211_data + (v1182_data * v1209_data)).copy_to(ir0 + (2));
              }
              int32_t v1216_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1222_data(0.0f);
              v1222_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1224_data(0.0f);
              v1224_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1224_data + (v1182_data * v1222_data)).copy_to(ir0 + (3));
              }
              int32_t v1229_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1235_data(0.0f);
              v1235_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1237_data(0.0f);
              v1237_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1237_data + (v1182_data * v1235_data)).copy_to(ir0 + (4));
              }
              int32_t v1242_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1248_data(0.0f);
              v1248_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[91]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1250_data(0.0f);
              v1250_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1250_data + (v1182_data * v1248_data)).copy_to(ir0 + (5));
              }
              int32_t v1255_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1261_data(0.0f);
              v1261_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[107]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1263_data(0.0f);
              v1263_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1263_data + (v1182_data * v1261_data)).copy_to(ir0 + (6));
              }
              int32_t v1268_a = -4_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1274_data(0.0f);
              v1274_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[123]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1276_data(0.0f);
              v1276_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1276_data + (v1182_data * v1274_data)).copy_to(ir0 + (7));
              }
              int32_t v1283_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1288_data(0.0f);
              v1288_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[140_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1289_data(0.0f);
              v1289_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1291_data(0.0f);
              v1291_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1291_data + (v1288_data * v1289_data)).copy_to(ir0 + (0));
              }
              int32_t v1296_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1302_data(0.0f);
              v1302_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1304_data(0.0f);
              v1304_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1304_data + (v1288_data * v1302_data)).copy_to(ir0 + (1));
              }
              int32_t v1309_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1315_data(0.0f);
              v1315_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1317_data(0.0f);
              v1317_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1317_data + (v1288_data * v1315_data)).copy_to(ir0 + (2));
              }
              int32_t v1322_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1328_data(0.0f);
              v1328_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1330_data(0.0f);
              v1330_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1330_data + (v1288_data * v1328_data)).copy_to(ir0 + (3));
              }
              int32_t v1335_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1341_data(0.0f);
              v1341_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1343_data(0.0f);
              v1343_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1343_data + (v1288_data * v1341_data)).copy_to(ir0 + (4));
              }
              int32_t v1348_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1354_data(0.0f);
              v1354_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[92]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1356_data(0.0f);
              v1356_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1356_data + (v1288_data * v1354_data)).copy_to(ir0 + (5));
              }
              int32_t v1361_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1367_data(0.0f);
              v1367_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[108]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1369_data(0.0f);
              v1369_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1369_data + (v1288_data * v1367_data)).copy_to(ir0 + (6));
              }
              int32_t v1374_a = -4_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1380_data(0.0f);
              v1380_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[124]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1382_data(0.0f);
              v1382_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1382_data + (v1288_data * v1380_data)).copy_to(ir0 + (7));
              }
              int32_t v1389_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1394_data(0.0f);
              v1394_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[152_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1395_data(0.0f);
              v1395_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1397_data(0.0f);
              v1397_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1397_data + (v1394_data * v1395_data)).copy_to(ir0 + (0));
              }
              int32_t v1402_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1408_data(0.0f);
              v1408_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1410_data(0.0f);
              v1410_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1410_data + (v1394_data * v1408_data)).copy_to(ir0 + (1));
              }
              int32_t v1415_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1421_data(0.0f);
              v1421_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1423_data(0.0f);
              v1423_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1423_data + (v1394_data * v1421_data)).copy_to(ir0 + (2));
              }
              int32_t v1428_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1434_data(0.0f);
              v1434_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1436_data(0.0f);
              v1436_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1436_data + (v1394_data * v1434_data)).copy_to(ir0 + (3));
              }
              int32_t v1441_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1447_data(0.0f);
              v1447_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1449_data(0.0f);
              v1449_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1449_data + (v1394_data * v1447_data)).copy_to(ir0 + (4));
              }
              int32_t v1454_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1460_data(0.0f);
              v1460_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[93]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1462_data(0.0f);
              v1462_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1462_data + (v1394_data * v1460_data)).copy_to(ir0 + (5));
              }
              int32_t v1467_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1473_data(0.0f);
              v1473_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[109]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1475_data(0.0f);
              v1475_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1475_data + (v1394_data * v1473_data)).copy_to(ir0 + (6));
              }
              int32_t v1480_a = -4_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1486_data(0.0f);
              v1486_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[125]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1488_data(0.0f);
              v1488_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1488_data + (v1394_data * v1486_data)).copy_to(ir0 + (7));
              }
              int32_t v1495_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1500_data(0.0f);
              v1500_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[164_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1501_data(0.0f);
              v1501_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1503_data(0.0f);
              v1503_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1503_data + (v1500_data * v1501_data)).copy_to(ir0 + (0));
              }
              int32_t v1508_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1514_data(0.0f);
              v1514_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1516_data(0.0f);
              v1516_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1516_data + (v1500_data * v1514_data)).copy_to(ir0 + (1));
              }
              int32_t v1521_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1527_data(0.0f);
              v1527_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1529_data(0.0f);
              v1529_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1529_data + (v1500_data * v1527_data)).copy_to(ir0 + (2));
              }
              int32_t v1534_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1540_data(0.0f);
              v1540_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1542_data(0.0f);
              v1542_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1542_data + (v1500_data * v1540_data)).copy_to(ir0 + (3));
              }
              int32_t v1547_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1553_data(0.0f);
              v1553_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1555_data(0.0f);
              v1555_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1555_data + (v1500_data * v1553_data)).copy_to(ir0 + (4));
              }
              int32_t v1560_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1566_data(0.0f);
              v1566_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[94]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1568_data(0.0f);
              v1568_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1568_data + (v1500_data * v1566_data)).copy_to(ir0 + (5));
              }
              int32_t v1573_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1579_data(0.0f);
              v1579_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[110]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1581_data(0.0f);
              v1581_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1581_data + (v1500_data * v1579_data)).copy_to(ir0 + (6));
              }
              int32_t v1586_a = -4_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1592_data(0.0f);
              v1592_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[126]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1594_data(0.0f);
              v1594_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1594_data + (v1500_data * v1592_data)).copy_to(ir0 + (7));
              }
              int32_t v1601_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1606_data(0.0f);
              v1606_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[176_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1607_data(0.0f);
              v1607_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1609_data(0.0f);
              v1609_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1609_data + (v1606_data * v1607_data)).copy_to(ir0 + (0));
              }
              int32_t v1614_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1620_data(0.0f);
              v1620_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1622_data(0.0f);
              v1622_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1622_data + (v1606_data * v1620_data)).copy_to(ir0 + (1));
              }
              int32_t v1627_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1633_data(0.0f);
              v1633_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1635_data(0.0f);
              v1635_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1635_data + (v1606_data * v1633_data)).copy_to(ir0 + (2));
              }
              int32_t v1640_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1646_data(0.0f);
              v1646_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1648_data(0.0f);
              v1648_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1648_data + (v1606_data * v1646_data)).copy_to(ir0 + (3));
              }
              int32_t v1653_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1659_data(0.0f);
              v1659_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1661_data(0.0f);
              v1661_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1661_data + (v1606_data * v1659_data)).copy_to(ir0 + (4));
              }
              int32_t v1666_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1672_data(0.0f);
              v1672_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[95]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1674_data(0.0f);
              v1674_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1674_data + (v1606_data * v1672_data)).copy_to(ir0 + (5));
              }
              int32_t v1679_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1685_data(0.0f);
              v1685_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[111]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1687_data(0.0f);
              v1687_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1687_data + (v1606_data * v1685_data)).copy_to(ir0 + (6));
              }
              int32_t v1692_a = -4_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1698_data(0.0f);
              v1698_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[127]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1700_data(0.0f);
              v1700_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1700_data + (v1606_data * v1698_data)).copy_to(ir0 + (7));
              }
              #pragma unroll
              for (int32_t v1704_n1 = 0; v1704_n1 < 8; ++v1704_n1) {
                int32_t v1705_a = 0 + v1704_n1;
                tensorforge::intel_esimd::simd<float, 16> v1707_data(0.0f);
                v1707_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[v1704_n1]), v7_g);
                if (v7_g) {
                  v1707_data.copy_to(r0 + (v1704_n1));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1711_i1 = 0; v1711_i1 < 8; ++v1711_i1) {
                int32_t v1712_a = 0 + v1711_i1;
                tensorforge::intel_esimd::simd<float, 16> v1714_data(0.0f);
                v1714_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v1711_i1]), v7_g);
                if (v7_g) {
                  v1714_data.copy_to(glb_m0 + ((-4_i32 + (v1711_i1 * 12))));
                }
              }
            }
          }
        }
      });
    }
  });
}

