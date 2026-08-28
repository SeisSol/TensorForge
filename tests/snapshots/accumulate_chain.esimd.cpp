// === base name ===
kernel_8a03a3cd0d

// === header ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_8a03a3cd0d(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8a03a3cd0d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8a03a3cd0d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, const float* m7, unsigned m7_extraOffset, const float* m8, unsigned m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×12(12×12) {0..12}×{0..12} strided
        // m2 12×8(12×8) {0..12}×{0..8} strided
        // m3 12×12(12×12) {0..12}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 12×12(12×12) {0..12}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m7 12×12(12×12) {0..12}×{0..12} strided
        // m8 12×8(12×8) {0..12}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m2 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m3 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m5 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m7 12×12(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×m8 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[batchId0 * 96 + 0 + m8_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m2[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir0[8]{};
              tensorforge::intel_esimd::simd_mask<16> v13_g = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 12;
              int32_t v16_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v20_data(0.0f);
              v20_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[0_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v21_data(0.0f);
              v21_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v23_data(0.0f);
              v23_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v23_data + (v20_data * v21_data)).copy_to(ir0 + (0));
              }
              int32_t v27_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v32_data(0.0f);
              v32_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v34_data(0.0f);
              v34_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v34_data + (v20_data * v32_data)).copy_to(ir0 + (1));
              }
              int32_t v38_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v43_data(0.0f);
              v43_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v45_data(0.0f);
              v45_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v45_data + (v20_data * v43_data)).copy_to(ir0 + (2));
              }
              int32_t v49_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v54_data(0.0f);
              v54_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v56_data(0.0f);
              v56_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v56_data + (v20_data * v54_data)).copy_to(ir0 + (3));
              }
              int32_t v60_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v65_data(0.0f);
              v65_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v67_data(0.0f);
              v67_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v67_data + (v20_data * v65_data)).copy_to(ir0 + (4));
              }
              int32_t v71_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v76_data(0.0f);
              v76_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v78_data(0.0f);
              v78_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v78_data + (v20_data * v76_data)).copy_to(ir0 + (5));
              }
              int32_t v82_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v87_data(0.0f);
              v87_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v89_data(0.0f);
              v89_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v89_data + (v20_data * v87_data)).copy_to(ir0 + (6));
              }
              int32_t v93_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v98_data(0.0f);
              v98_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[84]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v100_data(0.0f);
              v100_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v100_data + (v20_data * v98_data)).copy_to(ir0 + (7));
              }
              int32_t v106_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v110_data(0.0f);
              v110_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[12_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v111_data(0.0f);
              v111_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v113_data(0.0f);
              v113_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v113_data + (v110_data * v111_data)).copy_to(ir0 + (0));
              }
              int32_t v117_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v122_data(0.0f);
              v122_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v124_data(0.0f);
              v124_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v124_data + (v110_data * v122_data)).copy_to(ir0 + (1));
              }
              int32_t v128_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v133_data(0.0f);
              v133_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v135_data(0.0f);
              v135_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v135_data + (v110_data * v133_data)).copy_to(ir0 + (2));
              }
              int32_t v139_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v144_data(0.0f);
              v144_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v146_data(0.0f);
              v146_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v146_data + (v110_data * v144_data)).copy_to(ir0 + (3));
              }
              int32_t v150_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v155_data(0.0f);
              v155_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v157_data(0.0f);
              v157_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v157_data + (v110_data * v155_data)).copy_to(ir0 + (4));
              }
              int32_t v161_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v166_data(0.0f);
              v166_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v168_data(0.0f);
              v168_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v168_data + (v110_data * v166_data)).copy_to(ir0 + (5));
              }
              int32_t v172_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v177_data(0.0f);
              v177_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v179_data(0.0f);
              v179_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v179_data + (v110_data * v177_data)).copy_to(ir0 + (6));
              }
              int32_t v183_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v188_data(0.0f);
              v188_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[85]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v190_data(0.0f);
              v190_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v190_data + (v110_data * v188_data)).copy_to(ir0 + (7));
              }
              int32_t v196_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v200_data(0.0f);
              v200_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[24_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v201_data(0.0f);
              v201_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v203_data(0.0f);
              v203_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v203_data + (v200_data * v201_data)).copy_to(ir0 + (0));
              }
              int32_t v207_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v212_data(0.0f);
              v212_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v214_data(0.0f);
              v214_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v214_data + (v200_data * v212_data)).copy_to(ir0 + (1));
              }
              int32_t v218_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v223_data(0.0f);
              v223_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v225_data(0.0f);
              v225_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v225_data + (v200_data * v223_data)).copy_to(ir0 + (2));
              }
              int32_t v229_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v234_data(0.0f);
              v234_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v236_data(0.0f);
              v236_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v236_data + (v200_data * v234_data)).copy_to(ir0 + (3));
              }
              int32_t v240_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v245_data(0.0f);
              v245_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[50]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v247_data(0.0f);
              v247_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v247_data + (v200_data * v245_data)).copy_to(ir0 + (4));
              }
              int32_t v251_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v256_data(0.0f);
              v256_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v258_data(0.0f);
              v258_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v258_data + (v200_data * v256_data)).copy_to(ir0 + (5));
              }
              int32_t v262_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v267_data(0.0f);
              v267_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v269_data(0.0f);
              v269_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v269_data + (v200_data * v267_data)).copy_to(ir0 + (6));
              }
              int32_t v273_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v278_data(0.0f);
              v278_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[86]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v280_data(0.0f);
              v280_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v280_data + (v200_data * v278_data)).copy_to(ir0 + (7));
              }
              int32_t v286_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v290_data(0.0f);
              v290_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[36_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v291_data(0.0f);
              v291_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v293_data(0.0f);
              v293_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v293_data + (v290_data * v291_data)).copy_to(ir0 + (0));
              }
              int32_t v297_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v302_data(0.0f);
              v302_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v304_data(0.0f);
              v304_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v304_data + (v290_data * v302_data)).copy_to(ir0 + (1));
              }
              int32_t v308_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v313_data(0.0f);
              v313_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v315_data(0.0f);
              v315_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v315_data + (v290_data * v313_data)).copy_to(ir0 + (2));
              }
              int32_t v319_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v324_data(0.0f);
              v324_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v326_data(0.0f);
              v326_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v326_data + (v290_data * v324_data)).copy_to(ir0 + (3));
              }
              int32_t v330_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v335_data(0.0f);
              v335_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v337_data(0.0f);
              v337_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v337_data + (v290_data * v335_data)).copy_to(ir0 + (4));
              }
              int32_t v341_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v346_data(0.0f);
              v346_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v348_data(0.0f);
              v348_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v348_data + (v290_data * v346_data)).copy_to(ir0 + (5));
              }
              int32_t v352_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v357_data(0.0f);
              v357_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v359_data(0.0f);
              v359_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v359_data + (v290_data * v357_data)).copy_to(ir0 + (6));
              }
              int32_t v363_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v368_data(0.0f);
              v368_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[87]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v370_data(0.0f);
              v370_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v370_data + (v290_data * v368_data)).copy_to(ir0 + (7));
              }
              int32_t v376_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v380_data(0.0f);
              v380_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[48_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v381_data(0.0f);
              v381_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v383_data(0.0f);
              v383_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v383_data + (v380_data * v381_data)).copy_to(ir0 + (0));
              }
              int32_t v387_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v392_data(0.0f);
              v392_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v394_data(0.0f);
              v394_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v394_data + (v380_data * v392_data)).copy_to(ir0 + (1));
              }
              int32_t v398_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v403_data(0.0f);
              v403_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v405_data(0.0f);
              v405_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v405_data + (v380_data * v403_data)).copy_to(ir0 + (2));
              }
              int32_t v409_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v414_data(0.0f);
              v414_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v416_data(0.0f);
              v416_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v416_data + (v380_data * v414_data)).copy_to(ir0 + (3));
              }
              int32_t v420_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v425_data(0.0f);
              v425_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v427_data(0.0f);
              v427_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v427_data + (v380_data * v425_data)).copy_to(ir0 + (4));
              }
              int32_t v431_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v436_data(0.0f);
              v436_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v438_data(0.0f);
              v438_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v438_data + (v380_data * v436_data)).copy_to(ir0 + (5));
              }
              int32_t v442_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v447_data(0.0f);
              v447_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v449_data(0.0f);
              v449_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v449_data + (v380_data * v447_data)).copy_to(ir0 + (6));
              }
              int32_t v453_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v458_data(0.0f);
              v458_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[88]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v460_data(0.0f);
              v460_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v460_data + (v380_data * v458_data)).copy_to(ir0 + (7));
              }
              int32_t v466_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v470_data(0.0f);
              v470_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[60_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v471_data(0.0f);
              v471_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v473_data(0.0f);
              v473_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v473_data + (v470_data * v471_data)).copy_to(ir0 + (0));
              }
              int32_t v477_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v482_data(0.0f);
              v482_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v484_data(0.0f);
              v484_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v484_data + (v470_data * v482_data)).copy_to(ir0 + (1));
              }
              int32_t v488_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v493_data(0.0f);
              v493_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v495_data(0.0f);
              v495_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v495_data + (v470_data * v493_data)).copy_to(ir0 + (2));
              }
              int32_t v499_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v504_data(0.0f);
              v504_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v506_data(0.0f);
              v506_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v506_data + (v470_data * v504_data)).copy_to(ir0 + (3));
              }
              int32_t v510_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v515_data(0.0f);
              v515_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v517_data(0.0f);
              v517_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v517_data + (v470_data * v515_data)).copy_to(ir0 + (4));
              }
              int32_t v521_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v526_data(0.0f);
              v526_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v528_data(0.0f);
              v528_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v528_data + (v470_data * v526_data)).copy_to(ir0 + (5));
              }
              int32_t v532_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v537_data(0.0f);
              v537_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v539_data(0.0f);
              v539_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v539_data + (v470_data * v537_data)).copy_to(ir0 + (6));
              }
              int32_t v543_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v548_data(0.0f);
              v548_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[89]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v550_data(0.0f);
              v550_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v550_data + (v470_data * v548_data)).copy_to(ir0 + (7));
              }
              int32_t v556_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v560_data(0.0f);
              v560_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[72_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v561_data(0.0f);
              v561_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v563_data(0.0f);
              v563_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v563_data + (v560_data * v561_data)).copy_to(ir0 + (0));
              }
              int32_t v567_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v572_data(0.0f);
              v572_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v574_data(0.0f);
              v574_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v574_data + (v560_data * v572_data)).copy_to(ir0 + (1));
              }
              int32_t v578_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v583_data(0.0f);
              v583_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v585_data(0.0f);
              v585_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v585_data + (v560_data * v583_data)).copy_to(ir0 + (2));
              }
              int32_t v589_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v594_data(0.0f);
              v594_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v596_data(0.0f);
              v596_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v596_data + (v560_data * v594_data)).copy_to(ir0 + (3));
              }
              int32_t v600_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v605_data(0.0f);
              v605_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v607_data(0.0f);
              v607_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v607_data + (v560_data * v605_data)).copy_to(ir0 + (4));
              }
              int32_t v611_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v616_data(0.0f);
              v616_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v618_data(0.0f);
              v618_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v618_data + (v560_data * v616_data)).copy_to(ir0 + (5));
              }
              int32_t v622_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v627_data(0.0f);
              v627_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v629_data(0.0f);
              v629_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v629_data + (v560_data * v627_data)).copy_to(ir0 + (6));
              }
              int32_t v633_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v638_data(0.0f);
              v638_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[90]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v640_data(0.0f);
              v640_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v640_data + (v560_data * v638_data)).copy_to(ir0 + (7));
              }
              int32_t v646_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v650_data(0.0f);
              v650_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[84_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v651_data(0.0f);
              v651_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v653_data(0.0f);
              v653_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v653_data + (v650_data * v651_data)).copy_to(ir0 + (0));
              }
              int32_t v657_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v662_data(0.0f);
              v662_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v664_data(0.0f);
              v664_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v664_data + (v650_data * v662_data)).copy_to(ir0 + (1));
              }
              int32_t v668_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v673_data(0.0f);
              v673_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v675_data(0.0f);
              v675_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v675_data + (v650_data * v673_data)).copy_to(ir0 + (2));
              }
              int32_t v679_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v684_data(0.0f);
              v684_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v686_data(0.0f);
              v686_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v686_data + (v650_data * v684_data)).copy_to(ir0 + (3));
              }
              int32_t v690_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v695_data(0.0f);
              v695_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v697_data(0.0f);
              v697_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v697_data + (v650_data * v695_data)).copy_to(ir0 + (4));
              }
              int32_t v701_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v706_data(0.0f);
              v706_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[67]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v708_data(0.0f);
              v708_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v708_data + (v650_data * v706_data)).copy_to(ir0 + (5));
              }
              int32_t v712_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v717_data(0.0f);
              v717_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v719_data(0.0f);
              v719_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v719_data + (v650_data * v717_data)).copy_to(ir0 + (6));
              }
              int32_t v723_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v728_data(0.0f);
              v728_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[91]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v730_data(0.0f);
              v730_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v730_data + (v650_data * v728_data)).copy_to(ir0 + (7));
              }
              int32_t v736_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v740_data(0.0f);
              v740_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[96_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v741_data(0.0f);
              v741_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v743_data(0.0f);
              v743_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v743_data + (v740_data * v741_data)).copy_to(ir0 + (0));
              }
              int32_t v747_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v752_data(0.0f);
              v752_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v754_data(0.0f);
              v754_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v754_data + (v740_data * v752_data)).copy_to(ir0 + (1));
              }
              int32_t v758_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v763_data(0.0f);
              v763_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v765_data(0.0f);
              v765_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v765_data + (v740_data * v763_data)).copy_to(ir0 + (2));
              }
              int32_t v769_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v774_data(0.0f);
              v774_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v776_data(0.0f);
              v776_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v776_data + (v740_data * v774_data)).copy_to(ir0 + (3));
              }
              int32_t v780_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v785_data(0.0f);
              v785_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v787_data(0.0f);
              v787_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v787_data + (v740_data * v785_data)).copy_to(ir0 + (4));
              }
              int32_t v791_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v796_data(0.0f);
              v796_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v798_data(0.0f);
              v798_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v798_data + (v740_data * v796_data)).copy_to(ir0 + (5));
              }
              int32_t v802_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v807_data(0.0f);
              v807_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v809_data(0.0f);
              v809_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v809_data + (v740_data * v807_data)).copy_to(ir0 + (6));
              }
              int32_t v813_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v818_data(0.0f);
              v818_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[92]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v820_data(0.0f);
              v820_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v820_data + (v740_data * v818_data)).copy_to(ir0 + (7));
              }
              int32_t v826_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v830_data(0.0f);
              v830_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[108_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v831_data(0.0f);
              v831_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v833_data(0.0f);
              v833_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v833_data + (v830_data * v831_data)).copy_to(ir0 + (0));
              }
              int32_t v837_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v842_data(0.0f);
              v842_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v844_data(0.0f);
              v844_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v844_data + (v830_data * v842_data)).copy_to(ir0 + (1));
              }
              int32_t v848_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v853_data(0.0f);
              v853_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v855_data(0.0f);
              v855_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v855_data + (v830_data * v853_data)).copy_to(ir0 + (2));
              }
              int32_t v859_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v864_data(0.0f);
              v864_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v866_data(0.0f);
              v866_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v866_data + (v830_data * v864_data)).copy_to(ir0 + (3));
              }
              int32_t v870_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v875_data(0.0f);
              v875_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v877_data(0.0f);
              v877_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v877_data + (v830_data * v875_data)).copy_to(ir0 + (4));
              }
              int32_t v881_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v886_data(0.0f);
              v886_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v888_data(0.0f);
              v888_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v888_data + (v830_data * v886_data)).copy_to(ir0 + (5));
              }
              int32_t v892_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v897_data(0.0f);
              v897_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[81]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v899_data(0.0f);
              v899_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v899_data + (v830_data * v897_data)).copy_to(ir0 + (6));
              }
              int32_t v903_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v908_data(0.0f);
              v908_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[93]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v910_data(0.0f);
              v910_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v910_data + (v830_data * v908_data)).copy_to(ir0 + (7));
              }
              int32_t v916_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v920_data(0.0f);
              v920_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[120_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v921_data(0.0f);
              v921_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v923_data(0.0f);
              v923_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v923_data + (v920_data * v921_data)).copy_to(ir0 + (0));
              }
              int32_t v927_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v932_data(0.0f);
              v932_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v934_data(0.0f);
              v934_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v934_data + (v920_data * v932_data)).copy_to(ir0 + (1));
              }
              int32_t v938_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v943_data(0.0f);
              v943_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v945_data(0.0f);
              v945_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v945_data + (v920_data * v943_data)).copy_to(ir0 + (2));
              }
              int32_t v949_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v954_data(0.0f);
              v954_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v956_data(0.0f);
              v956_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v956_data + (v920_data * v954_data)).copy_to(ir0 + (3));
              }
              int32_t v960_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v965_data(0.0f);
              v965_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v967_data(0.0f);
              v967_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v967_data + (v920_data * v965_data)).copy_to(ir0 + (4));
              }
              int32_t v971_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v976_data(0.0f);
              v976_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v978_data(0.0f);
              v978_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v978_data + (v920_data * v976_data)).copy_to(ir0 + (5));
              }
              int32_t v982_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v987_data(0.0f);
              v987_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[82]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v989_data(0.0f);
              v989_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v989_data + (v920_data * v987_data)).copy_to(ir0 + (6));
              }
              int32_t v993_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v998_data(0.0f);
              v998_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[94]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1000_data(0.0f);
              v1000_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v1000_data + (v920_data * v998_data)).copy_to(ir0 + (7));
              }
              int32_t v1006_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1010_data(0.0f);
              v1010_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[132_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1011_data(0.0f);
              v1011_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1013_data(0.0f);
              v1013_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v13_g);
              if (v13_g) {
                (v1013_data + (v1010_data * v1011_data)).copy_to(ir0 + (0));
              }
              int32_t v1017_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1022_data(0.0f);
              v1022_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1024_data(0.0f);
              v1024_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v13_g);
              if (v13_g) {
                (v1024_data + (v1010_data * v1022_data)).copy_to(ir0 + (1));
              }
              int32_t v1028_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1033_data(0.0f);
              v1033_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1035_data(0.0f);
              v1035_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v13_g);
              if (v13_g) {
                (v1035_data + (v1010_data * v1033_data)).copy_to(ir0 + (2));
              }
              int32_t v1039_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1044_data(0.0f);
              v1044_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1046_data(0.0f);
              v1046_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v13_g);
              if (v13_g) {
                (v1046_data + (v1010_data * v1044_data)).copy_to(ir0 + (3));
              }
              int32_t v1050_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1055_data(0.0f);
              v1055_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1057_data(0.0f);
              v1057_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v13_g);
              if (v13_g) {
                (v1057_data + (v1010_data * v1055_data)).copy_to(ir0 + (4));
              }
              int32_t v1061_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1066_data(0.0f);
              v1066_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1068_data(0.0f);
              v1068_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v13_g);
              if (v13_g) {
                (v1068_data + (v1010_data * v1066_data)).copy_to(ir0 + (5));
              }
              int32_t v1072_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1077_data(0.0f);
              v1077_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[83]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1079_data(0.0f);
              v1079_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v13_g);
              if (v13_g) {
                (v1079_data + (v1010_data * v1077_data)).copy_to(ir0 + (6));
              }
              int32_t v1083_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1088_data(0.0f);
              v1088_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[95]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1090_data(0.0f);
              v1090_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v13_g);
              if (v13_g) {
                (v1090_data + (v1010_data * v1088_data)).copy_to(ir0 + (7));
              }
              #pragma unroll
              for (int32_t v1094_n1 = 0; v1094_n1 < 8; ++v1094_n1) {
                int32_t v1095_a = 0 + v1094_n1;
                tensorforge::intel_esimd::simd<float, 16> v1097_data(0.0f);
                v1097_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[v1094_n1]), v13_g);
                if (v13_g) {
                  v1097_data.copy_to(r0 + (v1094_n1));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1101_i1 = 0; v1101_i1 < 8; ++v1101_i1) {
                int32_t v1102_a = 0 + v1101_i1;
                tensorforge::intel_esimd::simd<float, 16> v1104_data(0.0f);
                v1104_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v1101_i1]), v13_g);
                if (v13_g) {
                  v1104_data.copy_to(glb_m0 + ((v1101_i1 * 12)));
                }
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m4[0, 1])
              *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m4[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s1[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m4[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r1[8]{};
              // r1 = +(glb_m3 * s1) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir1[8]{};
              int32_t v1116_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1120_data(0.0f);
              v1120_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[0_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1121_data(0.0f);
              v1121_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[0]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1123_data(0.0f);
              v1123_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1123_data + (v1120_data * v1121_data)).copy_to(ir1 + (0));
              }
              int32_t v1127_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1132_data(0.0f);
              v1132_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[12]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1134_data(0.0f);
              v1134_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1134_data + (v1120_data * v1132_data)).copy_to(ir1 + (1));
              }
              int32_t v1138_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1143_data(0.0f);
              v1143_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[24]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1145_data(0.0f);
              v1145_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1145_data + (v1120_data * v1143_data)).copy_to(ir1 + (2));
              }
              int32_t v1149_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1154_data(0.0f);
              v1154_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[36]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1156_data(0.0f);
              v1156_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1156_data + (v1120_data * v1154_data)).copy_to(ir1 + (3));
              }
              int32_t v1160_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1165_data(0.0f);
              v1165_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[48]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1167_data(0.0f);
              v1167_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1167_data + (v1120_data * v1165_data)).copy_to(ir1 + (4));
              }
              int32_t v1171_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1176_data(0.0f);
              v1176_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[60]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1178_data(0.0f);
              v1178_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1178_data + (v1120_data * v1176_data)).copy_to(ir1 + (5));
              }
              int32_t v1182_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1187_data(0.0f);
              v1187_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[72]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1189_data(0.0f);
              v1189_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1189_data + (v1120_data * v1187_data)).copy_to(ir1 + (6));
              }
              int32_t v1193_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v1198_data(0.0f);
              v1198_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[84]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1200_data(0.0f);
              v1200_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1200_data + (v1120_data * v1198_data)).copy_to(ir1 + (7));
              }
              int32_t v1206_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1210_data(0.0f);
              v1210_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[12_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1211_data(0.0f);
              v1211_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[1]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1213_data(0.0f);
              v1213_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1213_data + (v1210_data * v1211_data)).copy_to(ir1 + (0));
              }
              int32_t v1217_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1222_data(0.0f);
              v1222_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[13]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1224_data(0.0f);
              v1224_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1224_data + (v1210_data * v1222_data)).copy_to(ir1 + (1));
              }
              int32_t v1228_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1233_data(0.0f);
              v1233_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[25]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1235_data(0.0f);
              v1235_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1235_data + (v1210_data * v1233_data)).copy_to(ir1 + (2));
              }
              int32_t v1239_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1244_data(0.0f);
              v1244_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[37]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1246_data(0.0f);
              v1246_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1246_data + (v1210_data * v1244_data)).copy_to(ir1 + (3));
              }
              int32_t v1250_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1255_data(0.0f);
              v1255_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[49]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1257_data(0.0f);
              v1257_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1257_data + (v1210_data * v1255_data)).copy_to(ir1 + (4));
              }
              int32_t v1261_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1266_data(0.0f);
              v1266_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[61]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1268_data(0.0f);
              v1268_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1268_data + (v1210_data * v1266_data)).copy_to(ir1 + (5));
              }
              int32_t v1272_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1277_data(0.0f);
              v1277_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[73]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1279_data(0.0f);
              v1279_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1279_data + (v1210_data * v1277_data)).copy_to(ir1 + (6));
              }
              int32_t v1283_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v1288_data(0.0f);
              v1288_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[85]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1290_data(0.0f);
              v1290_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1290_data + (v1210_data * v1288_data)).copy_to(ir1 + (7));
              }
              int32_t v1296_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1300_data(0.0f);
              v1300_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[24_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1301_data(0.0f);
              v1301_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[2]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1303_data(0.0f);
              v1303_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1303_data + (v1300_data * v1301_data)).copy_to(ir1 + (0));
              }
              int32_t v1307_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1312_data(0.0f);
              v1312_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[14]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1314_data(0.0f);
              v1314_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1314_data + (v1300_data * v1312_data)).copy_to(ir1 + (1));
              }
              int32_t v1318_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1323_data(0.0f);
              v1323_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[26]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1325_data(0.0f);
              v1325_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1325_data + (v1300_data * v1323_data)).copy_to(ir1 + (2));
              }
              int32_t v1329_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1334_data(0.0f);
              v1334_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[38]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1336_data(0.0f);
              v1336_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1336_data + (v1300_data * v1334_data)).copy_to(ir1 + (3));
              }
              int32_t v1340_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1345_data(0.0f);
              v1345_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[50]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1347_data(0.0f);
              v1347_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1347_data + (v1300_data * v1345_data)).copy_to(ir1 + (4));
              }
              int32_t v1351_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1356_data(0.0f);
              v1356_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[62]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1358_data(0.0f);
              v1358_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1358_data + (v1300_data * v1356_data)).copy_to(ir1 + (5));
              }
              int32_t v1362_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1367_data(0.0f);
              v1367_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[74]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1369_data(0.0f);
              v1369_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1369_data + (v1300_data * v1367_data)).copy_to(ir1 + (6));
              }
              int32_t v1373_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v1378_data(0.0f);
              v1378_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[86]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1380_data(0.0f);
              v1380_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1380_data + (v1300_data * v1378_data)).copy_to(ir1 + (7));
              }
              int32_t v1386_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1390_data(0.0f);
              v1390_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[36_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1391_data(0.0f);
              v1391_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[3]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1393_data(0.0f);
              v1393_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1393_data + (v1390_data * v1391_data)).copy_to(ir1 + (0));
              }
              int32_t v1397_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1402_data(0.0f);
              v1402_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[15]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1404_data(0.0f);
              v1404_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1404_data + (v1390_data * v1402_data)).copy_to(ir1 + (1));
              }
              int32_t v1408_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1413_data(0.0f);
              v1413_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[27]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1415_data(0.0f);
              v1415_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1415_data + (v1390_data * v1413_data)).copy_to(ir1 + (2));
              }
              int32_t v1419_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1424_data(0.0f);
              v1424_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[39]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1426_data(0.0f);
              v1426_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1426_data + (v1390_data * v1424_data)).copy_to(ir1 + (3));
              }
              int32_t v1430_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1435_data(0.0f);
              v1435_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[51]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1437_data(0.0f);
              v1437_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1437_data + (v1390_data * v1435_data)).copy_to(ir1 + (4));
              }
              int32_t v1441_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1446_data(0.0f);
              v1446_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[63]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1448_data(0.0f);
              v1448_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1448_data + (v1390_data * v1446_data)).copy_to(ir1 + (5));
              }
              int32_t v1452_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1457_data(0.0f);
              v1457_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[75]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1459_data(0.0f);
              v1459_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1459_data + (v1390_data * v1457_data)).copy_to(ir1 + (6));
              }
              int32_t v1463_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v1468_data(0.0f);
              v1468_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[87]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1470_data(0.0f);
              v1470_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1470_data + (v1390_data * v1468_data)).copy_to(ir1 + (7));
              }
              int32_t v1476_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1480_data(0.0f);
              v1480_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[48_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1481_data(0.0f);
              v1481_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[4]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1483_data(0.0f);
              v1483_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1483_data + (v1480_data * v1481_data)).copy_to(ir1 + (0));
              }
              int32_t v1487_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1492_data(0.0f);
              v1492_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[16]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1494_data(0.0f);
              v1494_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1494_data + (v1480_data * v1492_data)).copy_to(ir1 + (1));
              }
              int32_t v1498_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1503_data(0.0f);
              v1503_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[28]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1505_data(0.0f);
              v1505_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1505_data + (v1480_data * v1503_data)).copy_to(ir1 + (2));
              }
              int32_t v1509_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1514_data(0.0f);
              v1514_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[40]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1516_data(0.0f);
              v1516_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1516_data + (v1480_data * v1514_data)).copy_to(ir1 + (3));
              }
              int32_t v1520_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1525_data(0.0f);
              v1525_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[52]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1527_data(0.0f);
              v1527_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1527_data + (v1480_data * v1525_data)).copy_to(ir1 + (4));
              }
              int32_t v1531_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1536_data(0.0f);
              v1536_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[64]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1538_data(0.0f);
              v1538_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1538_data + (v1480_data * v1536_data)).copy_to(ir1 + (5));
              }
              int32_t v1542_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1547_data(0.0f);
              v1547_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[76]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1549_data(0.0f);
              v1549_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1549_data + (v1480_data * v1547_data)).copy_to(ir1 + (6));
              }
              int32_t v1553_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v1558_data(0.0f);
              v1558_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[88]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1560_data(0.0f);
              v1560_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1560_data + (v1480_data * v1558_data)).copy_to(ir1 + (7));
              }
              int32_t v1566_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1570_data(0.0f);
              v1570_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[60_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1571_data(0.0f);
              v1571_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[5]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1573_data(0.0f);
              v1573_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1573_data + (v1570_data * v1571_data)).copy_to(ir1 + (0));
              }
              int32_t v1577_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1582_data(0.0f);
              v1582_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[17]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1584_data(0.0f);
              v1584_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1584_data + (v1570_data * v1582_data)).copy_to(ir1 + (1));
              }
              int32_t v1588_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1593_data(0.0f);
              v1593_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[29]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1595_data(0.0f);
              v1595_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1595_data + (v1570_data * v1593_data)).copy_to(ir1 + (2));
              }
              int32_t v1599_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1604_data(0.0f);
              v1604_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[41]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1606_data(0.0f);
              v1606_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1606_data + (v1570_data * v1604_data)).copy_to(ir1 + (3));
              }
              int32_t v1610_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1615_data(0.0f);
              v1615_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[53]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1617_data(0.0f);
              v1617_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1617_data + (v1570_data * v1615_data)).copy_to(ir1 + (4));
              }
              int32_t v1621_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1626_data(0.0f);
              v1626_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[65]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1628_data(0.0f);
              v1628_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1628_data + (v1570_data * v1626_data)).copy_to(ir1 + (5));
              }
              int32_t v1632_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1637_data(0.0f);
              v1637_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[77]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1639_data(0.0f);
              v1639_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1639_data + (v1570_data * v1637_data)).copy_to(ir1 + (6));
              }
              int32_t v1643_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v1648_data(0.0f);
              v1648_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[89]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1650_data(0.0f);
              v1650_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1650_data + (v1570_data * v1648_data)).copy_to(ir1 + (7));
              }
              int32_t v1656_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1660_data(0.0f);
              v1660_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[72_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1661_data(0.0f);
              v1661_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[6]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1663_data(0.0f);
              v1663_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1663_data + (v1660_data * v1661_data)).copy_to(ir1 + (0));
              }
              int32_t v1667_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1672_data(0.0f);
              v1672_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[18]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1674_data(0.0f);
              v1674_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1674_data + (v1660_data * v1672_data)).copy_to(ir1 + (1));
              }
              int32_t v1678_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1683_data(0.0f);
              v1683_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[30]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1685_data(0.0f);
              v1685_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1685_data + (v1660_data * v1683_data)).copy_to(ir1 + (2));
              }
              int32_t v1689_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1694_data(0.0f);
              v1694_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[42]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1696_data(0.0f);
              v1696_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1696_data + (v1660_data * v1694_data)).copy_to(ir1 + (3));
              }
              int32_t v1700_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1705_data(0.0f);
              v1705_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[54]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1707_data(0.0f);
              v1707_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1707_data + (v1660_data * v1705_data)).copy_to(ir1 + (4));
              }
              int32_t v1711_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1716_data(0.0f);
              v1716_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[66]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1718_data(0.0f);
              v1718_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1718_data + (v1660_data * v1716_data)).copy_to(ir1 + (5));
              }
              int32_t v1722_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1727_data(0.0f);
              v1727_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[78]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1729_data(0.0f);
              v1729_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1729_data + (v1660_data * v1727_data)).copy_to(ir1 + (6));
              }
              int32_t v1733_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v1738_data(0.0f);
              v1738_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[90]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1740_data(0.0f);
              v1740_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1740_data + (v1660_data * v1738_data)).copy_to(ir1 + (7));
              }
              int32_t v1746_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1750_data(0.0f);
              v1750_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[84_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1751_data(0.0f);
              v1751_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[7]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1753_data(0.0f);
              v1753_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1753_data + (v1750_data * v1751_data)).copy_to(ir1 + (0));
              }
              int32_t v1757_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1762_data(0.0f);
              v1762_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[19]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1764_data(0.0f);
              v1764_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1764_data + (v1750_data * v1762_data)).copy_to(ir1 + (1));
              }
              int32_t v1768_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1773_data(0.0f);
              v1773_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[31]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1775_data(0.0f);
              v1775_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1775_data + (v1750_data * v1773_data)).copy_to(ir1 + (2));
              }
              int32_t v1779_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1784_data(0.0f);
              v1784_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[43]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1786_data(0.0f);
              v1786_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1786_data + (v1750_data * v1784_data)).copy_to(ir1 + (3));
              }
              int32_t v1790_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1795_data(0.0f);
              v1795_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[55]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1797_data(0.0f);
              v1797_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1797_data + (v1750_data * v1795_data)).copy_to(ir1 + (4));
              }
              int32_t v1801_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1806_data(0.0f);
              v1806_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[67]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1808_data(0.0f);
              v1808_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1808_data + (v1750_data * v1806_data)).copy_to(ir1 + (5));
              }
              int32_t v1812_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1817_data(0.0f);
              v1817_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[79]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1819_data(0.0f);
              v1819_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1819_data + (v1750_data * v1817_data)).copy_to(ir1 + (6));
              }
              int32_t v1823_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v1828_data(0.0f);
              v1828_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[91]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1830_data(0.0f);
              v1830_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1830_data + (v1750_data * v1828_data)).copy_to(ir1 + (7));
              }
              int32_t v1836_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1840_data(0.0f);
              v1840_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[96_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1841_data(0.0f);
              v1841_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[8]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1843_data(0.0f);
              v1843_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1843_data + (v1840_data * v1841_data)).copy_to(ir1 + (0));
              }
              int32_t v1847_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1852_data(0.0f);
              v1852_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[20]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1854_data(0.0f);
              v1854_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1854_data + (v1840_data * v1852_data)).copy_to(ir1 + (1));
              }
              int32_t v1858_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1863_data(0.0f);
              v1863_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1865_data(0.0f);
              v1865_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1865_data + (v1840_data * v1863_data)).copy_to(ir1 + (2));
              }
              int32_t v1869_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1874_data(0.0f);
              v1874_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[44]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1876_data(0.0f);
              v1876_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1876_data + (v1840_data * v1874_data)).copy_to(ir1 + (3));
              }
              int32_t v1880_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1885_data(0.0f);
              v1885_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[56]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1887_data(0.0f);
              v1887_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1887_data + (v1840_data * v1885_data)).copy_to(ir1 + (4));
              }
              int32_t v1891_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1896_data(0.0f);
              v1896_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[68]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1898_data(0.0f);
              v1898_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1898_data + (v1840_data * v1896_data)).copy_to(ir1 + (5));
              }
              int32_t v1902_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1907_data(0.0f);
              v1907_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[80]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1909_data(0.0f);
              v1909_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1909_data + (v1840_data * v1907_data)).copy_to(ir1 + (6));
              }
              int32_t v1913_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1918_data(0.0f);
              v1918_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[92]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1920_data(0.0f);
              v1920_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v1920_data + (v1840_data * v1918_data)).copy_to(ir1 + (7));
              }
              int32_t v1926_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1930_data(0.0f);
              v1930_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[108_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1931_data(0.0f);
              v1931_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[9]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1933_data(0.0f);
              v1933_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v1933_data + (v1930_data * v1931_data)).copy_to(ir1 + (0));
              }
              int32_t v1937_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1942_data(0.0f);
              v1942_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[21]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1944_data(0.0f);
              v1944_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v1944_data + (v1930_data * v1942_data)).copy_to(ir1 + (1));
              }
              int32_t v1948_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1953_data(0.0f);
              v1953_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[33]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1955_data(0.0f);
              v1955_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v1955_data + (v1930_data * v1953_data)).copy_to(ir1 + (2));
              }
              int32_t v1959_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1964_data(0.0f);
              v1964_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[45]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1966_data(0.0f);
              v1966_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v1966_data + (v1930_data * v1964_data)).copy_to(ir1 + (3));
              }
              int32_t v1970_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1975_data(0.0f);
              v1975_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[57]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1977_data(0.0f);
              v1977_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v1977_data + (v1930_data * v1975_data)).copy_to(ir1 + (4));
              }
              int32_t v1981_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1986_data(0.0f);
              v1986_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[69]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1988_data(0.0f);
              v1988_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v1988_data + (v1930_data * v1986_data)).copy_to(ir1 + (5));
              }
              int32_t v1992_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1997_data(0.0f);
              v1997_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[81]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v1999_data(0.0f);
              v1999_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v1999_data + (v1930_data * v1997_data)).copy_to(ir1 + (6));
              }
              int32_t v2003_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v2008_data(0.0f);
              v2008_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[93]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2010_data(0.0f);
              v2010_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v2010_data + (v1930_data * v2008_data)).copy_to(ir1 + (7));
              }
              int32_t v2016_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2020_data(0.0f);
              v2020_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[120_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2021_data(0.0f);
              v2021_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[10]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2023_data(0.0f);
              v2023_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v2023_data + (v2020_data * v2021_data)).copy_to(ir1 + (0));
              }
              int32_t v2027_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2032_data(0.0f);
              v2032_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[22]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2034_data(0.0f);
              v2034_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v2034_data + (v2020_data * v2032_data)).copy_to(ir1 + (1));
              }
              int32_t v2038_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2043_data(0.0f);
              v2043_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[34]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2045_data(0.0f);
              v2045_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v2045_data + (v2020_data * v2043_data)).copy_to(ir1 + (2));
              }
              int32_t v2049_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2054_data(0.0f);
              v2054_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[46]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2056_data(0.0f);
              v2056_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v2056_data + (v2020_data * v2054_data)).copy_to(ir1 + (3));
              }
              int32_t v2060_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2065_data(0.0f);
              v2065_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[58]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2067_data(0.0f);
              v2067_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v2067_data + (v2020_data * v2065_data)).copy_to(ir1 + (4));
              }
              int32_t v2071_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2076_data(0.0f);
              v2076_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[70]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2078_data(0.0f);
              v2078_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v2078_data + (v2020_data * v2076_data)).copy_to(ir1 + (5));
              }
              int32_t v2082_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2087_data(0.0f);
              v2087_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[82]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2089_data(0.0f);
              v2089_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v2089_data + (v2020_data * v2087_data)).copy_to(ir1 + (6));
              }
              int32_t v2093_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v2098_data(0.0f);
              v2098_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[94]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2100_data(0.0f);
              v2100_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v2100_data + (v2020_data * v2098_data)).copy_to(ir1 + (7));
              }
              int32_t v2106_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2110_data(0.0f);
              v2110_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[132_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2111_data(0.0f);
              v2111_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[11]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2113_data(0.0f);
              v2113_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v13_g);
              if (v13_g) {
                (v2113_data + (v2110_data * v2111_data)).copy_to(ir1 + (0));
              }
              int32_t v2117_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2122_data(0.0f);
              v2122_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[23]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2124_data(0.0f);
              v2124_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v13_g);
              if (v13_g) {
                (v2124_data + (v2110_data * v2122_data)).copy_to(ir1 + (1));
              }
              int32_t v2128_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2133_data(0.0f);
              v2133_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[35]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2135_data(0.0f);
              v2135_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v13_g);
              if (v13_g) {
                (v2135_data + (v2110_data * v2133_data)).copy_to(ir1 + (2));
              }
              int32_t v2139_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2144_data(0.0f);
              v2144_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[47]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2146_data(0.0f);
              v2146_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v13_g);
              if (v13_g) {
                (v2146_data + (v2110_data * v2144_data)).copy_to(ir1 + (3));
              }
              int32_t v2150_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2155_data(0.0f);
              v2155_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[59]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2157_data(0.0f);
              v2157_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v13_g);
              if (v13_g) {
                (v2157_data + (v2110_data * v2155_data)).copy_to(ir1 + (4));
              }
              int32_t v2161_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2166_data(0.0f);
              v2166_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[71]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2168_data(0.0f);
              v2168_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v13_g);
              if (v13_g) {
                (v2168_data + (v2110_data * v2166_data)).copy_to(ir1 + (5));
              }
              int32_t v2172_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2177_data(0.0f);
              v2177_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[83]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2179_data(0.0f);
              v2179_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[6]), v13_g);
              if (v13_g) {
                (v2179_data + (v2110_data * v2177_data)).copy_to(ir1 + (6));
              }
              int32_t v2183_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v2188_data(0.0f);
              v2188_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[95]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2190_data(0.0f);
              v2190_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[7]), v13_g);
              if (v13_g) {
                (v2190_data + (v2110_data * v2188_data)).copy_to(ir1 + (7));
              }
              #pragma unroll
              for (int32_t v2194_n1 = 0; v2194_n1 < 8; ++v2194_n1) {
                int32_t v2195_a = 0 + v2194_n1;
                tensorforge::intel_esimd::simd<float, 16> v2197_data(0.0f);
                v2197_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[v2194_n1]), v13_g);
                int32_t v2200_a = v2194_n1 * 12;
                int32_t v2201_a = 0_i32 + v2200_a;
                tensorforge::intel_esimd::simd<float, 16> v2206_data(0.0f);
                v2206_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[v2200_a]), v13_g);
                if (v13_g) {
                  (v2206_data + v2197_data).copy_to(r1 + (v2194_n1));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v2211_i1 = 0; v2211_i1 < 8; ++v2211_i1) {
                int32_t v2212_a = 0 + v2211_i1;
                tensorforge::intel_esimd::simd<float, 16> v2214_data(0.0f);
                v2214_data.merge(tensorforge::intel_esimd::simd<float, 16>(r1[v2211_i1]), v13_g);
                if (v13_g) {
                  v2214_data.copy_to(glb_m0 + ((v2211_i1 * 12)));
                }
              }
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m6[0, 1])
              *(sycl::vec<float, 4>*)&s2[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m6[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s2[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m6[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              float r2[8]{};
              // r2 = +(glb_m5 * s2) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              int32_t v2226_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2230_data(0.0f);
              v2230_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[0_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2231_data(0.0f);
              v2231_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[0]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2233_data(0.0f);
              v2233_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2233_data + (v2230_data * v2231_data)).copy_to(ir2 + (0));
              }
              int32_t v2237_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2242_data(0.0f);
              v2242_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[12]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2244_data(0.0f);
              v2244_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2244_data + (v2230_data * v2242_data)).copy_to(ir2 + (1));
              }
              int32_t v2248_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2253_data(0.0f);
              v2253_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[24]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2255_data(0.0f);
              v2255_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2255_data + (v2230_data * v2253_data)).copy_to(ir2 + (2));
              }
              int32_t v2259_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2264_data(0.0f);
              v2264_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[36]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2266_data(0.0f);
              v2266_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2266_data + (v2230_data * v2264_data)).copy_to(ir2 + (3));
              }
              int32_t v2270_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2275_data(0.0f);
              v2275_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[48]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2277_data(0.0f);
              v2277_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2277_data + (v2230_data * v2275_data)).copy_to(ir2 + (4));
              }
              int32_t v2281_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2286_data(0.0f);
              v2286_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[60]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2288_data(0.0f);
              v2288_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2288_data + (v2230_data * v2286_data)).copy_to(ir2 + (5));
              }
              int32_t v2292_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2297_data(0.0f);
              v2297_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[72]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2299_data(0.0f);
              v2299_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2299_data + (v2230_data * v2297_data)).copy_to(ir2 + (6));
              }
              int32_t v2303_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v2308_data(0.0f);
              v2308_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[84]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2310_data(0.0f);
              v2310_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2310_data + (v2230_data * v2308_data)).copy_to(ir2 + (7));
              }
              int32_t v2316_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2320_data(0.0f);
              v2320_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[12_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2321_data(0.0f);
              v2321_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[1]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2323_data(0.0f);
              v2323_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2323_data + (v2320_data * v2321_data)).copy_to(ir2 + (0));
              }
              int32_t v2327_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2332_data(0.0f);
              v2332_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[13]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2334_data(0.0f);
              v2334_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2334_data + (v2320_data * v2332_data)).copy_to(ir2 + (1));
              }
              int32_t v2338_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2343_data(0.0f);
              v2343_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[25]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2345_data(0.0f);
              v2345_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2345_data + (v2320_data * v2343_data)).copy_to(ir2 + (2));
              }
              int32_t v2349_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2354_data(0.0f);
              v2354_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[37]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2356_data(0.0f);
              v2356_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2356_data + (v2320_data * v2354_data)).copy_to(ir2 + (3));
              }
              int32_t v2360_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2365_data(0.0f);
              v2365_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[49]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2367_data(0.0f);
              v2367_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2367_data + (v2320_data * v2365_data)).copy_to(ir2 + (4));
              }
              int32_t v2371_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2376_data(0.0f);
              v2376_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[61]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2378_data(0.0f);
              v2378_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2378_data + (v2320_data * v2376_data)).copy_to(ir2 + (5));
              }
              int32_t v2382_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2387_data(0.0f);
              v2387_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[73]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2389_data(0.0f);
              v2389_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2389_data + (v2320_data * v2387_data)).copy_to(ir2 + (6));
              }
              int32_t v2393_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v2398_data(0.0f);
              v2398_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[85]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2400_data(0.0f);
              v2400_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2400_data + (v2320_data * v2398_data)).copy_to(ir2 + (7));
              }
              int32_t v2406_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2410_data(0.0f);
              v2410_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[24_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2411_data(0.0f);
              v2411_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[2]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2413_data(0.0f);
              v2413_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2413_data + (v2410_data * v2411_data)).copy_to(ir2 + (0));
              }
              int32_t v2417_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2422_data(0.0f);
              v2422_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[14]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2424_data(0.0f);
              v2424_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2424_data + (v2410_data * v2422_data)).copy_to(ir2 + (1));
              }
              int32_t v2428_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2433_data(0.0f);
              v2433_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[26]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2435_data(0.0f);
              v2435_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2435_data + (v2410_data * v2433_data)).copy_to(ir2 + (2));
              }
              int32_t v2439_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2444_data(0.0f);
              v2444_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[38]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2446_data(0.0f);
              v2446_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2446_data + (v2410_data * v2444_data)).copy_to(ir2 + (3));
              }
              int32_t v2450_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2455_data(0.0f);
              v2455_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[50]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2457_data(0.0f);
              v2457_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2457_data + (v2410_data * v2455_data)).copy_to(ir2 + (4));
              }
              int32_t v2461_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2466_data(0.0f);
              v2466_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[62]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2468_data(0.0f);
              v2468_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2468_data + (v2410_data * v2466_data)).copy_to(ir2 + (5));
              }
              int32_t v2472_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2477_data(0.0f);
              v2477_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[74]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2479_data(0.0f);
              v2479_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2479_data + (v2410_data * v2477_data)).copy_to(ir2 + (6));
              }
              int32_t v2483_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v2488_data(0.0f);
              v2488_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[86]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2490_data(0.0f);
              v2490_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2490_data + (v2410_data * v2488_data)).copy_to(ir2 + (7));
              }
              int32_t v2496_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2500_data(0.0f);
              v2500_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[36_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2501_data(0.0f);
              v2501_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[3]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2503_data(0.0f);
              v2503_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2503_data + (v2500_data * v2501_data)).copy_to(ir2 + (0));
              }
              int32_t v2507_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2512_data(0.0f);
              v2512_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[15]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2514_data(0.0f);
              v2514_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2514_data + (v2500_data * v2512_data)).copy_to(ir2 + (1));
              }
              int32_t v2518_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2523_data(0.0f);
              v2523_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[27]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2525_data(0.0f);
              v2525_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2525_data + (v2500_data * v2523_data)).copy_to(ir2 + (2));
              }
              int32_t v2529_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2534_data(0.0f);
              v2534_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[39]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2536_data(0.0f);
              v2536_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2536_data + (v2500_data * v2534_data)).copy_to(ir2 + (3));
              }
              int32_t v2540_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2545_data(0.0f);
              v2545_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[51]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2547_data(0.0f);
              v2547_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2547_data + (v2500_data * v2545_data)).copy_to(ir2 + (4));
              }
              int32_t v2551_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2556_data(0.0f);
              v2556_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[63]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2558_data(0.0f);
              v2558_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2558_data + (v2500_data * v2556_data)).copy_to(ir2 + (5));
              }
              int32_t v2562_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2567_data(0.0f);
              v2567_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[75]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2569_data(0.0f);
              v2569_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2569_data + (v2500_data * v2567_data)).copy_to(ir2 + (6));
              }
              int32_t v2573_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v2578_data(0.0f);
              v2578_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[87]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2580_data(0.0f);
              v2580_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2580_data + (v2500_data * v2578_data)).copy_to(ir2 + (7));
              }
              int32_t v2586_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2590_data(0.0f);
              v2590_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[48_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2591_data(0.0f);
              v2591_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[4]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2593_data(0.0f);
              v2593_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2593_data + (v2590_data * v2591_data)).copy_to(ir2 + (0));
              }
              int32_t v2597_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2602_data(0.0f);
              v2602_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[16]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2604_data(0.0f);
              v2604_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2604_data + (v2590_data * v2602_data)).copy_to(ir2 + (1));
              }
              int32_t v2608_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2613_data(0.0f);
              v2613_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[28]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2615_data(0.0f);
              v2615_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2615_data + (v2590_data * v2613_data)).copy_to(ir2 + (2));
              }
              int32_t v2619_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2624_data(0.0f);
              v2624_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[40]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2626_data(0.0f);
              v2626_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2626_data + (v2590_data * v2624_data)).copy_to(ir2 + (3));
              }
              int32_t v2630_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2635_data(0.0f);
              v2635_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[52]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2637_data(0.0f);
              v2637_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2637_data + (v2590_data * v2635_data)).copy_to(ir2 + (4));
              }
              int32_t v2641_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2646_data(0.0f);
              v2646_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[64]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2648_data(0.0f);
              v2648_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2648_data + (v2590_data * v2646_data)).copy_to(ir2 + (5));
              }
              int32_t v2652_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2657_data(0.0f);
              v2657_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[76]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2659_data(0.0f);
              v2659_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2659_data + (v2590_data * v2657_data)).copy_to(ir2 + (6));
              }
              int32_t v2663_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v2668_data(0.0f);
              v2668_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[88]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2670_data(0.0f);
              v2670_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2670_data + (v2590_data * v2668_data)).copy_to(ir2 + (7));
              }
              int32_t v2676_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2680_data(0.0f);
              v2680_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[60_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2681_data(0.0f);
              v2681_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[5]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2683_data(0.0f);
              v2683_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2683_data + (v2680_data * v2681_data)).copy_to(ir2 + (0));
              }
              int32_t v2687_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2692_data(0.0f);
              v2692_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[17]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2694_data(0.0f);
              v2694_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2694_data + (v2680_data * v2692_data)).copy_to(ir2 + (1));
              }
              int32_t v2698_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2703_data(0.0f);
              v2703_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[29]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2705_data(0.0f);
              v2705_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2705_data + (v2680_data * v2703_data)).copy_to(ir2 + (2));
              }
              int32_t v2709_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2714_data(0.0f);
              v2714_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[41]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2716_data(0.0f);
              v2716_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2716_data + (v2680_data * v2714_data)).copy_to(ir2 + (3));
              }
              int32_t v2720_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2725_data(0.0f);
              v2725_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[53]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2727_data(0.0f);
              v2727_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2727_data + (v2680_data * v2725_data)).copy_to(ir2 + (4));
              }
              int32_t v2731_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2736_data(0.0f);
              v2736_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[65]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2738_data(0.0f);
              v2738_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2738_data + (v2680_data * v2736_data)).copy_to(ir2 + (5));
              }
              int32_t v2742_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2747_data(0.0f);
              v2747_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[77]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2749_data(0.0f);
              v2749_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2749_data + (v2680_data * v2747_data)).copy_to(ir2 + (6));
              }
              int32_t v2753_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v2758_data(0.0f);
              v2758_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[89]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2760_data(0.0f);
              v2760_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2760_data + (v2680_data * v2758_data)).copy_to(ir2 + (7));
              }
              int32_t v2766_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2770_data(0.0f);
              v2770_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[72_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2771_data(0.0f);
              v2771_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[6]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2773_data(0.0f);
              v2773_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2773_data + (v2770_data * v2771_data)).copy_to(ir2 + (0));
              }
              int32_t v2777_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2782_data(0.0f);
              v2782_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[18]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2784_data(0.0f);
              v2784_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2784_data + (v2770_data * v2782_data)).copy_to(ir2 + (1));
              }
              int32_t v2788_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2793_data(0.0f);
              v2793_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[30]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2795_data(0.0f);
              v2795_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2795_data + (v2770_data * v2793_data)).copy_to(ir2 + (2));
              }
              int32_t v2799_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2804_data(0.0f);
              v2804_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[42]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2806_data(0.0f);
              v2806_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2806_data + (v2770_data * v2804_data)).copy_to(ir2 + (3));
              }
              int32_t v2810_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2815_data(0.0f);
              v2815_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[54]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2817_data(0.0f);
              v2817_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2817_data + (v2770_data * v2815_data)).copy_to(ir2 + (4));
              }
              int32_t v2821_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2826_data(0.0f);
              v2826_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[66]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2828_data(0.0f);
              v2828_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2828_data + (v2770_data * v2826_data)).copy_to(ir2 + (5));
              }
              int32_t v2832_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2837_data(0.0f);
              v2837_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[78]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2839_data(0.0f);
              v2839_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2839_data + (v2770_data * v2837_data)).copy_to(ir2 + (6));
              }
              int32_t v2843_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v2848_data(0.0f);
              v2848_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[90]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2850_data(0.0f);
              v2850_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2850_data + (v2770_data * v2848_data)).copy_to(ir2 + (7));
              }
              int32_t v2856_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2860_data(0.0f);
              v2860_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[84_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2861_data(0.0f);
              v2861_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[7]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2863_data(0.0f);
              v2863_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2863_data + (v2860_data * v2861_data)).copy_to(ir2 + (0));
              }
              int32_t v2867_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2872_data(0.0f);
              v2872_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[19]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2874_data(0.0f);
              v2874_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2874_data + (v2860_data * v2872_data)).copy_to(ir2 + (1));
              }
              int32_t v2878_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2883_data(0.0f);
              v2883_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[31]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2885_data(0.0f);
              v2885_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2885_data + (v2860_data * v2883_data)).copy_to(ir2 + (2));
              }
              int32_t v2889_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2894_data(0.0f);
              v2894_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[43]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2896_data(0.0f);
              v2896_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2896_data + (v2860_data * v2894_data)).copy_to(ir2 + (3));
              }
              int32_t v2900_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2905_data(0.0f);
              v2905_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[55]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2907_data(0.0f);
              v2907_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2907_data + (v2860_data * v2905_data)).copy_to(ir2 + (4));
              }
              int32_t v2911_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2916_data(0.0f);
              v2916_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[67]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2918_data(0.0f);
              v2918_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v2918_data + (v2860_data * v2916_data)).copy_to(ir2 + (5));
              }
              int32_t v2922_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2927_data(0.0f);
              v2927_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[79]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2929_data(0.0f);
              v2929_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v2929_data + (v2860_data * v2927_data)).copy_to(ir2 + (6));
              }
              int32_t v2933_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v2938_data(0.0f);
              v2938_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[91]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2940_data(0.0f);
              v2940_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v2940_data + (v2860_data * v2938_data)).copy_to(ir2 + (7));
              }
              int32_t v2946_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v2950_data(0.0f);
              v2950_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[96_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2951_data(0.0f);
              v2951_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[8]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2953_data(0.0f);
              v2953_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v2953_data + (v2950_data * v2951_data)).copy_to(ir2 + (0));
              }
              int32_t v2957_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v2962_data(0.0f);
              v2962_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[20]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2964_data(0.0f);
              v2964_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v2964_data + (v2950_data * v2962_data)).copy_to(ir2 + (1));
              }
              int32_t v2968_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v2973_data(0.0f);
              v2973_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2975_data(0.0f);
              v2975_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v2975_data + (v2950_data * v2973_data)).copy_to(ir2 + (2));
              }
              int32_t v2979_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v2984_data(0.0f);
              v2984_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[44]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2986_data(0.0f);
              v2986_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v2986_data + (v2950_data * v2984_data)).copy_to(ir2 + (3));
              }
              int32_t v2990_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v2995_data(0.0f);
              v2995_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[56]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v2997_data(0.0f);
              v2997_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v2997_data + (v2950_data * v2995_data)).copy_to(ir2 + (4));
              }
              int32_t v3001_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v3006_data(0.0f);
              v3006_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[68]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3008_data(0.0f);
              v3008_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v3008_data + (v2950_data * v3006_data)).copy_to(ir2 + (5));
              }
              int32_t v3012_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v3017_data(0.0f);
              v3017_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[80]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3019_data(0.0f);
              v3019_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v3019_data + (v2950_data * v3017_data)).copy_to(ir2 + (6));
              }
              int32_t v3023_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v3028_data(0.0f);
              v3028_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[92]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3030_data(0.0f);
              v3030_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v3030_data + (v2950_data * v3028_data)).copy_to(ir2 + (7));
              }
              int32_t v3036_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3040_data(0.0f);
              v3040_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[108_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3041_data(0.0f);
              v3041_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[9]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3043_data(0.0f);
              v3043_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v3043_data + (v3040_data * v3041_data)).copy_to(ir2 + (0));
              }
              int32_t v3047_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3052_data(0.0f);
              v3052_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[21]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3054_data(0.0f);
              v3054_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v3054_data + (v3040_data * v3052_data)).copy_to(ir2 + (1));
              }
              int32_t v3058_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3063_data(0.0f);
              v3063_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[33]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3065_data(0.0f);
              v3065_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v3065_data + (v3040_data * v3063_data)).copy_to(ir2 + (2));
              }
              int32_t v3069_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3074_data(0.0f);
              v3074_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[45]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3076_data(0.0f);
              v3076_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v3076_data + (v3040_data * v3074_data)).copy_to(ir2 + (3));
              }
              int32_t v3080_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3085_data(0.0f);
              v3085_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[57]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3087_data(0.0f);
              v3087_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v3087_data + (v3040_data * v3085_data)).copy_to(ir2 + (4));
              }
              int32_t v3091_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3096_data(0.0f);
              v3096_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[69]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3098_data(0.0f);
              v3098_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v3098_data + (v3040_data * v3096_data)).copy_to(ir2 + (5));
              }
              int32_t v3102_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3107_data(0.0f);
              v3107_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[81]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3109_data(0.0f);
              v3109_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v3109_data + (v3040_data * v3107_data)).copy_to(ir2 + (6));
              }
              int32_t v3113_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v3118_data(0.0f);
              v3118_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[93]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3120_data(0.0f);
              v3120_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v3120_data + (v3040_data * v3118_data)).copy_to(ir2 + (7));
              }
              int32_t v3126_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3130_data(0.0f);
              v3130_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[120_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3131_data(0.0f);
              v3131_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[10]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3133_data(0.0f);
              v3133_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v3133_data + (v3130_data * v3131_data)).copy_to(ir2 + (0));
              }
              int32_t v3137_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3142_data(0.0f);
              v3142_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[22]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3144_data(0.0f);
              v3144_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v3144_data + (v3130_data * v3142_data)).copy_to(ir2 + (1));
              }
              int32_t v3148_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3153_data(0.0f);
              v3153_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[34]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3155_data(0.0f);
              v3155_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v3155_data + (v3130_data * v3153_data)).copy_to(ir2 + (2));
              }
              int32_t v3159_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3164_data(0.0f);
              v3164_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[46]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3166_data(0.0f);
              v3166_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v3166_data + (v3130_data * v3164_data)).copy_to(ir2 + (3));
              }
              int32_t v3170_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3175_data(0.0f);
              v3175_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[58]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3177_data(0.0f);
              v3177_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v3177_data + (v3130_data * v3175_data)).copy_to(ir2 + (4));
              }
              int32_t v3181_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3186_data(0.0f);
              v3186_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[70]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3188_data(0.0f);
              v3188_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v3188_data + (v3130_data * v3186_data)).copy_to(ir2 + (5));
              }
              int32_t v3192_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3197_data(0.0f);
              v3197_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[82]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3199_data(0.0f);
              v3199_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v3199_data + (v3130_data * v3197_data)).copy_to(ir2 + (6));
              }
              int32_t v3203_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v3208_data(0.0f);
              v3208_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[94]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3210_data(0.0f);
              v3210_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v3210_data + (v3130_data * v3208_data)).copy_to(ir2 + (7));
              }
              int32_t v3216_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3220_data(0.0f);
              v3220_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m5[132_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3221_data(0.0f);
              v3221_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[11]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3223_data(0.0f);
              v3223_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[0]), v13_g);
              if (v13_g) {
                (v3223_data + (v3220_data * v3221_data)).copy_to(ir2 + (0));
              }
              int32_t v3227_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3232_data(0.0f);
              v3232_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[23]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3234_data(0.0f);
              v3234_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[1]), v13_g);
              if (v13_g) {
                (v3234_data + (v3220_data * v3232_data)).copy_to(ir2 + (1));
              }
              int32_t v3238_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3243_data(0.0f);
              v3243_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[35]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3245_data(0.0f);
              v3245_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[2]), v13_g);
              if (v13_g) {
                (v3245_data + (v3220_data * v3243_data)).copy_to(ir2 + (2));
              }
              int32_t v3249_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3254_data(0.0f);
              v3254_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[47]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3256_data(0.0f);
              v3256_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[3]), v13_g);
              if (v13_g) {
                (v3256_data + (v3220_data * v3254_data)).copy_to(ir2 + (3));
              }
              int32_t v3260_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3265_data(0.0f);
              v3265_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[59]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3267_data(0.0f);
              v3267_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[4]), v13_g);
              if (v13_g) {
                (v3267_data + (v3220_data * v3265_data)).copy_to(ir2 + (4));
              }
              int32_t v3271_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3276_data(0.0f);
              v3276_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[71]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3278_data(0.0f);
              v3278_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[5]), v13_g);
              if (v13_g) {
                (v3278_data + (v3220_data * v3276_data)).copy_to(ir2 + (5));
              }
              int32_t v3282_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3287_data(0.0f);
              v3287_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[83]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3289_data(0.0f);
              v3289_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[6]), v13_g);
              if (v13_g) {
                (v3289_data + (v3220_data * v3287_data)).copy_to(ir2 + (6));
              }
              int32_t v3293_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v3298_data(0.0f);
              v3298_data.merge(tensorforge::intel_esimd::simd<float, 16>(s2[95]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3300_data(0.0f);
              v3300_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[7]), v13_g);
              if (v13_g) {
                (v3300_data + (v3220_data * v3298_data)).copy_to(ir2 + (7));
              }
              #pragma unroll
              for (int32_t v3304_n1 = 0; v3304_n1 < 8; ++v3304_n1) {
                int32_t v3305_a = 0 + v3304_n1;
                tensorforge::intel_esimd::simd<float, 16> v3307_data(0.0f);
                v3307_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir2[v3304_n1]), v13_g);
                int32_t v3310_a = v3304_n1 * 12;
                int32_t v3311_a = 0_i32 + v3310_a;
                tensorforge::intel_esimd::simd<float, 16> v3316_data(0.0f);
                v3316_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[v3310_a]), v13_g);
                if (v13_g) {
                  (v3316_data + v3307_data).copy_to(r2 + (v3304_n1));
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v3321_i1 = 0; v3321_i1 < 8; ++v3321_i1) {
                int32_t v3322_a = 0 + v3321_i1;
                tensorforge::intel_esimd::simd<float, 16> v3324_data(0.0f);
                v3324_data.merge(tensorforge::intel_esimd::simd<float, 16>(r2[v3321_i1]), v13_g);
                if (v13_g) {
                  v3324_data.copy_to(glb_m0 + ((v3321_i1 * 12)));
                }
              }
              float* __restrict__ s3 = &localShrMem0[0];
              // s3 = load{g>s}(glb_m8[0, 1])
              *(sycl::vec<float, 4>*)&s3[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m8[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 2>*)&s3[0 + 0 + 2 * item.get_local_id(0) + 64] = *(sycl::vec<float, 2>*)&glb_m8[0 + 0 + 2 * item.get_local_id(0) + 64];
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              float r3[8]{};
              // r3 = +(glb_m7 * s3) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir3[8]{};
              int32_t v3336_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3340_data(0.0f);
              v3340_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[0_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3341_data(0.0f);
              v3341_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[0]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3343_data(0.0f);
              v3343_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3343_data + (v3340_data * v3341_data)).copy_to(ir3 + (0));
              }
              int32_t v3347_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3352_data(0.0f);
              v3352_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[12]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3354_data(0.0f);
              v3354_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3354_data + (v3340_data * v3352_data)).copy_to(ir3 + (1));
              }
              int32_t v3358_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3363_data(0.0f);
              v3363_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[24]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3365_data(0.0f);
              v3365_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3365_data + (v3340_data * v3363_data)).copy_to(ir3 + (2));
              }
              int32_t v3369_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3374_data(0.0f);
              v3374_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[36]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3376_data(0.0f);
              v3376_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v3376_data + (v3340_data * v3374_data)).copy_to(ir3 + (3));
              }
              int32_t v3380_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3385_data(0.0f);
              v3385_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[48]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3387_data(0.0f);
              v3387_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v3387_data + (v3340_data * v3385_data)).copy_to(ir3 + (4));
              }
              int32_t v3391_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3396_data(0.0f);
              v3396_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[60]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3398_data(0.0f);
              v3398_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v3398_data + (v3340_data * v3396_data)).copy_to(ir3 + (5));
              }
              int32_t v3402_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3407_data(0.0f);
              v3407_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[72]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3409_data(0.0f);
              v3409_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v3409_data + (v3340_data * v3407_data)).copy_to(ir3 + (6));
              }
              int32_t v3413_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v3418_data(0.0f);
              v3418_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[84]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3420_data(0.0f);
              v3420_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v3420_data + (v3340_data * v3418_data)).copy_to(ir3 + (7));
              }
              int32_t v3426_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3430_data(0.0f);
              v3430_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[12_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3431_data(0.0f);
              v3431_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[1]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3433_data(0.0f);
              v3433_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3433_data + (v3430_data * v3431_data)).copy_to(ir3 + (0));
              }
              int32_t v3437_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3442_data(0.0f);
              v3442_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[13]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3444_data(0.0f);
              v3444_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3444_data + (v3430_data * v3442_data)).copy_to(ir3 + (1));
              }
              int32_t v3448_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3453_data(0.0f);
              v3453_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[25]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3455_data(0.0f);
              v3455_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3455_data + (v3430_data * v3453_data)).copy_to(ir3 + (2));
              }
              int32_t v3459_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3464_data(0.0f);
              v3464_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[37]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3466_data(0.0f);
              v3466_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v3466_data + (v3430_data * v3464_data)).copy_to(ir3 + (3));
              }
              int32_t v3470_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3475_data(0.0f);
              v3475_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[49]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3477_data(0.0f);
              v3477_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v3477_data + (v3430_data * v3475_data)).copy_to(ir3 + (4));
              }
              int32_t v3481_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3486_data(0.0f);
              v3486_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[61]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3488_data(0.0f);
              v3488_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v3488_data + (v3430_data * v3486_data)).copy_to(ir3 + (5));
              }
              int32_t v3492_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3497_data(0.0f);
              v3497_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[73]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3499_data(0.0f);
              v3499_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v3499_data + (v3430_data * v3497_data)).copy_to(ir3 + (6));
              }
              int32_t v3503_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v3508_data(0.0f);
              v3508_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[85]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3510_data(0.0f);
              v3510_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v3510_data + (v3430_data * v3508_data)).copy_to(ir3 + (7));
              }
              int32_t v3516_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3520_data(0.0f);
              v3520_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[24_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3521_data(0.0f);
              v3521_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[2]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3523_data(0.0f);
              v3523_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3523_data + (v3520_data * v3521_data)).copy_to(ir3 + (0));
              }
              int32_t v3527_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3532_data(0.0f);
              v3532_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[14]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3534_data(0.0f);
              v3534_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3534_data + (v3520_data * v3532_data)).copy_to(ir3 + (1));
              }
              int32_t v3538_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3543_data(0.0f);
              v3543_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[26]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3545_data(0.0f);
              v3545_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3545_data + (v3520_data * v3543_data)).copy_to(ir3 + (2));
              }
              int32_t v3549_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3554_data(0.0f);
              v3554_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[38]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3556_data(0.0f);
              v3556_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v3556_data + (v3520_data * v3554_data)).copy_to(ir3 + (3));
              }
              int32_t v3560_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3565_data(0.0f);
              v3565_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[50]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3567_data(0.0f);
              v3567_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v3567_data + (v3520_data * v3565_data)).copy_to(ir3 + (4));
              }
              int32_t v3571_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3576_data(0.0f);
              v3576_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[62]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3578_data(0.0f);
              v3578_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v3578_data + (v3520_data * v3576_data)).copy_to(ir3 + (5));
              }
              int32_t v3582_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3587_data(0.0f);
              v3587_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[74]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3589_data(0.0f);
              v3589_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v3589_data + (v3520_data * v3587_data)).copy_to(ir3 + (6));
              }
              int32_t v3593_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v3598_data(0.0f);
              v3598_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[86]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3600_data(0.0f);
              v3600_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v3600_data + (v3520_data * v3598_data)).copy_to(ir3 + (7));
              }
              int32_t v3606_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3610_data(0.0f);
              v3610_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[36_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3611_data(0.0f);
              v3611_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[3]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3613_data(0.0f);
              v3613_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3613_data + (v3610_data * v3611_data)).copy_to(ir3 + (0));
              }
              int32_t v3617_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3622_data(0.0f);
              v3622_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[15]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3624_data(0.0f);
              v3624_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3624_data + (v3610_data * v3622_data)).copy_to(ir3 + (1));
              }
              int32_t v3628_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3633_data(0.0f);
              v3633_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[27]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3635_data(0.0f);
              v3635_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3635_data + (v3610_data * v3633_data)).copy_to(ir3 + (2));
              }
              int32_t v3639_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3644_data(0.0f);
              v3644_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[39]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3646_data(0.0f);
              v3646_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v3646_data + (v3610_data * v3644_data)).copy_to(ir3 + (3));
              }
              int32_t v3650_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3655_data(0.0f);
              v3655_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[51]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3657_data(0.0f);
              v3657_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v3657_data + (v3610_data * v3655_data)).copy_to(ir3 + (4));
              }
              int32_t v3661_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3666_data(0.0f);
              v3666_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[63]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3668_data(0.0f);
              v3668_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v3668_data + (v3610_data * v3666_data)).copy_to(ir3 + (5));
              }
              int32_t v3672_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3677_data(0.0f);
              v3677_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[75]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3679_data(0.0f);
              v3679_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v3679_data + (v3610_data * v3677_data)).copy_to(ir3 + (6));
              }
              int32_t v3683_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v3688_data(0.0f);
              v3688_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[87]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3690_data(0.0f);
              v3690_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v3690_data + (v3610_data * v3688_data)).copy_to(ir3 + (7));
              }
              int32_t v3696_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3700_data(0.0f);
              v3700_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[48_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3701_data(0.0f);
              v3701_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[4]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3703_data(0.0f);
              v3703_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3703_data + (v3700_data * v3701_data)).copy_to(ir3 + (0));
              }
              int32_t v3707_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3712_data(0.0f);
              v3712_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[16]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3714_data(0.0f);
              v3714_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3714_data + (v3700_data * v3712_data)).copy_to(ir3 + (1));
              }
              int32_t v3718_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3723_data(0.0f);
              v3723_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[28]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3725_data(0.0f);
              v3725_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3725_data + (v3700_data * v3723_data)).copy_to(ir3 + (2));
              }
              int32_t v3729_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3734_data(0.0f);
              v3734_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[40]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3736_data(0.0f);
              v3736_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v3736_data + (v3700_data * v3734_data)).copy_to(ir3 + (3));
              }
              int32_t v3740_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3745_data(0.0f);
              v3745_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[52]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3747_data(0.0f);
              v3747_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v3747_data + (v3700_data * v3745_data)).copy_to(ir3 + (4));
              }
              int32_t v3751_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3756_data(0.0f);
              v3756_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[64]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3758_data(0.0f);
              v3758_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v3758_data + (v3700_data * v3756_data)).copy_to(ir3 + (5));
              }
              int32_t v3762_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3767_data(0.0f);
              v3767_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[76]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3769_data(0.0f);
              v3769_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v3769_data + (v3700_data * v3767_data)).copy_to(ir3 + (6));
              }
              int32_t v3773_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v3778_data(0.0f);
              v3778_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[88]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3780_data(0.0f);
              v3780_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v3780_data + (v3700_data * v3778_data)).copy_to(ir3 + (7));
              }
              int32_t v3786_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3790_data(0.0f);
              v3790_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[60_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3791_data(0.0f);
              v3791_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[5]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3793_data(0.0f);
              v3793_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3793_data + (v3790_data * v3791_data)).copy_to(ir3 + (0));
              }
              int32_t v3797_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3802_data(0.0f);
              v3802_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[17]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3804_data(0.0f);
              v3804_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3804_data + (v3790_data * v3802_data)).copy_to(ir3 + (1));
              }
              int32_t v3808_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3813_data(0.0f);
              v3813_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[29]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3815_data(0.0f);
              v3815_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3815_data + (v3790_data * v3813_data)).copy_to(ir3 + (2));
              }
              int32_t v3819_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3824_data(0.0f);
              v3824_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[41]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3826_data(0.0f);
              v3826_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v3826_data + (v3790_data * v3824_data)).copy_to(ir3 + (3));
              }
              int32_t v3830_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3835_data(0.0f);
              v3835_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[53]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3837_data(0.0f);
              v3837_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v3837_data + (v3790_data * v3835_data)).copy_to(ir3 + (4));
              }
              int32_t v3841_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3846_data(0.0f);
              v3846_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[65]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3848_data(0.0f);
              v3848_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v3848_data + (v3790_data * v3846_data)).copy_to(ir3 + (5));
              }
              int32_t v3852_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3857_data(0.0f);
              v3857_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[77]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3859_data(0.0f);
              v3859_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v3859_data + (v3790_data * v3857_data)).copy_to(ir3 + (6));
              }
              int32_t v3863_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v3868_data(0.0f);
              v3868_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[89]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3870_data(0.0f);
              v3870_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v3870_data + (v3790_data * v3868_data)).copy_to(ir3 + (7));
              }
              int32_t v3876_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3880_data(0.0f);
              v3880_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[72_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3881_data(0.0f);
              v3881_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[6]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3883_data(0.0f);
              v3883_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3883_data + (v3880_data * v3881_data)).copy_to(ir3 + (0));
              }
              int32_t v3887_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3892_data(0.0f);
              v3892_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[18]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3894_data(0.0f);
              v3894_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3894_data + (v3880_data * v3892_data)).copy_to(ir3 + (1));
              }
              int32_t v3898_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3903_data(0.0f);
              v3903_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[30]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3905_data(0.0f);
              v3905_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3905_data + (v3880_data * v3903_data)).copy_to(ir3 + (2));
              }
              int32_t v3909_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3914_data(0.0f);
              v3914_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[42]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3916_data(0.0f);
              v3916_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v3916_data + (v3880_data * v3914_data)).copy_to(ir3 + (3));
              }
              int32_t v3920_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3925_data(0.0f);
              v3925_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[54]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3927_data(0.0f);
              v3927_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v3927_data + (v3880_data * v3925_data)).copy_to(ir3 + (4));
              }
              int32_t v3931_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3936_data(0.0f);
              v3936_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[66]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3938_data(0.0f);
              v3938_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v3938_data + (v3880_data * v3936_data)).copy_to(ir3 + (5));
              }
              int32_t v3942_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3947_data(0.0f);
              v3947_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[78]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3949_data(0.0f);
              v3949_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v3949_data + (v3880_data * v3947_data)).copy_to(ir3 + (6));
              }
              int32_t v3953_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v3958_data(0.0f);
              v3958_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[90]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3960_data(0.0f);
              v3960_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v3960_data + (v3880_data * v3958_data)).copy_to(ir3 + (7));
              }
              int32_t v3966_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v3970_data(0.0f);
              v3970_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[84_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3971_data(0.0f);
              v3971_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[7]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3973_data(0.0f);
              v3973_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v3973_data + (v3970_data * v3971_data)).copy_to(ir3 + (0));
              }
              int32_t v3977_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v3982_data(0.0f);
              v3982_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[19]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3984_data(0.0f);
              v3984_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v3984_data + (v3970_data * v3982_data)).copy_to(ir3 + (1));
              }
              int32_t v3988_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v3993_data(0.0f);
              v3993_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[31]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v3995_data(0.0f);
              v3995_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v3995_data + (v3970_data * v3993_data)).copy_to(ir3 + (2));
              }
              int32_t v3999_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4004_data(0.0f);
              v4004_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[43]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4006_data(0.0f);
              v4006_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v4006_data + (v3970_data * v4004_data)).copy_to(ir3 + (3));
              }
              int32_t v4010_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4015_data(0.0f);
              v4015_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[55]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4017_data(0.0f);
              v4017_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v4017_data + (v3970_data * v4015_data)).copy_to(ir3 + (4));
              }
              int32_t v4021_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4026_data(0.0f);
              v4026_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[67]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4028_data(0.0f);
              v4028_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v4028_data + (v3970_data * v4026_data)).copy_to(ir3 + (5));
              }
              int32_t v4032_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4037_data(0.0f);
              v4037_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[79]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4039_data(0.0f);
              v4039_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v4039_data + (v3970_data * v4037_data)).copy_to(ir3 + (6));
              }
              int32_t v4043_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v4048_data(0.0f);
              v4048_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[91]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4050_data(0.0f);
              v4050_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v4050_data + (v3970_data * v4048_data)).copy_to(ir3 + (7));
              }
              int32_t v4056_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4060_data(0.0f);
              v4060_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[96_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4061_data(0.0f);
              v4061_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[8]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4063_data(0.0f);
              v4063_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v4063_data + (v4060_data * v4061_data)).copy_to(ir3 + (0));
              }
              int32_t v4067_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4072_data(0.0f);
              v4072_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[20]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4074_data(0.0f);
              v4074_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v4074_data + (v4060_data * v4072_data)).copy_to(ir3 + (1));
              }
              int32_t v4078_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4083_data(0.0f);
              v4083_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4085_data(0.0f);
              v4085_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v4085_data + (v4060_data * v4083_data)).copy_to(ir3 + (2));
              }
              int32_t v4089_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4094_data(0.0f);
              v4094_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[44]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4096_data(0.0f);
              v4096_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v4096_data + (v4060_data * v4094_data)).copy_to(ir3 + (3));
              }
              int32_t v4100_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4105_data(0.0f);
              v4105_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[56]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4107_data(0.0f);
              v4107_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v4107_data + (v4060_data * v4105_data)).copy_to(ir3 + (4));
              }
              int32_t v4111_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4116_data(0.0f);
              v4116_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[68]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4118_data(0.0f);
              v4118_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v4118_data + (v4060_data * v4116_data)).copy_to(ir3 + (5));
              }
              int32_t v4122_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4127_data(0.0f);
              v4127_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[80]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4129_data(0.0f);
              v4129_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v4129_data + (v4060_data * v4127_data)).copy_to(ir3 + (6));
              }
              int32_t v4133_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v4138_data(0.0f);
              v4138_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[92]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4140_data(0.0f);
              v4140_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v4140_data + (v4060_data * v4138_data)).copy_to(ir3 + (7));
              }
              int32_t v4146_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4150_data(0.0f);
              v4150_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[108_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4151_data(0.0f);
              v4151_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[9]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4153_data(0.0f);
              v4153_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v4153_data + (v4150_data * v4151_data)).copy_to(ir3 + (0));
              }
              int32_t v4157_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4162_data(0.0f);
              v4162_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[21]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4164_data(0.0f);
              v4164_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v4164_data + (v4150_data * v4162_data)).copy_to(ir3 + (1));
              }
              int32_t v4168_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4173_data(0.0f);
              v4173_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[33]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4175_data(0.0f);
              v4175_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v4175_data + (v4150_data * v4173_data)).copy_to(ir3 + (2));
              }
              int32_t v4179_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4184_data(0.0f);
              v4184_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[45]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4186_data(0.0f);
              v4186_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v4186_data + (v4150_data * v4184_data)).copy_to(ir3 + (3));
              }
              int32_t v4190_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4195_data(0.0f);
              v4195_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[57]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4197_data(0.0f);
              v4197_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v4197_data + (v4150_data * v4195_data)).copy_to(ir3 + (4));
              }
              int32_t v4201_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4206_data(0.0f);
              v4206_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[69]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4208_data(0.0f);
              v4208_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v4208_data + (v4150_data * v4206_data)).copy_to(ir3 + (5));
              }
              int32_t v4212_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4217_data(0.0f);
              v4217_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[81]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4219_data(0.0f);
              v4219_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v4219_data + (v4150_data * v4217_data)).copy_to(ir3 + (6));
              }
              int32_t v4223_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v4228_data(0.0f);
              v4228_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[93]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4230_data(0.0f);
              v4230_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v4230_data + (v4150_data * v4228_data)).copy_to(ir3 + (7));
              }
              int32_t v4236_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4240_data(0.0f);
              v4240_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[120_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4241_data(0.0f);
              v4241_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[10]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4243_data(0.0f);
              v4243_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v4243_data + (v4240_data * v4241_data)).copy_to(ir3 + (0));
              }
              int32_t v4247_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4252_data(0.0f);
              v4252_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[22]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4254_data(0.0f);
              v4254_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v4254_data + (v4240_data * v4252_data)).copy_to(ir3 + (1));
              }
              int32_t v4258_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4263_data(0.0f);
              v4263_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[34]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4265_data(0.0f);
              v4265_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v4265_data + (v4240_data * v4263_data)).copy_to(ir3 + (2));
              }
              int32_t v4269_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4274_data(0.0f);
              v4274_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[46]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4276_data(0.0f);
              v4276_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v4276_data + (v4240_data * v4274_data)).copy_to(ir3 + (3));
              }
              int32_t v4280_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4285_data(0.0f);
              v4285_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[58]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4287_data(0.0f);
              v4287_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v4287_data + (v4240_data * v4285_data)).copy_to(ir3 + (4));
              }
              int32_t v4291_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4296_data(0.0f);
              v4296_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[70]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4298_data(0.0f);
              v4298_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v4298_data + (v4240_data * v4296_data)).copy_to(ir3 + (5));
              }
              int32_t v4302_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4307_data(0.0f);
              v4307_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[82]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4309_data(0.0f);
              v4309_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v4309_data + (v4240_data * v4307_data)).copy_to(ir3 + (6));
              }
              int32_t v4313_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v4318_data(0.0f);
              v4318_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[94]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4320_data(0.0f);
              v4320_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v4320_data + (v4240_data * v4318_data)).copy_to(ir3 + (7));
              }
              int32_t v4326_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4330_data(0.0f);
              v4330_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m7[132_i32]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4331_data(0.0f);
              v4331_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[11]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4333_data(0.0f);
              v4333_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[0]), v13_g);
              if (v13_g) {
                (v4333_data + (v4330_data * v4331_data)).copy_to(ir3 + (0));
              }
              int32_t v4337_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4342_data(0.0f);
              v4342_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[23]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4344_data(0.0f);
              v4344_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[1]), v13_g);
              if (v13_g) {
                (v4344_data + (v4330_data * v4342_data)).copy_to(ir3 + (1));
              }
              int32_t v4348_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4353_data(0.0f);
              v4353_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[35]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4355_data(0.0f);
              v4355_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[2]), v13_g);
              if (v13_g) {
                (v4355_data + (v4330_data * v4353_data)).copy_to(ir3 + (2));
              }
              int32_t v4359_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4364_data(0.0f);
              v4364_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[47]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4366_data(0.0f);
              v4366_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[3]), v13_g);
              if (v13_g) {
                (v4366_data + (v4330_data * v4364_data)).copy_to(ir3 + (3));
              }
              int32_t v4370_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4375_data(0.0f);
              v4375_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[59]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4377_data(0.0f);
              v4377_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[4]), v13_g);
              if (v13_g) {
                (v4377_data + (v4330_data * v4375_data)).copy_to(ir3 + (4));
              }
              int32_t v4381_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4386_data(0.0f);
              v4386_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[71]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4388_data(0.0f);
              v4388_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[5]), v13_g);
              if (v13_g) {
                (v4388_data + (v4330_data * v4386_data)).copy_to(ir3 + (5));
              }
              int32_t v4392_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4397_data(0.0f);
              v4397_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[83]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4399_data(0.0f);
              v4399_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[6]), v13_g);
              if (v13_g) {
                (v4399_data + (v4330_data * v4397_data)).copy_to(ir3 + (6));
              }
              int32_t v4403_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v4408_data(0.0f);
              v4408_data.merge(tensorforge::intel_esimd::simd<float, 16>(s3[95]), v13_g);
              tensorforge::intel_esimd::simd<float, 16> v4410_data(0.0f);
              v4410_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[7]), v13_g);
              if (v13_g) {
                (v4410_data + (v4330_data * v4408_data)).copy_to(ir3 + (7));
              }
              #pragma unroll
              for (int32_t v4414_n1 = 0; v4414_n1 < 8; ++v4414_n1) {
                int32_t v4415_a = 0 + v4414_n1;
                tensorforge::intel_esimd::simd<float, 16> v4417_data(0.0f);
                v4417_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir3[v4414_n1]), v13_g);
                int32_t v4420_a = v4414_n1 * 12;
                int32_t v4421_a = 0_i32 + v4420_a;
                tensorforge::intel_esimd::simd<float, 16> v4426_data(0.0f);
                v4426_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[v4420_a]), v13_g);
                if (v13_g) {
                  (v4426_data + v4417_data).copy_to(r3 + (v4414_n1));
                }
              }
              // glb_m0 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v4431_i1 = 0; v4431_i1 < 8; ++v4431_i1) {
                int32_t v4432_a = 0 + v4431_i1;
                tensorforge::intel_esimd::simd<float, 16> v4434_data(0.0f);
                v4434_data.merge(tensorforge::intel_esimd::simd<float, 16>(r3[v4431_i1]), v13_g);
                if (v13_g) {
                  v4434_data.copy_to(glb_m0 + ((v4431_i1 * 12)));
                }
              }
            }
          }
        }
      });
    }
  });
}

