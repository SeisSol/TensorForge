// === base name ===
kernel_5e7da3148f

// === header ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5e7da3148f(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5e7da3148f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5e7da3148f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              // r0 = +(glb_m1 * s0) + name: glb_m0, type: SymbolType.Global, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              tensorforge::intel_esimd::simd_mask<16> v7_g = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 12;
              int32_t v10_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v14_data(0.0f);
              v14_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[0_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v15_data(0.0f);
              v15_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v17_data(0.0f);
              v17_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v17_data + (v14_data * v15_data)).copy_to(ir0 + (0));
              }
              int32_t v21_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v26_data(0.0f);
              v26_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v28_data(0.0f);
              v28_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v28_data + (v14_data * v26_data)).copy_to(ir0 + (1));
              }
              int32_t v32_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v37_data(0.0f);
              v37_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v39_data(0.0f);
              v39_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v39_data + (v14_data * v37_data)).copy_to(ir0 + (2));
              }
              int32_t v43_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v48_data(0.0f);
              v48_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v50_data(0.0f);
              v50_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v50_data + (v14_data * v48_data)).copy_to(ir0 + (3));
              }
              int32_t v54_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v59_data(0.0f);
              v59_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v61_data(0.0f);
              v61_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v61_data + (v14_data * v59_data)).copy_to(ir0 + (4));
              }
              int32_t v65_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v70_data(0.0f);
              v70_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v72_data(0.0f);
              v72_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v72_data + (v14_data * v70_data)).copy_to(ir0 + (5));
              }
              int32_t v76_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v81_data(0.0f);
              v81_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[96]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v83_data(0.0f);
              v83_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v83_data + (v14_data * v81_data)).copy_to(ir0 + (6));
              }
              int32_t v87_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v92_data(0.0f);
              v92_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[112]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v94_data(0.0f);
              v94_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v94_data + (v14_data * v92_data)).copy_to(ir0 + (7));
              }
              int32_t v100_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v104_data(0.0f);
              v104_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[12_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v105_data(0.0f);
              v105_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v107_data(0.0f);
              v107_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v107_data + (v104_data * v105_data)).copy_to(ir0 + (0));
              }
              int32_t v111_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v116_data(0.0f);
              v116_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v118_data(0.0f);
              v118_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v118_data + (v104_data * v116_data)).copy_to(ir0 + (1));
              }
              int32_t v122_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v127_data(0.0f);
              v127_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v129_data(0.0f);
              v129_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v129_data + (v104_data * v127_data)).copy_to(ir0 + (2));
              }
              int32_t v133_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v138_data(0.0f);
              v138_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v140_data(0.0f);
              v140_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v140_data + (v104_data * v138_data)).copy_to(ir0 + (3));
              }
              int32_t v144_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v149_data(0.0f);
              v149_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v151_data(0.0f);
              v151_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v151_data + (v104_data * v149_data)).copy_to(ir0 + (4));
              }
              int32_t v155_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v160_data(0.0f);
              v160_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[81]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v162_data(0.0f);
              v162_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v162_data + (v104_data * v160_data)).copy_to(ir0 + (5));
              }
              int32_t v166_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v171_data(0.0f);
              v171_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[97]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v173_data(0.0f);
              v173_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v173_data + (v104_data * v171_data)).copy_to(ir0 + (6));
              }
              int32_t v177_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v182_data(0.0f);
              v182_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[113]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v184_data(0.0f);
              v184_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v184_data + (v104_data * v182_data)).copy_to(ir0 + (7));
              }
              int32_t v190_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v194_data(0.0f);
              v194_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[24_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v195_data(0.0f);
              v195_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v197_data(0.0f);
              v197_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v197_data + (v194_data * v195_data)).copy_to(ir0 + (0));
              }
              int32_t v201_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v206_data(0.0f);
              v206_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v208_data(0.0f);
              v208_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v208_data + (v194_data * v206_data)).copy_to(ir0 + (1));
              }
              int32_t v212_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v217_data(0.0f);
              v217_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v219_data(0.0f);
              v219_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v219_data + (v194_data * v217_data)).copy_to(ir0 + (2));
              }
              int32_t v223_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v228_data(0.0f);
              v228_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[50]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v230_data(0.0f);
              v230_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v230_data + (v194_data * v228_data)).copy_to(ir0 + (3));
              }
              int32_t v234_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v239_data(0.0f);
              v239_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v241_data(0.0f);
              v241_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v241_data + (v194_data * v239_data)).copy_to(ir0 + (4));
              }
              int32_t v245_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v250_data(0.0f);
              v250_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[82]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v252_data(0.0f);
              v252_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v252_data + (v194_data * v250_data)).copy_to(ir0 + (5));
              }
              int32_t v256_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v261_data(0.0f);
              v261_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[98]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v263_data(0.0f);
              v263_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v263_data + (v194_data * v261_data)).copy_to(ir0 + (6));
              }
              int32_t v267_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v272_data(0.0f);
              v272_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[114]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v274_data(0.0f);
              v274_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v274_data + (v194_data * v272_data)).copy_to(ir0 + (7));
              }
              int32_t v280_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v284_data(0.0f);
              v284_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[36_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v285_data(0.0f);
              v285_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v287_data(0.0f);
              v287_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v287_data + (v284_data * v285_data)).copy_to(ir0 + (0));
              }
              int32_t v291_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v296_data(0.0f);
              v296_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v298_data(0.0f);
              v298_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v298_data + (v284_data * v296_data)).copy_to(ir0 + (1));
              }
              int32_t v302_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v307_data(0.0f);
              v307_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v309_data(0.0f);
              v309_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v309_data + (v284_data * v307_data)).copy_to(ir0 + (2));
              }
              int32_t v313_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v318_data(0.0f);
              v318_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v320_data(0.0f);
              v320_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v320_data + (v284_data * v318_data)).copy_to(ir0 + (3));
              }
              int32_t v324_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v329_data(0.0f);
              v329_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[67]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v331_data(0.0f);
              v331_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v331_data + (v284_data * v329_data)).copy_to(ir0 + (4));
              }
              int32_t v335_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v340_data(0.0f);
              v340_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[83]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v342_data(0.0f);
              v342_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v342_data + (v284_data * v340_data)).copy_to(ir0 + (5));
              }
              int32_t v346_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v351_data(0.0f);
              v351_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[99]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v353_data(0.0f);
              v353_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v353_data + (v284_data * v351_data)).copy_to(ir0 + (6));
              }
              int32_t v357_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v362_data(0.0f);
              v362_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[115]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v364_data(0.0f);
              v364_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v364_data + (v284_data * v362_data)).copy_to(ir0 + (7));
              }
              int32_t v370_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v374_data(0.0f);
              v374_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[48_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v375_data(0.0f);
              v375_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v377_data(0.0f);
              v377_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v377_data + (v374_data * v375_data)).copy_to(ir0 + (0));
              }
              int32_t v381_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v386_data(0.0f);
              v386_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v388_data(0.0f);
              v388_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v388_data + (v374_data * v386_data)).copy_to(ir0 + (1));
              }
              int32_t v392_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v397_data(0.0f);
              v397_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v399_data(0.0f);
              v399_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v399_data + (v374_data * v397_data)).copy_to(ir0 + (2));
              }
              int32_t v403_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v408_data(0.0f);
              v408_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v410_data(0.0f);
              v410_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v410_data + (v374_data * v408_data)).copy_to(ir0 + (3));
              }
              int32_t v414_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v419_data(0.0f);
              v419_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v421_data(0.0f);
              v421_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v421_data + (v374_data * v419_data)).copy_to(ir0 + (4));
              }
              int32_t v425_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v430_data(0.0f);
              v430_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[84]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v432_data(0.0f);
              v432_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v432_data + (v374_data * v430_data)).copy_to(ir0 + (5));
              }
              int32_t v436_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v441_data(0.0f);
              v441_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[100]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v443_data(0.0f);
              v443_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v443_data + (v374_data * v441_data)).copy_to(ir0 + (6));
              }
              int32_t v447_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v452_data(0.0f);
              v452_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[116]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v454_data(0.0f);
              v454_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v454_data + (v374_data * v452_data)).copy_to(ir0 + (7));
              }
              int32_t v460_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v464_data(0.0f);
              v464_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[60_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v465_data(0.0f);
              v465_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v467_data(0.0f);
              v467_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v467_data + (v464_data * v465_data)).copy_to(ir0 + (0));
              }
              int32_t v471_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v476_data(0.0f);
              v476_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v478_data(0.0f);
              v478_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v478_data + (v464_data * v476_data)).copy_to(ir0 + (1));
              }
              int32_t v482_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v487_data(0.0f);
              v487_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v489_data(0.0f);
              v489_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v489_data + (v464_data * v487_data)).copy_to(ir0 + (2));
              }
              int32_t v493_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v498_data(0.0f);
              v498_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v500_data(0.0f);
              v500_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v500_data + (v464_data * v498_data)).copy_to(ir0 + (3));
              }
              int32_t v504_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v509_data(0.0f);
              v509_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v511_data(0.0f);
              v511_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v511_data + (v464_data * v509_data)).copy_to(ir0 + (4));
              }
              int32_t v515_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v520_data(0.0f);
              v520_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[85]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v522_data(0.0f);
              v522_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v522_data + (v464_data * v520_data)).copy_to(ir0 + (5));
              }
              int32_t v526_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v531_data(0.0f);
              v531_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[101]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v533_data(0.0f);
              v533_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v533_data + (v464_data * v531_data)).copy_to(ir0 + (6));
              }
              int32_t v537_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v542_data(0.0f);
              v542_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[117]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v544_data(0.0f);
              v544_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v544_data + (v464_data * v542_data)).copy_to(ir0 + (7));
              }
              int32_t v550_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v554_data(0.0f);
              v554_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[72_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v555_data(0.0f);
              v555_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v557_data(0.0f);
              v557_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v557_data + (v554_data * v555_data)).copy_to(ir0 + (0));
              }
              int32_t v561_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v566_data(0.0f);
              v566_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v568_data(0.0f);
              v568_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v568_data + (v554_data * v566_data)).copy_to(ir0 + (1));
              }
              int32_t v572_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v577_data(0.0f);
              v577_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v579_data(0.0f);
              v579_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v579_data + (v554_data * v577_data)).copy_to(ir0 + (2));
              }
              int32_t v583_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v588_data(0.0f);
              v588_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v590_data(0.0f);
              v590_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v590_data + (v554_data * v588_data)).copy_to(ir0 + (3));
              }
              int32_t v594_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v599_data(0.0f);
              v599_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v601_data(0.0f);
              v601_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v601_data + (v554_data * v599_data)).copy_to(ir0 + (4));
              }
              int32_t v605_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v610_data(0.0f);
              v610_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[86]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v612_data(0.0f);
              v612_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v612_data + (v554_data * v610_data)).copy_to(ir0 + (5));
              }
              int32_t v616_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v621_data(0.0f);
              v621_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[102]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v623_data(0.0f);
              v623_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v623_data + (v554_data * v621_data)).copy_to(ir0 + (6));
              }
              int32_t v627_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v632_data(0.0f);
              v632_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[118]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v634_data(0.0f);
              v634_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v634_data + (v554_data * v632_data)).copy_to(ir0 + (7));
              }
              int32_t v640_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v644_data(0.0f);
              v644_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[84_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v645_data(0.0f);
              v645_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v647_data(0.0f);
              v647_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v647_data + (v644_data * v645_data)).copy_to(ir0 + (0));
              }
              int32_t v651_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v656_data(0.0f);
              v656_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v658_data(0.0f);
              v658_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v658_data + (v644_data * v656_data)).copy_to(ir0 + (1));
              }
              int32_t v662_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v667_data(0.0f);
              v667_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v669_data(0.0f);
              v669_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v669_data + (v644_data * v667_data)).copy_to(ir0 + (2));
              }
              int32_t v673_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v678_data(0.0f);
              v678_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v680_data(0.0f);
              v680_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v680_data + (v644_data * v678_data)).copy_to(ir0 + (3));
              }
              int32_t v684_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v689_data(0.0f);
              v689_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v691_data(0.0f);
              v691_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v691_data + (v644_data * v689_data)).copy_to(ir0 + (4));
              }
              int32_t v695_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v700_data(0.0f);
              v700_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[87]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v702_data(0.0f);
              v702_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v702_data + (v644_data * v700_data)).copy_to(ir0 + (5));
              }
              int32_t v706_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v711_data(0.0f);
              v711_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[103]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v713_data(0.0f);
              v713_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v713_data + (v644_data * v711_data)).copy_to(ir0 + (6));
              }
              int32_t v717_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v722_data(0.0f);
              v722_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[119]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v724_data(0.0f);
              v724_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v724_data + (v644_data * v722_data)).copy_to(ir0 + (7));
              }
              int32_t v730_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v734_data(0.0f);
              v734_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[96_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v735_data(0.0f);
              v735_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v737_data(0.0f);
              v737_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v737_data + (v734_data * v735_data)).copy_to(ir0 + (0));
              }
              int32_t v741_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v746_data(0.0f);
              v746_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v748_data(0.0f);
              v748_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v748_data + (v734_data * v746_data)).copy_to(ir0 + (1));
              }
              int32_t v752_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v757_data(0.0f);
              v757_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v759_data(0.0f);
              v759_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v759_data + (v734_data * v757_data)).copy_to(ir0 + (2));
              }
              int32_t v763_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v768_data(0.0f);
              v768_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v770_data(0.0f);
              v770_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v770_data + (v734_data * v768_data)).copy_to(ir0 + (3));
              }
              int32_t v774_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v779_data(0.0f);
              v779_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v781_data(0.0f);
              v781_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v781_data + (v734_data * v779_data)).copy_to(ir0 + (4));
              }
              int32_t v785_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v790_data(0.0f);
              v790_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[88]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v792_data(0.0f);
              v792_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v792_data + (v734_data * v790_data)).copy_to(ir0 + (5));
              }
              int32_t v796_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v801_data(0.0f);
              v801_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[104]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v803_data(0.0f);
              v803_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v803_data + (v734_data * v801_data)).copy_to(ir0 + (6));
              }
              int32_t v807_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v812_data(0.0f);
              v812_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[120]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v814_data(0.0f);
              v814_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v814_data + (v734_data * v812_data)).copy_to(ir0 + (7));
              }
              int32_t v820_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v824_data(0.0f);
              v824_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[108_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v825_data(0.0f);
              v825_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v827_data(0.0f);
              v827_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v827_data + (v824_data * v825_data)).copy_to(ir0 + (0));
              }
              int32_t v831_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v836_data(0.0f);
              v836_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v838_data(0.0f);
              v838_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v838_data + (v824_data * v836_data)).copy_to(ir0 + (1));
              }
              int32_t v842_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v847_data(0.0f);
              v847_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v849_data(0.0f);
              v849_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v849_data + (v824_data * v847_data)).copy_to(ir0 + (2));
              }
              int32_t v853_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v858_data(0.0f);
              v858_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v860_data(0.0f);
              v860_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v860_data + (v824_data * v858_data)).copy_to(ir0 + (3));
              }
              int32_t v864_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v869_data(0.0f);
              v869_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v871_data(0.0f);
              v871_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v871_data + (v824_data * v869_data)).copy_to(ir0 + (4));
              }
              int32_t v875_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v880_data(0.0f);
              v880_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[89]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v882_data(0.0f);
              v882_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v882_data + (v824_data * v880_data)).copy_to(ir0 + (5));
              }
              int32_t v886_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v891_data(0.0f);
              v891_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[105]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v893_data(0.0f);
              v893_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v893_data + (v824_data * v891_data)).copy_to(ir0 + (6));
              }
              int32_t v897_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v902_data(0.0f);
              v902_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[121]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v904_data(0.0f);
              v904_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v904_data + (v824_data * v902_data)).copy_to(ir0 + (7));
              }
              int32_t v910_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v914_data(0.0f);
              v914_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[120_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v915_data(0.0f);
              v915_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v917_data(0.0f);
              v917_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v917_data + (v914_data * v915_data)).copy_to(ir0 + (0));
              }
              int32_t v921_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v926_data(0.0f);
              v926_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v928_data(0.0f);
              v928_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v928_data + (v914_data * v926_data)).copy_to(ir0 + (1));
              }
              int32_t v932_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v937_data(0.0f);
              v937_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v939_data(0.0f);
              v939_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v939_data + (v914_data * v937_data)).copy_to(ir0 + (2));
              }
              int32_t v943_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v948_data(0.0f);
              v948_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v950_data(0.0f);
              v950_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v950_data + (v914_data * v948_data)).copy_to(ir0 + (3));
              }
              int32_t v954_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v959_data(0.0f);
              v959_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v961_data(0.0f);
              v961_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v961_data + (v914_data * v959_data)).copy_to(ir0 + (4));
              }
              int32_t v965_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v970_data(0.0f);
              v970_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[90]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v972_data(0.0f);
              v972_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v972_data + (v914_data * v970_data)).copy_to(ir0 + (5));
              }
              int32_t v976_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v981_data(0.0f);
              v981_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[106]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v983_data(0.0f);
              v983_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v983_data + (v914_data * v981_data)).copy_to(ir0 + (6));
              }
              int32_t v987_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v992_data(0.0f);
              v992_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[122]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v994_data(0.0f);
              v994_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v994_data + (v914_data * v992_data)).copy_to(ir0 + (7));
              }
              int32_t v1000_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1004_data(0.0f);
              v1004_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[132_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1005_data(0.0f);
              v1005_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1007_data(0.0f);
              v1007_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1007_data + (v1004_data * v1005_data)).copy_to(ir0 + (0));
              }
              int32_t v1011_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1016_data(0.0f);
              v1016_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1018_data(0.0f);
              v1018_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1018_data + (v1004_data * v1016_data)).copy_to(ir0 + (1));
              }
              int32_t v1022_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1027_data(0.0f);
              v1027_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1029_data(0.0f);
              v1029_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1029_data + (v1004_data * v1027_data)).copy_to(ir0 + (2));
              }
              int32_t v1033_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1038_data(0.0f);
              v1038_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1040_data(0.0f);
              v1040_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1040_data + (v1004_data * v1038_data)).copy_to(ir0 + (3));
              }
              int32_t v1044_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1049_data(0.0f);
              v1049_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1051_data(0.0f);
              v1051_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1051_data + (v1004_data * v1049_data)).copy_to(ir0 + (4));
              }
              int32_t v1055_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1060_data(0.0f);
              v1060_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[91]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1062_data(0.0f);
              v1062_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1062_data + (v1004_data * v1060_data)).copy_to(ir0 + (5));
              }
              int32_t v1066_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1071_data(0.0f);
              v1071_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[107]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1073_data(0.0f);
              v1073_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1073_data + (v1004_data * v1071_data)).copy_to(ir0 + (6));
              }
              int32_t v1077_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1082_data(0.0f);
              v1082_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[123]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1084_data(0.0f);
              v1084_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1084_data + (v1004_data * v1082_data)).copy_to(ir0 + (7));
              }
              int32_t v1090_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1094_data(0.0f);
              v1094_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[144_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1095_data(0.0f);
              v1095_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1097_data(0.0f);
              v1097_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1097_data + (v1094_data * v1095_data)).copy_to(ir0 + (0));
              }
              int32_t v1101_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1106_data(0.0f);
              v1106_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1108_data(0.0f);
              v1108_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1108_data + (v1094_data * v1106_data)).copy_to(ir0 + (1));
              }
              int32_t v1112_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1117_data(0.0f);
              v1117_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1119_data(0.0f);
              v1119_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1119_data + (v1094_data * v1117_data)).copy_to(ir0 + (2));
              }
              int32_t v1123_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1128_data(0.0f);
              v1128_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1130_data(0.0f);
              v1130_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1130_data + (v1094_data * v1128_data)).copy_to(ir0 + (3));
              }
              int32_t v1134_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1139_data(0.0f);
              v1139_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1141_data(0.0f);
              v1141_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1141_data + (v1094_data * v1139_data)).copy_to(ir0 + (4));
              }
              int32_t v1145_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1150_data(0.0f);
              v1150_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[92]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1152_data(0.0f);
              v1152_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1152_data + (v1094_data * v1150_data)).copy_to(ir0 + (5));
              }
              int32_t v1156_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1161_data(0.0f);
              v1161_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[108]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1163_data(0.0f);
              v1163_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1163_data + (v1094_data * v1161_data)).copy_to(ir0 + (6));
              }
              int32_t v1167_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v1172_data(0.0f);
              v1172_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[124]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1174_data(0.0f);
              v1174_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1174_data + (v1094_data * v1172_data)).copy_to(ir0 + (7));
              }
              int32_t v1180_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1184_data(0.0f);
              v1184_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[156_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1185_data(0.0f);
              v1185_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1187_data(0.0f);
              v1187_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1187_data + (v1184_data * v1185_data)).copy_to(ir0 + (0));
              }
              int32_t v1191_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1196_data(0.0f);
              v1196_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1198_data(0.0f);
              v1198_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1198_data + (v1184_data * v1196_data)).copy_to(ir0 + (1));
              }
              int32_t v1202_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1207_data(0.0f);
              v1207_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1209_data(0.0f);
              v1209_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1209_data + (v1184_data * v1207_data)).copy_to(ir0 + (2));
              }
              int32_t v1213_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1218_data(0.0f);
              v1218_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1220_data(0.0f);
              v1220_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1220_data + (v1184_data * v1218_data)).copy_to(ir0 + (3));
              }
              int32_t v1224_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1229_data(0.0f);
              v1229_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1231_data(0.0f);
              v1231_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1231_data + (v1184_data * v1229_data)).copy_to(ir0 + (4));
              }
              int32_t v1235_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1240_data(0.0f);
              v1240_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[93]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1242_data(0.0f);
              v1242_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1242_data + (v1184_data * v1240_data)).copy_to(ir0 + (5));
              }
              int32_t v1246_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1251_data(0.0f);
              v1251_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[109]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1253_data(0.0f);
              v1253_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1253_data + (v1184_data * v1251_data)).copy_to(ir0 + (6));
              }
              int32_t v1257_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v1262_data(0.0f);
              v1262_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[125]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1264_data(0.0f);
              v1264_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1264_data + (v1184_data * v1262_data)).copy_to(ir0 + (7));
              }
              int32_t v1270_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1274_data(0.0f);
              v1274_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[168_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1275_data(0.0f);
              v1275_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1277_data(0.0f);
              v1277_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1277_data + (v1274_data * v1275_data)).copy_to(ir0 + (0));
              }
              int32_t v1281_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1286_data(0.0f);
              v1286_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1288_data(0.0f);
              v1288_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1288_data + (v1274_data * v1286_data)).copy_to(ir0 + (1));
              }
              int32_t v1292_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1297_data(0.0f);
              v1297_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1299_data(0.0f);
              v1299_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1299_data + (v1274_data * v1297_data)).copy_to(ir0 + (2));
              }
              int32_t v1303_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1308_data(0.0f);
              v1308_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1310_data(0.0f);
              v1310_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1310_data + (v1274_data * v1308_data)).copy_to(ir0 + (3));
              }
              int32_t v1314_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1319_data(0.0f);
              v1319_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1321_data(0.0f);
              v1321_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1321_data + (v1274_data * v1319_data)).copy_to(ir0 + (4));
              }
              int32_t v1325_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1330_data(0.0f);
              v1330_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[94]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1332_data(0.0f);
              v1332_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1332_data + (v1274_data * v1330_data)).copy_to(ir0 + (5));
              }
              int32_t v1336_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1341_data(0.0f);
              v1341_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[110]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1343_data(0.0f);
              v1343_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1343_data + (v1274_data * v1341_data)).copy_to(ir0 + (6));
              }
              int32_t v1347_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v1352_data(0.0f);
              v1352_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[126]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1354_data(0.0f);
              v1354_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1354_data + (v1274_data * v1352_data)).copy_to(ir0 + (7));
              }
              int32_t v1360_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1364_data(0.0f);
              v1364_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[180_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1365_data(0.0f);
              v1365_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1367_data(0.0f);
              v1367_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v1367_data + (v1364_data * v1365_data)).copy_to(ir0 + (0));
              }
              int32_t v1371_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1376_data(0.0f);
              v1376_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1378_data(0.0f);
              v1378_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v1378_data + (v1364_data * v1376_data)).copy_to(ir0 + (1));
              }
              int32_t v1382_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1387_data(0.0f);
              v1387_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1389_data(0.0f);
              v1389_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v1389_data + (v1364_data * v1387_data)).copy_to(ir0 + (2));
              }
              int32_t v1393_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1398_data(0.0f);
              v1398_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1400_data(0.0f);
              v1400_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v1400_data + (v1364_data * v1398_data)).copy_to(ir0 + (3));
              }
              int32_t v1404_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1409_data(0.0f);
              v1409_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1411_data(0.0f);
              v1411_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v1411_data + (v1364_data * v1409_data)).copy_to(ir0 + (4));
              }
              int32_t v1415_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1420_data(0.0f);
              v1420_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[95]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1422_data(0.0f);
              v1422_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v1422_data + (v1364_data * v1420_data)).copy_to(ir0 + (5));
              }
              int32_t v1426_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1431_data(0.0f);
              v1431_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[111]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1433_data(0.0f);
              v1433_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v1433_data + (v1364_data * v1431_data)).copy_to(ir0 + (6));
              }
              int32_t v1437_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v1442_data(0.0f);
              v1442_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[127]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1444_data(0.0f);
              v1444_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v1444_data + (v1364_data * v1442_data)).copy_to(ir0 + (7));
              }
              #pragma unroll
              for (int32_t v1448_n1 = 0; v1448_n1 < 8; ++v1448_n1) {
                int32_t v1449_a = 0 + v1448_n1;
                tensorforge::intel_esimd::simd<float, 16> v1451_data(0.0f);
                v1451_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[v1448_n1]), v7_g);
                int32_t v1454_a = v1448_n1 * 12;
                int32_t v1455_a = 0_i32 + v1454_a;
                tensorforge::intel_esimd::simd<float, 16> v1460_data(0.0f);
                v1460_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[v1454_a]), v7_g);
                if (v7_g) {
                  (v1460_data + v1451_data).copy_to(r0 + (v1448_n1));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1465_i1 = 0; v1465_i1 < 8; ++v1465_i1) {
                int32_t v1466_a = 0 + v1465_i1;
                tensorforge::intel_esimd::simd<float, 16> v1468_data(0.0f);
                v1468_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v1465_i1]), v7_g);
                if (v7_g) {
                  v1468_data.copy_to(glb_m0 + ((v1465_i1 * 12)));
                }
              }
            }
          }
        }
      });
    }
  });
}

