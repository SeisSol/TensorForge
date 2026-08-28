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
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 81 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              s0[0 + 0 + 1 * item.get_local_id(0) + 64] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 64];
              if (item.get_local_id(0) < 1) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 80] = glb_m2[0 + 0 + 1 * item.get_local_id(0) + 80];
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[9]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir0[9]{};
              tensorforge::intel_esimd::simd_mask<16> v7_g = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 9;
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
              v26_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v28_data(0.0f);
              v28_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v28_data + (v14_data * v26_data)).copy_to(ir0 + (1));
              }
              int32_t v32_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v37_data(0.0f);
              v37_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v39_data(0.0f);
              v39_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v39_data + (v14_data * v37_data)).copy_to(ir0 + (2));
              }
              int32_t v43_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v48_data(0.0f);
              v48_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v50_data(0.0f);
              v50_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v50_data + (v14_data * v48_data)).copy_to(ir0 + (3));
              }
              int32_t v54_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v59_data(0.0f);
              v59_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[36]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v61_data(0.0f);
              v61_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v61_data + (v14_data * v59_data)).copy_to(ir0 + (4));
              }
              int32_t v65_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v70_data(0.0f);
              v70_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[45]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v72_data(0.0f);
              v72_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v72_data + (v14_data * v70_data)).copy_to(ir0 + (5));
              }
              int32_t v76_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v81_data(0.0f);
              v81_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[54]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v83_data(0.0f);
              v83_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v83_data + (v14_data * v81_data)).copy_to(ir0 + (6));
              }
              int32_t v87_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v92_data(0.0f);
              v92_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[63]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v94_data(0.0f);
              v94_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v94_data + (v14_data * v92_data)).copy_to(ir0 + (7));
              }
              int32_t v98_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v103_data(0.0f);
              v103_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[72]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v105_data(0.0f);
              v105_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v105_data + (v14_data * v103_data)).copy_to(ir0 + (8));
              }
              int32_t v111_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v115_data(0.0f);
              v115_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[9_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v116_data(0.0f);
              v116_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v118_data(0.0f);
              v118_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v118_data + (v115_data * v116_data)).copy_to(ir0 + (0));
              }
              int32_t v122_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v127_data(0.0f);
              v127_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v129_data(0.0f);
              v129_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v129_data + (v115_data * v127_data)).copy_to(ir0 + (1));
              }
              int32_t v133_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v138_data(0.0f);
              v138_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v140_data(0.0f);
              v140_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v140_data + (v115_data * v138_data)).copy_to(ir0 + (2));
              }
              int32_t v144_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v149_data(0.0f);
              v149_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v151_data(0.0f);
              v151_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v151_data + (v115_data * v149_data)).copy_to(ir0 + (3));
              }
              int32_t v155_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v160_data(0.0f);
              v160_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[37]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v162_data(0.0f);
              v162_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v162_data + (v115_data * v160_data)).copy_to(ir0 + (4));
              }
              int32_t v166_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v171_data(0.0f);
              v171_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[46]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v173_data(0.0f);
              v173_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v173_data + (v115_data * v171_data)).copy_to(ir0 + (5));
              }
              int32_t v177_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v182_data(0.0f);
              v182_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[55]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v184_data(0.0f);
              v184_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v184_data + (v115_data * v182_data)).copy_to(ir0 + (6));
              }
              int32_t v188_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v193_data(0.0f);
              v193_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[64]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v195_data(0.0f);
              v195_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v195_data + (v115_data * v193_data)).copy_to(ir0 + (7));
              }
              int32_t v199_a = 0_i32 + 9;
              tensorforge::intel_esimd::simd<float, 16> v204_data(0.0f);
              v204_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[73]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v206_data(0.0f);
              v206_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v206_data + (v115_data * v204_data)).copy_to(ir0 + (8));
              }
              int32_t v212_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v216_data(0.0f);
              v216_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[18_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v217_data(0.0f);
              v217_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v219_data(0.0f);
              v219_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v219_data + (v216_data * v217_data)).copy_to(ir0 + (0));
              }
              int32_t v223_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v228_data(0.0f);
              v228_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v230_data(0.0f);
              v230_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v230_data + (v216_data * v228_data)).copy_to(ir0 + (1));
              }
              int32_t v234_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v239_data(0.0f);
              v239_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v241_data(0.0f);
              v241_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v241_data + (v216_data * v239_data)).copy_to(ir0 + (2));
              }
              int32_t v245_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v250_data(0.0f);
              v250_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v252_data(0.0f);
              v252_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v252_data + (v216_data * v250_data)).copy_to(ir0 + (3));
              }
              int32_t v256_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v261_data(0.0f);
              v261_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[38]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v263_data(0.0f);
              v263_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v263_data + (v216_data * v261_data)).copy_to(ir0 + (4));
              }
              int32_t v267_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v272_data(0.0f);
              v272_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[47]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v274_data(0.0f);
              v274_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v274_data + (v216_data * v272_data)).copy_to(ir0 + (5));
              }
              int32_t v278_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v283_data(0.0f);
              v283_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[56]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v285_data(0.0f);
              v285_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v285_data + (v216_data * v283_data)).copy_to(ir0 + (6));
              }
              int32_t v289_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v294_data(0.0f);
              v294_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[65]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v296_data(0.0f);
              v296_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v296_data + (v216_data * v294_data)).copy_to(ir0 + (7));
              }
              int32_t v300_a = 0_i32 + 18;
              tensorforge::intel_esimd::simd<float, 16> v305_data(0.0f);
              v305_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[74]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v307_data(0.0f);
              v307_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v307_data + (v216_data * v305_data)).copy_to(ir0 + (8));
              }
              int32_t v313_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v317_data(0.0f);
              v317_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[27_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v318_data(0.0f);
              v318_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v320_data(0.0f);
              v320_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v320_data + (v317_data * v318_data)).copy_to(ir0 + (0));
              }
              int32_t v324_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v329_data(0.0f);
              v329_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v331_data(0.0f);
              v331_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v331_data + (v317_data * v329_data)).copy_to(ir0 + (1));
              }
              int32_t v335_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v340_data(0.0f);
              v340_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v342_data(0.0f);
              v342_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v342_data + (v317_data * v340_data)).copy_to(ir0 + (2));
              }
              int32_t v346_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v351_data(0.0f);
              v351_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v353_data(0.0f);
              v353_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v353_data + (v317_data * v351_data)).copy_to(ir0 + (3));
              }
              int32_t v357_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v362_data(0.0f);
              v362_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[39]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v364_data(0.0f);
              v364_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v364_data + (v317_data * v362_data)).copy_to(ir0 + (4));
              }
              int32_t v368_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v373_data(0.0f);
              v373_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[48]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v375_data(0.0f);
              v375_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v375_data + (v317_data * v373_data)).copy_to(ir0 + (5));
              }
              int32_t v379_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v384_data(0.0f);
              v384_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[57]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v386_data(0.0f);
              v386_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v386_data + (v317_data * v384_data)).copy_to(ir0 + (6));
              }
              int32_t v390_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v395_data(0.0f);
              v395_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[66]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v397_data(0.0f);
              v397_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v397_data + (v317_data * v395_data)).copy_to(ir0 + (7));
              }
              int32_t v401_a = 0_i32 + 27;
              tensorforge::intel_esimd::simd<float, 16> v406_data(0.0f);
              v406_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[75]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v408_data(0.0f);
              v408_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v408_data + (v317_data * v406_data)).copy_to(ir0 + (8));
              }
              int32_t v414_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v418_data(0.0f);
              v418_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[36_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v419_data(0.0f);
              v419_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v421_data(0.0f);
              v421_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v421_data + (v418_data * v419_data)).copy_to(ir0 + (0));
              }
              int32_t v425_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v430_data(0.0f);
              v430_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v432_data(0.0f);
              v432_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v432_data + (v418_data * v430_data)).copy_to(ir0 + (1));
              }
              int32_t v436_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v441_data(0.0f);
              v441_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v443_data(0.0f);
              v443_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v443_data + (v418_data * v441_data)).copy_to(ir0 + (2));
              }
              int32_t v447_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v452_data(0.0f);
              v452_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v454_data(0.0f);
              v454_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v454_data + (v418_data * v452_data)).copy_to(ir0 + (3));
              }
              int32_t v458_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v463_data(0.0f);
              v463_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[40]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v465_data(0.0f);
              v465_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v465_data + (v418_data * v463_data)).copy_to(ir0 + (4));
              }
              int32_t v469_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v474_data(0.0f);
              v474_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[49]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v476_data(0.0f);
              v476_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v476_data + (v418_data * v474_data)).copy_to(ir0 + (5));
              }
              int32_t v480_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v485_data(0.0f);
              v485_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[58]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v487_data(0.0f);
              v487_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v487_data + (v418_data * v485_data)).copy_to(ir0 + (6));
              }
              int32_t v491_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v496_data(0.0f);
              v496_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[67]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v498_data(0.0f);
              v498_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v498_data + (v418_data * v496_data)).copy_to(ir0 + (7));
              }
              int32_t v502_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v507_data(0.0f);
              v507_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[76]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v509_data(0.0f);
              v509_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v509_data + (v418_data * v507_data)).copy_to(ir0 + (8));
              }
              int32_t v515_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v519_data(0.0f);
              v519_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[45_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v520_data(0.0f);
              v520_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v522_data(0.0f);
              v522_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v522_data + (v519_data * v520_data)).copy_to(ir0 + (0));
              }
              int32_t v526_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v531_data(0.0f);
              v531_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v533_data(0.0f);
              v533_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v533_data + (v519_data * v531_data)).copy_to(ir0 + (1));
              }
              int32_t v537_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v542_data(0.0f);
              v542_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v544_data(0.0f);
              v544_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v544_data + (v519_data * v542_data)).copy_to(ir0 + (2));
              }
              int32_t v548_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v553_data(0.0f);
              v553_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v555_data(0.0f);
              v555_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v555_data + (v519_data * v553_data)).copy_to(ir0 + (3));
              }
              int32_t v559_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v564_data(0.0f);
              v564_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[41]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v566_data(0.0f);
              v566_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v566_data + (v519_data * v564_data)).copy_to(ir0 + (4));
              }
              int32_t v570_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v575_data(0.0f);
              v575_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[50]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v577_data(0.0f);
              v577_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v577_data + (v519_data * v575_data)).copy_to(ir0 + (5));
              }
              int32_t v581_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v586_data(0.0f);
              v586_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[59]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v588_data(0.0f);
              v588_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v588_data + (v519_data * v586_data)).copy_to(ir0 + (6));
              }
              int32_t v592_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v597_data(0.0f);
              v597_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[68]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v599_data(0.0f);
              v599_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v599_data + (v519_data * v597_data)).copy_to(ir0 + (7));
              }
              int32_t v603_a = 0_i32 + 45;
              tensorforge::intel_esimd::simd<float, 16> v608_data(0.0f);
              v608_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[77]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v610_data(0.0f);
              v610_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v610_data + (v519_data * v608_data)).copy_to(ir0 + (8));
              }
              int32_t v616_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v620_data(0.0f);
              v620_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[54_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v621_data(0.0f);
              v621_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v623_data(0.0f);
              v623_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v623_data + (v620_data * v621_data)).copy_to(ir0 + (0));
              }
              int32_t v627_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v632_data(0.0f);
              v632_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v634_data(0.0f);
              v634_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v634_data + (v620_data * v632_data)).copy_to(ir0 + (1));
              }
              int32_t v638_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v643_data(0.0f);
              v643_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v645_data(0.0f);
              v645_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v645_data + (v620_data * v643_data)).copy_to(ir0 + (2));
              }
              int32_t v649_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v654_data(0.0f);
              v654_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v656_data(0.0f);
              v656_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v656_data + (v620_data * v654_data)).copy_to(ir0 + (3));
              }
              int32_t v660_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v665_data(0.0f);
              v665_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[42]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v667_data(0.0f);
              v667_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v667_data + (v620_data * v665_data)).copy_to(ir0 + (4));
              }
              int32_t v671_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v676_data(0.0f);
              v676_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[51]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v678_data(0.0f);
              v678_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v678_data + (v620_data * v676_data)).copy_to(ir0 + (5));
              }
              int32_t v682_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v687_data(0.0f);
              v687_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[60]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v689_data(0.0f);
              v689_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v689_data + (v620_data * v687_data)).copy_to(ir0 + (6));
              }
              int32_t v693_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v698_data(0.0f);
              v698_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[69]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v700_data(0.0f);
              v700_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v700_data + (v620_data * v698_data)).copy_to(ir0 + (7));
              }
              int32_t v704_a = 0_i32 + 54;
              tensorforge::intel_esimd::simd<float, 16> v709_data(0.0f);
              v709_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[78]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v711_data(0.0f);
              v711_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v711_data + (v620_data * v709_data)).copy_to(ir0 + (8));
              }
              int32_t v717_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v721_data(0.0f);
              v721_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[63_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v722_data(0.0f);
              v722_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v724_data(0.0f);
              v724_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v724_data + (v721_data * v722_data)).copy_to(ir0 + (0));
              }
              int32_t v728_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v733_data(0.0f);
              v733_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v735_data(0.0f);
              v735_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v735_data + (v721_data * v733_data)).copy_to(ir0 + (1));
              }
              int32_t v739_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v744_data(0.0f);
              v744_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v746_data(0.0f);
              v746_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v746_data + (v721_data * v744_data)).copy_to(ir0 + (2));
              }
              int32_t v750_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v755_data(0.0f);
              v755_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v757_data(0.0f);
              v757_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v757_data + (v721_data * v755_data)).copy_to(ir0 + (3));
              }
              int32_t v761_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v766_data(0.0f);
              v766_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[43]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v768_data(0.0f);
              v768_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v768_data + (v721_data * v766_data)).copy_to(ir0 + (4));
              }
              int32_t v772_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v777_data(0.0f);
              v777_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[52]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v779_data(0.0f);
              v779_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v779_data + (v721_data * v777_data)).copy_to(ir0 + (5));
              }
              int32_t v783_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v788_data(0.0f);
              v788_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[61]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v790_data(0.0f);
              v790_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v790_data + (v721_data * v788_data)).copy_to(ir0 + (6));
              }
              int32_t v794_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v799_data(0.0f);
              v799_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[70]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v801_data(0.0f);
              v801_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v801_data + (v721_data * v799_data)).copy_to(ir0 + (7));
              }
              int32_t v805_a = 0_i32 + 63;
              tensorforge::intel_esimd::simd<float, 16> v810_data(0.0f);
              v810_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[79]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v812_data(0.0f);
              v812_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v812_data + (v721_data * v810_data)).copy_to(ir0 + (8));
              }
              int32_t v818_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v822_data(0.0f);
              v822_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m1[72_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v823_data(0.0f);
              v823_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v825_data(0.0f);
              v825_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[0]), v7_g);
              if (v7_g) {
                (v825_data + (v822_data * v823_data)).copy_to(ir0 + (0));
              }
              int32_t v829_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v834_data(0.0f);
              v834_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v836_data(0.0f);
              v836_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[1]), v7_g);
              if (v7_g) {
                (v836_data + (v822_data * v834_data)).copy_to(ir0 + (1));
              }
              int32_t v840_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v845_data(0.0f);
              v845_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v847_data(0.0f);
              v847_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[2]), v7_g);
              if (v7_g) {
                (v847_data + (v822_data * v845_data)).copy_to(ir0 + (2));
              }
              int32_t v851_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v856_data(0.0f);
              v856_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v858_data(0.0f);
              v858_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[3]), v7_g);
              if (v7_g) {
                (v858_data + (v822_data * v856_data)).copy_to(ir0 + (3));
              }
              int32_t v862_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v867_data(0.0f);
              v867_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[44]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v869_data(0.0f);
              v869_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[4]), v7_g);
              if (v7_g) {
                (v869_data + (v822_data * v867_data)).copy_to(ir0 + (4));
              }
              int32_t v873_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v878_data(0.0f);
              v878_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[53]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v880_data(0.0f);
              v880_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[5]), v7_g);
              if (v7_g) {
                (v880_data + (v822_data * v878_data)).copy_to(ir0 + (5));
              }
              int32_t v884_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v889_data(0.0f);
              v889_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[62]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v891_data(0.0f);
              v891_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[6]), v7_g);
              if (v7_g) {
                (v891_data + (v822_data * v889_data)).copy_to(ir0 + (6));
              }
              int32_t v895_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v900_data(0.0f);
              v900_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[71]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v902_data(0.0f);
              v902_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[7]), v7_g);
              if (v7_g) {
                (v902_data + (v822_data * v900_data)).copy_to(ir0 + (7));
              }
              int32_t v906_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v911_data(0.0f);
              v911_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[80]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v913_data(0.0f);
              v913_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[8]), v7_g);
              if (v7_g) {
                (v913_data + (v822_data * v911_data)).copy_to(ir0 + (8));
              }
              #pragma unroll
              for (int32_t v918_n1 = 0; v918_n1 < 9; ++v918_n1) {
                int32_t v919_a = 0 + v918_n1;
                tensorforge::intel_esimd::simd<float, 16> v921_data(0.0f);
                v921_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir0[v918_n1]), v7_g);
                if (v7_g) {
                  (v921_data * 13.0f).copy_to(r0 + (v918_n1));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v926_i1 = 0; v926_i1 < 9; ++v926_i1) {
                int32_t v927_a = 0 + v926_i1;
                tensorforge::intel_esimd::simd<float, 16> v929_data(0.0f);
                v929_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v926_i1]), v7_g);
                if (v7_g) {
                  v929_data.copy_to(glb_m0 + ((v926_i1 * 9)));
                }
              }
            }
          }
        }
      });
    }
  });
}

