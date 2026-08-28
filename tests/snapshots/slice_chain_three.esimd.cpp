// === base name ===
kernel_08703cce1d

// === header ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08703cce1d(const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08703cce1d(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08703cce1d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1536, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[96 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[80];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 144 + 0 + m3_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m1[0, 1])
              *(sycl::vec<float, 2>*)&s0[0 + 0 + 2 * item.get_local_id(0) + 0] = *(sycl::vec<float, 2>*)&glb_m1[0 + 0 + 2 * item.get_local_id(0) + 0];
              if (item.get_local_id(0) < 4) {
                s0[0 + 0 + 1 * item.get_local_id(0) + 32] = glb_m1[0 + 0 + 1 * item.get_local_id(0) + 32];
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r0[6]{};
              // r0 = +(glb_m0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd_mask<16> v7_g = (tensorforge::intel_esimd::simd<int32_t, 16>(0, 1)) < 12;
              int32_t v10_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v14_data(0.0f);
              v14_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[0_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v15_data(0.0f);
              v15_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[0]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v17_data(0.0f);
              v17_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v7_g);
              if (v7_g) {
                (v17_data + (v14_data * v15_data)).copy_to(r0 + (0));
              }
              int32_t v21_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v26_data(0.0f);
              v26_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[6]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v28_data(0.0f);
              v28_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v7_g);
              if (v7_g) {
                (v28_data + (v14_data * v26_data)).copy_to(r0 + (1));
              }
              int32_t v32_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v37_data(0.0f);
              v37_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[12]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v39_data(0.0f);
              v39_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v7_g);
              if (v7_g) {
                (v39_data + (v14_data * v37_data)).copy_to(r0 + (2));
              }
              int32_t v43_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v48_data(0.0f);
              v48_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[18]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v50_data(0.0f);
              v50_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v7_g);
              if (v7_g) {
                (v50_data + (v14_data * v48_data)).copy_to(r0 + (3));
              }
              int32_t v54_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v59_data(0.0f);
              v59_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[24]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v61_data(0.0f);
              v61_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v7_g);
              if (v7_g) {
                (v61_data + (v14_data * v59_data)).copy_to(r0 + (4));
              }
              int32_t v65_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v70_data(0.0f);
              v70_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[30]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v72_data(0.0f);
              v72_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v7_g);
              if (v7_g) {
                (v72_data + (v14_data * v70_data)).copy_to(r0 + (5));
              }
              int32_t v78_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v82_data(0.0f);
              v82_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[12_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v83_data(0.0f);
              v83_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[1]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v85_data(0.0f);
              v85_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v7_g);
              if (v7_g) {
                (v85_data + (v82_data * v83_data)).copy_to(r0 + (0));
              }
              int32_t v89_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v94_data(0.0f);
              v94_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[7]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v96_data(0.0f);
              v96_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v7_g);
              if (v7_g) {
                (v96_data + (v82_data * v94_data)).copy_to(r0 + (1));
              }
              int32_t v100_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v105_data(0.0f);
              v105_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[13]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v107_data(0.0f);
              v107_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v7_g);
              if (v7_g) {
                (v107_data + (v82_data * v105_data)).copy_to(r0 + (2));
              }
              int32_t v111_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v116_data(0.0f);
              v116_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[19]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v118_data(0.0f);
              v118_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v7_g);
              if (v7_g) {
                (v118_data + (v82_data * v116_data)).copy_to(r0 + (3));
              }
              int32_t v122_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v127_data(0.0f);
              v127_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[25]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v129_data(0.0f);
              v129_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v7_g);
              if (v7_g) {
                (v129_data + (v82_data * v127_data)).copy_to(r0 + (4));
              }
              int32_t v133_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v138_data(0.0f);
              v138_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[31]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v140_data(0.0f);
              v140_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v7_g);
              if (v7_g) {
                (v140_data + (v82_data * v138_data)).copy_to(r0 + (5));
              }
              int32_t v146_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v150_data(0.0f);
              v150_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[24_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v151_data(0.0f);
              v151_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[2]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v153_data(0.0f);
              v153_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v7_g);
              if (v7_g) {
                (v153_data + (v150_data * v151_data)).copy_to(r0 + (0));
              }
              int32_t v157_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v162_data(0.0f);
              v162_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[8]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v164_data(0.0f);
              v164_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v7_g);
              if (v7_g) {
                (v164_data + (v150_data * v162_data)).copy_to(r0 + (1));
              }
              int32_t v168_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v173_data(0.0f);
              v173_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[14]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v175_data(0.0f);
              v175_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v7_g);
              if (v7_g) {
                (v175_data + (v150_data * v173_data)).copy_to(r0 + (2));
              }
              int32_t v179_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v184_data(0.0f);
              v184_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[20]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v186_data(0.0f);
              v186_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v7_g);
              if (v7_g) {
                (v186_data + (v150_data * v184_data)).copy_to(r0 + (3));
              }
              int32_t v190_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v195_data(0.0f);
              v195_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[26]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v197_data(0.0f);
              v197_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v7_g);
              if (v7_g) {
                (v197_data + (v150_data * v195_data)).copy_to(r0 + (4));
              }
              int32_t v201_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v206_data(0.0f);
              v206_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v208_data(0.0f);
              v208_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v7_g);
              if (v7_g) {
                (v208_data + (v150_data * v206_data)).copy_to(r0 + (5));
              }
              int32_t v214_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v218_data(0.0f);
              v218_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[36_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v219_data(0.0f);
              v219_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[3]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v221_data(0.0f);
              v221_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v7_g);
              if (v7_g) {
                (v221_data + (v218_data * v219_data)).copy_to(r0 + (0));
              }
              int32_t v225_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v230_data(0.0f);
              v230_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[9]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v232_data(0.0f);
              v232_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v7_g);
              if (v7_g) {
                (v232_data + (v218_data * v230_data)).copy_to(r0 + (1));
              }
              int32_t v236_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v241_data(0.0f);
              v241_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[15]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v243_data(0.0f);
              v243_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v7_g);
              if (v7_g) {
                (v243_data + (v218_data * v241_data)).copy_to(r0 + (2));
              }
              int32_t v247_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v252_data(0.0f);
              v252_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[21]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v254_data(0.0f);
              v254_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v7_g);
              if (v7_g) {
                (v254_data + (v218_data * v252_data)).copy_to(r0 + (3));
              }
              int32_t v258_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v263_data(0.0f);
              v263_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[27]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v265_data(0.0f);
              v265_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v7_g);
              if (v7_g) {
                (v265_data + (v218_data * v263_data)).copy_to(r0 + (4));
              }
              int32_t v269_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v274_data(0.0f);
              v274_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[33]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v276_data(0.0f);
              v276_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v7_g);
              if (v7_g) {
                (v276_data + (v218_data * v274_data)).copy_to(r0 + (5));
              }
              int32_t v282_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v286_data(0.0f);
              v286_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[48_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v287_data(0.0f);
              v287_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[4]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v289_data(0.0f);
              v289_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v7_g);
              if (v7_g) {
                (v289_data + (v286_data * v287_data)).copy_to(r0 + (0));
              }
              int32_t v293_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v298_data(0.0f);
              v298_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[10]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v300_data(0.0f);
              v300_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v7_g);
              if (v7_g) {
                (v300_data + (v286_data * v298_data)).copy_to(r0 + (1));
              }
              int32_t v304_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v309_data(0.0f);
              v309_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[16]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v311_data(0.0f);
              v311_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v7_g);
              if (v7_g) {
                (v311_data + (v286_data * v309_data)).copy_to(r0 + (2));
              }
              int32_t v315_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v320_data(0.0f);
              v320_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[22]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v322_data(0.0f);
              v322_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v7_g);
              if (v7_g) {
                (v322_data + (v286_data * v320_data)).copy_to(r0 + (3));
              }
              int32_t v326_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v331_data(0.0f);
              v331_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[28]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v333_data(0.0f);
              v333_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v7_g);
              if (v7_g) {
                (v333_data + (v286_data * v331_data)).copy_to(r0 + (4));
              }
              int32_t v337_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v342_data(0.0f);
              v342_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[34]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v344_data(0.0f);
              v344_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v7_g);
              if (v7_g) {
                (v344_data + (v286_data * v342_data)).copy_to(r0 + (5));
              }
              int32_t v350_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v354_data(0.0f);
              v354_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m0[60_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v355_data(0.0f);
              v355_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[5]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v357_data(0.0f);
              v357_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[0]), v7_g);
              if (v7_g) {
                (v357_data + (v354_data * v355_data)).copy_to(r0 + (0));
              }
              int32_t v361_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v366_data(0.0f);
              v366_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[11]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v368_data(0.0f);
              v368_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[1]), v7_g);
              if (v7_g) {
                (v368_data + (v354_data * v366_data)).copy_to(r0 + (1));
              }
              int32_t v372_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v377_data(0.0f);
              v377_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[17]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v379_data(0.0f);
              v379_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[2]), v7_g);
              if (v7_g) {
                (v379_data + (v354_data * v377_data)).copy_to(r0 + (2));
              }
              int32_t v383_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v388_data(0.0f);
              v388_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[23]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v390_data(0.0f);
              v390_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[3]), v7_g);
              if (v7_g) {
                (v390_data + (v354_data * v388_data)).copy_to(r0 + (3));
              }
              int32_t v394_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v399_data(0.0f);
              v399_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[29]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v401_data(0.0f);
              v401_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[4]), v7_g);
              if (v7_g) {
                (v401_data + (v354_data * v399_data)).copy_to(r0 + (4));
              }
              int32_t v405_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v410_data(0.0f);
              v410_data.merge(tensorforge::intel_esimd::simd<float, 16>(s0[35]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v412_data(0.0f);
              v412_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[5]), v7_g);
              if (v7_g) {
                (v412_data + (v354_data * v410_data)).copy_to(r0 + (5));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v417_i1 = 0; v417_i1 < 6; ++v417_i1) {
                int32_t v418_a = 0 + v417_i1;
                tensorforge::intel_esimd::simd<float, 16> v420_data(0.0f);
                v420_data.merge(tensorforge::intel_esimd::simd<float, 16>(r0[v417_i1]), v7_g);
                int32_t v424_a = 0_i32 + (v417_i1 * 12);
                if (v7_g) {
                  s1[v424_a] = v420_data;
                }
              }
              float r1[6]{};
              // r1 = +(glb_m3 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir1[6]{};
              int32_t v431_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v435_data(0.0f);
              v435_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[0_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v436_data(0.0f);
              v436_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[0]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v438_data(0.0f);
              v438_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v438_data + (v435_data * v436_data)).copy_to(ir1 + (0));
              }
              int32_t v442_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v447_data(0.0f);
              v447_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[12]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v449_data(0.0f);
              v449_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v449_data + (v435_data * v447_data)).copy_to(ir1 + (1));
              }
              int32_t v453_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v458_data(0.0f);
              v458_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[24]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v460_data(0.0f);
              v460_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v460_data + (v435_data * v458_data)).copy_to(ir1 + (2));
              }
              int32_t v464_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v469_data(0.0f);
              v469_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[36]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v471_data(0.0f);
              v471_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v471_data + (v435_data * v469_data)).copy_to(ir1 + (3));
              }
              int32_t v475_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v480_data(0.0f);
              v480_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[48]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v482_data(0.0f);
              v482_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v482_data + (v435_data * v480_data)).copy_to(ir1 + (4));
              }
              int32_t v486_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v491_data(0.0f);
              v491_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[60]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v493_data(0.0f);
              v493_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v493_data + (v435_data * v491_data)).copy_to(ir1 + (5));
              }
              int32_t v499_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v503_data(0.0f);
              v503_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[12_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v504_data(0.0f);
              v504_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[1]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v506_data(0.0f);
              v506_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v506_data + (v503_data * v504_data)).copy_to(ir1 + (0));
              }
              int32_t v510_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v515_data(0.0f);
              v515_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[13]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v517_data(0.0f);
              v517_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v517_data + (v503_data * v515_data)).copy_to(ir1 + (1));
              }
              int32_t v521_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v526_data(0.0f);
              v526_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[25]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v528_data(0.0f);
              v528_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v528_data + (v503_data * v526_data)).copy_to(ir1 + (2));
              }
              int32_t v532_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v537_data(0.0f);
              v537_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[37]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v539_data(0.0f);
              v539_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v539_data + (v503_data * v537_data)).copy_to(ir1 + (3));
              }
              int32_t v543_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v548_data(0.0f);
              v548_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[49]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v550_data(0.0f);
              v550_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v550_data + (v503_data * v548_data)).copy_to(ir1 + (4));
              }
              int32_t v554_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v559_data(0.0f);
              v559_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[61]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v561_data(0.0f);
              v561_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v561_data + (v503_data * v559_data)).copy_to(ir1 + (5));
              }
              int32_t v567_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v571_data(0.0f);
              v571_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[24_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v572_data(0.0f);
              v572_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[2]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v574_data(0.0f);
              v574_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v574_data + (v571_data * v572_data)).copy_to(ir1 + (0));
              }
              int32_t v578_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v583_data(0.0f);
              v583_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[14]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v585_data(0.0f);
              v585_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v585_data + (v571_data * v583_data)).copy_to(ir1 + (1));
              }
              int32_t v589_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v594_data(0.0f);
              v594_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[26]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v596_data(0.0f);
              v596_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v596_data + (v571_data * v594_data)).copy_to(ir1 + (2));
              }
              int32_t v600_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v605_data(0.0f);
              v605_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[38]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v607_data(0.0f);
              v607_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v607_data + (v571_data * v605_data)).copy_to(ir1 + (3));
              }
              int32_t v611_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v616_data(0.0f);
              v616_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[50]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v618_data(0.0f);
              v618_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v618_data + (v571_data * v616_data)).copy_to(ir1 + (4));
              }
              int32_t v622_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v627_data(0.0f);
              v627_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[62]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v629_data(0.0f);
              v629_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v629_data + (v571_data * v627_data)).copy_to(ir1 + (5));
              }
              int32_t v635_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v639_data(0.0f);
              v639_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[36_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v640_data(0.0f);
              v640_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[3]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v642_data(0.0f);
              v642_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v642_data + (v639_data * v640_data)).copy_to(ir1 + (0));
              }
              int32_t v646_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v651_data(0.0f);
              v651_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[15]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v653_data(0.0f);
              v653_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v653_data + (v639_data * v651_data)).copy_to(ir1 + (1));
              }
              int32_t v657_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v662_data(0.0f);
              v662_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[27]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v664_data(0.0f);
              v664_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v664_data + (v639_data * v662_data)).copy_to(ir1 + (2));
              }
              int32_t v668_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v673_data(0.0f);
              v673_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[39]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v675_data(0.0f);
              v675_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v675_data + (v639_data * v673_data)).copy_to(ir1 + (3));
              }
              int32_t v679_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v684_data(0.0f);
              v684_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[51]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v686_data(0.0f);
              v686_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v686_data + (v639_data * v684_data)).copy_to(ir1 + (4));
              }
              int32_t v690_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v695_data(0.0f);
              v695_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[63]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v697_data(0.0f);
              v697_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v697_data + (v639_data * v695_data)).copy_to(ir1 + (5));
              }
              int32_t v703_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v707_data(0.0f);
              v707_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[48_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v708_data(0.0f);
              v708_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[4]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v710_data(0.0f);
              v710_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v710_data + (v707_data * v708_data)).copy_to(ir1 + (0));
              }
              int32_t v714_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v719_data(0.0f);
              v719_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[16]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v721_data(0.0f);
              v721_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v721_data + (v707_data * v719_data)).copy_to(ir1 + (1));
              }
              int32_t v725_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v730_data(0.0f);
              v730_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[28]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v732_data(0.0f);
              v732_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v732_data + (v707_data * v730_data)).copy_to(ir1 + (2));
              }
              int32_t v736_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v741_data(0.0f);
              v741_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[40]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v743_data(0.0f);
              v743_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v743_data + (v707_data * v741_data)).copy_to(ir1 + (3));
              }
              int32_t v747_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v752_data(0.0f);
              v752_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[52]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v754_data(0.0f);
              v754_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v754_data + (v707_data * v752_data)).copy_to(ir1 + (4));
              }
              int32_t v758_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v763_data(0.0f);
              v763_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[64]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v765_data(0.0f);
              v765_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v765_data + (v707_data * v763_data)).copy_to(ir1 + (5));
              }
              int32_t v771_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v775_data(0.0f);
              v775_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[60_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v776_data(0.0f);
              v776_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[5]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v778_data(0.0f);
              v778_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v778_data + (v775_data * v776_data)).copy_to(ir1 + (0));
              }
              int32_t v782_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v787_data(0.0f);
              v787_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[17]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v789_data(0.0f);
              v789_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v789_data + (v775_data * v787_data)).copy_to(ir1 + (1));
              }
              int32_t v793_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v798_data(0.0f);
              v798_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[29]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v800_data(0.0f);
              v800_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v800_data + (v775_data * v798_data)).copy_to(ir1 + (2));
              }
              int32_t v804_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v809_data(0.0f);
              v809_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[41]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v811_data(0.0f);
              v811_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v811_data + (v775_data * v809_data)).copy_to(ir1 + (3));
              }
              int32_t v815_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v820_data(0.0f);
              v820_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[53]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v822_data(0.0f);
              v822_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v822_data + (v775_data * v820_data)).copy_to(ir1 + (4));
              }
              int32_t v826_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v831_data(0.0f);
              v831_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[65]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v833_data(0.0f);
              v833_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v833_data + (v775_data * v831_data)).copy_to(ir1 + (5));
              }
              int32_t v839_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v843_data(0.0f);
              v843_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[72_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v844_data(0.0f);
              v844_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[6]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v846_data(0.0f);
              v846_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v846_data + (v843_data * v844_data)).copy_to(ir1 + (0));
              }
              int32_t v850_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v855_data(0.0f);
              v855_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[18]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v857_data(0.0f);
              v857_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v857_data + (v843_data * v855_data)).copy_to(ir1 + (1));
              }
              int32_t v861_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v866_data(0.0f);
              v866_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[30]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v868_data(0.0f);
              v868_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v868_data + (v843_data * v866_data)).copy_to(ir1 + (2));
              }
              int32_t v872_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v877_data(0.0f);
              v877_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[42]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v879_data(0.0f);
              v879_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v879_data + (v843_data * v877_data)).copy_to(ir1 + (3));
              }
              int32_t v883_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v888_data(0.0f);
              v888_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[54]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v890_data(0.0f);
              v890_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v890_data + (v843_data * v888_data)).copy_to(ir1 + (4));
              }
              int32_t v894_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v899_data(0.0f);
              v899_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[66]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v901_data(0.0f);
              v901_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v901_data + (v843_data * v899_data)).copy_to(ir1 + (5));
              }
              int32_t v907_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v911_data(0.0f);
              v911_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[84_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v912_data(0.0f);
              v912_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[7]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v914_data(0.0f);
              v914_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v914_data + (v911_data * v912_data)).copy_to(ir1 + (0));
              }
              int32_t v918_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v923_data(0.0f);
              v923_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[19]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v925_data(0.0f);
              v925_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v925_data + (v911_data * v923_data)).copy_to(ir1 + (1));
              }
              int32_t v929_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v934_data(0.0f);
              v934_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[31]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v936_data(0.0f);
              v936_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v936_data + (v911_data * v934_data)).copy_to(ir1 + (2));
              }
              int32_t v940_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v945_data(0.0f);
              v945_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[43]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v947_data(0.0f);
              v947_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v947_data + (v911_data * v945_data)).copy_to(ir1 + (3));
              }
              int32_t v951_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v956_data(0.0f);
              v956_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[55]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v958_data(0.0f);
              v958_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v958_data + (v911_data * v956_data)).copy_to(ir1 + (4));
              }
              int32_t v962_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v967_data(0.0f);
              v967_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[67]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v969_data(0.0f);
              v969_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v969_data + (v911_data * v967_data)).copy_to(ir1 + (5));
              }
              int32_t v975_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v979_data(0.0f);
              v979_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[96_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v980_data(0.0f);
              v980_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[8]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v982_data(0.0f);
              v982_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v982_data + (v979_data * v980_data)).copy_to(ir1 + (0));
              }
              int32_t v986_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v991_data(0.0f);
              v991_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[20]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v993_data(0.0f);
              v993_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v993_data + (v979_data * v991_data)).copy_to(ir1 + (1));
              }
              int32_t v997_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1002_data(0.0f);
              v1002_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1004_data(0.0f);
              v1004_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v1004_data + (v979_data * v1002_data)).copy_to(ir1 + (2));
              }
              int32_t v1008_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1013_data(0.0f);
              v1013_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[44]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1015_data(0.0f);
              v1015_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v1015_data + (v979_data * v1013_data)).copy_to(ir1 + (3));
              }
              int32_t v1019_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1024_data(0.0f);
              v1024_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[56]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1026_data(0.0f);
              v1026_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v1026_data + (v979_data * v1024_data)).copy_to(ir1 + (4));
              }
              int32_t v1030_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v1035_data(0.0f);
              v1035_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[68]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1037_data(0.0f);
              v1037_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v1037_data + (v979_data * v1035_data)).copy_to(ir1 + (5));
              }
              int32_t v1043_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1047_data(0.0f);
              v1047_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[108_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1048_data(0.0f);
              v1048_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[9]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1050_data(0.0f);
              v1050_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v1050_data + (v1047_data * v1048_data)).copy_to(ir1 + (0));
              }
              int32_t v1054_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1059_data(0.0f);
              v1059_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[21]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1061_data(0.0f);
              v1061_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v1061_data + (v1047_data * v1059_data)).copy_to(ir1 + (1));
              }
              int32_t v1065_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1070_data(0.0f);
              v1070_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[33]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1072_data(0.0f);
              v1072_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v1072_data + (v1047_data * v1070_data)).copy_to(ir1 + (2));
              }
              int32_t v1076_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1081_data(0.0f);
              v1081_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[45]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1083_data(0.0f);
              v1083_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v1083_data + (v1047_data * v1081_data)).copy_to(ir1 + (3));
              }
              int32_t v1087_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1092_data(0.0f);
              v1092_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[57]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1094_data(0.0f);
              v1094_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v1094_data + (v1047_data * v1092_data)).copy_to(ir1 + (4));
              }
              int32_t v1098_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v1103_data(0.0f);
              v1103_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[69]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1105_data(0.0f);
              v1105_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v1105_data + (v1047_data * v1103_data)).copy_to(ir1 + (5));
              }
              int32_t v1111_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1115_data(0.0f);
              v1115_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[120_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1116_data(0.0f);
              v1116_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[10]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1118_data(0.0f);
              v1118_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v1118_data + (v1115_data * v1116_data)).copy_to(ir1 + (0));
              }
              int32_t v1122_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1127_data(0.0f);
              v1127_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[22]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1129_data(0.0f);
              v1129_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v1129_data + (v1115_data * v1127_data)).copy_to(ir1 + (1));
              }
              int32_t v1133_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1138_data(0.0f);
              v1138_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[34]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1140_data(0.0f);
              v1140_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v1140_data + (v1115_data * v1138_data)).copy_to(ir1 + (2));
              }
              int32_t v1144_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1149_data(0.0f);
              v1149_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[46]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1151_data(0.0f);
              v1151_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v1151_data + (v1115_data * v1149_data)).copy_to(ir1 + (3));
              }
              int32_t v1155_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1160_data(0.0f);
              v1160_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[58]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1162_data(0.0f);
              v1162_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v1162_data + (v1115_data * v1160_data)).copy_to(ir1 + (4));
              }
              int32_t v1166_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v1171_data(0.0f);
              v1171_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[70]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1173_data(0.0f);
              v1173_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v1173_data + (v1115_data * v1171_data)).copy_to(ir1 + (5));
              }
              int32_t v1179_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1183_data(0.0f);
              v1183_data.merge(tensorforge::intel_esimd::simd<float, 16>(glb_m3[132_i32]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1184_data(0.0f);
              v1184_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[11]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1186_data(0.0f);
              v1186_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[0]), v7_g);
              if (v7_g) {
                (v1186_data + (v1183_data * v1184_data)).copy_to(ir1 + (0));
              }
              int32_t v1190_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1195_data(0.0f);
              v1195_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[23]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1197_data(0.0f);
              v1197_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[1]), v7_g);
              if (v7_g) {
                (v1197_data + (v1183_data * v1195_data)).copy_to(ir1 + (1));
              }
              int32_t v1201_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1206_data(0.0f);
              v1206_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[35]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1208_data(0.0f);
              v1208_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[2]), v7_g);
              if (v7_g) {
                (v1208_data + (v1183_data * v1206_data)).copy_to(ir1 + (2));
              }
              int32_t v1212_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1217_data(0.0f);
              v1217_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[47]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1219_data(0.0f);
              v1219_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[3]), v7_g);
              if (v7_g) {
                (v1219_data + (v1183_data * v1217_data)).copy_to(ir1 + (3));
              }
              int32_t v1223_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1228_data(0.0f);
              v1228_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[59]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1230_data(0.0f);
              v1230_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[4]), v7_g);
              if (v7_g) {
                (v1230_data + (v1183_data * v1228_data)).copy_to(ir1 + (4));
              }
              int32_t v1234_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v1239_data(0.0f);
              v1239_data.merge(tensorforge::intel_esimd::simd<float, 16>(s1[71]), v7_g);
              tensorforge::intel_esimd::simd<float, 16> v1241_data(0.0f);
              v1241_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[5]), v7_g);
              if (v7_g) {
                (v1241_data + (v1183_data * v1239_data)).copy_to(ir1 + (5));
              }
              #pragma unroll
              for (int32_t v1245_n1 = 0; v1245_n1 < 6; ++v1245_n1) {
                int32_t v1246_a = 0 + v1245_n1;
                tensorforge::intel_esimd::simd<float, 16> v1248_data(0.0f);
                v1248_data.merge(tensorforge::intel_esimd::simd<float, 16>(ir1[v1245_n1]), v7_g);
                if (v7_g) {
                  v1248_data.copy_to(r1 + (v1245_n1));
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1252_i1 = 0; v1252_i1 < 6; ++v1252_i1) {
                int32_t v1253_a = 0 + v1252_i1;
                tensorforge::intel_esimd::simd<float, 16> v1255_data(0.0f);
                v1255_data.merge(tensorforge::intel_esimd::simd<float, 16>(r1[v1252_i1]), v7_g);
                if (v7_g) {
                  v1255_data.copy_to(glb_m2 + ((v1252_i1 * 12)));
                }
              }
            }
          }
        }
      });
    }
  });
}

