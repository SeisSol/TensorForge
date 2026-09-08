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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
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
              float r0[192]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v12_i1 = 0; v12_i1 < 12; ++v12_i1) {
                tensorforge::intel_esimd::simd<float, 12> v17_data;
                v17_data.copy_from(glb_m1 + ((v12_i1 * 12)));
                v17_data.copy_to(r0 + ((v12_i1 * 16)));
              }
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v21_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v22_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 12> v29_data;
                v29_data.copy_from(glb_m3 + ((v24_i1 * 12)));
                v29_data.copy_to(r2 + ((v24_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v46_acc{};
              tensorforge::intel_esimd::simd<float, 16> v47_lin;
              v47_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v48_bc = v47_lin[0];
              v46_acc += (v48_bc * v34_data);
              float v50_bc = v47_lin[1];
              v46_acc += (v50_bc * v35_data);
              float v52_bc = v47_lin[2];
              v46_acc += (v52_bc * v36_data);
              float v54_bc = v47_lin[3];
              v46_acc += (v54_bc * v37_data);
              float v56_bc = v47_lin[4];
              v46_acc += (v56_bc * v38_data);
              float v58_bc = v47_lin[5];
              v46_acc += (v58_bc * v39_data);
              float v60_bc = v47_lin[6];
              v46_acc += (v60_bc * v40_data);
              float v62_bc = v47_lin[7];
              v46_acc += (v62_bc * v41_data);
              float v64_bc = v47_lin[8];
              v46_acc += (v64_bc * v42_data);
              float v66_bc = v47_lin[9];
              v46_acc += (v66_bc * v43_data);
              float v68_bc = v47_lin[10];
              v46_acc += (v68_bc * v44_data);
              float v70_bc = v47_lin[11];
              v46_acc += (v70_bc * v45_data);
              v46_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v72_acc{};
              v72_acc += (v48_bc * v34_data);
              v72_acc += (v50_bc * v35_data);
              v72_acc += (v52_bc * v36_data);
              v72_acc += (v54_bc * v37_data);
              v72_acc += (v56_bc * v38_data);
              v72_acc += (v58_bc * v39_data);
              v72_acc += (v60_bc * v40_data);
              v72_acc += (v62_bc * v41_data);
              v72_acc += (v64_bc * v42_data);
              v72_acc += (v66_bc * v43_data);
              v72_acc += (v68_bc * v44_data);
              v72_acc += (v70_bc * v45_data);
              v72_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v98_acc{};
              v98_acc += (v48_bc * v34_data);
              v98_acc += (v50_bc * v35_data);
              v98_acc += (v52_bc * v36_data);
              v98_acc += (v54_bc * v37_data);
              v98_acc += (v56_bc * v38_data);
              v98_acc += (v58_bc * v39_data);
              v98_acc += (v60_bc * v40_data);
              v98_acc += (v62_bc * v41_data);
              v98_acc += (v64_bc * v42_data);
              v98_acc += (v66_bc * v43_data);
              v98_acc += (v68_bc * v44_data);
              v98_acc += (v70_bc * v45_data);
              v98_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v124_acc{};
              v124_acc += (v48_bc * v34_data);
              v124_acc += (v50_bc * v35_data);
              v124_acc += (v52_bc * v36_data);
              v124_acc += (v54_bc * v37_data);
              v124_acc += (v56_bc * v38_data);
              v124_acc += (v58_bc * v39_data);
              v124_acc += (v60_bc * v40_data);
              v124_acc += (v62_bc * v41_data);
              v124_acc += (v64_bc * v42_data);
              v124_acc += (v66_bc * v43_data);
              v124_acc += (v68_bc * v44_data);
              v124_acc += (v70_bc * v45_data);
              v124_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v150_acc{};
              v150_acc += (v48_bc * v34_data);
              v150_acc += (v50_bc * v35_data);
              v150_acc += (v52_bc * v36_data);
              v150_acc += (v54_bc * v37_data);
              v150_acc += (v56_bc * v38_data);
              v150_acc += (v58_bc * v39_data);
              v150_acc += (v60_bc * v40_data);
              v150_acc += (v62_bc * v41_data);
              v150_acc += (v64_bc * v42_data);
              v150_acc += (v66_bc * v43_data);
              v150_acc += (v68_bc * v44_data);
              v150_acc += (v70_bc * v45_data);
              v150_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v176_acc{};
              v176_acc += (v48_bc * v34_data);
              v176_acc += (v50_bc * v35_data);
              v176_acc += (v52_bc * v36_data);
              v176_acc += (v54_bc * v37_data);
              v176_acc += (v56_bc * v38_data);
              v176_acc += (v58_bc * v39_data);
              v176_acc += (v60_bc * v40_data);
              v176_acc += (v62_bc * v41_data);
              v176_acc += (v64_bc * v42_data);
              v176_acc += (v66_bc * v43_data);
              v176_acc += (v68_bc * v44_data);
              v176_acc += (v70_bc * v45_data);
              v176_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v202_acc{};
              v202_acc += (v48_bc * v34_data);
              v202_acc += (v50_bc * v35_data);
              v202_acc += (v52_bc * v36_data);
              v202_acc += (v54_bc * v37_data);
              v202_acc += (v56_bc * v38_data);
              v202_acc += (v58_bc * v39_data);
              v202_acc += (v60_bc * v40_data);
              v202_acc += (v62_bc * v41_data);
              v202_acc += (v64_bc * v42_data);
              v202_acc += (v66_bc * v43_data);
              v202_acc += (v68_bc * v44_data);
              v202_acc += (v70_bc * v45_data);
              v202_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              v228_acc += (v48_bc * v34_data);
              v228_acc += (v50_bc * v35_data);
              v228_acc += (v52_bc * v36_data);
              v228_acc += (v54_bc * v37_data);
              v228_acc += (v56_bc * v38_data);
              v228_acc += (v58_bc * v39_data);
              v228_acc += (v60_bc * v40_data);
              v228_acc += (v62_bc * v41_data);
              v228_acc += (v64_bc * v42_data);
              v228_acc += (v66_bc * v43_data);
              v228_acc += (v68_bc * v44_data);
              v228_acc += (v70_bc * v45_data);
              v228_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v254_n1 = 0; v254_n1 < 8; ++v254_n1) {
                int32_t v255_a = v254_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v257_data;
                v257_data.copy_from(ir1 + (v255_a));
                v257_data.copy_to(r1 + (v255_a));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v261_ld;
              v261_ld.copy_from(glb_m4 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v261_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v262_ld;
              v262_ld.copy_from(glb_m4 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v262_ld.copy_to(s1 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r2 = load{g>r}(glb_m3););
              float r4[192]{};
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v264_i1 = 0; v264_i1 < 12; ++v264_i1) {
                tensorforge::intel_esimd::simd<float, 12> v269_data;
                v269_data.copy_from(glb_m5 + ((v264_i1 * 12)));
                v269_data.copy_to(r4 + ((v264_i1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v274_data;
              v274_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v275_data;
              v275_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v276_data;
              v276_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v277_data;
              v277_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v278_data;
              v278_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v279_data;
              v279_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v280_data;
              v280_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v281_data;
              v281_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v282_data;
              v282_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v283_data;
              v283_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v284_data;
              v284_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v285_data;
              v285_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v286_acc{};
              tensorforge::intel_esimd::simd<float, 16> v287_lin;
              v287_lin.copy_from(s1 + (0 + item.get_local_id(0) * 1));
              float v288_bc = v287_lin[0];
              v286_acc += (v288_bc * v274_data);
              float v290_bc = v287_lin[1];
              v286_acc += (v290_bc * v275_data);
              float v292_bc = v287_lin[2];
              v286_acc += (v292_bc * v276_data);
              float v294_bc = v287_lin[3];
              v286_acc += (v294_bc * v277_data);
              float v296_bc = v287_lin[4];
              v286_acc += (v296_bc * v278_data);
              float v298_bc = v287_lin[5];
              v286_acc += (v298_bc * v279_data);
              float v300_bc = v287_lin[6];
              v286_acc += (v300_bc * v280_data);
              float v302_bc = v287_lin[7];
              v286_acc += (v302_bc * v281_data);
              float v304_bc = v287_lin[8];
              v286_acc += (v304_bc * v282_data);
              float v306_bc = v287_lin[9];
              v286_acc += (v306_bc * v283_data);
              float v308_bc = v287_lin[10];
              v286_acc += (v308_bc * v284_data);
              float v310_bc = v287_lin[11];
              v286_acc += (v310_bc * v285_data);
              v286_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v312_acc{};
              v312_acc += (v288_bc * v274_data);
              v312_acc += (v290_bc * v275_data);
              v312_acc += (v292_bc * v276_data);
              v312_acc += (v294_bc * v277_data);
              v312_acc += (v296_bc * v278_data);
              v312_acc += (v298_bc * v279_data);
              v312_acc += (v300_bc * v280_data);
              v312_acc += (v302_bc * v281_data);
              v312_acc += (v304_bc * v282_data);
              v312_acc += (v306_bc * v283_data);
              v312_acc += (v308_bc * v284_data);
              v312_acc += (v310_bc * v285_data);
              v312_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v338_acc{};
              v338_acc += (v288_bc * v274_data);
              v338_acc += (v290_bc * v275_data);
              v338_acc += (v292_bc * v276_data);
              v338_acc += (v294_bc * v277_data);
              v338_acc += (v296_bc * v278_data);
              v338_acc += (v298_bc * v279_data);
              v338_acc += (v300_bc * v280_data);
              v338_acc += (v302_bc * v281_data);
              v338_acc += (v304_bc * v282_data);
              v338_acc += (v306_bc * v283_data);
              v338_acc += (v308_bc * v284_data);
              v338_acc += (v310_bc * v285_data);
              v338_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v364_acc{};
              v364_acc += (v288_bc * v274_data);
              v364_acc += (v290_bc * v275_data);
              v364_acc += (v292_bc * v276_data);
              v364_acc += (v294_bc * v277_data);
              v364_acc += (v296_bc * v278_data);
              v364_acc += (v298_bc * v279_data);
              v364_acc += (v300_bc * v280_data);
              v364_acc += (v302_bc * v281_data);
              v364_acc += (v304_bc * v282_data);
              v364_acc += (v306_bc * v283_data);
              v364_acc += (v308_bc * v284_data);
              v364_acc += (v310_bc * v285_data);
              v364_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v390_acc{};
              v390_acc += (v288_bc * v274_data);
              v390_acc += (v290_bc * v275_data);
              v390_acc += (v292_bc * v276_data);
              v390_acc += (v294_bc * v277_data);
              v390_acc += (v296_bc * v278_data);
              v390_acc += (v298_bc * v279_data);
              v390_acc += (v300_bc * v280_data);
              v390_acc += (v302_bc * v281_data);
              v390_acc += (v304_bc * v282_data);
              v390_acc += (v306_bc * v283_data);
              v390_acc += (v308_bc * v284_data);
              v390_acc += (v310_bc * v285_data);
              v390_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v416_acc{};
              v416_acc += (v288_bc * v274_data);
              v416_acc += (v290_bc * v275_data);
              v416_acc += (v292_bc * v276_data);
              v416_acc += (v294_bc * v277_data);
              v416_acc += (v296_bc * v278_data);
              v416_acc += (v298_bc * v279_data);
              v416_acc += (v300_bc * v280_data);
              v416_acc += (v302_bc * v281_data);
              v416_acc += (v304_bc * v282_data);
              v416_acc += (v306_bc * v283_data);
              v416_acc += (v308_bc * v284_data);
              v416_acc += (v310_bc * v285_data);
              v416_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v442_acc{};
              v442_acc += (v288_bc * v274_data);
              v442_acc += (v290_bc * v275_data);
              v442_acc += (v292_bc * v276_data);
              v442_acc += (v294_bc * v277_data);
              v442_acc += (v296_bc * v278_data);
              v442_acc += (v298_bc * v279_data);
              v442_acc += (v300_bc * v280_data);
              v442_acc += (v302_bc * v281_data);
              v442_acc += (v304_bc * v282_data);
              v442_acc += (v306_bc * v283_data);
              v442_acc += (v308_bc * v284_data);
              v442_acc += (v310_bc * v285_data);
              v442_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v468_acc{};
              v468_acc += (v288_bc * v274_data);
              v468_acc += (v290_bc * v275_data);
              v468_acc += (v292_bc * v276_data);
              v468_acc += (v294_bc * v277_data);
              v468_acc += (v296_bc * v278_data);
              v468_acc += (v298_bc * v279_data);
              v468_acc += (v300_bc * v280_data);
              v468_acc += (v302_bc * v281_data);
              v468_acc += (v304_bc * v282_data);
              v468_acc += (v306_bc * v283_data);
              v468_acc += (v308_bc * v284_data);
              v468_acc += (v310_bc * v285_data);
              v468_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v494_n1 = 0; v494_n1 < 8; ++v494_n1) {
                int32_t v495_a = v494_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v497_data;
                v497_data.copy_from(ir3 + (v495_a));
                tensorforge::intel_esimd::simd<float, 12> v500_data;
                v500_data.copy_from(r1 + (v495_a));
                (v500_data + v497_data).copy_to(r3 + (v495_a));
              }
              float* __restrict__ s2 = &localShrMem0[0];
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v505_ld;
              v505_ld.copy_from(glb_m6 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v505_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v506_ld;
              v506_ld.copy_from(glb_m6 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v506_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r4 = load{g>r}(glb_m5););
              float r6[192]{};
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v508_i1 = 0; v508_i1 < 12; ++v508_i1) {
                tensorforge::intel_esimd::simd<float, 12> v513_data;
                v513_data.copy_from(glb_m7 + ((v508_i1 * 12)));
                v513_data.copy_to(r6 + ((v508_i1 * 16)));
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              float r5[128]{};
              // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[128]{};
              tensorforge::intel_esimd::simd<float, 16> v518_data;
              v518_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v519_data;
              v519_data.copy_from(r4 + (16));
              tensorforge::intel_esimd::simd<float, 16> v520_data;
              v520_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v521_data;
              v521_data.copy_from(r4 + (48));
              tensorforge::intel_esimd::simd<float, 16> v522_data;
              v522_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v523_data;
              v523_data.copy_from(r4 + (80));
              tensorforge::intel_esimd::simd<float, 16> v524_data;
              v524_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v525_data;
              v525_data.copy_from(r4 + (112));
              tensorforge::intel_esimd::simd<float, 16> v526_data;
              v526_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v527_data;
              v527_data.copy_from(r4 + (144));
              tensorforge::intel_esimd::simd<float, 16> v528_data;
              v528_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v529_data;
              v529_data.copy_from(r4 + (176));
              tensorforge::intel_esimd::simd<float, 16> v530_acc{};
              tensorforge::intel_esimd::simd<float, 16> v531_lin;
              v531_lin.copy_from(s2 + (0 + item.get_local_id(0) * 1));
              float v532_bc = v531_lin[0];
              v530_acc += (v532_bc * v518_data);
              float v534_bc = v531_lin[1];
              v530_acc += (v534_bc * v519_data);
              float v536_bc = v531_lin[2];
              v530_acc += (v536_bc * v520_data);
              float v538_bc = v531_lin[3];
              v530_acc += (v538_bc * v521_data);
              float v540_bc = v531_lin[4];
              v530_acc += (v540_bc * v522_data);
              float v542_bc = v531_lin[5];
              v530_acc += (v542_bc * v523_data);
              float v544_bc = v531_lin[6];
              v530_acc += (v544_bc * v524_data);
              float v546_bc = v531_lin[7];
              v530_acc += (v546_bc * v525_data);
              float v548_bc = v531_lin[8];
              v530_acc += (v548_bc * v526_data);
              float v550_bc = v531_lin[9];
              v530_acc += (v550_bc * v527_data);
              float v552_bc = v531_lin[10];
              v530_acc += (v552_bc * v528_data);
              float v554_bc = v531_lin[11];
              v530_acc += (v554_bc * v529_data);
              v530_acc.copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v556_acc{};
              v556_acc += (v532_bc * v518_data);
              v556_acc += (v534_bc * v519_data);
              v556_acc += (v536_bc * v520_data);
              v556_acc += (v538_bc * v521_data);
              v556_acc += (v540_bc * v522_data);
              v556_acc += (v542_bc * v523_data);
              v556_acc += (v544_bc * v524_data);
              v556_acc += (v546_bc * v525_data);
              v556_acc += (v548_bc * v526_data);
              v556_acc += (v550_bc * v527_data);
              v556_acc += (v552_bc * v528_data);
              v556_acc += (v554_bc * v529_data);
              v556_acc.copy_to(ir5 + (16));
              tensorforge::intel_esimd::simd<float, 16> v582_acc{};
              v582_acc += (v532_bc * v518_data);
              v582_acc += (v534_bc * v519_data);
              v582_acc += (v536_bc * v520_data);
              v582_acc += (v538_bc * v521_data);
              v582_acc += (v540_bc * v522_data);
              v582_acc += (v542_bc * v523_data);
              v582_acc += (v544_bc * v524_data);
              v582_acc += (v546_bc * v525_data);
              v582_acc += (v548_bc * v526_data);
              v582_acc += (v550_bc * v527_data);
              v582_acc += (v552_bc * v528_data);
              v582_acc += (v554_bc * v529_data);
              v582_acc.copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v608_acc{};
              v608_acc += (v532_bc * v518_data);
              v608_acc += (v534_bc * v519_data);
              v608_acc += (v536_bc * v520_data);
              v608_acc += (v538_bc * v521_data);
              v608_acc += (v540_bc * v522_data);
              v608_acc += (v542_bc * v523_data);
              v608_acc += (v544_bc * v524_data);
              v608_acc += (v546_bc * v525_data);
              v608_acc += (v548_bc * v526_data);
              v608_acc += (v550_bc * v527_data);
              v608_acc += (v552_bc * v528_data);
              v608_acc += (v554_bc * v529_data);
              v608_acc.copy_to(ir5 + (48));
              tensorforge::intel_esimd::simd<float, 16> v634_acc{};
              v634_acc += (v532_bc * v518_data);
              v634_acc += (v534_bc * v519_data);
              v634_acc += (v536_bc * v520_data);
              v634_acc += (v538_bc * v521_data);
              v634_acc += (v540_bc * v522_data);
              v634_acc += (v542_bc * v523_data);
              v634_acc += (v544_bc * v524_data);
              v634_acc += (v546_bc * v525_data);
              v634_acc += (v548_bc * v526_data);
              v634_acc += (v550_bc * v527_data);
              v634_acc += (v552_bc * v528_data);
              v634_acc += (v554_bc * v529_data);
              v634_acc.copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v660_acc{};
              v660_acc += (v532_bc * v518_data);
              v660_acc += (v534_bc * v519_data);
              v660_acc += (v536_bc * v520_data);
              v660_acc += (v538_bc * v521_data);
              v660_acc += (v540_bc * v522_data);
              v660_acc += (v542_bc * v523_data);
              v660_acc += (v544_bc * v524_data);
              v660_acc += (v546_bc * v525_data);
              v660_acc += (v548_bc * v526_data);
              v660_acc += (v550_bc * v527_data);
              v660_acc += (v552_bc * v528_data);
              v660_acc += (v554_bc * v529_data);
              v660_acc.copy_to(ir5 + (80));
              tensorforge::intel_esimd::simd<float, 16> v686_acc{};
              v686_acc += (v532_bc * v518_data);
              v686_acc += (v534_bc * v519_data);
              v686_acc += (v536_bc * v520_data);
              v686_acc += (v538_bc * v521_data);
              v686_acc += (v540_bc * v522_data);
              v686_acc += (v542_bc * v523_data);
              v686_acc += (v544_bc * v524_data);
              v686_acc += (v546_bc * v525_data);
              v686_acc += (v548_bc * v526_data);
              v686_acc += (v550_bc * v527_data);
              v686_acc += (v552_bc * v528_data);
              v686_acc += (v554_bc * v529_data);
              v686_acc.copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v712_acc{};
              v712_acc += (v532_bc * v518_data);
              v712_acc += (v534_bc * v519_data);
              v712_acc += (v536_bc * v520_data);
              v712_acc += (v538_bc * v521_data);
              v712_acc += (v540_bc * v522_data);
              v712_acc += (v542_bc * v523_data);
              v712_acc += (v544_bc * v524_data);
              v712_acc += (v546_bc * v525_data);
              v712_acc += (v548_bc * v526_data);
              v712_acc += (v550_bc * v527_data);
              v712_acc += (v552_bc * v528_data);
              v712_acc += (v554_bc * v529_data);
              v712_acc.copy_to(ir5 + (112));
              #pragma unroll
              for (int32_t v738_n1 = 0; v738_n1 < 8; ++v738_n1) {
                int32_t v739_a = v738_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v741_data;
                v741_data.copy_from(ir5 + (v739_a));
                tensorforge::intel_esimd::simd<float, 12> v744_data;
                v744_data.copy_from(r3 + (v739_a));
                (v744_data + v741_data).copy_to(r5 + (v739_a));
              }
              float* __restrict__ s3 = &localShrMem0[0];
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v749_ld;
              v749_ld.copy_from(glb_m8 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v749_ld.copy_to(s3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v750_ld;
              v750_ld.copy_from(glb_m8 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v750_ld.copy_to(s3 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              float r7[128]{};
              // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir7[128]{};
              tensorforge::intel_esimd::simd<float, 16> v753_data;
              v753_data.copy_from(r6 + (0));
              tensorforge::intel_esimd::simd<float, 16> v754_data;
              v754_data.copy_from(r6 + (16));
              tensorforge::intel_esimd::simd<float, 16> v755_data;
              v755_data.copy_from(r6 + (32));
              tensorforge::intel_esimd::simd<float, 16> v756_data;
              v756_data.copy_from(r6 + (48));
              tensorforge::intel_esimd::simd<float, 16> v757_data;
              v757_data.copy_from(r6 + (64));
              tensorforge::intel_esimd::simd<float, 16> v758_data;
              v758_data.copy_from(r6 + (80));
              tensorforge::intel_esimd::simd<float, 16> v759_data;
              v759_data.copy_from(r6 + (96));
              tensorforge::intel_esimd::simd<float, 16> v760_data;
              v760_data.copy_from(r6 + (112));
              tensorforge::intel_esimd::simd<float, 16> v761_data;
              v761_data.copy_from(r6 + (128));
              tensorforge::intel_esimd::simd<float, 16> v762_data;
              v762_data.copy_from(r6 + (144));
              tensorforge::intel_esimd::simd<float, 16> v763_data;
              v763_data.copy_from(r6 + (160));
              tensorforge::intel_esimd::simd<float, 16> v764_data;
              v764_data.copy_from(r6 + (176));
              tensorforge::intel_esimd::simd<float, 16> v765_acc{};
              tensorforge::intel_esimd::simd<float, 16> v766_lin;
              v766_lin.copy_from(s3 + (0 + item.get_local_id(0) * 1));
              float v767_bc = v766_lin[0];
              v765_acc += (v767_bc * v753_data);
              float v769_bc = v766_lin[1];
              v765_acc += (v769_bc * v754_data);
              float v771_bc = v766_lin[2];
              v765_acc += (v771_bc * v755_data);
              float v773_bc = v766_lin[3];
              v765_acc += (v773_bc * v756_data);
              float v775_bc = v766_lin[4];
              v765_acc += (v775_bc * v757_data);
              float v777_bc = v766_lin[5];
              v765_acc += (v777_bc * v758_data);
              float v779_bc = v766_lin[6];
              v765_acc += (v779_bc * v759_data);
              float v781_bc = v766_lin[7];
              v765_acc += (v781_bc * v760_data);
              float v783_bc = v766_lin[8];
              v765_acc += (v783_bc * v761_data);
              float v785_bc = v766_lin[9];
              v765_acc += (v785_bc * v762_data);
              float v787_bc = v766_lin[10];
              v765_acc += (v787_bc * v763_data);
              float v789_bc = v766_lin[11];
              v765_acc += (v789_bc * v764_data);
              v765_acc.copy_to(ir7 + (0));
              tensorforge::intel_esimd::simd<float, 16> v791_acc{};
              v791_acc += (v767_bc * v753_data);
              v791_acc += (v769_bc * v754_data);
              v791_acc += (v771_bc * v755_data);
              v791_acc += (v773_bc * v756_data);
              v791_acc += (v775_bc * v757_data);
              v791_acc += (v777_bc * v758_data);
              v791_acc += (v779_bc * v759_data);
              v791_acc += (v781_bc * v760_data);
              v791_acc += (v783_bc * v761_data);
              v791_acc += (v785_bc * v762_data);
              v791_acc += (v787_bc * v763_data);
              v791_acc += (v789_bc * v764_data);
              v791_acc.copy_to(ir7 + (16));
              tensorforge::intel_esimd::simd<float, 16> v817_acc{};
              v817_acc += (v767_bc * v753_data);
              v817_acc += (v769_bc * v754_data);
              v817_acc += (v771_bc * v755_data);
              v817_acc += (v773_bc * v756_data);
              v817_acc += (v775_bc * v757_data);
              v817_acc += (v777_bc * v758_data);
              v817_acc += (v779_bc * v759_data);
              v817_acc += (v781_bc * v760_data);
              v817_acc += (v783_bc * v761_data);
              v817_acc += (v785_bc * v762_data);
              v817_acc += (v787_bc * v763_data);
              v817_acc += (v789_bc * v764_data);
              v817_acc.copy_to(ir7 + (32));
              tensorforge::intel_esimd::simd<float, 16> v843_acc{};
              v843_acc += (v767_bc * v753_data);
              v843_acc += (v769_bc * v754_data);
              v843_acc += (v771_bc * v755_data);
              v843_acc += (v773_bc * v756_data);
              v843_acc += (v775_bc * v757_data);
              v843_acc += (v777_bc * v758_data);
              v843_acc += (v779_bc * v759_data);
              v843_acc += (v781_bc * v760_data);
              v843_acc += (v783_bc * v761_data);
              v843_acc += (v785_bc * v762_data);
              v843_acc += (v787_bc * v763_data);
              v843_acc += (v789_bc * v764_data);
              v843_acc.copy_to(ir7 + (48));
              tensorforge::intel_esimd::simd<float, 16> v869_acc{};
              v869_acc += (v767_bc * v753_data);
              v869_acc += (v769_bc * v754_data);
              v869_acc += (v771_bc * v755_data);
              v869_acc += (v773_bc * v756_data);
              v869_acc += (v775_bc * v757_data);
              v869_acc += (v777_bc * v758_data);
              v869_acc += (v779_bc * v759_data);
              v869_acc += (v781_bc * v760_data);
              v869_acc += (v783_bc * v761_data);
              v869_acc += (v785_bc * v762_data);
              v869_acc += (v787_bc * v763_data);
              v869_acc += (v789_bc * v764_data);
              v869_acc.copy_to(ir7 + (64));
              tensorforge::intel_esimd::simd<float, 16> v895_acc{};
              v895_acc += (v767_bc * v753_data);
              v895_acc += (v769_bc * v754_data);
              v895_acc += (v771_bc * v755_data);
              v895_acc += (v773_bc * v756_data);
              v895_acc += (v775_bc * v757_data);
              v895_acc += (v777_bc * v758_data);
              v895_acc += (v779_bc * v759_data);
              v895_acc += (v781_bc * v760_data);
              v895_acc += (v783_bc * v761_data);
              v895_acc += (v785_bc * v762_data);
              v895_acc += (v787_bc * v763_data);
              v895_acc += (v789_bc * v764_data);
              v895_acc.copy_to(ir7 + (80));
              tensorforge::intel_esimd::simd<float, 16> v921_acc{};
              v921_acc += (v767_bc * v753_data);
              v921_acc += (v769_bc * v754_data);
              v921_acc += (v771_bc * v755_data);
              v921_acc += (v773_bc * v756_data);
              v921_acc += (v775_bc * v757_data);
              v921_acc += (v777_bc * v758_data);
              v921_acc += (v779_bc * v759_data);
              v921_acc += (v781_bc * v760_data);
              v921_acc += (v783_bc * v761_data);
              v921_acc += (v785_bc * v762_data);
              v921_acc += (v787_bc * v763_data);
              v921_acc += (v789_bc * v764_data);
              v921_acc.copy_to(ir7 + (96));
              tensorforge::intel_esimd::simd<float, 16> v947_acc{};
              v947_acc += (v767_bc * v753_data);
              v947_acc += (v769_bc * v754_data);
              v947_acc += (v771_bc * v755_data);
              v947_acc += (v773_bc * v756_data);
              v947_acc += (v775_bc * v757_data);
              v947_acc += (v777_bc * v758_data);
              v947_acc += (v779_bc * v759_data);
              v947_acc += (v781_bc * v760_data);
              v947_acc += (v783_bc * v761_data);
              v947_acc += (v785_bc * v762_data);
              v947_acc += (v787_bc * v763_data);
              v947_acc += (v789_bc * v764_data);
              v947_acc.copy_to(ir7 + (112));
              #pragma unroll
              for (int32_t v973_n1 = 0; v973_n1 < 8; ++v973_n1) {
                int32_t v974_a = v973_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v976_data;
                v976_data.copy_from(ir7 + (v974_a));
                tensorforge::intel_esimd::simd<float, 12> v979_data;
                v979_data.copy_from(r5 + (v974_a));
                (v979_data + v976_data).copy_to(r7 + (v974_a));
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v983_i1 = 0; v983_i1 < 8; ++v983_i1) {
                tensorforge::intel_esimd::simd<float, 12> v986_data;
                v986_data.copy_from(r7 + ((v983_i1 * 16)));
                v986_data.copy_to(glb_m0 + ((v983_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

