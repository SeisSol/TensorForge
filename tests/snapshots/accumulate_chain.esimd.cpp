// === base name ===
kernel_dff2295fdb63b74e

// === header ===
void launcher_kernel_dff2295fdb63b74e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_dff2295fdb63b74e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_dff2295fdb63b74e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  m7,  m7_extraOffset,  m8,  m8_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_dff2295fdb63b74e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, const float* m7, size_t m7_extraOffset, const float* m8, size_t m8_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          float * __restrict__ s0 = &localShrMem0[0];
          float * __restrict__ s1 = &localShrMem0[0];
          float * __restrict__ s2 = &localShrMem0[0];
          float * __restrict__ s3 = &localShrMem0[0];
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v6_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v6_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v6_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v6_batchId0 * 96 + 0 + m8_extraOffset];
              float r0[192]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 12> v29_data;
                v29_data.copy_from(glb_m1 + ((v24_i1 * 12)));
                v29_data.copy_to(r0 + ((v24_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v32_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v33_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
                tensorforge::intel_esimd::simd<float, 12> v40_data;
                v40_data.copy_from(glb_m3 + ((v35_i1 * 12)));
                v40_data.copy_to(r2 + ((v35_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v53_data;
              v53_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v54_data;
              v54_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v57_acc{};
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(s0 + (0_i32));
              v57_acc += ((static_cast<float>(v61_data[0])) * v45_data);
              v57_acc += ((static_cast<float>(v61_data[1])) * v46_data);
              v57_acc += ((static_cast<float>(v61_data[2])) * v47_data);
              v57_acc += ((static_cast<float>(v61_data[3])) * v48_data);
              v57_acc += ((static_cast<float>(v61_data[4])) * v49_data);
              v57_acc += ((static_cast<float>(v61_data[5])) * v50_data);
              v57_acc += ((static_cast<float>(v61_data[6])) * v51_data);
              v57_acc += ((static_cast<float>(v61_data[7])) * v52_data);
              v57_acc += ((static_cast<float>(v61_data[8])) * v53_data);
              v57_acc += ((static_cast<float>(v61_data[9])) * v54_data);
              v57_acc += ((static_cast<float>(v61_data[10])) * v55_data);
              v57_acc += ((static_cast<float>(v61_data[11])) * v56_data);
              v57_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v86_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data;
              v90_data.copy_from(s0 + (12_i32));
              v86_acc += ((static_cast<float>(v90_data[0])) * v45_data);
              v86_acc += ((static_cast<float>(v90_data[1])) * v46_data);
              v86_acc += ((static_cast<float>(v90_data[2])) * v47_data);
              v86_acc += ((static_cast<float>(v90_data[3])) * v48_data);
              v86_acc += ((static_cast<float>(v90_data[4])) * v49_data);
              v86_acc += ((static_cast<float>(v90_data[5])) * v50_data);
              v86_acc += ((static_cast<float>(v90_data[6])) * v51_data);
              v86_acc += ((static_cast<float>(v90_data[7])) * v52_data);
              v86_acc += ((static_cast<float>(v90_data[8])) * v53_data);
              v86_acc += ((static_cast<float>(v90_data[9])) * v54_data);
              v86_acc += ((static_cast<float>(v90_data[10])) * v55_data);
              v86_acc += ((static_cast<float>(v90_data[11])) * v56_data);
              v86_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v115_acc{};
              tensorforge::intel_esimd::simd<float, 16> v119_data;
              v119_data.copy_from(s0 + (24_i32));
              v115_acc += ((static_cast<float>(v119_data[0])) * v45_data);
              v115_acc += ((static_cast<float>(v119_data[1])) * v46_data);
              v115_acc += ((static_cast<float>(v119_data[2])) * v47_data);
              v115_acc += ((static_cast<float>(v119_data[3])) * v48_data);
              v115_acc += ((static_cast<float>(v119_data[4])) * v49_data);
              v115_acc += ((static_cast<float>(v119_data[5])) * v50_data);
              v115_acc += ((static_cast<float>(v119_data[6])) * v51_data);
              v115_acc += ((static_cast<float>(v119_data[7])) * v52_data);
              v115_acc += ((static_cast<float>(v119_data[8])) * v53_data);
              v115_acc += ((static_cast<float>(v119_data[9])) * v54_data);
              v115_acc += ((static_cast<float>(v119_data[10])) * v55_data);
              v115_acc += ((static_cast<float>(v119_data[11])) * v56_data);
              v115_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v144_acc{};
              tensorforge::intel_esimd::simd<float, 16> v148_data;
              v148_data.copy_from(s0 + (36_i32));
              v144_acc += ((static_cast<float>(v148_data[0])) * v45_data);
              v144_acc += ((static_cast<float>(v148_data[1])) * v46_data);
              v144_acc += ((static_cast<float>(v148_data[2])) * v47_data);
              v144_acc += ((static_cast<float>(v148_data[3])) * v48_data);
              v144_acc += ((static_cast<float>(v148_data[4])) * v49_data);
              v144_acc += ((static_cast<float>(v148_data[5])) * v50_data);
              v144_acc += ((static_cast<float>(v148_data[6])) * v51_data);
              v144_acc += ((static_cast<float>(v148_data[7])) * v52_data);
              v144_acc += ((static_cast<float>(v148_data[8])) * v53_data);
              v144_acc += ((static_cast<float>(v148_data[9])) * v54_data);
              v144_acc += ((static_cast<float>(v148_data[10])) * v55_data);
              v144_acc += ((static_cast<float>(v148_data[11])) * v56_data);
              v144_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              tensorforge::intel_esimd::simd<float, 16> v177_data;
              v177_data.copy_from(s0 + (48_i32));
              v173_acc += ((static_cast<float>(v177_data[0])) * v45_data);
              v173_acc += ((static_cast<float>(v177_data[1])) * v46_data);
              v173_acc += ((static_cast<float>(v177_data[2])) * v47_data);
              v173_acc += ((static_cast<float>(v177_data[3])) * v48_data);
              v173_acc += ((static_cast<float>(v177_data[4])) * v49_data);
              v173_acc += ((static_cast<float>(v177_data[5])) * v50_data);
              v173_acc += ((static_cast<float>(v177_data[6])) * v51_data);
              v173_acc += ((static_cast<float>(v177_data[7])) * v52_data);
              v173_acc += ((static_cast<float>(v177_data[8])) * v53_data);
              v173_acc += ((static_cast<float>(v177_data[9])) * v54_data);
              v173_acc += ((static_cast<float>(v177_data[10])) * v55_data);
              v173_acc += ((static_cast<float>(v177_data[11])) * v56_data);
              v173_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v202_acc{};
              tensorforge::intel_esimd::simd<float, 16> v206_data;
              v206_data.copy_from(s0 + (60_i32));
              v202_acc += ((static_cast<float>(v206_data[0])) * v45_data);
              v202_acc += ((static_cast<float>(v206_data[1])) * v46_data);
              v202_acc += ((static_cast<float>(v206_data[2])) * v47_data);
              v202_acc += ((static_cast<float>(v206_data[3])) * v48_data);
              v202_acc += ((static_cast<float>(v206_data[4])) * v49_data);
              v202_acc += ((static_cast<float>(v206_data[5])) * v50_data);
              v202_acc += ((static_cast<float>(v206_data[6])) * v51_data);
              v202_acc += ((static_cast<float>(v206_data[7])) * v52_data);
              v202_acc += ((static_cast<float>(v206_data[8])) * v53_data);
              v202_acc += ((static_cast<float>(v206_data[9])) * v54_data);
              v202_acc += ((static_cast<float>(v206_data[10])) * v55_data);
              v202_acc += ((static_cast<float>(v206_data[11])) * v56_data);
              v202_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v231_acc{};
              tensorforge::intel_esimd::simd<float, 16> v235_data;
              v235_data.copy_from(s0 + (72_i32));
              v231_acc += ((static_cast<float>(v235_data[0])) * v45_data);
              v231_acc += ((static_cast<float>(v235_data[1])) * v46_data);
              v231_acc += ((static_cast<float>(v235_data[2])) * v47_data);
              v231_acc += ((static_cast<float>(v235_data[3])) * v48_data);
              v231_acc += ((static_cast<float>(v235_data[4])) * v49_data);
              v231_acc += ((static_cast<float>(v235_data[5])) * v50_data);
              v231_acc += ((static_cast<float>(v235_data[6])) * v51_data);
              v231_acc += ((static_cast<float>(v235_data[7])) * v52_data);
              v231_acc += ((static_cast<float>(v235_data[8])) * v53_data);
              v231_acc += ((static_cast<float>(v235_data[9])) * v54_data);
              v231_acc += ((static_cast<float>(v235_data[10])) * v55_data);
              v231_acc += ((static_cast<float>(v235_data[11])) * v56_data);
              v231_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v260_acc{};
              tensorforge::intel_esimd::simd<float, 16> v264_data;
              v264_data.copy_from(s0 + (84_i32));
              v260_acc += ((static_cast<float>(v264_data[0])) * v45_data);
              v260_acc += ((static_cast<float>(v264_data[1])) * v46_data);
              v260_acc += ((static_cast<float>(v264_data[2])) * v47_data);
              v260_acc += ((static_cast<float>(v264_data[3])) * v48_data);
              v260_acc += ((static_cast<float>(v264_data[4])) * v49_data);
              v260_acc += ((static_cast<float>(v264_data[5])) * v50_data);
              v260_acc += ((static_cast<float>(v264_data[6])) * v51_data);
              v260_acc += ((static_cast<float>(v264_data[7])) * v52_data);
              v260_acc += ((static_cast<float>(v264_data[8])) * v53_data);
              v260_acc += ((static_cast<float>(v264_data[9])) * v54_data);
              v260_acc += ((static_cast<float>(v264_data[10])) * v55_data);
              v260_acc += ((static_cast<float>(v264_data[11])) * v56_data);
              v260_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v289_n1 = 0; v289_n1 < 8; ++v289_n1) {
                int32_t v290_a = v289_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v292_data;
                v292_data.copy_from(ir1 + (v290_a));
                v292_data.copy_to(r1 + (v290_a));
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v295_ld;
              v295_ld.copy_from(glb_m4 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v295_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v296_ld;
              v296_ld.copy_from(glb_m4 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v296_ld.copy_to(s1 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r2 = load{g>r}(glb_m3););
              float r4[192]{};
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v298_i1 = 0; v298_i1 < 12; ++v298_i1) {
                tensorforge::intel_esimd::simd<float, 12> v303_data;
                v303_data.copy_from(glb_m5 + ((v298_i1 * 12)));
                v303_data.copy_to(r4 + ((v298_i1 * 16)));
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v308_data;
              v308_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v309_data;
              v309_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v310_data;
              v310_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v311_data;
              v311_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v312_data;
              v312_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v313_data;
              v313_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v314_data;
              v314_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v315_data;
              v315_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v316_data;
              v316_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v317_data;
              v317_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v318_data;
              v318_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v319_data;
              v319_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v320_acc{};
              tensorforge::intel_esimd::simd<float, 16> v324_data;
              v324_data.copy_from(s1 + (0_i32));
              v320_acc += ((static_cast<float>(v324_data[0])) * v308_data);
              v320_acc += ((static_cast<float>(v324_data[1])) * v309_data);
              v320_acc += ((static_cast<float>(v324_data[2])) * v310_data);
              v320_acc += ((static_cast<float>(v324_data[3])) * v311_data);
              v320_acc += ((static_cast<float>(v324_data[4])) * v312_data);
              v320_acc += ((static_cast<float>(v324_data[5])) * v313_data);
              v320_acc += ((static_cast<float>(v324_data[6])) * v314_data);
              v320_acc += ((static_cast<float>(v324_data[7])) * v315_data);
              v320_acc += ((static_cast<float>(v324_data[8])) * v316_data);
              v320_acc += ((static_cast<float>(v324_data[9])) * v317_data);
              v320_acc += ((static_cast<float>(v324_data[10])) * v318_data);
              v320_acc += ((static_cast<float>(v324_data[11])) * v319_data);
              v320_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v349_acc{};
              tensorforge::intel_esimd::simd<float, 16> v353_data;
              v353_data.copy_from(s1 + (12_i32));
              v349_acc += ((static_cast<float>(v353_data[0])) * v308_data);
              v349_acc += ((static_cast<float>(v353_data[1])) * v309_data);
              v349_acc += ((static_cast<float>(v353_data[2])) * v310_data);
              v349_acc += ((static_cast<float>(v353_data[3])) * v311_data);
              v349_acc += ((static_cast<float>(v353_data[4])) * v312_data);
              v349_acc += ((static_cast<float>(v353_data[5])) * v313_data);
              v349_acc += ((static_cast<float>(v353_data[6])) * v314_data);
              v349_acc += ((static_cast<float>(v353_data[7])) * v315_data);
              v349_acc += ((static_cast<float>(v353_data[8])) * v316_data);
              v349_acc += ((static_cast<float>(v353_data[9])) * v317_data);
              v349_acc += ((static_cast<float>(v353_data[10])) * v318_data);
              v349_acc += ((static_cast<float>(v353_data[11])) * v319_data);
              v349_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v378_acc{};
              tensorforge::intel_esimd::simd<float, 16> v382_data;
              v382_data.copy_from(s1 + (24_i32));
              v378_acc += ((static_cast<float>(v382_data[0])) * v308_data);
              v378_acc += ((static_cast<float>(v382_data[1])) * v309_data);
              v378_acc += ((static_cast<float>(v382_data[2])) * v310_data);
              v378_acc += ((static_cast<float>(v382_data[3])) * v311_data);
              v378_acc += ((static_cast<float>(v382_data[4])) * v312_data);
              v378_acc += ((static_cast<float>(v382_data[5])) * v313_data);
              v378_acc += ((static_cast<float>(v382_data[6])) * v314_data);
              v378_acc += ((static_cast<float>(v382_data[7])) * v315_data);
              v378_acc += ((static_cast<float>(v382_data[8])) * v316_data);
              v378_acc += ((static_cast<float>(v382_data[9])) * v317_data);
              v378_acc += ((static_cast<float>(v382_data[10])) * v318_data);
              v378_acc += ((static_cast<float>(v382_data[11])) * v319_data);
              v378_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v407_acc{};
              tensorforge::intel_esimd::simd<float, 16> v411_data;
              v411_data.copy_from(s1 + (36_i32));
              v407_acc += ((static_cast<float>(v411_data[0])) * v308_data);
              v407_acc += ((static_cast<float>(v411_data[1])) * v309_data);
              v407_acc += ((static_cast<float>(v411_data[2])) * v310_data);
              v407_acc += ((static_cast<float>(v411_data[3])) * v311_data);
              v407_acc += ((static_cast<float>(v411_data[4])) * v312_data);
              v407_acc += ((static_cast<float>(v411_data[5])) * v313_data);
              v407_acc += ((static_cast<float>(v411_data[6])) * v314_data);
              v407_acc += ((static_cast<float>(v411_data[7])) * v315_data);
              v407_acc += ((static_cast<float>(v411_data[8])) * v316_data);
              v407_acc += ((static_cast<float>(v411_data[9])) * v317_data);
              v407_acc += ((static_cast<float>(v411_data[10])) * v318_data);
              v407_acc += ((static_cast<float>(v411_data[11])) * v319_data);
              v407_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v436_acc{};
              tensorforge::intel_esimd::simd<float, 16> v440_data;
              v440_data.copy_from(s1 + (48_i32));
              v436_acc += ((static_cast<float>(v440_data[0])) * v308_data);
              v436_acc += ((static_cast<float>(v440_data[1])) * v309_data);
              v436_acc += ((static_cast<float>(v440_data[2])) * v310_data);
              v436_acc += ((static_cast<float>(v440_data[3])) * v311_data);
              v436_acc += ((static_cast<float>(v440_data[4])) * v312_data);
              v436_acc += ((static_cast<float>(v440_data[5])) * v313_data);
              v436_acc += ((static_cast<float>(v440_data[6])) * v314_data);
              v436_acc += ((static_cast<float>(v440_data[7])) * v315_data);
              v436_acc += ((static_cast<float>(v440_data[8])) * v316_data);
              v436_acc += ((static_cast<float>(v440_data[9])) * v317_data);
              v436_acc += ((static_cast<float>(v440_data[10])) * v318_data);
              v436_acc += ((static_cast<float>(v440_data[11])) * v319_data);
              v436_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v465_acc{};
              tensorforge::intel_esimd::simd<float, 16> v469_data;
              v469_data.copy_from(s1 + (60_i32));
              v465_acc += ((static_cast<float>(v469_data[0])) * v308_data);
              v465_acc += ((static_cast<float>(v469_data[1])) * v309_data);
              v465_acc += ((static_cast<float>(v469_data[2])) * v310_data);
              v465_acc += ((static_cast<float>(v469_data[3])) * v311_data);
              v465_acc += ((static_cast<float>(v469_data[4])) * v312_data);
              v465_acc += ((static_cast<float>(v469_data[5])) * v313_data);
              v465_acc += ((static_cast<float>(v469_data[6])) * v314_data);
              v465_acc += ((static_cast<float>(v469_data[7])) * v315_data);
              v465_acc += ((static_cast<float>(v469_data[8])) * v316_data);
              v465_acc += ((static_cast<float>(v469_data[9])) * v317_data);
              v465_acc += ((static_cast<float>(v469_data[10])) * v318_data);
              v465_acc += ((static_cast<float>(v469_data[11])) * v319_data);
              v465_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v494_acc{};
              tensorforge::intel_esimd::simd<float, 16> v498_data;
              v498_data.copy_from(s1 + (72_i32));
              v494_acc += ((static_cast<float>(v498_data[0])) * v308_data);
              v494_acc += ((static_cast<float>(v498_data[1])) * v309_data);
              v494_acc += ((static_cast<float>(v498_data[2])) * v310_data);
              v494_acc += ((static_cast<float>(v498_data[3])) * v311_data);
              v494_acc += ((static_cast<float>(v498_data[4])) * v312_data);
              v494_acc += ((static_cast<float>(v498_data[5])) * v313_data);
              v494_acc += ((static_cast<float>(v498_data[6])) * v314_data);
              v494_acc += ((static_cast<float>(v498_data[7])) * v315_data);
              v494_acc += ((static_cast<float>(v498_data[8])) * v316_data);
              v494_acc += ((static_cast<float>(v498_data[9])) * v317_data);
              v494_acc += ((static_cast<float>(v498_data[10])) * v318_data);
              v494_acc += ((static_cast<float>(v498_data[11])) * v319_data);
              v494_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v523_acc{};
              tensorforge::intel_esimd::simd<float, 16> v527_data;
              v527_data.copy_from(s1 + (84_i32));
              v523_acc += ((static_cast<float>(v527_data[0])) * v308_data);
              v523_acc += ((static_cast<float>(v527_data[1])) * v309_data);
              v523_acc += ((static_cast<float>(v527_data[2])) * v310_data);
              v523_acc += ((static_cast<float>(v527_data[3])) * v311_data);
              v523_acc += ((static_cast<float>(v527_data[4])) * v312_data);
              v523_acc += ((static_cast<float>(v527_data[5])) * v313_data);
              v523_acc += ((static_cast<float>(v527_data[6])) * v314_data);
              v523_acc += ((static_cast<float>(v527_data[7])) * v315_data);
              v523_acc += ((static_cast<float>(v527_data[8])) * v316_data);
              v523_acc += ((static_cast<float>(v527_data[9])) * v317_data);
              v523_acc += ((static_cast<float>(v527_data[10])) * v318_data);
              v523_acc += ((static_cast<float>(v527_data[11])) * v319_data);
              v523_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v552_n1 = 0; v552_n1 < 8; ++v552_n1) {
                int32_t v553_a = v552_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v555_data;
                v555_data.copy_from(ir3 + (v553_a));
                tensorforge::intel_esimd::simd<float, 12> v558_data;
                v558_data.copy_from(r1 + (v553_a));
                (v558_data + v555_data).copy_to(r3 + (v553_a));
              }
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v562_ld;
              v562_ld.copy_from(glb_m6 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v562_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v563_ld;
              v563_ld.copy_from(glb_m6 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v563_ld.copy_to(s2 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r4 = load{g>r}(glb_m5););
              float r6[192]{};
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v565_i1 = 0; v565_i1 < 12; ++v565_i1) {
                tensorforge::intel_esimd::simd<float, 12> v570_data;
                v570_data.copy_from(glb_m7 + ((v565_i1 * 12)));
                v570_data.copy_to(r6 + ((v565_i1 * 16)));
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              float r5[128]{};
              // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[128]{};
              tensorforge::intel_esimd::simd<float, 16> v575_data;
              v575_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v576_data;
              v576_data.copy_from(r4 + (16));
              tensorforge::intel_esimd::simd<float, 16> v577_data;
              v577_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v578_data;
              v578_data.copy_from(r4 + (48));
              tensorforge::intel_esimd::simd<float, 16> v579_data;
              v579_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v580_data;
              v580_data.copy_from(r4 + (80));
              tensorforge::intel_esimd::simd<float, 16> v581_data;
              v581_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v582_data;
              v582_data.copy_from(r4 + (112));
              tensorforge::intel_esimd::simd<float, 16> v583_data;
              v583_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v584_data;
              v584_data.copy_from(r4 + (144));
              tensorforge::intel_esimd::simd<float, 16> v585_data;
              v585_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v586_data;
              v586_data.copy_from(r4 + (176));
              tensorforge::intel_esimd::simd<float, 16> v587_acc{};
              tensorforge::intel_esimd::simd<float, 16> v591_data;
              v591_data.copy_from(s2 + (0_i32));
              v587_acc += ((static_cast<float>(v591_data[0])) * v575_data);
              v587_acc += ((static_cast<float>(v591_data[1])) * v576_data);
              v587_acc += ((static_cast<float>(v591_data[2])) * v577_data);
              v587_acc += ((static_cast<float>(v591_data[3])) * v578_data);
              v587_acc += ((static_cast<float>(v591_data[4])) * v579_data);
              v587_acc += ((static_cast<float>(v591_data[5])) * v580_data);
              v587_acc += ((static_cast<float>(v591_data[6])) * v581_data);
              v587_acc += ((static_cast<float>(v591_data[7])) * v582_data);
              v587_acc += ((static_cast<float>(v591_data[8])) * v583_data);
              v587_acc += ((static_cast<float>(v591_data[9])) * v584_data);
              v587_acc += ((static_cast<float>(v591_data[10])) * v585_data);
              v587_acc += ((static_cast<float>(v591_data[11])) * v586_data);
              v587_acc.copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v616_acc{};
              tensorforge::intel_esimd::simd<float, 16> v620_data;
              v620_data.copy_from(s2 + (12_i32));
              v616_acc += ((static_cast<float>(v620_data[0])) * v575_data);
              v616_acc += ((static_cast<float>(v620_data[1])) * v576_data);
              v616_acc += ((static_cast<float>(v620_data[2])) * v577_data);
              v616_acc += ((static_cast<float>(v620_data[3])) * v578_data);
              v616_acc += ((static_cast<float>(v620_data[4])) * v579_data);
              v616_acc += ((static_cast<float>(v620_data[5])) * v580_data);
              v616_acc += ((static_cast<float>(v620_data[6])) * v581_data);
              v616_acc += ((static_cast<float>(v620_data[7])) * v582_data);
              v616_acc += ((static_cast<float>(v620_data[8])) * v583_data);
              v616_acc += ((static_cast<float>(v620_data[9])) * v584_data);
              v616_acc += ((static_cast<float>(v620_data[10])) * v585_data);
              v616_acc += ((static_cast<float>(v620_data[11])) * v586_data);
              v616_acc.copy_to(ir5 + (16));
              tensorforge::intel_esimd::simd<float, 16> v645_acc{};
              tensorforge::intel_esimd::simd<float, 16> v649_data;
              v649_data.copy_from(s2 + (24_i32));
              v645_acc += ((static_cast<float>(v649_data[0])) * v575_data);
              v645_acc += ((static_cast<float>(v649_data[1])) * v576_data);
              v645_acc += ((static_cast<float>(v649_data[2])) * v577_data);
              v645_acc += ((static_cast<float>(v649_data[3])) * v578_data);
              v645_acc += ((static_cast<float>(v649_data[4])) * v579_data);
              v645_acc += ((static_cast<float>(v649_data[5])) * v580_data);
              v645_acc += ((static_cast<float>(v649_data[6])) * v581_data);
              v645_acc += ((static_cast<float>(v649_data[7])) * v582_data);
              v645_acc += ((static_cast<float>(v649_data[8])) * v583_data);
              v645_acc += ((static_cast<float>(v649_data[9])) * v584_data);
              v645_acc += ((static_cast<float>(v649_data[10])) * v585_data);
              v645_acc += ((static_cast<float>(v649_data[11])) * v586_data);
              v645_acc.copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v674_acc{};
              tensorforge::intel_esimd::simd<float, 16> v678_data;
              v678_data.copy_from(s2 + (36_i32));
              v674_acc += ((static_cast<float>(v678_data[0])) * v575_data);
              v674_acc += ((static_cast<float>(v678_data[1])) * v576_data);
              v674_acc += ((static_cast<float>(v678_data[2])) * v577_data);
              v674_acc += ((static_cast<float>(v678_data[3])) * v578_data);
              v674_acc += ((static_cast<float>(v678_data[4])) * v579_data);
              v674_acc += ((static_cast<float>(v678_data[5])) * v580_data);
              v674_acc += ((static_cast<float>(v678_data[6])) * v581_data);
              v674_acc += ((static_cast<float>(v678_data[7])) * v582_data);
              v674_acc += ((static_cast<float>(v678_data[8])) * v583_data);
              v674_acc += ((static_cast<float>(v678_data[9])) * v584_data);
              v674_acc += ((static_cast<float>(v678_data[10])) * v585_data);
              v674_acc += ((static_cast<float>(v678_data[11])) * v586_data);
              v674_acc.copy_to(ir5 + (48));
              tensorforge::intel_esimd::simd<float, 16> v703_acc{};
              tensorforge::intel_esimd::simd<float, 16> v707_data;
              v707_data.copy_from(s2 + (48_i32));
              v703_acc += ((static_cast<float>(v707_data[0])) * v575_data);
              v703_acc += ((static_cast<float>(v707_data[1])) * v576_data);
              v703_acc += ((static_cast<float>(v707_data[2])) * v577_data);
              v703_acc += ((static_cast<float>(v707_data[3])) * v578_data);
              v703_acc += ((static_cast<float>(v707_data[4])) * v579_data);
              v703_acc += ((static_cast<float>(v707_data[5])) * v580_data);
              v703_acc += ((static_cast<float>(v707_data[6])) * v581_data);
              v703_acc += ((static_cast<float>(v707_data[7])) * v582_data);
              v703_acc += ((static_cast<float>(v707_data[8])) * v583_data);
              v703_acc += ((static_cast<float>(v707_data[9])) * v584_data);
              v703_acc += ((static_cast<float>(v707_data[10])) * v585_data);
              v703_acc += ((static_cast<float>(v707_data[11])) * v586_data);
              v703_acc.copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v732_acc{};
              tensorforge::intel_esimd::simd<float, 16> v736_data;
              v736_data.copy_from(s2 + (60_i32));
              v732_acc += ((static_cast<float>(v736_data[0])) * v575_data);
              v732_acc += ((static_cast<float>(v736_data[1])) * v576_data);
              v732_acc += ((static_cast<float>(v736_data[2])) * v577_data);
              v732_acc += ((static_cast<float>(v736_data[3])) * v578_data);
              v732_acc += ((static_cast<float>(v736_data[4])) * v579_data);
              v732_acc += ((static_cast<float>(v736_data[5])) * v580_data);
              v732_acc += ((static_cast<float>(v736_data[6])) * v581_data);
              v732_acc += ((static_cast<float>(v736_data[7])) * v582_data);
              v732_acc += ((static_cast<float>(v736_data[8])) * v583_data);
              v732_acc += ((static_cast<float>(v736_data[9])) * v584_data);
              v732_acc += ((static_cast<float>(v736_data[10])) * v585_data);
              v732_acc += ((static_cast<float>(v736_data[11])) * v586_data);
              v732_acc.copy_to(ir5 + (80));
              tensorforge::intel_esimd::simd<float, 16> v761_acc{};
              tensorforge::intel_esimd::simd<float, 16> v765_data;
              v765_data.copy_from(s2 + (72_i32));
              v761_acc += ((static_cast<float>(v765_data[0])) * v575_data);
              v761_acc += ((static_cast<float>(v765_data[1])) * v576_data);
              v761_acc += ((static_cast<float>(v765_data[2])) * v577_data);
              v761_acc += ((static_cast<float>(v765_data[3])) * v578_data);
              v761_acc += ((static_cast<float>(v765_data[4])) * v579_data);
              v761_acc += ((static_cast<float>(v765_data[5])) * v580_data);
              v761_acc += ((static_cast<float>(v765_data[6])) * v581_data);
              v761_acc += ((static_cast<float>(v765_data[7])) * v582_data);
              v761_acc += ((static_cast<float>(v765_data[8])) * v583_data);
              v761_acc += ((static_cast<float>(v765_data[9])) * v584_data);
              v761_acc += ((static_cast<float>(v765_data[10])) * v585_data);
              v761_acc += ((static_cast<float>(v765_data[11])) * v586_data);
              v761_acc.copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v790_acc{};
              tensorforge::intel_esimd::simd<float, 16> v794_data;
              v794_data.copy_from(s2 + (84_i32));
              v790_acc += ((static_cast<float>(v794_data[0])) * v575_data);
              v790_acc += ((static_cast<float>(v794_data[1])) * v576_data);
              v790_acc += ((static_cast<float>(v794_data[2])) * v577_data);
              v790_acc += ((static_cast<float>(v794_data[3])) * v578_data);
              v790_acc += ((static_cast<float>(v794_data[4])) * v579_data);
              v790_acc += ((static_cast<float>(v794_data[5])) * v580_data);
              v790_acc += ((static_cast<float>(v794_data[6])) * v581_data);
              v790_acc += ((static_cast<float>(v794_data[7])) * v582_data);
              v790_acc += ((static_cast<float>(v794_data[8])) * v583_data);
              v790_acc += ((static_cast<float>(v794_data[9])) * v584_data);
              v790_acc += ((static_cast<float>(v794_data[10])) * v585_data);
              v790_acc += ((static_cast<float>(v794_data[11])) * v586_data);
              v790_acc.copy_to(ir5 + (112));
              #pragma unroll
              for (int32_t v819_n1 = 0; v819_n1 < 8; ++v819_n1) {
                int32_t v820_a = v819_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v822_data;
                v822_data.copy_from(ir5 + (v820_a));
                tensorforge::intel_esimd::simd<float, 12> v825_data;
                v825_data.copy_from(r3 + (v820_a));
                (v825_data + v822_data).copy_to(r5 + (v820_a));
              }
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v829_ld;
              v829_ld.copy_from(glb_m8 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v829_ld.copy_to(s3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 32> v830_ld;
              v830_ld.copy_from(glb_m8 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              v830_ld.copy_to(s3 + (0 + 0 + 2 * item.get_local_id(0) + 64));
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              float r7[128]{};
              // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir7[128]{};
              tensorforge::intel_esimd::simd<float, 16> v833_data;
              v833_data.copy_from(r6 + (0));
              tensorforge::intel_esimd::simd<float, 16> v834_data;
              v834_data.copy_from(r6 + (16));
              tensorforge::intel_esimd::simd<float, 16> v835_data;
              v835_data.copy_from(r6 + (32));
              tensorforge::intel_esimd::simd<float, 16> v836_data;
              v836_data.copy_from(r6 + (48));
              tensorforge::intel_esimd::simd<float, 16> v837_data;
              v837_data.copy_from(r6 + (64));
              tensorforge::intel_esimd::simd<float, 16> v838_data;
              v838_data.copy_from(r6 + (80));
              tensorforge::intel_esimd::simd<float, 16> v839_data;
              v839_data.copy_from(r6 + (96));
              tensorforge::intel_esimd::simd<float, 16> v840_data;
              v840_data.copy_from(r6 + (112));
              tensorforge::intel_esimd::simd<float, 16> v841_data;
              v841_data.copy_from(r6 + (128));
              tensorforge::intel_esimd::simd<float, 16> v842_data;
              v842_data.copy_from(r6 + (144));
              tensorforge::intel_esimd::simd<float, 16> v843_data;
              v843_data.copy_from(r6 + (160));
              tensorforge::intel_esimd::simd<float, 16> v844_data;
              v844_data.copy_from(r6 + (176));
              tensorforge::intel_esimd::simd<float, 16> v845_acc{};
              tensorforge::intel_esimd::simd<float, 16> v849_data;
              v849_data.copy_from(s3 + (0_i32));
              v845_acc += ((static_cast<float>(v849_data[0])) * v833_data);
              v845_acc += ((static_cast<float>(v849_data[1])) * v834_data);
              v845_acc += ((static_cast<float>(v849_data[2])) * v835_data);
              v845_acc += ((static_cast<float>(v849_data[3])) * v836_data);
              v845_acc += ((static_cast<float>(v849_data[4])) * v837_data);
              v845_acc += ((static_cast<float>(v849_data[5])) * v838_data);
              v845_acc += ((static_cast<float>(v849_data[6])) * v839_data);
              v845_acc += ((static_cast<float>(v849_data[7])) * v840_data);
              v845_acc += ((static_cast<float>(v849_data[8])) * v841_data);
              v845_acc += ((static_cast<float>(v849_data[9])) * v842_data);
              v845_acc += ((static_cast<float>(v849_data[10])) * v843_data);
              v845_acc += ((static_cast<float>(v849_data[11])) * v844_data);
              v845_acc.copy_to(ir7 + (0));
              tensorforge::intel_esimd::simd<float, 16> v874_acc{};
              tensorforge::intel_esimd::simd<float, 16> v878_data;
              v878_data.copy_from(s3 + (12_i32));
              v874_acc += ((static_cast<float>(v878_data[0])) * v833_data);
              v874_acc += ((static_cast<float>(v878_data[1])) * v834_data);
              v874_acc += ((static_cast<float>(v878_data[2])) * v835_data);
              v874_acc += ((static_cast<float>(v878_data[3])) * v836_data);
              v874_acc += ((static_cast<float>(v878_data[4])) * v837_data);
              v874_acc += ((static_cast<float>(v878_data[5])) * v838_data);
              v874_acc += ((static_cast<float>(v878_data[6])) * v839_data);
              v874_acc += ((static_cast<float>(v878_data[7])) * v840_data);
              v874_acc += ((static_cast<float>(v878_data[8])) * v841_data);
              v874_acc += ((static_cast<float>(v878_data[9])) * v842_data);
              v874_acc += ((static_cast<float>(v878_data[10])) * v843_data);
              v874_acc += ((static_cast<float>(v878_data[11])) * v844_data);
              v874_acc.copy_to(ir7 + (16));
              tensorforge::intel_esimd::simd<float, 16> v903_acc{};
              tensorforge::intel_esimd::simd<float, 16> v907_data;
              v907_data.copy_from(s3 + (24_i32));
              v903_acc += ((static_cast<float>(v907_data[0])) * v833_data);
              v903_acc += ((static_cast<float>(v907_data[1])) * v834_data);
              v903_acc += ((static_cast<float>(v907_data[2])) * v835_data);
              v903_acc += ((static_cast<float>(v907_data[3])) * v836_data);
              v903_acc += ((static_cast<float>(v907_data[4])) * v837_data);
              v903_acc += ((static_cast<float>(v907_data[5])) * v838_data);
              v903_acc += ((static_cast<float>(v907_data[6])) * v839_data);
              v903_acc += ((static_cast<float>(v907_data[7])) * v840_data);
              v903_acc += ((static_cast<float>(v907_data[8])) * v841_data);
              v903_acc += ((static_cast<float>(v907_data[9])) * v842_data);
              v903_acc += ((static_cast<float>(v907_data[10])) * v843_data);
              v903_acc += ((static_cast<float>(v907_data[11])) * v844_data);
              v903_acc.copy_to(ir7 + (32));
              tensorforge::intel_esimd::simd<float, 16> v932_acc{};
              tensorforge::intel_esimd::simd<float, 16> v936_data;
              v936_data.copy_from(s3 + (36_i32));
              v932_acc += ((static_cast<float>(v936_data[0])) * v833_data);
              v932_acc += ((static_cast<float>(v936_data[1])) * v834_data);
              v932_acc += ((static_cast<float>(v936_data[2])) * v835_data);
              v932_acc += ((static_cast<float>(v936_data[3])) * v836_data);
              v932_acc += ((static_cast<float>(v936_data[4])) * v837_data);
              v932_acc += ((static_cast<float>(v936_data[5])) * v838_data);
              v932_acc += ((static_cast<float>(v936_data[6])) * v839_data);
              v932_acc += ((static_cast<float>(v936_data[7])) * v840_data);
              v932_acc += ((static_cast<float>(v936_data[8])) * v841_data);
              v932_acc += ((static_cast<float>(v936_data[9])) * v842_data);
              v932_acc += ((static_cast<float>(v936_data[10])) * v843_data);
              v932_acc += ((static_cast<float>(v936_data[11])) * v844_data);
              v932_acc.copy_to(ir7 + (48));
              tensorforge::intel_esimd::simd<float, 16> v961_acc{};
              tensorforge::intel_esimd::simd<float, 16> v965_data;
              v965_data.copy_from(s3 + (48_i32));
              v961_acc += ((static_cast<float>(v965_data[0])) * v833_data);
              v961_acc += ((static_cast<float>(v965_data[1])) * v834_data);
              v961_acc += ((static_cast<float>(v965_data[2])) * v835_data);
              v961_acc += ((static_cast<float>(v965_data[3])) * v836_data);
              v961_acc += ((static_cast<float>(v965_data[4])) * v837_data);
              v961_acc += ((static_cast<float>(v965_data[5])) * v838_data);
              v961_acc += ((static_cast<float>(v965_data[6])) * v839_data);
              v961_acc += ((static_cast<float>(v965_data[7])) * v840_data);
              v961_acc += ((static_cast<float>(v965_data[8])) * v841_data);
              v961_acc += ((static_cast<float>(v965_data[9])) * v842_data);
              v961_acc += ((static_cast<float>(v965_data[10])) * v843_data);
              v961_acc += ((static_cast<float>(v965_data[11])) * v844_data);
              v961_acc.copy_to(ir7 + (64));
              tensorforge::intel_esimd::simd<float, 16> v990_acc{};
              tensorforge::intel_esimd::simd<float, 16> v994_data;
              v994_data.copy_from(s3 + (60_i32));
              v990_acc += ((static_cast<float>(v994_data[0])) * v833_data);
              v990_acc += ((static_cast<float>(v994_data[1])) * v834_data);
              v990_acc += ((static_cast<float>(v994_data[2])) * v835_data);
              v990_acc += ((static_cast<float>(v994_data[3])) * v836_data);
              v990_acc += ((static_cast<float>(v994_data[4])) * v837_data);
              v990_acc += ((static_cast<float>(v994_data[5])) * v838_data);
              v990_acc += ((static_cast<float>(v994_data[6])) * v839_data);
              v990_acc += ((static_cast<float>(v994_data[7])) * v840_data);
              v990_acc += ((static_cast<float>(v994_data[8])) * v841_data);
              v990_acc += ((static_cast<float>(v994_data[9])) * v842_data);
              v990_acc += ((static_cast<float>(v994_data[10])) * v843_data);
              v990_acc += ((static_cast<float>(v994_data[11])) * v844_data);
              v990_acc.copy_to(ir7 + (80));
              tensorforge::intel_esimd::simd<float, 16> v1019_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1023_data;
              v1023_data.copy_from(s3 + (72_i32));
              v1019_acc += ((static_cast<float>(v1023_data[0])) * v833_data);
              v1019_acc += ((static_cast<float>(v1023_data[1])) * v834_data);
              v1019_acc += ((static_cast<float>(v1023_data[2])) * v835_data);
              v1019_acc += ((static_cast<float>(v1023_data[3])) * v836_data);
              v1019_acc += ((static_cast<float>(v1023_data[4])) * v837_data);
              v1019_acc += ((static_cast<float>(v1023_data[5])) * v838_data);
              v1019_acc += ((static_cast<float>(v1023_data[6])) * v839_data);
              v1019_acc += ((static_cast<float>(v1023_data[7])) * v840_data);
              v1019_acc += ((static_cast<float>(v1023_data[8])) * v841_data);
              v1019_acc += ((static_cast<float>(v1023_data[9])) * v842_data);
              v1019_acc += ((static_cast<float>(v1023_data[10])) * v843_data);
              v1019_acc += ((static_cast<float>(v1023_data[11])) * v844_data);
              v1019_acc.copy_to(ir7 + (96));
              tensorforge::intel_esimd::simd<float, 16> v1048_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1052_data;
              v1052_data.copy_from(s3 + (84_i32));
              v1048_acc += ((static_cast<float>(v1052_data[0])) * v833_data);
              v1048_acc += ((static_cast<float>(v1052_data[1])) * v834_data);
              v1048_acc += ((static_cast<float>(v1052_data[2])) * v835_data);
              v1048_acc += ((static_cast<float>(v1052_data[3])) * v836_data);
              v1048_acc += ((static_cast<float>(v1052_data[4])) * v837_data);
              v1048_acc += ((static_cast<float>(v1052_data[5])) * v838_data);
              v1048_acc += ((static_cast<float>(v1052_data[6])) * v839_data);
              v1048_acc += ((static_cast<float>(v1052_data[7])) * v840_data);
              v1048_acc += ((static_cast<float>(v1052_data[8])) * v841_data);
              v1048_acc += ((static_cast<float>(v1052_data[9])) * v842_data);
              v1048_acc += ((static_cast<float>(v1052_data[10])) * v843_data);
              v1048_acc += ((static_cast<float>(v1052_data[11])) * v844_data);
              v1048_acc.copy_to(ir7 + (112));
              #pragma unroll
              for (int32_t v1077_n1 = 0; v1077_n1 < 8; ++v1077_n1) {
                int32_t v1078_a = v1077_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1080_data;
                v1080_data.copy_from(ir7 + (v1078_a));
                tensorforge::intel_esimd::simd<float, 12> v1083_data;
                v1083_data.copy_from(r5 + (v1078_a));
                (v1083_data + v1080_data).copy_to(r7 + (v1078_a));
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1087_i1 = 0; v1087_i1 < 8; ++v1087_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1090_data;
                v1090_data.copy_from(r7 + ((v1087_i1 * 16)));
                v1090_data.copy_to(glb_m0 + ((v1087_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

