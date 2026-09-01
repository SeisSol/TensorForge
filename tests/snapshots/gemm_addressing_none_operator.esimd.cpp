// === base name ===
kernel_151d4e8604

// === header ===
void launcher_kernel_151d4e8604(float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_151d4e8604(float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_151d4e8604(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_151d4e8604(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} none
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} none({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[272 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[256];
          const float *const __restrict__ glb_m1 = &m1[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v6_ld;
              v6_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v6_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v7_ld;
              v7_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v7_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              tensorforge::intel_esimd::simd<float, 64> v8_ld;
              v8_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              v8_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              tensorforge::intel_esimd::simd<float, 64> v9_ld;
              v9_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              v9_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v15_data;
              v15_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data;
              v63_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_acc{};
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(s0 + (0_i32));
              v76_acc += ((v80_data[0]) * v15_data);
              v76_acc += ((v80_data[1]) * v19_data);
              v76_acc += ((v80_data[2]) * v23_data);
              v76_acc += ((v80_data[3]) * v27_data);
              v76_acc += ((v80_data[4]) * v31_data);
              v76_acc += ((v80_data[5]) * v35_data);
              v76_acc += ((v80_data[6]) * v39_data);
              v76_acc += ((v80_data[7]) * v43_data);
              v76_acc += ((v80_data[8]) * v47_data);
              v76_acc += ((v80_data[9]) * v51_data);
              v76_acc += ((v80_data[10]) * v55_data);
              v76_acc += ((v80_data[11]) * v59_data);
              v76_acc += ((v80_data[12]) * v63_data);
              v76_acc += ((v80_data[13]) * v67_data);
              v76_acc += ((v80_data[14]) * v71_data);
              v76_acc += ((v80_data[15]) * v75_data);
              v76_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              tensorforge::intel_esimd::simd<float, 16> v117_data;
              v117_data.copy_from(s0 + (16_i32));
              v113_acc += ((v117_data[0]) * v15_data);
              v113_acc += ((v117_data[1]) * v19_data);
              v113_acc += ((v117_data[2]) * v23_data);
              v113_acc += ((v117_data[3]) * v27_data);
              v113_acc += ((v117_data[4]) * v31_data);
              v113_acc += ((v117_data[5]) * v35_data);
              v113_acc += ((v117_data[6]) * v39_data);
              v113_acc += ((v117_data[7]) * v43_data);
              v113_acc += ((v117_data[8]) * v47_data);
              v113_acc += ((v117_data[9]) * v51_data);
              v113_acc += ((v117_data[10]) * v55_data);
              v113_acc += ((v117_data[11]) * v59_data);
              v113_acc += ((v117_data[12]) * v63_data);
              v113_acc += ((v117_data[13]) * v67_data);
              v113_acc += ((v117_data[14]) * v71_data);
              v113_acc += ((v117_data[15]) * v75_data);
              v113_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v150_acc{};
              tensorforge::intel_esimd::simd<float, 16> v154_data;
              v154_data.copy_from(s0 + (32_i32));
              v150_acc += ((v154_data[0]) * v15_data);
              v150_acc += ((v154_data[1]) * v19_data);
              v150_acc += ((v154_data[2]) * v23_data);
              v150_acc += ((v154_data[3]) * v27_data);
              v150_acc += ((v154_data[4]) * v31_data);
              v150_acc += ((v154_data[5]) * v35_data);
              v150_acc += ((v154_data[6]) * v39_data);
              v150_acc += ((v154_data[7]) * v43_data);
              v150_acc += ((v154_data[8]) * v47_data);
              v150_acc += ((v154_data[9]) * v51_data);
              v150_acc += ((v154_data[10]) * v55_data);
              v150_acc += ((v154_data[11]) * v59_data);
              v150_acc += ((v154_data[12]) * v63_data);
              v150_acc += ((v154_data[13]) * v67_data);
              v150_acc += ((v154_data[14]) * v71_data);
              v150_acc += ((v154_data[15]) * v75_data);
              v150_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v187_acc{};
              tensorforge::intel_esimd::simd<float, 16> v191_data;
              v191_data.copy_from(s0 + (48_i32));
              v187_acc += ((v191_data[0]) * v15_data);
              v187_acc += ((v191_data[1]) * v19_data);
              v187_acc += ((v191_data[2]) * v23_data);
              v187_acc += ((v191_data[3]) * v27_data);
              v187_acc += ((v191_data[4]) * v31_data);
              v187_acc += ((v191_data[5]) * v35_data);
              v187_acc += ((v191_data[6]) * v39_data);
              v187_acc += ((v191_data[7]) * v43_data);
              v187_acc += ((v191_data[8]) * v47_data);
              v187_acc += ((v191_data[9]) * v51_data);
              v187_acc += ((v191_data[10]) * v55_data);
              v187_acc += ((v191_data[11]) * v59_data);
              v187_acc += ((v191_data[12]) * v63_data);
              v187_acc += ((v191_data[13]) * v67_data);
              v187_acc += ((v191_data[14]) * v71_data);
              v187_acc += ((v191_data[15]) * v75_data);
              v187_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v224_acc{};
              tensorforge::intel_esimd::simd<float, 16> v228_data;
              v228_data.copy_from(s0 + (64_i32));
              v224_acc += ((v228_data[0]) * v15_data);
              v224_acc += ((v228_data[1]) * v19_data);
              v224_acc += ((v228_data[2]) * v23_data);
              v224_acc += ((v228_data[3]) * v27_data);
              v224_acc += ((v228_data[4]) * v31_data);
              v224_acc += ((v228_data[5]) * v35_data);
              v224_acc += ((v228_data[6]) * v39_data);
              v224_acc += ((v228_data[7]) * v43_data);
              v224_acc += ((v228_data[8]) * v47_data);
              v224_acc += ((v228_data[9]) * v51_data);
              v224_acc += ((v228_data[10]) * v55_data);
              v224_acc += ((v228_data[11]) * v59_data);
              v224_acc += ((v228_data[12]) * v63_data);
              v224_acc += ((v228_data[13]) * v67_data);
              v224_acc += ((v228_data[14]) * v71_data);
              v224_acc += ((v228_data[15]) * v75_data);
              v224_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v261_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data;
              v265_data.copy_from(s0 + (80_i32));
              v261_acc += ((v265_data[0]) * v15_data);
              v261_acc += ((v265_data[1]) * v19_data);
              v261_acc += ((v265_data[2]) * v23_data);
              v261_acc += ((v265_data[3]) * v27_data);
              v261_acc += ((v265_data[4]) * v31_data);
              v261_acc += ((v265_data[5]) * v35_data);
              v261_acc += ((v265_data[6]) * v39_data);
              v261_acc += ((v265_data[7]) * v43_data);
              v261_acc += ((v265_data[8]) * v47_data);
              v261_acc += ((v265_data[9]) * v51_data);
              v261_acc += ((v265_data[10]) * v55_data);
              v261_acc += ((v265_data[11]) * v59_data);
              v261_acc += ((v265_data[12]) * v63_data);
              v261_acc += ((v265_data[13]) * v67_data);
              v261_acc += ((v265_data[14]) * v71_data);
              v261_acc += ((v265_data[15]) * v75_data);
              v261_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v298_acc{};
              tensorforge::intel_esimd::simd<float, 16> v302_data;
              v302_data.copy_from(s0 + (96_i32));
              v298_acc += ((v302_data[0]) * v15_data);
              v298_acc += ((v302_data[1]) * v19_data);
              v298_acc += ((v302_data[2]) * v23_data);
              v298_acc += ((v302_data[3]) * v27_data);
              v298_acc += ((v302_data[4]) * v31_data);
              v298_acc += ((v302_data[5]) * v35_data);
              v298_acc += ((v302_data[6]) * v39_data);
              v298_acc += ((v302_data[7]) * v43_data);
              v298_acc += ((v302_data[8]) * v47_data);
              v298_acc += ((v302_data[9]) * v51_data);
              v298_acc += ((v302_data[10]) * v55_data);
              v298_acc += ((v302_data[11]) * v59_data);
              v298_acc += ((v302_data[12]) * v63_data);
              v298_acc += ((v302_data[13]) * v67_data);
              v298_acc += ((v302_data[14]) * v71_data);
              v298_acc += ((v302_data[15]) * v75_data);
              v298_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v335_acc{};
              tensorforge::intel_esimd::simd<float, 16> v339_data;
              v339_data.copy_from(s0 + (112_i32));
              v335_acc += ((v339_data[0]) * v15_data);
              v335_acc += ((v339_data[1]) * v19_data);
              v335_acc += ((v339_data[2]) * v23_data);
              v335_acc += ((v339_data[3]) * v27_data);
              v335_acc += ((v339_data[4]) * v31_data);
              v335_acc += ((v339_data[5]) * v35_data);
              v335_acc += ((v339_data[6]) * v39_data);
              v335_acc += ((v339_data[7]) * v43_data);
              v335_acc += ((v339_data[8]) * v47_data);
              v335_acc += ((v339_data[9]) * v51_data);
              v335_acc += ((v339_data[10]) * v55_data);
              v335_acc += ((v339_data[11]) * v59_data);
              v335_acc += ((v339_data[12]) * v63_data);
              v335_acc += ((v339_data[13]) * v67_data);
              v335_acc += ((v339_data[14]) * v71_data);
              v335_acc += ((v339_data[15]) * v75_data);
              v335_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v372_acc{};
              tensorforge::intel_esimd::simd<float, 16> v376_data;
              v376_data.copy_from(s0 + (128_i32));
              v372_acc += ((v376_data[0]) * v15_data);
              v372_acc += ((v376_data[1]) * v19_data);
              v372_acc += ((v376_data[2]) * v23_data);
              v372_acc += ((v376_data[3]) * v27_data);
              v372_acc += ((v376_data[4]) * v31_data);
              v372_acc += ((v376_data[5]) * v35_data);
              v372_acc += ((v376_data[6]) * v39_data);
              v372_acc += ((v376_data[7]) * v43_data);
              v372_acc += ((v376_data[8]) * v47_data);
              v372_acc += ((v376_data[9]) * v51_data);
              v372_acc += ((v376_data[10]) * v55_data);
              v372_acc += ((v376_data[11]) * v59_data);
              v372_acc += ((v376_data[12]) * v63_data);
              v372_acc += ((v376_data[13]) * v67_data);
              v372_acc += ((v376_data[14]) * v71_data);
              v372_acc += ((v376_data[15]) * v75_data);
              v372_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v409_acc{};
              tensorforge::intel_esimd::simd<float, 16> v413_data;
              v413_data.copy_from(s0 + (144_i32));
              v409_acc += ((v413_data[0]) * v15_data);
              v409_acc += ((v413_data[1]) * v19_data);
              v409_acc += ((v413_data[2]) * v23_data);
              v409_acc += ((v413_data[3]) * v27_data);
              v409_acc += ((v413_data[4]) * v31_data);
              v409_acc += ((v413_data[5]) * v35_data);
              v409_acc += ((v413_data[6]) * v39_data);
              v409_acc += ((v413_data[7]) * v43_data);
              v409_acc += ((v413_data[8]) * v47_data);
              v409_acc += ((v413_data[9]) * v51_data);
              v409_acc += ((v413_data[10]) * v55_data);
              v409_acc += ((v413_data[11]) * v59_data);
              v409_acc += ((v413_data[12]) * v63_data);
              v409_acc += ((v413_data[13]) * v67_data);
              v409_acc += ((v413_data[14]) * v71_data);
              v409_acc += ((v413_data[15]) * v75_data);
              v409_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v446_acc{};
              tensorforge::intel_esimd::simd<float, 16> v450_data;
              v450_data.copy_from(s0 + (160_i32));
              v446_acc += ((v450_data[0]) * v15_data);
              v446_acc += ((v450_data[1]) * v19_data);
              v446_acc += ((v450_data[2]) * v23_data);
              v446_acc += ((v450_data[3]) * v27_data);
              v446_acc += ((v450_data[4]) * v31_data);
              v446_acc += ((v450_data[5]) * v35_data);
              v446_acc += ((v450_data[6]) * v39_data);
              v446_acc += ((v450_data[7]) * v43_data);
              v446_acc += ((v450_data[8]) * v47_data);
              v446_acc += ((v450_data[9]) * v51_data);
              v446_acc += ((v450_data[10]) * v55_data);
              v446_acc += ((v450_data[11]) * v59_data);
              v446_acc += ((v450_data[12]) * v63_data);
              v446_acc += ((v450_data[13]) * v67_data);
              v446_acc += ((v450_data[14]) * v71_data);
              v446_acc += ((v450_data[15]) * v75_data);
              v446_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v483_acc{};
              tensorforge::intel_esimd::simd<float, 16> v487_data;
              v487_data.copy_from(s0 + (176_i32));
              v483_acc += ((v487_data[0]) * v15_data);
              v483_acc += ((v487_data[1]) * v19_data);
              v483_acc += ((v487_data[2]) * v23_data);
              v483_acc += ((v487_data[3]) * v27_data);
              v483_acc += ((v487_data[4]) * v31_data);
              v483_acc += ((v487_data[5]) * v35_data);
              v483_acc += ((v487_data[6]) * v39_data);
              v483_acc += ((v487_data[7]) * v43_data);
              v483_acc += ((v487_data[8]) * v47_data);
              v483_acc += ((v487_data[9]) * v51_data);
              v483_acc += ((v487_data[10]) * v55_data);
              v483_acc += ((v487_data[11]) * v59_data);
              v483_acc += ((v487_data[12]) * v63_data);
              v483_acc += ((v487_data[13]) * v67_data);
              v483_acc += ((v487_data[14]) * v71_data);
              v483_acc += ((v487_data[15]) * v75_data);
              v483_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v520_acc{};
              tensorforge::intel_esimd::simd<float, 16> v524_data;
              v524_data.copy_from(s0 + (192_i32));
              v520_acc += ((v524_data[0]) * v15_data);
              v520_acc += ((v524_data[1]) * v19_data);
              v520_acc += ((v524_data[2]) * v23_data);
              v520_acc += ((v524_data[3]) * v27_data);
              v520_acc += ((v524_data[4]) * v31_data);
              v520_acc += ((v524_data[5]) * v35_data);
              v520_acc += ((v524_data[6]) * v39_data);
              v520_acc += ((v524_data[7]) * v43_data);
              v520_acc += ((v524_data[8]) * v47_data);
              v520_acc += ((v524_data[9]) * v51_data);
              v520_acc += ((v524_data[10]) * v55_data);
              v520_acc += ((v524_data[11]) * v59_data);
              v520_acc += ((v524_data[12]) * v63_data);
              v520_acc += ((v524_data[13]) * v67_data);
              v520_acc += ((v524_data[14]) * v71_data);
              v520_acc += ((v524_data[15]) * v75_data);
              v520_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v557_acc{};
              tensorforge::intel_esimd::simd<float, 16> v561_data;
              v561_data.copy_from(s0 + (208_i32));
              v557_acc += ((v561_data[0]) * v15_data);
              v557_acc += ((v561_data[1]) * v19_data);
              v557_acc += ((v561_data[2]) * v23_data);
              v557_acc += ((v561_data[3]) * v27_data);
              v557_acc += ((v561_data[4]) * v31_data);
              v557_acc += ((v561_data[5]) * v35_data);
              v557_acc += ((v561_data[6]) * v39_data);
              v557_acc += ((v561_data[7]) * v43_data);
              v557_acc += ((v561_data[8]) * v47_data);
              v557_acc += ((v561_data[9]) * v51_data);
              v557_acc += ((v561_data[10]) * v55_data);
              v557_acc += ((v561_data[11]) * v59_data);
              v557_acc += ((v561_data[12]) * v63_data);
              v557_acc += ((v561_data[13]) * v67_data);
              v557_acc += ((v561_data[14]) * v71_data);
              v557_acc += ((v561_data[15]) * v75_data);
              v557_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v594_acc{};
              tensorforge::intel_esimd::simd<float, 16> v598_data;
              v598_data.copy_from(s0 + (224_i32));
              v594_acc += ((v598_data[0]) * v15_data);
              v594_acc += ((v598_data[1]) * v19_data);
              v594_acc += ((v598_data[2]) * v23_data);
              v594_acc += ((v598_data[3]) * v27_data);
              v594_acc += ((v598_data[4]) * v31_data);
              v594_acc += ((v598_data[5]) * v35_data);
              v594_acc += ((v598_data[6]) * v39_data);
              v594_acc += ((v598_data[7]) * v43_data);
              v594_acc += ((v598_data[8]) * v47_data);
              v594_acc += ((v598_data[9]) * v51_data);
              v594_acc += ((v598_data[10]) * v55_data);
              v594_acc += ((v598_data[11]) * v59_data);
              v594_acc += ((v598_data[12]) * v63_data);
              v594_acc += ((v598_data[13]) * v67_data);
              v594_acc += ((v598_data[14]) * v71_data);
              v594_acc += ((v598_data[15]) * v75_data);
              v594_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v631_acc{};
              tensorforge::intel_esimd::simd<float, 16> v635_data;
              v635_data.copy_from(s0 + (240_i32));
              v631_acc += ((v635_data[0]) * v15_data);
              v631_acc += ((v635_data[1]) * v19_data);
              v631_acc += ((v635_data[2]) * v23_data);
              v631_acc += ((v635_data[3]) * v27_data);
              v631_acc += ((v635_data[4]) * v31_data);
              v631_acc += ((v635_data[5]) * v35_data);
              v631_acc += ((v635_data[6]) * v39_data);
              v631_acc += ((v635_data[7]) * v43_data);
              v631_acc += ((v635_data[8]) * v47_data);
              v631_acc += ((v635_data[9]) * v51_data);
              v631_acc += ((v635_data[10]) * v55_data);
              v631_acc += ((v635_data[11]) * v59_data);
              v631_acc += ((v635_data[12]) * v63_data);
              v631_acc += ((v635_data[13]) * v67_data);
              v631_acc += ((v635_data[14]) * v71_data);
              v631_acc += ((v635_data[15]) * v75_data);
              v631_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v668_n0 = 0; v668_n0 < 1; ++v668_n0) {
                int32_t v670_a = v668_n0 * 16;
                #pragma unroll
                for (int32_t v669_n1 = 0; v669_n1 < 16; ++v669_n1) {
                  int32_t v672_a = v670_a + (v669_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v673_data;
                  v673_data.copy_from(ir0 + (v672_a));
                  v673_data.copy_to(r0 + (v672_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v677_i0 = 0; v677_i0 < 1; ++v677_i0) {
                int32_t v679_a = v677_i0 * 16;
                #pragma unroll
                for (int32_t v678_i1 = 0; v678_i1 < 16; ++v678_i1) {
                  int32_t v681_a = v679_a + (v678_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v682_data;
                  v682_data.copy_from(r0 + (v681_a));
                  v682_data.copy_to(glb_m0 + (v681_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

