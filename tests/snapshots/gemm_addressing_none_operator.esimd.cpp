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
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 128] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 128];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 192] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 192];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v9_data;
              v9_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v13_data;
              v13_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v17_data;
              v17_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v53_data;
              v53_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v57_data;
              v57_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data;
              v69_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v70_acc += ((v77_data[0]) * v9_data);
              v70_acc += ((v77_data[1]) * v13_data);
              v70_acc += ((v77_data[2]) * v17_data);
              v70_acc += ((v77_data[3]) * v21_data);
              v70_acc += ((v77_data[4]) * v25_data);
              v70_acc += ((v77_data[5]) * v29_data);
              v70_acc += ((v77_data[6]) * v33_data);
              v70_acc += ((v77_data[7]) * v37_data);
              v70_acc += ((v77_data[8]) * v41_data);
              v70_acc += ((v77_data[9]) * v45_data);
              v70_acc += ((v77_data[10]) * v49_data);
              v70_acc += ((v77_data[11]) * v53_data);
              v70_acc += ((v77_data[12]) * v57_data);
              v70_acc += ((v77_data[13]) * v61_data);
              v70_acc += ((v77_data[14]) * v65_data);
              v70_acc += ((v77_data[15]) * v69_data);
              v70_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v117_data;
              v117_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v110_acc += ((v117_data[0]) * v9_data);
              v110_acc += ((v117_data[1]) * v13_data);
              v110_acc += ((v117_data[2]) * v17_data);
              v110_acc += ((v117_data[3]) * v21_data);
              v110_acc += ((v117_data[4]) * v25_data);
              v110_acc += ((v117_data[5]) * v29_data);
              v110_acc += ((v117_data[6]) * v33_data);
              v110_acc += ((v117_data[7]) * v37_data);
              v110_acc += ((v117_data[8]) * v41_data);
              v110_acc += ((v117_data[9]) * v45_data);
              v110_acc += ((v117_data[10]) * v49_data);
              v110_acc += ((v117_data[11]) * v53_data);
              v110_acc += ((v117_data[12]) * v57_data);
              v110_acc += ((v117_data[13]) * v61_data);
              v110_acc += ((v117_data[14]) * v65_data);
              v110_acc += ((v117_data[15]) * v69_data);
              v110_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v150_acc{};
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 5) & 31))));
              v150_acc += ((v157_data[0]) * v9_data);
              v150_acc += ((v157_data[1]) * v13_data);
              v150_acc += ((v157_data[2]) * v17_data);
              v150_acc += ((v157_data[3]) * v21_data);
              v150_acc += ((v157_data[4]) * v25_data);
              v150_acc += ((v157_data[5]) * v29_data);
              v150_acc += ((v157_data[6]) * v33_data);
              v150_acc += ((v157_data[7]) * v37_data);
              v150_acc += ((v157_data[8]) * v41_data);
              v150_acc += ((v157_data[9]) * v45_data);
              v150_acc += ((v157_data[10]) * v49_data);
              v150_acc += ((v157_data[11]) * v53_data);
              v150_acc += ((v157_data[12]) * v57_data);
              v150_acc += ((v157_data[13]) * v61_data);
              v150_acc += ((v157_data[14]) * v65_data);
              v150_acc += ((v157_data[15]) * v69_data);
              v150_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              tensorforge::intel_esimd::simd<float, 16> v197_data;
              v197_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 5) & 31))));
              v190_acc += ((v197_data[0]) * v9_data);
              v190_acc += ((v197_data[1]) * v13_data);
              v190_acc += ((v197_data[2]) * v17_data);
              v190_acc += ((v197_data[3]) * v21_data);
              v190_acc += ((v197_data[4]) * v25_data);
              v190_acc += ((v197_data[5]) * v29_data);
              v190_acc += ((v197_data[6]) * v33_data);
              v190_acc += ((v197_data[7]) * v37_data);
              v190_acc += ((v197_data[8]) * v41_data);
              v190_acc += ((v197_data[9]) * v45_data);
              v190_acc += ((v197_data[10]) * v49_data);
              v190_acc += ((v197_data[11]) * v53_data);
              v190_acc += ((v197_data[12]) * v57_data);
              v190_acc += ((v197_data[13]) * v61_data);
              v190_acc += ((v197_data[14]) * v65_data);
              v190_acc += ((v197_data[15]) * v69_data);
              v190_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v230_acc{};
              tensorforge::intel_esimd::simd<float, 16> v237_data;
              v237_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 5) & 31))));
              v230_acc += ((v237_data[0]) * v9_data);
              v230_acc += ((v237_data[1]) * v13_data);
              v230_acc += ((v237_data[2]) * v17_data);
              v230_acc += ((v237_data[3]) * v21_data);
              v230_acc += ((v237_data[4]) * v25_data);
              v230_acc += ((v237_data[5]) * v29_data);
              v230_acc += ((v237_data[6]) * v33_data);
              v230_acc += ((v237_data[7]) * v37_data);
              v230_acc += ((v237_data[8]) * v41_data);
              v230_acc += ((v237_data[9]) * v45_data);
              v230_acc += ((v237_data[10]) * v49_data);
              v230_acc += ((v237_data[11]) * v53_data);
              v230_acc += ((v237_data[12]) * v57_data);
              v230_acc += ((v237_data[13]) * v61_data);
              v230_acc += ((v237_data[14]) * v65_data);
              v230_acc += ((v237_data[15]) * v69_data);
              v230_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v270_acc{};
              tensorforge::intel_esimd::simd<float, 16> v277_data;
              v277_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v270_acc += ((v277_data[0]) * v9_data);
              v270_acc += ((v277_data[1]) * v13_data);
              v270_acc += ((v277_data[2]) * v17_data);
              v270_acc += ((v277_data[3]) * v21_data);
              v270_acc += ((v277_data[4]) * v25_data);
              v270_acc += ((v277_data[5]) * v29_data);
              v270_acc += ((v277_data[6]) * v33_data);
              v270_acc += ((v277_data[7]) * v37_data);
              v270_acc += ((v277_data[8]) * v41_data);
              v270_acc += ((v277_data[9]) * v45_data);
              v270_acc += ((v277_data[10]) * v49_data);
              v270_acc += ((v277_data[11]) * v53_data);
              v270_acc += ((v277_data[12]) * v57_data);
              v270_acc += ((v277_data[13]) * v61_data);
              v270_acc += ((v277_data[14]) * v65_data);
              v270_acc += ((v277_data[15]) * v69_data);
              v270_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v310_acc{};
              tensorforge::intel_esimd::simd<float, 16> v317_data;
              v317_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v310_acc += ((v317_data[0]) * v9_data);
              v310_acc += ((v317_data[1]) * v13_data);
              v310_acc += ((v317_data[2]) * v17_data);
              v310_acc += ((v317_data[3]) * v21_data);
              v310_acc += ((v317_data[4]) * v25_data);
              v310_acc += ((v317_data[5]) * v29_data);
              v310_acc += ((v317_data[6]) * v33_data);
              v310_acc += ((v317_data[7]) * v37_data);
              v310_acc += ((v317_data[8]) * v41_data);
              v310_acc += ((v317_data[9]) * v45_data);
              v310_acc += ((v317_data[10]) * v49_data);
              v310_acc += ((v317_data[11]) * v53_data);
              v310_acc += ((v317_data[12]) * v57_data);
              v310_acc += ((v317_data[13]) * v61_data);
              v310_acc += ((v317_data[14]) * v65_data);
              v310_acc += ((v317_data[15]) * v69_data);
              v310_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v350_acc{};
              tensorforge::intel_esimd::simd<float, 16> v357_data;
              v357_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 5) & 31))));
              v350_acc += ((v357_data[0]) * v9_data);
              v350_acc += ((v357_data[1]) * v13_data);
              v350_acc += ((v357_data[2]) * v17_data);
              v350_acc += ((v357_data[3]) * v21_data);
              v350_acc += ((v357_data[4]) * v25_data);
              v350_acc += ((v357_data[5]) * v29_data);
              v350_acc += ((v357_data[6]) * v33_data);
              v350_acc += ((v357_data[7]) * v37_data);
              v350_acc += ((v357_data[8]) * v41_data);
              v350_acc += ((v357_data[9]) * v45_data);
              v350_acc += ((v357_data[10]) * v49_data);
              v350_acc += ((v357_data[11]) * v53_data);
              v350_acc += ((v357_data[12]) * v57_data);
              v350_acc += ((v357_data[13]) * v61_data);
              v350_acc += ((v357_data[14]) * v65_data);
              v350_acc += ((v357_data[15]) * v69_data);
              v350_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v390_acc{};
              tensorforge::intel_esimd::simd<float, 16> v397_data;
              v397_data.copy_from(s0 + ((128_i32 ^ ((128_i32 >> 5) & 31))));
              v390_acc += ((v397_data[0]) * v9_data);
              v390_acc += ((v397_data[1]) * v13_data);
              v390_acc += ((v397_data[2]) * v17_data);
              v390_acc += ((v397_data[3]) * v21_data);
              v390_acc += ((v397_data[4]) * v25_data);
              v390_acc += ((v397_data[5]) * v29_data);
              v390_acc += ((v397_data[6]) * v33_data);
              v390_acc += ((v397_data[7]) * v37_data);
              v390_acc += ((v397_data[8]) * v41_data);
              v390_acc += ((v397_data[9]) * v45_data);
              v390_acc += ((v397_data[10]) * v49_data);
              v390_acc += ((v397_data[11]) * v53_data);
              v390_acc += ((v397_data[12]) * v57_data);
              v390_acc += ((v397_data[13]) * v61_data);
              v390_acc += ((v397_data[14]) * v65_data);
              v390_acc += ((v397_data[15]) * v69_data);
              v390_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v430_acc{};
              tensorforge::intel_esimd::simd<float, 16> v437_data;
              v437_data.copy_from(s0 + ((144_i32 ^ ((144_i32 >> 5) & 31))));
              v430_acc += ((v437_data[0]) * v9_data);
              v430_acc += ((v437_data[1]) * v13_data);
              v430_acc += ((v437_data[2]) * v17_data);
              v430_acc += ((v437_data[3]) * v21_data);
              v430_acc += ((v437_data[4]) * v25_data);
              v430_acc += ((v437_data[5]) * v29_data);
              v430_acc += ((v437_data[6]) * v33_data);
              v430_acc += ((v437_data[7]) * v37_data);
              v430_acc += ((v437_data[8]) * v41_data);
              v430_acc += ((v437_data[9]) * v45_data);
              v430_acc += ((v437_data[10]) * v49_data);
              v430_acc += ((v437_data[11]) * v53_data);
              v430_acc += ((v437_data[12]) * v57_data);
              v430_acc += ((v437_data[13]) * v61_data);
              v430_acc += ((v437_data[14]) * v65_data);
              v430_acc += ((v437_data[15]) * v69_data);
              v430_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v470_acc{};
              tensorforge::intel_esimd::simd<float, 16> v477_data;
              v477_data.copy_from(s0 + ((160_i32 ^ ((160_i32 >> 5) & 31))));
              v470_acc += ((v477_data[0]) * v9_data);
              v470_acc += ((v477_data[1]) * v13_data);
              v470_acc += ((v477_data[2]) * v17_data);
              v470_acc += ((v477_data[3]) * v21_data);
              v470_acc += ((v477_data[4]) * v25_data);
              v470_acc += ((v477_data[5]) * v29_data);
              v470_acc += ((v477_data[6]) * v33_data);
              v470_acc += ((v477_data[7]) * v37_data);
              v470_acc += ((v477_data[8]) * v41_data);
              v470_acc += ((v477_data[9]) * v45_data);
              v470_acc += ((v477_data[10]) * v49_data);
              v470_acc += ((v477_data[11]) * v53_data);
              v470_acc += ((v477_data[12]) * v57_data);
              v470_acc += ((v477_data[13]) * v61_data);
              v470_acc += ((v477_data[14]) * v65_data);
              v470_acc += ((v477_data[15]) * v69_data);
              v470_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v510_acc{};
              tensorforge::intel_esimd::simd<float, 16> v517_data;
              v517_data.copy_from(s0 + ((176_i32 ^ ((176_i32 >> 5) & 31))));
              v510_acc += ((v517_data[0]) * v9_data);
              v510_acc += ((v517_data[1]) * v13_data);
              v510_acc += ((v517_data[2]) * v17_data);
              v510_acc += ((v517_data[3]) * v21_data);
              v510_acc += ((v517_data[4]) * v25_data);
              v510_acc += ((v517_data[5]) * v29_data);
              v510_acc += ((v517_data[6]) * v33_data);
              v510_acc += ((v517_data[7]) * v37_data);
              v510_acc += ((v517_data[8]) * v41_data);
              v510_acc += ((v517_data[9]) * v45_data);
              v510_acc += ((v517_data[10]) * v49_data);
              v510_acc += ((v517_data[11]) * v53_data);
              v510_acc += ((v517_data[12]) * v57_data);
              v510_acc += ((v517_data[13]) * v61_data);
              v510_acc += ((v517_data[14]) * v65_data);
              v510_acc += ((v517_data[15]) * v69_data);
              v510_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v550_acc{};
              tensorforge::intel_esimd::simd<float, 16> v557_data;
              v557_data.copy_from(s0 + ((192_i32 ^ ((192_i32 >> 5) & 31))));
              v550_acc += ((v557_data[0]) * v9_data);
              v550_acc += ((v557_data[1]) * v13_data);
              v550_acc += ((v557_data[2]) * v17_data);
              v550_acc += ((v557_data[3]) * v21_data);
              v550_acc += ((v557_data[4]) * v25_data);
              v550_acc += ((v557_data[5]) * v29_data);
              v550_acc += ((v557_data[6]) * v33_data);
              v550_acc += ((v557_data[7]) * v37_data);
              v550_acc += ((v557_data[8]) * v41_data);
              v550_acc += ((v557_data[9]) * v45_data);
              v550_acc += ((v557_data[10]) * v49_data);
              v550_acc += ((v557_data[11]) * v53_data);
              v550_acc += ((v557_data[12]) * v57_data);
              v550_acc += ((v557_data[13]) * v61_data);
              v550_acc += ((v557_data[14]) * v65_data);
              v550_acc += ((v557_data[15]) * v69_data);
              v550_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v590_acc{};
              tensorforge::intel_esimd::simd<float, 16> v597_data;
              v597_data.copy_from(s0 + ((208_i32 ^ ((208_i32 >> 5) & 31))));
              v590_acc += ((v597_data[0]) * v9_data);
              v590_acc += ((v597_data[1]) * v13_data);
              v590_acc += ((v597_data[2]) * v17_data);
              v590_acc += ((v597_data[3]) * v21_data);
              v590_acc += ((v597_data[4]) * v25_data);
              v590_acc += ((v597_data[5]) * v29_data);
              v590_acc += ((v597_data[6]) * v33_data);
              v590_acc += ((v597_data[7]) * v37_data);
              v590_acc += ((v597_data[8]) * v41_data);
              v590_acc += ((v597_data[9]) * v45_data);
              v590_acc += ((v597_data[10]) * v49_data);
              v590_acc += ((v597_data[11]) * v53_data);
              v590_acc += ((v597_data[12]) * v57_data);
              v590_acc += ((v597_data[13]) * v61_data);
              v590_acc += ((v597_data[14]) * v65_data);
              v590_acc += ((v597_data[15]) * v69_data);
              v590_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v630_acc{};
              tensorforge::intel_esimd::simd<float, 16> v637_data;
              v637_data.copy_from(s0 + ((224_i32 ^ ((224_i32 >> 5) & 31))));
              v630_acc += ((v637_data[0]) * v9_data);
              v630_acc += ((v637_data[1]) * v13_data);
              v630_acc += ((v637_data[2]) * v17_data);
              v630_acc += ((v637_data[3]) * v21_data);
              v630_acc += ((v637_data[4]) * v25_data);
              v630_acc += ((v637_data[5]) * v29_data);
              v630_acc += ((v637_data[6]) * v33_data);
              v630_acc += ((v637_data[7]) * v37_data);
              v630_acc += ((v637_data[8]) * v41_data);
              v630_acc += ((v637_data[9]) * v45_data);
              v630_acc += ((v637_data[10]) * v49_data);
              v630_acc += ((v637_data[11]) * v53_data);
              v630_acc += ((v637_data[12]) * v57_data);
              v630_acc += ((v637_data[13]) * v61_data);
              v630_acc += ((v637_data[14]) * v65_data);
              v630_acc += ((v637_data[15]) * v69_data);
              v630_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v670_acc{};
              tensorforge::intel_esimd::simd<float, 16> v677_data;
              v677_data.copy_from(s0 + ((240_i32 ^ ((240_i32 >> 5) & 31))));
              v670_acc += ((v677_data[0]) * v9_data);
              v670_acc += ((v677_data[1]) * v13_data);
              v670_acc += ((v677_data[2]) * v17_data);
              v670_acc += ((v677_data[3]) * v21_data);
              v670_acc += ((v677_data[4]) * v25_data);
              v670_acc += ((v677_data[5]) * v29_data);
              v670_acc += ((v677_data[6]) * v33_data);
              v670_acc += ((v677_data[7]) * v37_data);
              v670_acc += ((v677_data[8]) * v41_data);
              v670_acc += ((v677_data[9]) * v45_data);
              v670_acc += ((v677_data[10]) * v49_data);
              v670_acc += ((v677_data[11]) * v53_data);
              v670_acc += ((v677_data[12]) * v57_data);
              v670_acc += ((v677_data[13]) * v61_data);
              v670_acc += ((v677_data[14]) * v65_data);
              v670_acc += ((v677_data[15]) * v69_data);
              v670_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v710_n0 = 0; v710_n0 < 1; ++v710_n0) {
                int32_t v712_a = v710_n0 * 16;
                #pragma unroll
                for (int32_t v711_n1 = 0; v711_n1 < 16; ++v711_n1) {
                  int32_t v714_a = v712_a + (v711_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v715_data;
                  v715_data.copy_from(ir0 + (v714_a));
                  v715_data.copy_to(r0 + (v714_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v719_i0 = 0; v719_i0 < 1; ++v719_i0) {
                int32_t v721_a = v719_i0 * 16;
                #pragma unroll
                for (int32_t v720_i1 = 0; v720_i1 < 16; ++v720_i1) {
                  int32_t v723_a = v721_a + (v720_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v724_data;
                  v724_data.copy_from(r0 + (v723_a));
                  v724_data.copy_to(glb_m0 + (v723_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

