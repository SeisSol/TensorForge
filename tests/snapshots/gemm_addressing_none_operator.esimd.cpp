// === base name ===
kernel_e0dbeb2d3fe0873f

// === header ===
void launcher_kernel_e0dbeb2d3fe0873f(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e0dbeb2d3fe0873f(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e0dbeb2d3fe0873f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e0dbeb2d3fe0873f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v10_ld;
              v10_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v10_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v11_ld;
              v11_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v11_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              tensorforge::intel_esimd::simd<float, 64> v12_ld;
              v12_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              v12_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              tensorforge::intel_esimd::simd<float, 64> v13_ld;
              v13_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              v13_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data;
              v63_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data;
              v79_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data;
              v84_data.copy_from(s0 + (0_i32));
              v80_acc += ((v84_data[0]) * v19_data);
              v80_acc += ((v84_data[1]) * v23_data);
              v80_acc += ((v84_data[2]) * v27_data);
              v80_acc += ((v84_data[3]) * v31_data);
              v80_acc += ((v84_data[4]) * v35_data);
              v80_acc += ((v84_data[5]) * v39_data);
              v80_acc += ((v84_data[6]) * v43_data);
              v80_acc += ((v84_data[7]) * v47_data);
              v80_acc += ((v84_data[8]) * v51_data);
              v80_acc += ((v84_data[9]) * v55_data);
              v80_acc += ((v84_data[10]) * v59_data);
              v80_acc += ((v84_data[11]) * v63_data);
              v80_acc += ((v84_data[12]) * v67_data);
              v80_acc += ((v84_data[13]) * v71_data);
              v80_acc += ((v84_data[14]) * v75_data);
              v80_acc += ((v84_data[15]) * v79_data);
              v80_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              tensorforge::intel_esimd::simd<float, 16> v121_data;
              v121_data.copy_from(s0 + (16_i32));
              v117_acc += ((v121_data[0]) * v19_data);
              v117_acc += ((v121_data[1]) * v23_data);
              v117_acc += ((v121_data[2]) * v27_data);
              v117_acc += ((v121_data[3]) * v31_data);
              v117_acc += ((v121_data[4]) * v35_data);
              v117_acc += ((v121_data[5]) * v39_data);
              v117_acc += ((v121_data[6]) * v43_data);
              v117_acc += ((v121_data[7]) * v47_data);
              v117_acc += ((v121_data[8]) * v51_data);
              v117_acc += ((v121_data[9]) * v55_data);
              v117_acc += ((v121_data[10]) * v59_data);
              v117_acc += ((v121_data[11]) * v63_data);
              v117_acc += ((v121_data[12]) * v67_data);
              v117_acc += ((v121_data[13]) * v71_data);
              v117_acc += ((v121_data[14]) * v75_data);
              v117_acc += ((v121_data[15]) * v79_data);
              v117_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v158_data;
              v158_data.copy_from(s0 + (32_i32));
              v154_acc += ((v158_data[0]) * v19_data);
              v154_acc += ((v158_data[1]) * v23_data);
              v154_acc += ((v158_data[2]) * v27_data);
              v154_acc += ((v158_data[3]) * v31_data);
              v154_acc += ((v158_data[4]) * v35_data);
              v154_acc += ((v158_data[5]) * v39_data);
              v154_acc += ((v158_data[6]) * v43_data);
              v154_acc += ((v158_data[7]) * v47_data);
              v154_acc += ((v158_data[8]) * v51_data);
              v154_acc += ((v158_data[9]) * v55_data);
              v154_acc += ((v158_data[10]) * v59_data);
              v154_acc += ((v158_data[11]) * v63_data);
              v154_acc += ((v158_data[12]) * v67_data);
              v154_acc += ((v158_data[13]) * v71_data);
              v154_acc += ((v158_data[14]) * v75_data);
              v154_acc += ((v158_data[15]) * v79_data);
              v154_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data;
              v195_data.copy_from(s0 + (48_i32));
              v191_acc += ((v195_data[0]) * v19_data);
              v191_acc += ((v195_data[1]) * v23_data);
              v191_acc += ((v195_data[2]) * v27_data);
              v191_acc += ((v195_data[3]) * v31_data);
              v191_acc += ((v195_data[4]) * v35_data);
              v191_acc += ((v195_data[5]) * v39_data);
              v191_acc += ((v195_data[6]) * v43_data);
              v191_acc += ((v195_data[7]) * v47_data);
              v191_acc += ((v195_data[8]) * v51_data);
              v191_acc += ((v195_data[9]) * v55_data);
              v191_acc += ((v195_data[10]) * v59_data);
              v191_acc += ((v195_data[11]) * v63_data);
              v191_acc += ((v195_data[12]) * v67_data);
              v191_acc += ((v195_data[13]) * v71_data);
              v191_acc += ((v195_data[14]) * v75_data);
              v191_acc += ((v195_data[15]) * v79_data);
              v191_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              tensorforge::intel_esimd::simd<float, 16> v232_data;
              v232_data.copy_from(s0 + (64_i32));
              v228_acc += ((v232_data[0]) * v19_data);
              v228_acc += ((v232_data[1]) * v23_data);
              v228_acc += ((v232_data[2]) * v27_data);
              v228_acc += ((v232_data[3]) * v31_data);
              v228_acc += ((v232_data[4]) * v35_data);
              v228_acc += ((v232_data[5]) * v39_data);
              v228_acc += ((v232_data[6]) * v43_data);
              v228_acc += ((v232_data[7]) * v47_data);
              v228_acc += ((v232_data[8]) * v51_data);
              v228_acc += ((v232_data[9]) * v55_data);
              v228_acc += ((v232_data[10]) * v59_data);
              v228_acc += ((v232_data[11]) * v63_data);
              v228_acc += ((v232_data[12]) * v67_data);
              v228_acc += ((v232_data[13]) * v71_data);
              v228_acc += ((v232_data[14]) * v75_data);
              v228_acc += ((v232_data[15]) * v79_data);
              v228_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v265_acc{};
              tensorforge::intel_esimd::simd<float, 16> v269_data;
              v269_data.copy_from(s0 + (80_i32));
              v265_acc += ((v269_data[0]) * v19_data);
              v265_acc += ((v269_data[1]) * v23_data);
              v265_acc += ((v269_data[2]) * v27_data);
              v265_acc += ((v269_data[3]) * v31_data);
              v265_acc += ((v269_data[4]) * v35_data);
              v265_acc += ((v269_data[5]) * v39_data);
              v265_acc += ((v269_data[6]) * v43_data);
              v265_acc += ((v269_data[7]) * v47_data);
              v265_acc += ((v269_data[8]) * v51_data);
              v265_acc += ((v269_data[9]) * v55_data);
              v265_acc += ((v269_data[10]) * v59_data);
              v265_acc += ((v269_data[11]) * v63_data);
              v265_acc += ((v269_data[12]) * v67_data);
              v265_acc += ((v269_data[13]) * v71_data);
              v265_acc += ((v269_data[14]) * v75_data);
              v265_acc += ((v269_data[15]) * v79_data);
              v265_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v302_acc{};
              tensorforge::intel_esimd::simd<float, 16> v306_data;
              v306_data.copy_from(s0 + (96_i32));
              v302_acc += ((v306_data[0]) * v19_data);
              v302_acc += ((v306_data[1]) * v23_data);
              v302_acc += ((v306_data[2]) * v27_data);
              v302_acc += ((v306_data[3]) * v31_data);
              v302_acc += ((v306_data[4]) * v35_data);
              v302_acc += ((v306_data[5]) * v39_data);
              v302_acc += ((v306_data[6]) * v43_data);
              v302_acc += ((v306_data[7]) * v47_data);
              v302_acc += ((v306_data[8]) * v51_data);
              v302_acc += ((v306_data[9]) * v55_data);
              v302_acc += ((v306_data[10]) * v59_data);
              v302_acc += ((v306_data[11]) * v63_data);
              v302_acc += ((v306_data[12]) * v67_data);
              v302_acc += ((v306_data[13]) * v71_data);
              v302_acc += ((v306_data[14]) * v75_data);
              v302_acc += ((v306_data[15]) * v79_data);
              v302_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v339_acc{};
              tensorforge::intel_esimd::simd<float, 16> v343_data;
              v343_data.copy_from(s0 + (112_i32));
              v339_acc += ((v343_data[0]) * v19_data);
              v339_acc += ((v343_data[1]) * v23_data);
              v339_acc += ((v343_data[2]) * v27_data);
              v339_acc += ((v343_data[3]) * v31_data);
              v339_acc += ((v343_data[4]) * v35_data);
              v339_acc += ((v343_data[5]) * v39_data);
              v339_acc += ((v343_data[6]) * v43_data);
              v339_acc += ((v343_data[7]) * v47_data);
              v339_acc += ((v343_data[8]) * v51_data);
              v339_acc += ((v343_data[9]) * v55_data);
              v339_acc += ((v343_data[10]) * v59_data);
              v339_acc += ((v343_data[11]) * v63_data);
              v339_acc += ((v343_data[12]) * v67_data);
              v339_acc += ((v343_data[13]) * v71_data);
              v339_acc += ((v343_data[14]) * v75_data);
              v339_acc += ((v343_data[15]) * v79_data);
              v339_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v376_acc{};
              tensorforge::intel_esimd::simd<float, 16> v380_data;
              v380_data.copy_from(s0 + (128_i32));
              v376_acc += ((v380_data[0]) * v19_data);
              v376_acc += ((v380_data[1]) * v23_data);
              v376_acc += ((v380_data[2]) * v27_data);
              v376_acc += ((v380_data[3]) * v31_data);
              v376_acc += ((v380_data[4]) * v35_data);
              v376_acc += ((v380_data[5]) * v39_data);
              v376_acc += ((v380_data[6]) * v43_data);
              v376_acc += ((v380_data[7]) * v47_data);
              v376_acc += ((v380_data[8]) * v51_data);
              v376_acc += ((v380_data[9]) * v55_data);
              v376_acc += ((v380_data[10]) * v59_data);
              v376_acc += ((v380_data[11]) * v63_data);
              v376_acc += ((v380_data[12]) * v67_data);
              v376_acc += ((v380_data[13]) * v71_data);
              v376_acc += ((v380_data[14]) * v75_data);
              v376_acc += ((v380_data[15]) * v79_data);
              v376_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v413_acc{};
              tensorforge::intel_esimd::simd<float, 16> v417_data;
              v417_data.copy_from(s0 + (144_i32));
              v413_acc += ((v417_data[0]) * v19_data);
              v413_acc += ((v417_data[1]) * v23_data);
              v413_acc += ((v417_data[2]) * v27_data);
              v413_acc += ((v417_data[3]) * v31_data);
              v413_acc += ((v417_data[4]) * v35_data);
              v413_acc += ((v417_data[5]) * v39_data);
              v413_acc += ((v417_data[6]) * v43_data);
              v413_acc += ((v417_data[7]) * v47_data);
              v413_acc += ((v417_data[8]) * v51_data);
              v413_acc += ((v417_data[9]) * v55_data);
              v413_acc += ((v417_data[10]) * v59_data);
              v413_acc += ((v417_data[11]) * v63_data);
              v413_acc += ((v417_data[12]) * v67_data);
              v413_acc += ((v417_data[13]) * v71_data);
              v413_acc += ((v417_data[14]) * v75_data);
              v413_acc += ((v417_data[15]) * v79_data);
              v413_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v450_acc{};
              tensorforge::intel_esimd::simd<float, 16> v454_data;
              v454_data.copy_from(s0 + (160_i32));
              v450_acc += ((v454_data[0]) * v19_data);
              v450_acc += ((v454_data[1]) * v23_data);
              v450_acc += ((v454_data[2]) * v27_data);
              v450_acc += ((v454_data[3]) * v31_data);
              v450_acc += ((v454_data[4]) * v35_data);
              v450_acc += ((v454_data[5]) * v39_data);
              v450_acc += ((v454_data[6]) * v43_data);
              v450_acc += ((v454_data[7]) * v47_data);
              v450_acc += ((v454_data[8]) * v51_data);
              v450_acc += ((v454_data[9]) * v55_data);
              v450_acc += ((v454_data[10]) * v59_data);
              v450_acc += ((v454_data[11]) * v63_data);
              v450_acc += ((v454_data[12]) * v67_data);
              v450_acc += ((v454_data[13]) * v71_data);
              v450_acc += ((v454_data[14]) * v75_data);
              v450_acc += ((v454_data[15]) * v79_data);
              v450_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              tensorforge::intel_esimd::simd<float, 16> v491_data;
              v491_data.copy_from(s0 + (176_i32));
              v487_acc += ((v491_data[0]) * v19_data);
              v487_acc += ((v491_data[1]) * v23_data);
              v487_acc += ((v491_data[2]) * v27_data);
              v487_acc += ((v491_data[3]) * v31_data);
              v487_acc += ((v491_data[4]) * v35_data);
              v487_acc += ((v491_data[5]) * v39_data);
              v487_acc += ((v491_data[6]) * v43_data);
              v487_acc += ((v491_data[7]) * v47_data);
              v487_acc += ((v491_data[8]) * v51_data);
              v487_acc += ((v491_data[9]) * v55_data);
              v487_acc += ((v491_data[10]) * v59_data);
              v487_acc += ((v491_data[11]) * v63_data);
              v487_acc += ((v491_data[12]) * v67_data);
              v487_acc += ((v491_data[13]) * v71_data);
              v487_acc += ((v491_data[14]) * v75_data);
              v487_acc += ((v491_data[15]) * v79_data);
              v487_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v524_acc{};
              tensorforge::intel_esimd::simd<float, 16> v528_data;
              v528_data.copy_from(s0 + (192_i32));
              v524_acc += ((v528_data[0]) * v19_data);
              v524_acc += ((v528_data[1]) * v23_data);
              v524_acc += ((v528_data[2]) * v27_data);
              v524_acc += ((v528_data[3]) * v31_data);
              v524_acc += ((v528_data[4]) * v35_data);
              v524_acc += ((v528_data[5]) * v39_data);
              v524_acc += ((v528_data[6]) * v43_data);
              v524_acc += ((v528_data[7]) * v47_data);
              v524_acc += ((v528_data[8]) * v51_data);
              v524_acc += ((v528_data[9]) * v55_data);
              v524_acc += ((v528_data[10]) * v59_data);
              v524_acc += ((v528_data[11]) * v63_data);
              v524_acc += ((v528_data[12]) * v67_data);
              v524_acc += ((v528_data[13]) * v71_data);
              v524_acc += ((v528_data[14]) * v75_data);
              v524_acc += ((v528_data[15]) * v79_data);
              v524_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v561_acc{};
              tensorforge::intel_esimd::simd<float, 16> v565_data;
              v565_data.copy_from(s0 + (208_i32));
              v561_acc += ((v565_data[0]) * v19_data);
              v561_acc += ((v565_data[1]) * v23_data);
              v561_acc += ((v565_data[2]) * v27_data);
              v561_acc += ((v565_data[3]) * v31_data);
              v561_acc += ((v565_data[4]) * v35_data);
              v561_acc += ((v565_data[5]) * v39_data);
              v561_acc += ((v565_data[6]) * v43_data);
              v561_acc += ((v565_data[7]) * v47_data);
              v561_acc += ((v565_data[8]) * v51_data);
              v561_acc += ((v565_data[9]) * v55_data);
              v561_acc += ((v565_data[10]) * v59_data);
              v561_acc += ((v565_data[11]) * v63_data);
              v561_acc += ((v565_data[12]) * v67_data);
              v561_acc += ((v565_data[13]) * v71_data);
              v561_acc += ((v565_data[14]) * v75_data);
              v561_acc += ((v565_data[15]) * v79_data);
              v561_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v598_acc{};
              tensorforge::intel_esimd::simd<float, 16> v602_data;
              v602_data.copy_from(s0 + (224_i32));
              v598_acc += ((v602_data[0]) * v19_data);
              v598_acc += ((v602_data[1]) * v23_data);
              v598_acc += ((v602_data[2]) * v27_data);
              v598_acc += ((v602_data[3]) * v31_data);
              v598_acc += ((v602_data[4]) * v35_data);
              v598_acc += ((v602_data[5]) * v39_data);
              v598_acc += ((v602_data[6]) * v43_data);
              v598_acc += ((v602_data[7]) * v47_data);
              v598_acc += ((v602_data[8]) * v51_data);
              v598_acc += ((v602_data[9]) * v55_data);
              v598_acc += ((v602_data[10]) * v59_data);
              v598_acc += ((v602_data[11]) * v63_data);
              v598_acc += ((v602_data[12]) * v67_data);
              v598_acc += ((v602_data[13]) * v71_data);
              v598_acc += ((v602_data[14]) * v75_data);
              v598_acc += ((v602_data[15]) * v79_data);
              v598_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v635_acc{};
              tensorforge::intel_esimd::simd<float, 16> v639_data;
              v639_data.copy_from(s0 + (240_i32));
              v635_acc += ((v639_data[0]) * v19_data);
              v635_acc += ((v639_data[1]) * v23_data);
              v635_acc += ((v639_data[2]) * v27_data);
              v635_acc += ((v639_data[3]) * v31_data);
              v635_acc += ((v639_data[4]) * v35_data);
              v635_acc += ((v639_data[5]) * v39_data);
              v635_acc += ((v639_data[6]) * v43_data);
              v635_acc += ((v639_data[7]) * v47_data);
              v635_acc += ((v639_data[8]) * v51_data);
              v635_acc += ((v639_data[9]) * v55_data);
              v635_acc += ((v639_data[10]) * v59_data);
              v635_acc += ((v639_data[11]) * v63_data);
              v635_acc += ((v639_data[12]) * v67_data);
              v635_acc += ((v639_data[13]) * v71_data);
              v635_acc += ((v639_data[14]) * v75_data);
              v635_acc += ((v639_data[15]) * v79_data);
              v635_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v672_n0 = 0; v672_n0 < 1; ++v672_n0) {
                int32_t v674_a = v672_n0 * 16;
                #pragma unroll
                for (int32_t v673_n1 = 0; v673_n1 < 16; ++v673_n1) {
                  int32_t v676_a = v674_a + (v673_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v677_data;
                  v677_data.copy_from(ir0 + (v676_a));
                  v677_data.copy_to(r0 + (v676_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v681_i0 = 0; v681_i0 < 1; ++v681_i0) {
                int32_t v683_a = v681_i0 * 16;
                #pragma unroll
                for (int32_t v682_i1 = 0; v682_i1 < 16; ++v682_i1) {
                  int32_t v685_a = v683_a + (v682_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v686_data;
                  v686_data.copy_from(r0 + (v685_a));
                  v686_data.copy_to(glb_m0 + (v685_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

