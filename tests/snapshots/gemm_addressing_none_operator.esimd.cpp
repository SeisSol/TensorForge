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
              tensorforge::intel_esimd::simd<float, 16> v11_data;
              v11_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v15_data;
              v15_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data;
              v63_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_acc{};
              tensorforge::intel_esimd::simd<float, 16> v79_data;
              v79_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v72_acc += ((v79_data[0]) * v11_data);
              v72_acc += ((v79_data[1]) * v15_data);
              v72_acc += ((v79_data[2]) * v19_data);
              v72_acc += ((v79_data[3]) * v23_data);
              v72_acc += ((v79_data[4]) * v27_data);
              v72_acc += ((v79_data[5]) * v31_data);
              v72_acc += ((v79_data[6]) * v35_data);
              v72_acc += ((v79_data[7]) * v39_data);
              v72_acc += ((v79_data[8]) * v43_data);
              v72_acc += ((v79_data[9]) * v47_data);
              v72_acc += ((v79_data[10]) * v51_data);
              v72_acc += ((v79_data[11]) * v55_data);
              v72_acc += ((v79_data[12]) * v59_data);
              v72_acc += ((v79_data[13]) * v63_data);
              v72_acc += ((v79_data[14]) * v67_data);
              v72_acc += ((v79_data[15]) * v71_data);
              v72_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v112_acc{};
              tensorforge::intel_esimd::simd<float, 16> v119_data;
              v119_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v112_acc += ((v119_data[0]) * v11_data);
              v112_acc += ((v119_data[1]) * v15_data);
              v112_acc += ((v119_data[2]) * v19_data);
              v112_acc += ((v119_data[3]) * v23_data);
              v112_acc += ((v119_data[4]) * v27_data);
              v112_acc += ((v119_data[5]) * v31_data);
              v112_acc += ((v119_data[6]) * v35_data);
              v112_acc += ((v119_data[7]) * v39_data);
              v112_acc += ((v119_data[8]) * v43_data);
              v112_acc += ((v119_data[9]) * v47_data);
              v112_acc += ((v119_data[10]) * v51_data);
              v112_acc += ((v119_data[11]) * v55_data);
              v112_acc += ((v119_data[12]) * v59_data);
              v112_acc += ((v119_data[13]) * v63_data);
              v112_acc += ((v119_data[14]) * v67_data);
              v112_acc += ((v119_data[15]) * v71_data);
              v112_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              tensorforge::intel_esimd::simd<float, 16> v159_data;
              v159_data.copy_from(s0 + ((32_i32 ^ ((32_i32 >> 5) & 31))));
              v152_acc += ((v159_data[0]) * v11_data);
              v152_acc += ((v159_data[1]) * v15_data);
              v152_acc += ((v159_data[2]) * v19_data);
              v152_acc += ((v159_data[3]) * v23_data);
              v152_acc += ((v159_data[4]) * v27_data);
              v152_acc += ((v159_data[5]) * v31_data);
              v152_acc += ((v159_data[6]) * v35_data);
              v152_acc += ((v159_data[7]) * v39_data);
              v152_acc += ((v159_data[8]) * v43_data);
              v152_acc += ((v159_data[9]) * v47_data);
              v152_acc += ((v159_data[10]) * v51_data);
              v152_acc += ((v159_data[11]) * v55_data);
              v152_acc += ((v159_data[12]) * v59_data);
              v152_acc += ((v159_data[13]) * v63_data);
              v152_acc += ((v159_data[14]) * v67_data);
              v152_acc += ((v159_data[15]) * v71_data);
              v152_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v192_acc{};
              tensorforge::intel_esimd::simd<float, 16> v199_data;
              v199_data.copy_from(s0 + ((48_i32 ^ ((48_i32 >> 5) & 31))));
              v192_acc += ((v199_data[0]) * v11_data);
              v192_acc += ((v199_data[1]) * v15_data);
              v192_acc += ((v199_data[2]) * v19_data);
              v192_acc += ((v199_data[3]) * v23_data);
              v192_acc += ((v199_data[4]) * v27_data);
              v192_acc += ((v199_data[5]) * v31_data);
              v192_acc += ((v199_data[6]) * v35_data);
              v192_acc += ((v199_data[7]) * v39_data);
              v192_acc += ((v199_data[8]) * v43_data);
              v192_acc += ((v199_data[9]) * v47_data);
              v192_acc += ((v199_data[10]) * v51_data);
              v192_acc += ((v199_data[11]) * v55_data);
              v192_acc += ((v199_data[12]) * v59_data);
              v192_acc += ((v199_data[13]) * v63_data);
              v192_acc += ((v199_data[14]) * v67_data);
              v192_acc += ((v199_data[15]) * v71_data);
              v192_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v232_acc{};
              tensorforge::intel_esimd::simd<float, 16> v239_data;
              v239_data.copy_from(s0 + ((64_i32 ^ ((64_i32 >> 5) & 31))));
              v232_acc += ((v239_data[0]) * v11_data);
              v232_acc += ((v239_data[1]) * v15_data);
              v232_acc += ((v239_data[2]) * v19_data);
              v232_acc += ((v239_data[3]) * v23_data);
              v232_acc += ((v239_data[4]) * v27_data);
              v232_acc += ((v239_data[5]) * v31_data);
              v232_acc += ((v239_data[6]) * v35_data);
              v232_acc += ((v239_data[7]) * v39_data);
              v232_acc += ((v239_data[8]) * v43_data);
              v232_acc += ((v239_data[9]) * v47_data);
              v232_acc += ((v239_data[10]) * v51_data);
              v232_acc += ((v239_data[11]) * v55_data);
              v232_acc += ((v239_data[12]) * v59_data);
              v232_acc += ((v239_data[13]) * v63_data);
              v232_acc += ((v239_data[14]) * v67_data);
              v232_acc += ((v239_data[15]) * v71_data);
              v232_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              tensorforge::intel_esimd::simd<float, 16> v279_data;
              v279_data.copy_from(s0 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v272_acc += ((v279_data[0]) * v11_data);
              v272_acc += ((v279_data[1]) * v15_data);
              v272_acc += ((v279_data[2]) * v19_data);
              v272_acc += ((v279_data[3]) * v23_data);
              v272_acc += ((v279_data[4]) * v27_data);
              v272_acc += ((v279_data[5]) * v31_data);
              v272_acc += ((v279_data[6]) * v35_data);
              v272_acc += ((v279_data[7]) * v39_data);
              v272_acc += ((v279_data[8]) * v43_data);
              v272_acc += ((v279_data[9]) * v47_data);
              v272_acc += ((v279_data[10]) * v51_data);
              v272_acc += ((v279_data[11]) * v55_data);
              v272_acc += ((v279_data[12]) * v59_data);
              v272_acc += ((v279_data[13]) * v63_data);
              v272_acc += ((v279_data[14]) * v67_data);
              v272_acc += ((v279_data[15]) * v71_data);
              v272_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v312_acc{};
              tensorforge::intel_esimd::simd<float, 16> v319_data;
              v319_data.copy_from(s0 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v312_acc += ((v319_data[0]) * v11_data);
              v312_acc += ((v319_data[1]) * v15_data);
              v312_acc += ((v319_data[2]) * v19_data);
              v312_acc += ((v319_data[3]) * v23_data);
              v312_acc += ((v319_data[4]) * v27_data);
              v312_acc += ((v319_data[5]) * v31_data);
              v312_acc += ((v319_data[6]) * v35_data);
              v312_acc += ((v319_data[7]) * v39_data);
              v312_acc += ((v319_data[8]) * v43_data);
              v312_acc += ((v319_data[9]) * v47_data);
              v312_acc += ((v319_data[10]) * v51_data);
              v312_acc += ((v319_data[11]) * v55_data);
              v312_acc += ((v319_data[12]) * v59_data);
              v312_acc += ((v319_data[13]) * v63_data);
              v312_acc += ((v319_data[14]) * v67_data);
              v312_acc += ((v319_data[15]) * v71_data);
              v312_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v352_acc{};
              tensorforge::intel_esimd::simd<float, 16> v359_data;
              v359_data.copy_from(s0 + ((112_i32 ^ ((112_i32 >> 5) & 31))));
              v352_acc += ((v359_data[0]) * v11_data);
              v352_acc += ((v359_data[1]) * v15_data);
              v352_acc += ((v359_data[2]) * v19_data);
              v352_acc += ((v359_data[3]) * v23_data);
              v352_acc += ((v359_data[4]) * v27_data);
              v352_acc += ((v359_data[5]) * v31_data);
              v352_acc += ((v359_data[6]) * v35_data);
              v352_acc += ((v359_data[7]) * v39_data);
              v352_acc += ((v359_data[8]) * v43_data);
              v352_acc += ((v359_data[9]) * v47_data);
              v352_acc += ((v359_data[10]) * v51_data);
              v352_acc += ((v359_data[11]) * v55_data);
              v352_acc += ((v359_data[12]) * v59_data);
              v352_acc += ((v359_data[13]) * v63_data);
              v352_acc += ((v359_data[14]) * v67_data);
              v352_acc += ((v359_data[15]) * v71_data);
              v352_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v392_acc{};
              tensorforge::intel_esimd::simd<float, 16> v399_data;
              v399_data.copy_from(s0 + ((128_i32 ^ ((128_i32 >> 5) & 31))));
              v392_acc += ((v399_data[0]) * v11_data);
              v392_acc += ((v399_data[1]) * v15_data);
              v392_acc += ((v399_data[2]) * v19_data);
              v392_acc += ((v399_data[3]) * v23_data);
              v392_acc += ((v399_data[4]) * v27_data);
              v392_acc += ((v399_data[5]) * v31_data);
              v392_acc += ((v399_data[6]) * v35_data);
              v392_acc += ((v399_data[7]) * v39_data);
              v392_acc += ((v399_data[8]) * v43_data);
              v392_acc += ((v399_data[9]) * v47_data);
              v392_acc += ((v399_data[10]) * v51_data);
              v392_acc += ((v399_data[11]) * v55_data);
              v392_acc += ((v399_data[12]) * v59_data);
              v392_acc += ((v399_data[13]) * v63_data);
              v392_acc += ((v399_data[14]) * v67_data);
              v392_acc += ((v399_data[15]) * v71_data);
              v392_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v432_acc{};
              tensorforge::intel_esimd::simd<float, 16> v439_data;
              v439_data.copy_from(s0 + ((144_i32 ^ ((144_i32 >> 5) & 31))));
              v432_acc += ((v439_data[0]) * v11_data);
              v432_acc += ((v439_data[1]) * v15_data);
              v432_acc += ((v439_data[2]) * v19_data);
              v432_acc += ((v439_data[3]) * v23_data);
              v432_acc += ((v439_data[4]) * v27_data);
              v432_acc += ((v439_data[5]) * v31_data);
              v432_acc += ((v439_data[6]) * v35_data);
              v432_acc += ((v439_data[7]) * v39_data);
              v432_acc += ((v439_data[8]) * v43_data);
              v432_acc += ((v439_data[9]) * v47_data);
              v432_acc += ((v439_data[10]) * v51_data);
              v432_acc += ((v439_data[11]) * v55_data);
              v432_acc += ((v439_data[12]) * v59_data);
              v432_acc += ((v439_data[13]) * v63_data);
              v432_acc += ((v439_data[14]) * v67_data);
              v432_acc += ((v439_data[15]) * v71_data);
              v432_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v472_acc{};
              tensorforge::intel_esimd::simd<float, 16> v479_data;
              v479_data.copy_from(s0 + ((160_i32 ^ ((160_i32 >> 5) & 31))));
              v472_acc += ((v479_data[0]) * v11_data);
              v472_acc += ((v479_data[1]) * v15_data);
              v472_acc += ((v479_data[2]) * v19_data);
              v472_acc += ((v479_data[3]) * v23_data);
              v472_acc += ((v479_data[4]) * v27_data);
              v472_acc += ((v479_data[5]) * v31_data);
              v472_acc += ((v479_data[6]) * v35_data);
              v472_acc += ((v479_data[7]) * v39_data);
              v472_acc += ((v479_data[8]) * v43_data);
              v472_acc += ((v479_data[9]) * v47_data);
              v472_acc += ((v479_data[10]) * v51_data);
              v472_acc += ((v479_data[11]) * v55_data);
              v472_acc += ((v479_data[12]) * v59_data);
              v472_acc += ((v479_data[13]) * v63_data);
              v472_acc += ((v479_data[14]) * v67_data);
              v472_acc += ((v479_data[15]) * v71_data);
              v472_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v512_acc{};
              tensorforge::intel_esimd::simd<float, 16> v519_data;
              v519_data.copy_from(s0 + ((176_i32 ^ ((176_i32 >> 5) & 31))));
              v512_acc += ((v519_data[0]) * v11_data);
              v512_acc += ((v519_data[1]) * v15_data);
              v512_acc += ((v519_data[2]) * v19_data);
              v512_acc += ((v519_data[3]) * v23_data);
              v512_acc += ((v519_data[4]) * v27_data);
              v512_acc += ((v519_data[5]) * v31_data);
              v512_acc += ((v519_data[6]) * v35_data);
              v512_acc += ((v519_data[7]) * v39_data);
              v512_acc += ((v519_data[8]) * v43_data);
              v512_acc += ((v519_data[9]) * v47_data);
              v512_acc += ((v519_data[10]) * v51_data);
              v512_acc += ((v519_data[11]) * v55_data);
              v512_acc += ((v519_data[12]) * v59_data);
              v512_acc += ((v519_data[13]) * v63_data);
              v512_acc += ((v519_data[14]) * v67_data);
              v512_acc += ((v519_data[15]) * v71_data);
              v512_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v552_acc{};
              tensorforge::intel_esimd::simd<float, 16> v559_data;
              v559_data.copy_from(s0 + ((192_i32 ^ ((192_i32 >> 5) & 31))));
              v552_acc += ((v559_data[0]) * v11_data);
              v552_acc += ((v559_data[1]) * v15_data);
              v552_acc += ((v559_data[2]) * v19_data);
              v552_acc += ((v559_data[3]) * v23_data);
              v552_acc += ((v559_data[4]) * v27_data);
              v552_acc += ((v559_data[5]) * v31_data);
              v552_acc += ((v559_data[6]) * v35_data);
              v552_acc += ((v559_data[7]) * v39_data);
              v552_acc += ((v559_data[8]) * v43_data);
              v552_acc += ((v559_data[9]) * v47_data);
              v552_acc += ((v559_data[10]) * v51_data);
              v552_acc += ((v559_data[11]) * v55_data);
              v552_acc += ((v559_data[12]) * v59_data);
              v552_acc += ((v559_data[13]) * v63_data);
              v552_acc += ((v559_data[14]) * v67_data);
              v552_acc += ((v559_data[15]) * v71_data);
              v552_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v592_acc{};
              tensorforge::intel_esimd::simd<float, 16> v599_data;
              v599_data.copy_from(s0 + ((208_i32 ^ ((208_i32 >> 5) & 31))));
              v592_acc += ((v599_data[0]) * v11_data);
              v592_acc += ((v599_data[1]) * v15_data);
              v592_acc += ((v599_data[2]) * v19_data);
              v592_acc += ((v599_data[3]) * v23_data);
              v592_acc += ((v599_data[4]) * v27_data);
              v592_acc += ((v599_data[5]) * v31_data);
              v592_acc += ((v599_data[6]) * v35_data);
              v592_acc += ((v599_data[7]) * v39_data);
              v592_acc += ((v599_data[8]) * v43_data);
              v592_acc += ((v599_data[9]) * v47_data);
              v592_acc += ((v599_data[10]) * v51_data);
              v592_acc += ((v599_data[11]) * v55_data);
              v592_acc += ((v599_data[12]) * v59_data);
              v592_acc += ((v599_data[13]) * v63_data);
              v592_acc += ((v599_data[14]) * v67_data);
              v592_acc += ((v599_data[15]) * v71_data);
              v592_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v632_acc{};
              tensorforge::intel_esimd::simd<float, 16> v639_data;
              v639_data.copy_from(s0 + ((224_i32 ^ ((224_i32 >> 5) & 31))));
              v632_acc += ((v639_data[0]) * v11_data);
              v632_acc += ((v639_data[1]) * v15_data);
              v632_acc += ((v639_data[2]) * v19_data);
              v632_acc += ((v639_data[3]) * v23_data);
              v632_acc += ((v639_data[4]) * v27_data);
              v632_acc += ((v639_data[5]) * v31_data);
              v632_acc += ((v639_data[6]) * v35_data);
              v632_acc += ((v639_data[7]) * v39_data);
              v632_acc += ((v639_data[8]) * v43_data);
              v632_acc += ((v639_data[9]) * v47_data);
              v632_acc += ((v639_data[10]) * v51_data);
              v632_acc += ((v639_data[11]) * v55_data);
              v632_acc += ((v639_data[12]) * v59_data);
              v632_acc += ((v639_data[13]) * v63_data);
              v632_acc += ((v639_data[14]) * v67_data);
              v632_acc += ((v639_data[15]) * v71_data);
              v632_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v672_acc{};
              tensorforge::intel_esimd::simd<float, 16> v679_data;
              v679_data.copy_from(s0 + ((240_i32 ^ ((240_i32 >> 5) & 31))));
              v672_acc += ((v679_data[0]) * v11_data);
              v672_acc += ((v679_data[1]) * v15_data);
              v672_acc += ((v679_data[2]) * v19_data);
              v672_acc += ((v679_data[3]) * v23_data);
              v672_acc += ((v679_data[4]) * v27_data);
              v672_acc += ((v679_data[5]) * v31_data);
              v672_acc += ((v679_data[6]) * v35_data);
              v672_acc += ((v679_data[7]) * v39_data);
              v672_acc += ((v679_data[8]) * v43_data);
              v672_acc += ((v679_data[9]) * v47_data);
              v672_acc += ((v679_data[10]) * v51_data);
              v672_acc += ((v679_data[11]) * v55_data);
              v672_acc += ((v679_data[12]) * v59_data);
              v672_acc += ((v679_data[13]) * v63_data);
              v672_acc += ((v679_data[14]) * v67_data);
              v672_acc += ((v679_data[15]) * v71_data);
              v672_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v712_n0 = 0; v712_n0 < 1; ++v712_n0) {
                int32_t v714_a = v712_n0 * 16;
                #pragma unroll
                for (int32_t v713_n1 = 0; v713_n1 < 16; ++v713_n1) {
                  int32_t v716_a = v714_a + (v713_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v717_data;
                  v717_data.copy_from(ir0 + (v716_a));
                  v717_data.copy_to(r0 + (v716_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v721_i0 = 0; v721_i0 < 1; ++v721_i0) {
                int32_t v723_a = v721_i0 * 16;
                #pragma unroll
                for (int32_t v722_i1 = 0; v722_i1 < 16; ++v722_i1) {
                  int32_t v725_a = v723_a + (v722_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v726_data;
                  v726_data.copy_from(r0 + (v725_a));
                  v726_data.copy_to(glb_m0 + (v725_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

