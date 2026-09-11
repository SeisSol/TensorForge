// === base name ===
kernel_0a31923805dfdd61

// === header ===
void launcher_kernel_0a31923805dfdd61(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0a31923805dfdd61(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_0a31923805dfdd61(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0a31923805dfdd61(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} none
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} none({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[272 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[256];
          const float *const __restrict__ glb_m1 = &m1[0];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v14_ld;
              v14_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v14_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v15_ld;
              v15_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v15_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              tensorforge::intel_esimd::simd<float, 64> v16_ld;
              v16_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              v16_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              v17_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(glb_m1 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(glb_m1 + (32_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(glb_m1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(glb_m1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(glb_m1 + (112_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m1 + (144_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data;
              v63_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(glb_m1 + (176_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(glb_m1 + (208_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data;
              v79_data.copy_from(glb_m1 + (224_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v84_acc{};
              tensorforge::intel_esimd::simd<float, 16> v88_data;
              v88_data.copy_from(s0 + (0_i32));
              v84_acc += ((static_cast<float>(v88_data[0])) * v23_data);
              v84_acc += ((static_cast<float>(v88_data[1])) * v27_data);
              v84_acc += ((static_cast<float>(v88_data[2])) * v31_data);
              v84_acc += ((static_cast<float>(v88_data[3])) * v35_data);
              v84_acc += ((static_cast<float>(v88_data[4])) * v39_data);
              v84_acc += ((static_cast<float>(v88_data[5])) * v43_data);
              v84_acc += ((static_cast<float>(v88_data[6])) * v47_data);
              v84_acc += ((static_cast<float>(v88_data[7])) * v51_data);
              v84_acc += ((static_cast<float>(v88_data[8])) * v55_data);
              v84_acc += ((static_cast<float>(v88_data[9])) * v59_data);
              v84_acc += ((static_cast<float>(v88_data[10])) * v63_data);
              v84_acc += ((static_cast<float>(v88_data[11])) * v67_data);
              v84_acc += ((static_cast<float>(v88_data[12])) * v71_data);
              v84_acc += ((static_cast<float>(v88_data[13])) * v75_data);
              v84_acc += ((static_cast<float>(v88_data[14])) * v79_data);
              v84_acc += ((static_cast<float>(v88_data[15])) * v83_data);
              v84_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v121_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data;
              v125_data.copy_from(s0 + (16_i32));
              v121_acc += ((static_cast<float>(v125_data[0])) * v23_data);
              v121_acc += ((static_cast<float>(v125_data[1])) * v27_data);
              v121_acc += ((static_cast<float>(v125_data[2])) * v31_data);
              v121_acc += ((static_cast<float>(v125_data[3])) * v35_data);
              v121_acc += ((static_cast<float>(v125_data[4])) * v39_data);
              v121_acc += ((static_cast<float>(v125_data[5])) * v43_data);
              v121_acc += ((static_cast<float>(v125_data[6])) * v47_data);
              v121_acc += ((static_cast<float>(v125_data[7])) * v51_data);
              v121_acc += ((static_cast<float>(v125_data[8])) * v55_data);
              v121_acc += ((static_cast<float>(v125_data[9])) * v59_data);
              v121_acc += ((static_cast<float>(v125_data[10])) * v63_data);
              v121_acc += ((static_cast<float>(v125_data[11])) * v67_data);
              v121_acc += ((static_cast<float>(v125_data[12])) * v71_data);
              v121_acc += ((static_cast<float>(v125_data[13])) * v75_data);
              v121_acc += ((static_cast<float>(v125_data[14])) * v79_data);
              v121_acc += ((static_cast<float>(v125_data[15])) * v83_data);
              v121_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v162_data;
              v162_data.copy_from(s0 + (32_i32));
              v158_acc += ((static_cast<float>(v162_data[0])) * v23_data);
              v158_acc += ((static_cast<float>(v162_data[1])) * v27_data);
              v158_acc += ((static_cast<float>(v162_data[2])) * v31_data);
              v158_acc += ((static_cast<float>(v162_data[3])) * v35_data);
              v158_acc += ((static_cast<float>(v162_data[4])) * v39_data);
              v158_acc += ((static_cast<float>(v162_data[5])) * v43_data);
              v158_acc += ((static_cast<float>(v162_data[6])) * v47_data);
              v158_acc += ((static_cast<float>(v162_data[7])) * v51_data);
              v158_acc += ((static_cast<float>(v162_data[8])) * v55_data);
              v158_acc += ((static_cast<float>(v162_data[9])) * v59_data);
              v158_acc += ((static_cast<float>(v162_data[10])) * v63_data);
              v158_acc += ((static_cast<float>(v162_data[11])) * v67_data);
              v158_acc += ((static_cast<float>(v162_data[12])) * v71_data);
              v158_acc += ((static_cast<float>(v162_data[13])) * v75_data);
              v158_acc += ((static_cast<float>(v162_data[14])) * v79_data);
              v158_acc += ((static_cast<float>(v162_data[15])) * v83_data);
              v158_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v195_acc{};
              tensorforge::intel_esimd::simd<float, 16> v199_data;
              v199_data.copy_from(s0 + (48_i32));
              v195_acc += ((static_cast<float>(v199_data[0])) * v23_data);
              v195_acc += ((static_cast<float>(v199_data[1])) * v27_data);
              v195_acc += ((static_cast<float>(v199_data[2])) * v31_data);
              v195_acc += ((static_cast<float>(v199_data[3])) * v35_data);
              v195_acc += ((static_cast<float>(v199_data[4])) * v39_data);
              v195_acc += ((static_cast<float>(v199_data[5])) * v43_data);
              v195_acc += ((static_cast<float>(v199_data[6])) * v47_data);
              v195_acc += ((static_cast<float>(v199_data[7])) * v51_data);
              v195_acc += ((static_cast<float>(v199_data[8])) * v55_data);
              v195_acc += ((static_cast<float>(v199_data[9])) * v59_data);
              v195_acc += ((static_cast<float>(v199_data[10])) * v63_data);
              v195_acc += ((static_cast<float>(v199_data[11])) * v67_data);
              v195_acc += ((static_cast<float>(v199_data[12])) * v71_data);
              v195_acc += ((static_cast<float>(v199_data[13])) * v75_data);
              v195_acc += ((static_cast<float>(v199_data[14])) * v79_data);
              v195_acc += ((static_cast<float>(v199_data[15])) * v83_data);
              v195_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v232_acc{};
              tensorforge::intel_esimd::simd<float, 16> v236_data;
              v236_data.copy_from(s0 + (64_i32));
              v232_acc += ((static_cast<float>(v236_data[0])) * v23_data);
              v232_acc += ((static_cast<float>(v236_data[1])) * v27_data);
              v232_acc += ((static_cast<float>(v236_data[2])) * v31_data);
              v232_acc += ((static_cast<float>(v236_data[3])) * v35_data);
              v232_acc += ((static_cast<float>(v236_data[4])) * v39_data);
              v232_acc += ((static_cast<float>(v236_data[5])) * v43_data);
              v232_acc += ((static_cast<float>(v236_data[6])) * v47_data);
              v232_acc += ((static_cast<float>(v236_data[7])) * v51_data);
              v232_acc += ((static_cast<float>(v236_data[8])) * v55_data);
              v232_acc += ((static_cast<float>(v236_data[9])) * v59_data);
              v232_acc += ((static_cast<float>(v236_data[10])) * v63_data);
              v232_acc += ((static_cast<float>(v236_data[11])) * v67_data);
              v232_acc += ((static_cast<float>(v236_data[12])) * v71_data);
              v232_acc += ((static_cast<float>(v236_data[13])) * v75_data);
              v232_acc += ((static_cast<float>(v236_data[14])) * v79_data);
              v232_acc += ((static_cast<float>(v236_data[15])) * v83_data);
              v232_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v269_acc{};
              tensorforge::intel_esimd::simd<float, 16> v273_data;
              v273_data.copy_from(s0 + (80_i32));
              v269_acc += ((static_cast<float>(v273_data[0])) * v23_data);
              v269_acc += ((static_cast<float>(v273_data[1])) * v27_data);
              v269_acc += ((static_cast<float>(v273_data[2])) * v31_data);
              v269_acc += ((static_cast<float>(v273_data[3])) * v35_data);
              v269_acc += ((static_cast<float>(v273_data[4])) * v39_data);
              v269_acc += ((static_cast<float>(v273_data[5])) * v43_data);
              v269_acc += ((static_cast<float>(v273_data[6])) * v47_data);
              v269_acc += ((static_cast<float>(v273_data[7])) * v51_data);
              v269_acc += ((static_cast<float>(v273_data[8])) * v55_data);
              v269_acc += ((static_cast<float>(v273_data[9])) * v59_data);
              v269_acc += ((static_cast<float>(v273_data[10])) * v63_data);
              v269_acc += ((static_cast<float>(v273_data[11])) * v67_data);
              v269_acc += ((static_cast<float>(v273_data[12])) * v71_data);
              v269_acc += ((static_cast<float>(v273_data[13])) * v75_data);
              v269_acc += ((static_cast<float>(v273_data[14])) * v79_data);
              v269_acc += ((static_cast<float>(v273_data[15])) * v83_data);
              v269_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v306_acc{};
              tensorforge::intel_esimd::simd<float, 16> v310_data;
              v310_data.copy_from(s0 + (96_i32));
              v306_acc += ((static_cast<float>(v310_data[0])) * v23_data);
              v306_acc += ((static_cast<float>(v310_data[1])) * v27_data);
              v306_acc += ((static_cast<float>(v310_data[2])) * v31_data);
              v306_acc += ((static_cast<float>(v310_data[3])) * v35_data);
              v306_acc += ((static_cast<float>(v310_data[4])) * v39_data);
              v306_acc += ((static_cast<float>(v310_data[5])) * v43_data);
              v306_acc += ((static_cast<float>(v310_data[6])) * v47_data);
              v306_acc += ((static_cast<float>(v310_data[7])) * v51_data);
              v306_acc += ((static_cast<float>(v310_data[8])) * v55_data);
              v306_acc += ((static_cast<float>(v310_data[9])) * v59_data);
              v306_acc += ((static_cast<float>(v310_data[10])) * v63_data);
              v306_acc += ((static_cast<float>(v310_data[11])) * v67_data);
              v306_acc += ((static_cast<float>(v310_data[12])) * v71_data);
              v306_acc += ((static_cast<float>(v310_data[13])) * v75_data);
              v306_acc += ((static_cast<float>(v310_data[14])) * v79_data);
              v306_acc += ((static_cast<float>(v310_data[15])) * v83_data);
              v306_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v343_acc{};
              tensorforge::intel_esimd::simd<float, 16> v347_data;
              v347_data.copy_from(s0 + (112_i32));
              v343_acc += ((static_cast<float>(v347_data[0])) * v23_data);
              v343_acc += ((static_cast<float>(v347_data[1])) * v27_data);
              v343_acc += ((static_cast<float>(v347_data[2])) * v31_data);
              v343_acc += ((static_cast<float>(v347_data[3])) * v35_data);
              v343_acc += ((static_cast<float>(v347_data[4])) * v39_data);
              v343_acc += ((static_cast<float>(v347_data[5])) * v43_data);
              v343_acc += ((static_cast<float>(v347_data[6])) * v47_data);
              v343_acc += ((static_cast<float>(v347_data[7])) * v51_data);
              v343_acc += ((static_cast<float>(v347_data[8])) * v55_data);
              v343_acc += ((static_cast<float>(v347_data[9])) * v59_data);
              v343_acc += ((static_cast<float>(v347_data[10])) * v63_data);
              v343_acc += ((static_cast<float>(v347_data[11])) * v67_data);
              v343_acc += ((static_cast<float>(v347_data[12])) * v71_data);
              v343_acc += ((static_cast<float>(v347_data[13])) * v75_data);
              v343_acc += ((static_cast<float>(v347_data[14])) * v79_data);
              v343_acc += ((static_cast<float>(v347_data[15])) * v83_data);
              v343_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v380_acc{};
              tensorforge::intel_esimd::simd<float, 16> v384_data;
              v384_data.copy_from(s0 + (128_i32));
              v380_acc += ((static_cast<float>(v384_data[0])) * v23_data);
              v380_acc += ((static_cast<float>(v384_data[1])) * v27_data);
              v380_acc += ((static_cast<float>(v384_data[2])) * v31_data);
              v380_acc += ((static_cast<float>(v384_data[3])) * v35_data);
              v380_acc += ((static_cast<float>(v384_data[4])) * v39_data);
              v380_acc += ((static_cast<float>(v384_data[5])) * v43_data);
              v380_acc += ((static_cast<float>(v384_data[6])) * v47_data);
              v380_acc += ((static_cast<float>(v384_data[7])) * v51_data);
              v380_acc += ((static_cast<float>(v384_data[8])) * v55_data);
              v380_acc += ((static_cast<float>(v384_data[9])) * v59_data);
              v380_acc += ((static_cast<float>(v384_data[10])) * v63_data);
              v380_acc += ((static_cast<float>(v384_data[11])) * v67_data);
              v380_acc += ((static_cast<float>(v384_data[12])) * v71_data);
              v380_acc += ((static_cast<float>(v384_data[13])) * v75_data);
              v380_acc += ((static_cast<float>(v384_data[14])) * v79_data);
              v380_acc += ((static_cast<float>(v384_data[15])) * v83_data);
              v380_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v417_acc{};
              tensorforge::intel_esimd::simd<float, 16> v421_data;
              v421_data.copy_from(s0 + (144_i32));
              v417_acc += ((static_cast<float>(v421_data[0])) * v23_data);
              v417_acc += ((static_cast<float>(v421_data[1])) * v27_data);
              v417_acc += ((static_cast<float>(v421_data[2])) * v31_data);
              v417_acc += ((static_cast<float>(v421_data[3])) * v35_data);
              v417_acc += ((static_cast<float>(v421_data[4])) * v39_data);
              v417_acc += ((static_cast<float>(v421_data[5])) * v43_data);
              v417_acc += ((static_cast<float>(v421_data[6])) * v47_data);
              v417_acc += ((static_cast<float>(v421_data[7])) * v51_data);
              v417_acc += ((static_cast<float>(v421_data[8])) * v55_data);
              v417_acc += ((static_cast<float>(v421_data[9])) * v59_data);
              v417_acc += ((static_cast<float>(v421_data[10])) * v63_data);
              v417_acc += ((static_cast<float>(v421_data[11])) * v67_data);
              v417_acc += ((static_cast<float>(v421_data[12])) * v71_data);
              v417_acc += ((static_cast<float>(v421_data[13])) * v75_data);
              v417_acc += ((static_cast<float>(v421_data[14])) * v79_data);
              v417_acc += ((static_cast<float>(v421_data[15])) * v83_data);
              v417_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v454_acc{};
              tensorforge::intel_esimd::simd<float, 16> v458_data;
              v458_data.copy_from(s0 + (160_i32));
              v454_acc += ((static_cast<float>(v458_data[0])) * v23_data);
              v454_acc += ((static_cast<float>(v458_data[1])) * v27_data);
              v454_acc += ((static_cast<float>(v458_data[2])) * v31_data);
              v454_acc += ((static_cast<float>(v458_data[3])) * v35_data);
              v454_acc += ((static_cast<float>(v458_data[4])) * v39_data);
              v454_acc += ((static_cast<float>(v458_data[5])) * v43_data);
              v454_acc += ((static_cast<float>(v458_data[6])) * v47_data);
              v454_acc += ((static_cast<float>(v458_data[7])) * v51_data);
              v454_acc += ((static_cast<float>(v458_data[8])) * v55_data);
              v454_acc += ((static_cast<float>(v458_data[9])) * v59_data);
              v454_acc += ((static_cast<float>(v458_data[10])) * v63_data);
              v454_acc += ((static_cast<float>(v458_data[11])) * v67_data);
              v454_acc += ((static_cast<float>(v458_data[12])) * v71_data);
              v454_acc += ((static_cast<float>(v458_data[13])) * v75_data);
              v454_acc += ((static_cast<float>(v458_data[14])) * v79_data);
              v454_acc += ((static_cast<float>(v458_data[15])) * v83_data);
              v454_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v491_acc{};
              tensorforge::intel_esimd::simd<float, 16> v495_data;
              v495_data.copy_from(s0 + (176_i32));
              v491_acc += ((static_cast<float>(v495_data[0])) * v23_data);
              v491_acc += ((static_cast<float>(v495_data[1])) * v27_data);
              v491_acc += ((static_cast<float>(v495_data[2])) * v31_data);
              v491_acc += ((static_cast<float>(v495_data[3])) * v35_data);
              v491_acc += ((static_cast<float>(v495_data[4])) * v39_data);
              v491_acc += ((static_cast<float>(v495_data[5])) * v43_data);
              v491_acc += ((static_cast<float>(v495_data[6])) * v47_data);
              v491_acc += ((static_cast<float>(v495_data[7])) * v51_data);
              v491_acc += ((static_cast<float>(v495_data[8])) * v55_data);
              v491_acc += ((static_cast<float>(v495_data[9])) * v59_data);
              v491_acc += ((static_cast<float>(v495_data[10])) * v63_data);
              v491_acc += ((static_cast<float>(v495_data[11])) * v67_data);
              v491_acc += ((static_cast<float>(v495_data[12])) * v71_data);
              v491_acc += ((static_cast<float>(v495_data[13])) * v75_data);
              v491_acc += ((static_cast<float>(v495_data[14])) * v79_data);
              v491_acc += ((static_cast<float>(v495_data[15])) * v83_data);
              v491_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v528_acc{};
              tensorforge::intel_esimd::simd<float, 16> v532_data;
              v532_data.copy_from(s0 + (192_i32));
              v528_acc += ((static_cast<float>(v532_data[0])) * v23_data);
              v528_acc += ((static_cast<float>(v532_data[1])) * v27_data);
              v528_acc += ((static_cast<float>(v532_data[2])) * v31_data);
              v528_acc += ((static_cast<float>(v532_data[3])) * v35_data);
              v528_acc += ((static_cast<float>(v532_data[4])) * v39_data);
              v528_acc += ((static_cast<float>(v532_data[5])) * v43_data);
              v528_acc += ((static_cast<float>(v532_data[6])) * v47_data);
              v528_acc += ((static_cast<float>(v532_data[7])) * v51_data);
              v528_acc += ((static_cast<float>(v532_data[8])) * v55_data);
              v528_acc += ((static_cast<float>(v532_data[9])) * v59_data);
              v528_acc += ((static_cast<float>(v532_data[10])) * v63_data);
              v528_acc += ((static_cast<float>(v532_data[11])) * v67_data);
              v528_acc += ((static_cast<float>(v532_data[12])) * v71_data);
              v528_acc += ((static_cast<float>(v532_data[13])) * v75_data);
              v528_acc += ((static_cast<float>(v532_data[14])) * v79_data);
              v528_acc += ((static_cast<float>(v532_data[15])) * v83_data);
              v528_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v565_acc{};
              tensorforge::intel_esimd::simd<float, 16> v569_data;
              v569_data.copy_from(s0 + (208_i32));
              v565_acc += ((static_cast<float>(v569_data[0])) * v23_data);
              v565_acc += ((static_cast<float>(v569_data[1])) * v27_data);
              v565_acc += ((static_cast<float>(v569_data[2])) * v31_data);
              v565_acc += ((static_cast<float>(v569_data[3])) * v35_data);
              v565_acc += ((static_cast<float>(v569_data[4])) * v39_data);
              v565_acc += ((static_cast<float>(v569_data[5])) * v43_data);
              v565_acc += ((static_cast<float>(v569_data[6])) * v47_data);
              v565_acc += ((static_cast<float>(v569_data[7])) * v51_data);
              v565_acc += ((static_cast<float>(v569_data[8])) * v55_data);
              v565_acc += ((static_cast<float>(v569_data[9])) * v59_data);
              v565_acc += ((static_cast<float>(v569_data[10])) * v63_data);
              v565_acc += ((static_cast<float>(v569_data[11])) * v67_data);
              v565_acc += ((static_cast<float>(v569_data[12])) * v71_data);
              v565_acc += ((static_cast<float>(v569_data[13])) * v75_data);
              v565_acc += ((static_cast<float>(v569_data[14])) * v79_data);
              v565_acc += ((static_cast<float>(v569_data[15])) * v83_data);
              v565_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v602_acc{};
              tensorforge::intel_esimd::simd<float, 16> v606_data;
              v606_data.copy_from(s0 + (224_i32));
              v602_acc += ((static_cast<float>(v606_data[0])) * v23_data);
              v602_acc += ((static_cast<float>(v606_data[1])) * v27_data);
              v602_acc += ((static_cast<float>(v606_data[2])) * v31_data);
              v602_acc += ((static_cast<float>(v606_data[3])) * v35_data);
              v602_acc += ((static_cast<float>(v606_data[4])) * v39_data);
              v602_acc += ((static_cast<float>(v606_data[5])) * v43_data);
              v602_acc += ((static_cast<float>(v606_data[6])) * v47_data);
              v602_acc += ((static_cast<float>(v606_data[7])) * v51_data);
              v602_acc += ((static_cast<float>(v606_data[8])) * v55_data);
              v602_acc += ((static_cast<float>(v606_data[9])) * v59_data);
              v602_acc += ((static_cast<float>(v606_data[10])) * v63_data);
              v602_acc += ((static_cast<float>(v606_data[11])) * v67_data);
              v602_acc += ((static_cast<float>(v606_data[12])) * v71_data);
              v602_acc += ((static_cast<float>(v606_data[13])) * v75_data);
              v602_acc += ((static_cast<float>(v606_data[14])) * v79_data);
              v602_acc += ((static_cast<float>(v606_data[15])) * v83_data);
              v602_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v639_acc{};
              tensorforge::intel_esimd::simd<float, 16> v643_data;
              v643_data.copy_from(s0 + (240_i32));
              v639_acc += ((static_cast<float>(v643_data[0])) * v23_data);
              v639_acc += ((static_cast<float>(v643_data[1])) * v27_data);
              v639_acc += ((static_cast<float>(v643_data[2])) * v31_data);
              v639_acc += ((static_cast<float>(v643_data[3])) * v35_data);
              v639_acc += ((static_cast<float>(v643_data[4])) * v39_data);
              v639_acc += ((static_cast<float>(v643_data[5])) * v43_data);
              v639_acc += ((static_cast<float>(v643_data[6])) * v47_data);
              v639_acc += ((static_cast<float>(v643_data[7])) * v51_data);
              v639_acc += ((static_cast<float>(v643_data[8])) * v55_data);
              v639_acc += ((static_cast<float>(v643_data[9])) * v59_data);
              v639_acc += ((static_cast<float>(v643_data[10])) * v63_data);
              v639_acc += ((static_cast<float>(v643_data[11])) * v67_data);
              v639_acc += ((static_cast<float>(v643_data[12])) * v71_data);
              v639_acc += ((static_cast<float>(v643_data[13])) * v75_data);
              v639_acc += ((static_cast<float>(v643_data[14])) * v79_data);
              v639_acc += ((static_cast<float>(v643_data[15])) * v83_data);
              v639_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v676_n0 = 0; v676_n0 < 1; ++v676_n0) {
                int32_t v678_a = v676_n0 * 16;
                #pragma unroll
                for (int32_t v677_n1 = 0; v677_n1 < 16; ++v677_n1) {
                  int32_t v680_a = v678_a + (v677_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v681_data;
                  v681_data.copy_from(ir0 + (v680_a));
                  v681_data.copy_to(r0 + (v680_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v685_i0 = 0; v685_i0 < 1; ++v685_i0) {
                int32_t v687_a = v685_i0 * 16;
                #pragma unroll
                for (int32_t v686_i1 = 0; v686_i1 < 16; ++v686_i1) {
                  int32_t v689_a = v687_a + (v686_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v690_data;
                  v690_data.copy_from(r0 + (v689_a));
                  v690_data.copy_to(glb_m0 + (v689_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

