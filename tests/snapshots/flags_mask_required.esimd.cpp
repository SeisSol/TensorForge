// === base name ===
kernel_01d95c828acbd0fc

// === header ===
void launcher_kernel_01d95c828acbd0fc(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_01d95c828acbd0fc(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_01d95c828acbd0fc(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_01d95c828acbd0fc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[272 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[256];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v13_lead = v11_i0 * 16;
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
                  int32_t v16_a = v13_lead + (v12_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v17_data;
                  v17_data.copy_from(glb_m1 + (v16_a));
                  v17_data.copy_to(r0 + (v16_a));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v21_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v22_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              tensorforge::intel_esimd::simd<float, 64> v23_ld;
              v23_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              v23_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 128));
              tensorforge::intel_esimd::simd<float, 64> v24_ld;
              v24_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              v24_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 192));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v43_acc{};
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(s0 + (0_i32));
              v43_acc += ((v47_data[0]) * v27_data);
              v43_acc += ((v47_data[1]) * v28_data);
              v43_acc += ((v47_data[2]) * v29_data);
              v43_acc += ((v47_data[3]) * v30_data);
              v43_acc += ((v47_data[4]) * v31_data);
              v43_acc += ((v47_data[5]) * v32_data);
              v43_acc += ((v47_data[6]) * v33_data);
              v43_acc += ((v47_data[7]) * v34_data);
              v43_acc += ((v47_data[8]) * v35_data);
              v43_acc += ((v47_data[9]) * v36_data);
              v43_acc += ((v47_data[10]) * v37_data);
              v43_acc += ((v47_data[11]) * v38_data);
              v43_acc += ((v47_data[12]) * v39_data);
              v43_acc += ((v47_data[13]) * v40_data);
              v43_acc += ((v47_data[14]) * v41_data);
              v43_acc += ((v47_data[15]) * v42_data);
              v43_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v80_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data;
              v84_data.copy_from(s0 + (16_i32));
              v80_acc += ((v84_data[0]) * v27_data);
              v80_acc += ((v84_data[1]) * v28_data);
              v80_acc += ((v84_data[2]) * v29_data);
              v80_acc += ((v84_data[3]) * v30_data);
              v80_acc += ((v84_data[4]) * v31_data);
              v80_acc += ((v84_data[5]) * v32_data);
              v80_acc += ((v84_data[6]) * v33_data);
              v80_acc += ((v84_data[7]) * v34_data);
              v80_acc += ((v84_data[8]) * v35_data);
              v80_acc += ((v84_data[9]) * v36_data);
              v80_acc += ((v84_data[10]) * v37_data);
              v80_acc += ((v84_data[11]) * v38_data);
              v80_acc += ((v84_data[12]) * v39_data);
              v80_acc += ((v84_data[13]) * v40_data);
              v80_acc += ((v84_data[14]) * v41_data);
              v80_acc += ((v84_data[15]) * v42_data);
              v80_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v117_acc{};
              tensorforge::intel_esimd::simd<float, 16> v121_data;
              v121_data.copy_from(s0 + (32_i32));
              v117_acc += ((v121_data[0]) * v27_data);
              v117_acc += ((v121_data[1]) * v28_data);
              v117_acc += ((v121_data[2]) * v29_data);
              v117_acc += ((v121_data[3]) * v30_data);
              v117_acc += ((v121_data[4]) * v31_data);
              v117_acc += ((v121_data[5]) * v32_data);
              v117_acc += ((v121_data[6]) * v33_data);
              v117_acc += ((v121_data[7]) * v34_data);
              v117_acc += ((v121_data[8]) * v35_data);
              v117_acc += ((v121_data[9]) * v36_data);
              v117_acc += ((v121_data[10]) * v37_data);
              v117_acc += ((v121_data[11]) * v38_data);
              v117_acc += ((v121_data[12]) * v39_data);
              v117_acc += ((v121_data[13]) * v40_data);
              v117_acc += ((v121_data[14]) * v41_data);
              v117_acc += ((v121_data[15]) * v42_data);
              v117_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v158_data;
              v158_data.copy_from(s0 + (48_i32));
              v154_acc += ((v158_data[0]) * v27_data);
              v154_acc += ((v158_data[1]) * v28_data);
              v154_acc += ((v158_data[2]) * v29_data);
              v154_acc += ((v158_data[3]) * v30_data);
              v154_acc += ((v158_data[4]) * v31_data);
              v154_acc += ((v158_data[5]) * v32_data);
              v154_acc += ((v158_data[6]) * v33_data);
              v154_acc += ((v158_data[7]) * v34_data);
              v154_acc += ((v158_data[8]) * v35_data);
              v154_acc += ((v158_data[9]) * v36_data);
              v154_acc += ((v158_data[10]) * v37_data);
              v154_acc += ((v158_data[11]) * v38_data);
              v154_acc += ((v158_data[12]) * v39_data);
              v154_acc += ((v158_data[13]) * v40_data);
              v154_acc += ((v158_data[14]) * v41_data);
              v154_acc += ((v158_data[15]) * v42_data);
              v154_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data;
              v195_data.copy_from(s0 + (64_i32));
              v191_acc += ((v195_data[0]) * v27_data);
              v191_acc += ((v195_data[1]) * v28_data);
              v191_acc += ((v195_data[2]) * v29_data);
              v191_acc += ((v195_data[3]) * v30_data);
              v191_acc += ((v195_data[4]) * v31_data);
              v191_acc += ((v195_data[5]) * v32_data);
              v191_acc += ((v195_data[6]) * v33_data);
              v191_acc += ((v195_data[7]) * v34_data);
              v191_acc += ((v195_data[8]) * v35_data);
              v191_acc += ((v195_data[9]) * v36_data);
              v191_acc += ((v195_data[10]) * v37_data);
              v191_acc += ((v195_data[11]) * v38_data);
              v191_acc += ((v195_data[12]) * v39_data);
              v191_acc += ((v195_data[13]) * v40_data);
              v191_acc += ((v195_data[14]) * v41_data);
              v191_acc += ((v195_data[15]) * v42_data);
              v191_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              tensorforge::intel_esimd::simd<float, 16> v232_data;
              v232_data.copy_from(s0 + (80_i32));
              v228_acc += ((v232_data[0]) * v27_data);
              v228_acc += ((v232_data[1]) * v28_data);
              v228_acc += ((v232_data[2]) * v29_data);
              v228_acc += ((v232_data[3]) * v30_data);
              v228_acc += ((v232_data[4]) * v31_data);
              v228_acc += ((v232_data[5]) * v32_data);
              v228_acc += ((v232_data[6]) * v33_data);
              v228_acc += ((v232_data[7]) * v34_data);
              v228_acc += ((v232_data[8]) * v35_data);
              v228_acc += ((v232_data[9]) * v36_data);
              v228_acc += ((v232_data[10]) * v37_data);
              v228_acc += ((v232_data[11]) * v38_data);
              v228_acc += ((v232_data[12]) * v39_data);
              v228_acc += ((v232_data[13]) * v40_data);
              v228_acc += ((v232_data[14]) * v41_data);
              v228_acc += ((v232_data[15]) * v42_data);
              v228_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v265_acc{};
              tensorforge::intel_esimd::simd<float, 16> v269_data;
              v269_data.copy_from(s0 + (96_i32));
              v265_acc += ((v269_data[0]) * v27_data);
              v265_acc += ((v269_data[1]) * v28_data);
              v265_acc += ((v269_data[2]) * v29_data);
              v265_acc += ((v269_data[3]) * v30_data);
              v265_acc += ((v269_data[4]) * v31_data);
              v265_acc += ((v269_data[5]) * v32_data);
              v265_acc += ((v269_data[6]) * v33_data);
              v265_acc += ((v269_data[7]) * v34_data);
              v265_acc += ((v269_data[8]) * v35_data);
              v265_acc += ((v269_data[9]) * v36_data);
              v265_acc += ((v269_data[10]) * v37_data);
              v265_acc += ((v269_data[11]) * v38_data);
              v265_acc += ((v269_data[12]) * v39_data);
              v265_acc += ((v269_data[13]) * v40_data);
              v265_acc += ((v269_data[14]) * v41_data);
              v265_acc += ((v269_data[15]) * v42_data);
              v265_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v302_acc{};
              tensorforge::intel_esimd::simd<float, 16> v306_data;
              v306_data.copy_from(s0 + (112_i32));
              v302_acc += ((v306_data[0]) * v27_data);
              v302_acc += ((v306_data[1]) * v28_data);
              v302_acc += ((v306_data[2]) * v29_data);
              v302_acc += ((v306_data[3]) * v30_data);
              v302_acc += ((v306_data[4]) * v31_data);
              v302_acc += ((v306_data[5]) * v32_data);
              v302_acc += ((v306_data[6]) * v33_data);
              v302_acc += ((v306_data[7]) * v34_data);
              v302_acc += ((v306_data[8]) * v35_data);
              v302_acc += ((v306_data[9]) * v36_data);
              v302_acc += ((v306_data[10]) * v37_data);
              v302_acc += ((v306_data[11]) * v38_data);
              v302_acc += ((v306_data[12]) * v39_data);
              v302_acc += ((v306_data[13]) * v40_data);
              v302_acc += ((v306_data[14]) * v41_data);
              v302_acc += ((v306_data[15]) * v42_data);
              v302_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v339_acc{};
              tensorforge::intel_esimd::simd<float, 16> v343_data;
              v343_data.copy_from(s0 + (128_i32));
              v339_acc += ((v343_data[0]) * v27_data);
              v339_acc += ((v343_data[1]) * v28_data);
              v339_acc += ((v343_data[2]) * v29_data);
              v339_acc += ((v343_data[3]) * v30_data);
              v339_acc += ((v343_data[4]) * v31_data);
              v339_acc += ((v343_data[5]) * v32_data);
              v339_acc += ((v343_data[6]) * v33_data);
              v339_acc += ((v343_data[7]) * v34_data);
              v339_acc += ((v343_data[8]) * v35_data);
              v339_acc += ((v343_data[9]) * v36_data);
              v339_acc += ((v343_data[10]) * v37_data);
              v339_acc += ((v343_data[11]) * v38_data);
              v339_acc += ((v343_data[12]) * v39_data);
              v339_acc += ((v343_data[13]) * v40_data);
              v339_acc += ((v343_data[14]) * v41_data);
              v339_acc += ((v343_data[15]) * v42_data);
              v339_acc.copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 16> v376_acc{};
              tensorforge::intel_esimd::simd<float, 16> v380_data;
              v380_data.copy_from(s0 + (144_i32));
              v376_acc += ((v380_data[0]) * v27_data);
              v376_acc += ((v380_data[1]) * v28_data);
              v376_acc += ((v380_data[2]) * v29_data);
              v376_acc += ((v380_data[3]) * v30_data);
              v376_acc += ((v380_data[4]) * v31_data);
              v376_acc += ((v380_data[5]) * v32_data);
              v376_acc += ((v380_data[6]) * v33_data);
              v376_acc += ((v380_data[7]) * v34_data);
              v376_acc += ((v380_data[8]) * v35_data);
              v376_acc += ((v380_data[9]) * v36_data);
              v376_acc += ((v380_data[10]) * v37_data);
              v376_acc += ((v380_data[11]) * v38_data);
              v376_acc += ((v380_data[12]) * v39_data);
              v376_acc += ((v380_data[13]) * v40_data);
              v376_acc += ((v380_data[14]) * v41_data);
              v376_acc += ((v380_data[15]) * v42_data);
              v376_acc.copy_to(ir1 + (144));
              tensorforge::intel_esimd::simd<float, 16> v413_acc{};
              tensorforge::intel_esimd::simd<float, 16> v417_data;
              v417_data.copy_from(s0 + (160_i32));
              v413_acc += ((v417_data[0]) * v27_data);
              v413_acc += ((v417_data[1]) * v28_data);
              v413_acc += ((v417_data[2]) * v29_data);
              v413_acc += ((v417_data[3]) * v30_data);
              v413_acc += ((v417_data[4]) * v31_data);
              v413_acc += ((v417_data[5]) * v32_data);
              v413_acc += ((v417_data[6]) * v33_data);
              v413_acc += ((v417_data[7]) * v34_data);
              v413_acc += ((v417_data[8]) * v35_data);
              v413_acc += ((v417_data[9]) * v36_data);
              v413_acc += ((v417_data[10]) * v37_data);
              v413_acc += ((v417_data[11]) * v38_data);
              v413_acc += ((v417_data[12]) * v39_data);
              v413_acc += ((v417_data[13]) * v40_data);
              v413_acc += ((v417_data[14]) * v41_data);
              v413_acc += ((v417_data[15]) * v42_data);
              v413_acc.copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 16> v450_acc{};
              tensorforge::intel_esimd::simd<float, 16> v454_data;
              v454_data.copy_from(s0 + (176_i32));
              v450_acc += ((v454_data[0]) * v27_data);
              v450_acc += ((v454_data[1]) * v28_data);
              v450_acc += ((v454_data[2]) * v29_data);
              v450_acc += ((v454_data[3]) * v30_data);
              v450_acc += ((v454_data[4]) * v31_data);
              v450_acc += ((v454_data[5]) * v32_data);
              v450_acc += ((v454_data[6]) * v33_data);
              v450_acc += ((v454_data[7]) * v34_data);
              v450_acc += ((v454_data[8]) * v35_data);
              v450_acc += ((v454_data[9]) * v36_data);
              v450_acc += ((v454_data[10]) * v37_data);
              v450_acc += ((v454_data[11]) * v38_data);
              v450_acc += ((v454_data[12]) * v39_data);
              v450_acc += ((v454_data[13]) * v40_data);
              v450_acc += ((v454_data[14]) * v41_data);
              v450_acc += ((v454_data[15]) * v42_data);
              v450_acc.copy_to(ir1 + (176));
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              tensorforge::intel_esimd::simd<float, 16> v491_data;
              v491_data.copy_from(s0 + (192_i32));
              v487_acc += ((v491_data[0]) * v27_data);
              v487_acc += ((v491_data[1]) * v28_data);
              v487_acc += ((v491_data[2]) * v29_data);
              v487_acc += ((v491_data[3]) * v30_data);
              v487_acc += ((v491_data[4]) * v31_data);
              v487_acc += ((v491_data[5]) * v32_data);
              v487_acc += ((v491_data[6]) * v33_data);
              v487_acc += ((v491_data[7]) * v34_data);
              v487_acc += ((v491_data[8]) * v35_data);
              v487_acc += ((v491_data[9]) * v36_data);
              v487_acc += ((v491_data[10]) * v37_data);
              v487_acc += ((v491_data[11]) * v38_data);
              v487_acc += ((v491_data[12]) * v39_data);
              v487_acc += ((v491_data[13]) * v40_data);
              v487_acc += ((v491_data[14]) * v41_data);
              v487_acc += ((v491_data[15]) * v42_data);
              v487_acc.copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 16> v524_acc{};
              tensorforge::intel_esimd::simd<float, 16> v528_data;
              v528_data.copy_from(s0 + (208_i32));
              v524_acc += ((v528_data[0]) * v27_data);
              v524_acc += ((v528_data[1]) * v28_data);
              v524_acc += ((v528_data[2]) * v29_data);
              v524_acc += ((v528_data[3]) * v30_data);
              v524_acc += ((v528_data[4]) * v31_data);
              v524_acc += ((v528_data[5]) * v32_data);
              v524_acc += ((v528_data[6]) * v33_data);
              v524_acc += ((v528_data[7]) * v34_data);
              v524_acc += ((v528_data[8]) * v35_data);
              v524_acc += ((v528_data[9]) * v36_data);
              v524_acc += ((v528_data[10]) * v37_data);
              v524_acc += ((v528_data[11]) * v38_data);
              v524_acc += ((v528_data[12]) * v39_data);
              v524_acc += ((v528_data[13]) * v40_data);
              v524_acc += ((v528_data[14]) * v41_data);
              v524_acc += ((v528_data[15]) * v42_data);
              v524_acc.copy_to(ir1 + (208));
              tensorforge::intel_esimd::simd<float, 16> v561_acc{};
              tensorforge::intel_esimd::simd<float, 16> v565_data;
              v565_data.copy_from(s0 + (224_i32));
              v561_acc += ((v565_data[0]) * v27_data);
              v561_acc += ((v565_data[1]) * v28_data);
              v561_acc += ((v565_data[2]) * v29_data);
              v561_acc += ((v565_data[3]) * v30_data);
              v561_acc += ((v565_data[4]) * v31_data);
              v561_acc += ((v565_data[5]) * v32_data);
              v561_acc += ((v565_data[6]) * v33_data);
              v561_acc += ((v565_data[7]) * v34_data);
              v561_acc += ((v565_data[8]) * v35_data);
              v561_acc += ((v565_data[9]) * v36_data);
              v561_acc += ((v565_data[10]) * v37_data);
              v561_acc += ((v565_data[11]) * v38_data);
              v561_acc += ((v565_data[12]) * v39_data);
              v561_acc += ((v565_data[13]) * v40_data);
              v561_acc += ((v565_data[14]) * v41_data);
              v561_acc += ((v565_data[15]) * v42_data);
              v561_acc.copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 16> v598_acc{};
              tensorforge::intel_esimd::simd<float, 16> v602_data;
              v602_data.copy_from(s0 + (240_i32));
              v598_acc += ((v602_data[0]) * v27_data);
              v598_acc += ((v602_data[1]) * v28_data);
              v598_acc += ((v602_data[2]) * v29_data);
              v598_acc += ((v602_data[3]) * v30_data);
              v598_acc += ((v602_data[4]) * v31_data);
              v598_acc += ((v602_data[5]) * v32_data);
              v598_acc += ((v602_data[6]) * v33_data);
              v598_acc += ((v602_data[7]) * v34_data);
              v598_acc += ((v602_data[8]) * v35_data);
              v598_acc += ((v602_data[9]) * v36_data);
              v598_acc += ((v602_data[10]) * v37_data);
              v598_acc += ((v602_data[11]) * v38_data);
              v598_acc += ((v602_data[12]) * v39_data);
              v598_acc += ((v602_data[13]) * v40_data);
              v598_acc += ((v602_data[14]) * v41_data);
              v598_acc += ((v602_data[15]) * v42_data);
              v598_acc.copy_to(ir1 + (240));
              #pragma unroll
              for (int32_t v635_n0 = 0; v635_n0 < 1; ++v635_n0) {
                int32_t v637_a = v635_n0 * 16;
                #pragma unroll
                for (int32_t v636_n1 = 0; v636_n1 < 16; ++v636_n1) {
                  int32_t v639_a = v637_a + (v636_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v640_data;
                  v640_data.copy_from(ir1 + (v639_a));
                  v640_data.copy_to(r1 + (v639_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v644_i0 = 0; v644_i0 < 1; ++v644_i0) {
                int32_t v646_a = v644_i0 * 16;
                #pragma unroll
                for (int32_t v645_i1 = 0; v645_i1 < 16; ++v645_i1) {
                  int32_t v648_a = v646_a + (v645_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v649_data;
                  v649_data.copy_from(r1 + (v648_a));
                  v649_data.copy_to(glb_m0 + (v648_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

