// === base name ===
kernel_98b8c9eb8b

// === header ===
void launcher_kernel_98b8c9eb8b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_98b8c9eb8b(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_98b8c9eb8b(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_98b8c9eb8b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
            float r0[256]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v6_i0 = 0; v6_i0 < 1; ++v6_i0) {
              int32_t v8_lead = v6_i0 * 16;
              #pragma unroll
              for (int32_t v7_i1 = 0; v7_i1 < 16; ++v7_i1) {
                int32_t v11_a = v8_lead + (v7_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v12_data;
                v12_data.copy_from(glb_m1 + (v11_a));
                v12_data.copy_to(r0 + (v11_a));
              }
            }
            float* __restrict__ s0 = &localShrMem0[0];
            // s0 = load{g>s}(glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v17_ld;
            v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
            v17_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
            tensorforge::intel_esimd::simd<float, 64> v18_ld;
            v18_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
            v18_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
            tensorforge::intel_esimd::simd<float, 64> v19_ld;
            v19_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 128));
            v19_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 128));
            tensorforge::intel_esimd::simd<float, 64> v20_ld;
            v20_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 192));
            v20_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 192));
            // wait(r0 = load{g>r}(glb_m1););
            // wait(s0 = load{g>s}(glb_m2[0, 1]));
            float r1[256]{};
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir1[256]{};
            tensorforge::intel_esimd::simd<float, 16> v23_data;
            v23_data.copy_from(r0 + (0));
            tensorforge::intel_esimd::simd<float, 16> v24_data;
            v24_data.copy_from(r0 + (16));
            tensorforge::intel_esimd::simd<float, 16> v25_data;
            v25_data.copy_from(r0 + (32));
            tensorforge::intel_esimd::simd<float, 16> v26_data;
            v26_data.copy_from(r0 + (48));
            tensorforge::intel_esimd::simd<float, 16> v27_data;
            v27_data.copy_from(r0 + (64));
            tensorforge::intel_esimd::simd<float, 16> v28_data;
            v28_data.copy_from(r0 + (80));
            tensorforge::intel_esimd::simd<float, 16> v29_data;
            v29_data.copy_from(r0 + (96));
            tensorforge::intel_esimd::simd<float, 16> v30_data;
            v30_data.copy_from(r0 + (112));
            tensorforge::intel_esimd::simd<float, 16> v31_data;
            v31_data.copy_from(r0 + (128));
            tensorforge::intel_esimd::simd<float, 16> v32_data;
            v32_data.copy_from(r0 + (144));
            tensorforge::intel_esimd::simd<float, 16> v33_data;
            v33_data.copy_from(r0 + (160));
            tensorforge::intel_esimd::simd<float, 16> v34_data;
            v34_data.copy_from(r0 + (176));
            tensorforge::intel_esimd::simd<float, 16> v35_data;
            v35_data.copy_from(r0 + (192));
            tensorforge::intel_esimd::simd<float, 16> v36_data;
            v36_data.copy_from(r0 + (208));
            tensorforge::intel_esimd::simd<float, 16> v37_data;
            v37_data.copy_from(r0 + (224));
            tensorforge::intel_esimd::simd<float, 16> v38_data;
            v38_data.copy_from(r0 + (240));
            tensorforge::intel_esimd::simd<float, 16> v39_acc{};
            tensorforge::intel_esimd::simd<float, 16> v43_data;
            v43_data.copy_from(s0 + (0_i32));
            v39_acc += ((v43_data[0]) * v23_data);
            v39_acc += ((v43_data[1]) * v24_data);
            v39_acc += ((v43_data[2]) * v25_data);
            v39_acc += ((v43_data[3]) * v26_data);
            v39_acc += ((v43_data[4]) * v27_data);
            v39_acc += ((v43_data[5]) * v28_data);
            v39_acc += ((v43_data[6]) * v29_data);
            v39_acc += ((v43_data[7]) * v30_data);
            v39_acc += ((v43_data[8]) * v31_data);
            v39_acc += ((v43_data[9]) * v32_data);
            v39_acc += ((v43_data[10]) * v33_data);
            v39_acc += ((v43_data[11]) * v34_data);
            v39_acc += ((v43_data[12]) * v35_data);
            v39_acc += ((v43_data[13]) * v36_data);
            v39_acc += ((v43_data[14]) * v37_data);
            v39_acc += ((v43_data[15]) * v38_data);
            v39_acc.copy_to(ir1 + (0));
            tensorforge::intel_esimd::simd<float, 16> v76_acc{};
            tensorforge::intel_esimd::simd<float, 16> v80_data;
            v80_data.copy_from(s0 + (16_i32));
            v76_acc += ((v80_data[0]) * v23_data);
            v76_acc += ((v80_data[1]) * v24_data);
            v76_acc += ((v80_data[2]) * v25_data);
            v76_acc += ((v80_data[3]) * v26_data);
            v76_acc += ((v80_data[4]) * v27_data);
            v76_acc += ((v80_data[5]) * v28_data);
            v76_acc += ((v80_data[6]) * v29_data);
            v76_acc += ((v80_data[7]) * v30_data);
            v76_acc += ((v80_data[8]) * v31_data);
            v76_acc += ((v80_data[9]) * v32_data);
            v76_acc += ((v80_data[10]) * v33_data);
            v76_acc += ((v80_data[11]) * v34_data);
            v76_acc += ((v80_data[12]) * v35_data);
            v76_acc += ((v80_data[13]) * v36_data);
            v76_acc += ((v80_data[14]) * v37_data);
            v76_acc += ((v80_data[15]) * v38_data);
            v76_acc.copy_to(ir1 + (16));
            tensorforge::intel_esimd::simd<float, 16> v113_acc{};
            tensorforge::intel_esimd::simd<float, 16> v117_data;
            v117_data.copy_from(s0 + (32_i32));
            v113_acc += ((v117_data[0]) * v23_data);
            v113_acc += ((v117_data[1]) * v24_data);
            v113_acc += ((v117_data[2]) * v25_data);
            v113_acc += ((v117_data[3]) * v26_data);
            v113_acc += ((v117_data[4]) * v27_data);
            v113_acc += ((v117_data[5]) * v28_data);
            v113_acc += ((v117_data[6]) * v29_data);
            v113_acc += ((v117_data[7]) * v30_data);
            v113_acc += ((v117_data[8]) * v31_data);
            v113_acc += ((v117_data[9]) * v32_data);
            v113_acc += ((v117_data[10]) * v33_data);
            v113_acc += ((v117_data[11]) * v34_data);
            v113_acc += ((v117_data[12]) * v35_data);
            v113_acc += ((v117_data[13]) * v36_data);
            v113_acc += ((v117_data[14]) * v37_data);
            v113_acc += ((v117_data[15]) * v38_data);
            v113_acc.copy_to(ir1 + (32));
            tensorforge::intel_esimd::simd<float, 16> v150_acc{};
            tensorforge::intel_esimd::simd<float, 16> v154_data;
            v154_data.copy_from(s0 + (48_i32));
            v150_acc += ((v154_data[0]) * v23_data);
            v150_acc += ((v154_data[1]) * v24_data);
            v150_acc += ((v154_data[2]) * v25_data);
            v150_acc += ((v154_data[3]) * v26_data);
            v150_acc += ((v154_data[4]) * v27_data);
            v150_acc += ((v154_data[5]) * v28_data);
            v150_acc += ((v154_data[6]) * v29_data);
            v150_acc += ((v154_data[7]) * v30_data);
            v150_acc += ((v154_data[8]) * v31_data);
            v150_acc += ((v154_data[9]) * v32_data);
            v150_acc += ((v154_data[10]) * v33_data);
            v150_acc += ((v154_data[11]) * v34_data);
            v150_acc += ((v154_data[12]) * v35_data);
            v150_acc += ((v154_data[13]) * v36_data);
            v150_acc += ((v154_data[14]) * v37_data);
            v150_acc += ((v154_data[15]) * v38_data);
            v150_acc.copy_to(ir1 + (48));
            tensorforge::intel_esimd::simd<float, 16> v187_acc{};
            tensorforge::intel_esimd::simd<float, 16> v191_data;
            v191_data.copy_from(s0 + (64_i32));
            v187_acc += ((v191_data[0]) * v23_data);
            v187_acc += ((v191_data[1]) * v24_data);
            v187_acc += ((v191_data[2]) * v25_data);
            v187_acc += ((v191_data[3]) * v26_data);
            v187_acc += ((v191_data[4]) * v27_data);
            v187_acc += ((v191_data[5]) * v28_data);
            v187_acc += ((v191_data[6]) * v29_data);
            v187_acc += ((v191_data[7]) * v30_data);
            v187_acc += ((v191_data[8]) * v31_data);
            v187_acc += ((v191_data[9]) * v32_data);
            v187_acc += ((v191_data[10]) * v33_data);
            v187_acc += ((v191_data[11]) * v34_data);
            v187_acc += ((v191_data[12]) * v35_data);
            v187_acc += ((v191_data[13]) * v36_data);
            v187_acc += ((v191_data[14]) * v37_data);
            v187_acc += ((v191_data[15]) * v38_data);
            v187_acc.copy_to(ir1 + (64));
            tensorforge::intel_esimd::simd<float, 16> v224_acc{};
            tensorforge::intel_esimd::simd<float, 16> v228_data;
            v228_data.copy_from(s0 + (80_i32));
            v224_acc += ((v228_data[0]) * v23_data);
            v224_acc += ((v228_data[1]) * v24_data);
            v224_acc += ((v228_data[2]) * v25_data);
            v224_acc += ((v228_data[3]) * v26_data);
            v224_acc += ((v228_data[4]) * v27_data);
            v224_acc += ((v228_data[5]) * v28_data);
            v224_acc += ((v228_data[6]) * v29_data);
            v224_acc += ((v228_data[7]) * v30_data);
            v224_acc += ((v228_data[8]) * v31_data);
            v224_acc += ((v228_data[9]) * v32_data);
            v224_acc += ((v228_data[10]) * v33_data);
            v224_acc += ((v228_data[11]) * v34_data);
            v224_acc += ((v228_data[12]) * v35_data);
            v224_acc += ((v228_data[13]) * v36_data);
            v224_acc += ((v228_data[14]) * v37_data);
            v224_acc += ((v228_data[15]) * v38_data);
            v224_acc.copy_to(ir1 + (80));
            tensorforge::intel_esimd::simd<float, 16> v261_acc{};
            tensorforge::intel_esimd::simd<float, 16> v265_data;
            v265_data.copy_from(s0 + (96_i32));
            v261_acc += ((v265_data[0]) * v23_data);
            v261_acc += ((v265_data[1]) * v24_data);
            v261_acc += ((v265_data[2]) * v25_data);
            v261_acc += ((v265_data[3]) * v26_data);
            v261_acc += ((v265_data[4]) * v27_data);
            v261_acc += ((v265_data[5]) * v28_data);
            v261_acc += ((v265_data[6]) * v29_data);
            v261_acc += ((v265_data[7]) * v30_data);
            v261_acc += ((v265_data[8]) * v31_data);
            v261_acc += ((v265_data[9]) * v32_data);
            v261_acc += ((v265_data[10]) * v33_data);
            v261_acc += ((v265_data[11]) * v34_data);
            v261_acc += ((v265_data[12]) * v35_data);
            v261_acc += ((v265_data[13]) * v36_data);
            v261_acc += ((v265_data[14]) * v37_data);
            v261_acc += ((v265_data[15]) * v38_data);
            v261_acc.copy_to(ir1 + (96));
            tensorforge::intel_esimd::simd<float, 16> v298_acc{};
            tensorforge::intel_esimd::simd<float, 16> v302_data;
            v302_data.copy_from(s0 + (112_i32));
            v298_acc += ((v302_data[0]) * v23_data);
            v298_acc += ((v302_data[1]) * v24_data);
            v298_acc += ((v302_data[2]) * v25_data);
            v298_acc += ((v302_data[3]) * v26_data);
            v298_acc += ((v302_data[4]) * v27_data);
            v298_acc += ((v302_data[5]) * v28_data);
            v298_acc += ((v302_data[6]) * v29_data);
            v298_acc += ((v302_data[7]) * v30_data);
            v298_acc += ((v302_data[8]) * v31_data);
            v298_acc += ((v302_data[9]) * v32_data);
            v298_acc += ((v302_data[10]) * v33_data);
            v298_acc += ((v302_data[11]) * v34_data);
            v298_acc += ((v302_data[12]) * v35_data);
            v298_acc += ((v302_data[13]) * v36_data);
            v298_acc += ((v302_data[14]) * v37_data);
            v298_acc += ((v302_data[15]) * v38_data);
            v298_acc.copy_to(ir1 + (112));
            tensorforge::intel_esimd::simd<float, 16> v335_acc{};
            tensorforge::intel_esimd::simd<float, 16> v339_data;
            v339_data.copy_from(s0 + (128_i32));
            v335_acc += ((v339_data[0]) * v23_data);
            v335_acc += ((v339_data[1]) * v24_data);
            v335_acc += ((v339_data[2]) * v25_data);
            v335_acc += ((v339_data[3]) * v26_data);
            v335_acc += ((v339_data[4]) * v27_data);
            v335_acc += ((v339_data[5]) * v28_data);
            v335_acc += ((v339_data[6]) * v29_data);
            v335_acc += ((v339_data[7]) * v30_data);
            v335_acc += ((v339_data[8]) * v31_data);
            v335_acc += ((v339_data[9]) * v32_data);
            v335_acc += ((v339_data[10]) * v33_data);
            v335_acc += ((v339_data[11]) * v34_data);
            v335_acc += ((v339_data[12]) * v35_data);
            v335_acc += ((v339_data[13]) * v36_data);
            v335_acc += ((v339_data[14]) * v37_data);
            v335_acc += ((v339_data[15]) * v38_data);
            v335_acc.copy_to(ir1 + (128));
            tensorforge::intel_esimd::simd<float, 16> v372_acc{};
            tensorforge::intel_esimd::simd<float, 16> v376_data;
            v376_data.copy_from(s0 + (144_i32));
            v372_acc += ((v376_data[0]) * v23_data);
            v372_acc += ((v376_data[1]) * v24_data);
            v372_acc += ((v376_data[2]) * v25_data);
            v372_acc += ((v376_data[3]) * v26_data);
            v372_acc += ((v376_data[4]) * v27_data);
            v372_acc += ((v376_data[5]) * v28_data);
            v372_acc += ((v376_data[6]) * v29_data);
            v372_acc += ((v376_data[7]) * v30_data);
            v372_acc += ((v376_data[8]) * v31_data);
            v372_acc += ((v376_data[9]) * v32_data);
            v372_acc += ((v376_data[10]) * v33_data);
            v372_acc += ((v376_data[11]) * v34_data);
            v372_acc += ((v376_data[12]) * v35_data);
            v372_acc += ((v376_data[13]) * v36_data);
            v372_acc += ((v376_data[14]) * v37_data);
            v372_acc += ((v376_data[15]) * v38_data);
            v372_acc.copy_to(ir1 + (144));
            tensorforge::intel_esimd::simd<float, 16> v409_acc{};
            tensorforge::intel_esimd::simd<float, 16> v413_data;
            v413_data.copy_from(s0 + (160_i32));
            v409_acc += ((v413_data[0]) * v23_data);
            v409_acc += ((v413_data[1]) * v24_data);
            v409_acc += ((v413_data[2]) * v25_data);
            v409_acc += ((v413_data[3]) * v26_data);
            v409_acc += ((v413_data[4]) * v27_data);
            v409_acc += ((v413_data[5]) * v28_data);
            v409_acc += ((v413_data[6]) * v29_data);
            v409_acc += ((v413_data[7]) * v30_data);
            v409_acc += ((v413_data[8]) * v31_data);
            v409_acc += ((v413_data[9]) * v32_data);
            v409_acc += ((v413_data[10]) * v33_data);
            v409_acc += ((v413_data[11]) * v34_data);
            v409_acc += ((v413_data[12]) * v35_data);
            v409_acc += ((v413_data[13]) * v36_data);
            v409_acc += ((v413_data[14]) * v37_data);
            v409_acc += ((v413_data[15]) * v38_data);
            v409_acc.copy_to(ir1 + (160));
            tensorforge::intel_esimd::simd<float, 16> v446_acc{};
            tensorforge::intel_esimd::simd<float, 16> v450_data;
            v450_data.copy_from(s0 + (176_i32));
            v446_acc += ((v450_data[0]) * v23_data);
            v446_acc += ((v450_data[1]) * v24_data);
            v446_acc += ((v450_data[2]) * v25_data);
            v446_acc += ((v450_data[3]) * v26_data);
            v446_acc += ((v450_data[4]) * v27_data);
            v446_acc += ((v450_data[5]) * v28_data);
            v446_acc += ((v450_data[6]) * v29_data);
            v446_acc += ((v450_data[7]) * v30_data);
            v446_acc += ((v450_data[8]) * v31_data);
            v446_acc += ((v450_data[9]) * v32_data);
            v446_acc += ((v450_data[10]) * v33_data);
            v446_acc += ((v450_data[11]) * v34_data);
            v446_acc += ((v450_data[12]) * v35_data);
            v446_acc += ((v450_data[13]) * v36_data);
            v446_acc += ((v450_data[14]) * v37_data);
            v446_acc += ((v450_data[15]) * v38_data);
            v446_acc.copy_to(ir1 + (176));
            tensorforge::intel_esimd::simd<float, 16> v483_acc{};
            tensorforge::intel_esimd::simd<float, 16> v487_data;
            v487_data.copy_from(s0 + (192_i32));
            v483_acc += ((v487_data[0]) * v23_data);
            v483_acc += ((v487_data[1]) * v24_data);
            v483_acc += ((v487_data[2]) * v25_data);
            v483_acc += ((v487_data[3]) * v26_data);
            v483_acc += ((v487_data[4]) * v27_data);
            v483_acc += ((v487_data[5]) * v28_data);
            v483_acc += ((v487_data[6]) * v29_data);
            v483_acc += ((v487_data[7]) * v30_data);
            v483_acc += ((v487_data[8]) * v31_data);
            v483_acc += ((v487_data[9]) * v32_data);
            v483_acc += ((v487_data[10]) * v33_data);
            v483_acc += ((v487_data[11]) * v34_data);
            v483_acc += ((v487_data[12]) * v35_data);
            v483_acc += ((v487_data[13]) * v36_data);
            v483_acc += ((v487_data[14]) * v37_data);
            v483_acc += ((v487_data[15]) * v38_data);
            v483_acc.copy_to(ir1 + (192));
            tensorforge::intel_esimd::simd<float, 16> v520_acc{};
            tensorforge::intel_esimd::simd<float, 16> v524_data;
            v524_data.copy_from(s0 + (208_i32));
            v520_acc += ((v524_data[0]) * v23_data);
            v520_acc += ((v524_data[1]) * v24_data);
            v520_acc += ((v524_data[2]) * v25_data);
            v520_acc += ((v524_data[3]) * v26_data);
            v520_acc += ((v524_data[4]) * v27_data);
            v520_acc += ((v524_data[5]) * v28_data);
            v520_acc += ((v524_data[6]) * v29_data);
            v520_acc += ((v524_data[7]) * v30_data);
            v520_acc += ((v524_data[8]) * v31_data);
            v520_acc += ((v524_data[9]) * v32_data);
            v520_acc += ((v524_data[10]) * v33_data);
            v520_acc += ((v524_data[11]) * v34_data);
            v520_acc += ((v524_data[12]) * v35_data);
            v520_acc += ((v524_data[13]) * v36_data);
            v520_acc += ((v524_data[14]) * v37_data);
            v520_acc += ((v524_data[15]) * v38_data);
            v520_acc.copy_to(ir1 + (208));
            tensorforge::intel_esimd::simd<float, 16> v557_acc{};
            tensorforge::intel_esimd::simd<float, 16> v561_data;
            v561_data.copy_from(s0 + (224_i32));
            v557_acc += ((v561_data[0]) * v23_data);
            v557_acc += ((v561_data[1]) * v24_data);
            v557_acc += ((v561_data[2]) * v25_data);
            v557_acc += ((v561_data[3]) * v26_data);
            v557_acc += ((v561_data[4]) * v27_data);
            v557_acc += ((v561_data[5]) * v28_data);
            v557_acc += ((v561_data[6]) * v29_data);
            v557_acc += ((v561_data[7]) * v30_data);
            v557_acc += ((v561_data[8]) * v31_data);
            v557_acc += ((v561_data[9]) * v32_data);
            v557_acc += ((v561_data[10]) * v33_data);
            v557_acc += ((v561_data[11]) * v34_data);
            v557_acc += ((v561_data[12]) * v35_data);
            v557_acc += ((v561_data[13]) * v36_data);
            v557_acc += ((v561_data[14]) * v37_data);
            v557_acc += ((v561_data[15]) * v38_data);
            v557_acc.copy_to(ir1 + (224));
            tensorforge::intel_esimd::simd<float, 16> v594_acc{};
            tensorforge::intel_esimd::simd<float, 16> v598_data;
            v598_data.copy_from(s0 + (240_i32));
            v594_acc += ((v598_data[0]) * v23_data);
            v594_acc += ((v598_data[1]) * v24_data);
            v594_acc += ((v598_data[2]) * v25_data);
            v594_acc += ((v598_data[3]) * v26_data);
            v594_acc += ((v598_data[4]) * v27_data);
            v594_acc += ((v598_data[5]) * v28_data);
            v594_acc += ((v598_data[6]) * v29_data);
            v594_acc += ((v598_data[7]) * v30_data);
            v594_acc += ((v598_data[8]) * v31_data);
            v594_acc += ((v598_data[9]) * v32_data);
            v594_acc += ((v598_data[10]) * v33_data);
            v594_acc += ((v598_data[11]) * v34_data);
            v594_acc += ((v598_data[12]) * v35_data);
            v594_acc += ((v598_data[13]) * v36_data);
            v594_acc += ((v598_data[14]) * v37_data);
            v594_acc += ((v598_data[15]) * v38_data);
            v594_acc.copy_to(ir1 + (240));
            #pragma unroll
            for (int32_t v631_n0 = 0; v631_n0 < 1; ++v631_n0) {
              int32_t v633_a = v631_n0 * 16;
              #pragma unroll
              for (int32_t v632_n1 = 0; v632_n1 < 16; ++v632_n1) {
                int32_t v635_a = v633_a + (v632_n1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v636_data;
                v636_data.copy_from(ir1 + (v635_a));
                v636_data.copy_to(r1 + (v635_a));
              }
            }
            // glb_m0 = store{r>g}(r1);
            #pragma unroll
            for (int32_t v640_i0 = 0; v640_i0 < 1; ++v640_i0) {
              int32_t v642_a = v640_i0 * 16;
              #pragma unroll
              for (int32_t v641_i1 = 0; v641_i1 < 16; ++v641_i1) {
                int32_t v644_a = v642_a + (v641_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v645_data;
                v645_data.copy_from(r1 + (v644_a));
                v645_data.copy_to(glb_m0 + (v644_a));
              }
            }
          }
        }
      });
    }
  });
}

