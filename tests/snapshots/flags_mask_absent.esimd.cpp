// === base name ===
kernel_f3033111b8ce7810

// === header ===
void launcher_kernel_f3033111b8ce7810(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f3033111b8ce7810(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f3033111b8ce7810(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f3033111b8ce7810(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0) {
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
            float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
            float r0[256]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v10_i0 = 0; v10_i0 < 1; ++v10_i0) {
              int32_t v12_lead = v10_i0 * 16;
              #pragma unroll
              for (int32_t v11_i1 = 0; v11_i1 < 16; ++v11_i1) {
                int32_t v15_a = v12_lead + (v11_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v16_data;
                v16_data.copy_from(glb_m1 + (v15_a));
                v16_data.copy_to(r0 + (v15_a));
              }
            }
            // s0 = load{g>s}(glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v20_ld;
            v20_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
            v20_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
            tensorforge::intel_esimd::simd<float, 64> v21_ld;
            v21_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
            v21_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
            tensorforge::intel_esimd::simd<float, 64> v22_ld;
            v22_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 128));
            v22_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 128));
            tensorforge::intel_esimd::simd<float, 64> v23_ld;
            v23_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 192));
            v23_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 192));
            // wait(r0 = load{g>r}(glb_m1););
            // wait(s0 = load{g>s}(glb_m2[0, 1]));
            float r1[256]{};
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir1[256]{};
            tensorforge::intel_esimd::simd<float, 16> v26_data;
            v26_data.copy_from(r0 + (0));
            tensorforge::intel_esimd::simd<float, 16> v27_data;
            v27_data.copy_from(r0 + (16));
            tensorforge::intel_esimd::simd<float, 16> v28_data;
            v28_data.copy_from(r0 + (32));
            tensorforge::intel_esimd::simd<float, 16> v29_data;
            v29_data.copy_from(r0 + (48));
            tensorforge::intel_esimd::simd<float, 16> v30_data;
            v30_data.copy_from(r0 + (64));
            tensorforge::intel_esimd::simd<float, 16> v31_data;
            v31_data.copy_from(r0 + (80));
            tensorforge::intel_esimd::simd<float, 16> v32_data;
            v32_data.copy_from(r0 + (96));
            tensorforge::intel_esimd::simd<float, 16> v33_data;
            v33_data.copy_from(r0 + (112));
            tensorforge::intel_esimd::simd<float, 16> v34_data;
            v34_data.copy_from(r0 + (128));
            tensorforge::intel_esimd::simd<float, 16> v35_data;
            v35_data.copy_from(r0 + (144));
            tensorforge::intel_esimd::simd<float, 16> v36_data;
            v36_data.copy_from(r0 + (160));
            tensorforge::intel_esimd::simd<float, 16> v37_data;
            v37_data.copy_from(r0 + (176));
            tensorforge::intel_esimd::simd<float, 16> v38_data;
            v38_data.copy_from(r0 + (192));
            tensorforge::intel_esimd::simd<float, 16> v39_data;
            v39_data.copy_from(r0 + (208));
            tensorforge::intel_esimd::simd<float, 16> v40_data;
            v40_data.copy_from(r0 + (224));
            tensorforge::intel_esimd::simd<float, 16> v41_data;
            v41_data.copy_from(r0 + (240));
            tensorforge::intel_esimd::simd<float, 16> v42_acc{};
            tensorforge::intel_esimd::simd<float, 16> v46_data;
            v46_data.copy_from(s0 + (0_i32));
            v42_acc += ((v46_data[0]) * v26_data);
            v42_acc += ((v46_data[1]) * v27_data);
            v42_acc += ((v46_data[2]) * v28_data);
            v42_acc += ((v46_data[3]) * v29_data);
            v42_acc += ((v46_data[4]) * v30_data);
            v42_acc += ((v46_data[5]) * v31_data);
            v42_acc += ((v46_data[6]) * v32_data);
            v42_acc += ((v46_data[7]) * v33_data);
            v42_acc += ((v46_data[8]) * v34_data);
            v42_acc += ((v46_data[9]) * v35_data);
            v42_acc += ((v46_data[10]) * v36_data);
            v42_acc += ((v46_data[11]) * v37_data);
            v42_acc += ((v46_data[12]) * v38_data);
            v42_acc += ((v46_data[13]) * v39_data);
            v42_acc += ((v46_data[14]) * v40_data);
            v42_acc += ((v46_data[15]) * v41_data);
            v42_acc.copy_to(ir1 + (0));
            tensorforge::intel_esimd::simd<float, 16> v79_acc{};
            tensorforge::intel_esimd::simd<float, 16> v83_data;
            v83_data.copy_from(s0 + (16_i32));
            v79_acc += ((v83_data[0]) * v26_data);
            v79_acc += ((v83_data[1]) * v27_data);
            v79_acc += ((v83_data[2]) * v28_data);
            v79_acc += ((v83_data[3]) * v29_data);
            v79_acc += ((v83_data[4]) * v30_data);
            v79_acc += ((v83_data[5]) * v31_data);
            v79_acc += ((v83_data[6]) * v32_data);
            v79_acc += ((v83_data[7]) * v33_data);
            v79_acc += ((v83_data[8]) * v34_data);
            v79_acc += ((v83_data[9]) * v35_data);
            v79_acc += ((v83_data[10]) * v36_data);
            v79_acc += ((v83_data[11]) * v37_data);
            v79_acc += ((v83_data[12]) * v38_data);
            v79_acc += ((v83_data[13]) * v39_data);
            v79_acc += ((v83_data[14]) * v40_data);
            v79_acc += ((v83_data[15]) * v41_data);
            v79_acc.copy_to(ir1 + (16));
            tensorforge::intel_esimd::simd<float, 16> v116_acc{};
            tensorforge::intel_esimd::simd<float, 16> v120_data;
            v120_data.copy_from(s0 + (32_i32));
            v116_acc += ((v120_data[0]) * v26_data);
            v116_acc += ((v120_data[1]) * v27_data);
            v116_acc += ((v120_data[2]) * v28_data);
            v116_acc += ((v120_data[3]) * v29_data);
            v116_acc += ((v120_data[4]) * v30_data);
            v116_acc += ((v120_data[5]) * v31_data);
            v116_acc += ((v120_data[6]) * v32_data);
            v116_acc += ((v120_data[7]) * v33_data);
            v116_acc += ((v120_data[8]) * v34_data);
            v116_acc += ((v120_data[9]) * v35_data);
            v116_acc += ((v120_data[10]) * v36_data);
            v116_acc += ((v120_data[11]) * v37_data);
            v116_acc += ((v120_data[12]) * v38_data);
            v116_acc += ((v120_data[13]) * v39_data);
            v116_acc += ((v120_data[14]) * v40_data);
            v116_acc += ((v120_data[15]) * v41_data);
            v116_acc.copy_to(ir1 + (32));
            tensorforge::intel_esimd::simd<float, 16> v153_acc{};
            tensorforge::intel_esimd::simd<float, 16> v157_data;
            v157_data.copy_from(s0 + (48_i32));
            v153_acc += ((v157_data[0]) * v26_data);
            v153_acc += ((v157_data[1]) * v27_data);
            v153_acc += ((v157_data[2]) * v28_data);
            v153_acc += ((v157_data[3]) * v29_data);
            v153_acc += ((v157_data[4]) * v30_data);
            v153_acc += ((v157_data[5]) * v31_data);
            v153_acc += ((v157_data[6]) * v32_data);
            v153_acc += ((v157_data[7]) * v33_data);
            v153_acc += ((v157_data[8]) * v34_data);
            v153_acc += ((v157_data[9]) * v35_data);
            v153_acc += ((v157_data[10]) * v36_data);
            v153_acc += ((v157_data[11]) * v37_data);
            v153_acc += ((v157_data[12]) * v38_data);
            v153_acc += ((v157_data[13]) * v39_data);
            v153_acc += ((v157_data[14]) * v40_data);
            v153_acc += ((v157_data[15]) * v41_data);
            v153_acc.copy_to(ir1 + (48));
            tensorforge::intel_esimd::simd<float, 16> v190_acc{};
            tensorforge::intel_esimd::simd<float, 16> v194_data;
            v194_data.copy_from(s0 + (64_i32));
            v190_acc += ((v194_data[0]) * v26_data);
            v190_acc += ((v194_data[1]) * v27_data);
            v190_acc += ((v194_data[2]) * v28_data);
            v190_acc += ((v194_data[3]) * v29_data);
            v190_acc += ((v194_data[4]) * v30_data);
            v190_acc += ((v194_data[5]) * v31_data);
            v190_acc += ((v194_data[6]) * v32_data);
            v190_acc += ((v194_data[7]) * v33_data);
            v190_acc += ((v194_data[8]) * v34_data);
            v190_acc += ((v194_data[9]) * v35_data);
            v190_acc += ((v194_data[10]) * v36_data);
            v190_acc += ((v194_data[11]) * v37_data);
            v190_acc += ((v194_data[12]) * v38_data);
            v190_acc += ((v194_data[13]) * v39_data);
            v190_acc += ((v194_data[14]) * v40_data);
            v190_acc += ((v194_data[15]) * v41_data);
            v190_acc.copy_to(ir1 + (64));
            tensorforge::intel_esimd::simd<float, 16> v227_acc{};
            tensorforge::intel_esimd::simd<float, 16> v231_data;
            v231_data.copy_from(s0 + (80_i32));
            v227_acc += ((v231_data[0]) * v26_data);
            v227_acc += ((v231_data[1]) * v27_data);
            v227_acc += ((v231_data[2]) * v28_data);
            v227_acc += ((v231_data[3]) * v29_data);
            v227_acc += ((v231_data[4]) * v30_data);
            v227_acc += ((v231_data[5]) * v31_data);
            v227_acc += ((v231_data[6]) * v32_data);
            v227_acc += ((v231_data[7]) * v33_data);
            v227_acc += ((v231_data[8]) * v34_data);
            v227_acc += ((v231_data[9]) * v35_data);
            v227_acc += ((v231_data[10]) * v36_data);
            v227_acc += ((v231_data[11]) * v37_data);
            v227_acc += ((v231_data[12]) * v38_data);
            v227_acc += ((v231_data[13]) * v39_data);
            v227_acc += ((v231_data[14]) * v40_data);
            v227_acc += ((v231_data[15]) * v41_data);
            v227_acc.copy_to(ir1 + (80));
            tensorforge::intel_esimd::simd<float, 16> v264_acc{};
            tensorforge::intel_esimd::simd<float, 16> v268_data;
            v268_data.copy_from(s0 + (96_i32));
            v264_acc += ((v268_data[0]) * v26_data);
            v264_acc += ((v268_data[1]) * v27_data);
            v264_acc += ((v268_data[2]) * v28_data);
            v264_acc += ((v268_data[3]) * v29_data);
            v264_acc += ((v268_data[4]) * v30_data);
            v264_acc += ((v268_data[5]) * v31_data);
            v264_acc += ((v268_data[6]) * v32_data);
            v264_acc += ((v268_data[7]) * v33_data);
            v264_acc += ((v268_data[8]) * v34_data);
            v264_acc += ((v268_data[9]) * v35_data);
            v264_acc += ((v268_data[10]) * v36_data);
            v264_acc += ((v268_data[11]) * v37_data);
            v264_acc += ((v268_data[12]) * v38_data);
            v264_acc += ((v268_data[13]) * v39_data);
            v264_acc += ((v268_data[14]) * v40_data);
            v264_acc += ((v268_data[15]) * v41_data);
            v264_acc.copy_to(ir1 + (96));
            tensorforge::intel_esimd::simd<float, 16> v301_acc{};
            tensorforge::intel_esimd::simd<float, 16> v305_data;
            v305_data.copy_from(s0 + (112_i32));
            v301_acc += ((v305_data[0]) * v26_data);
            v301_acc += ((v305_data[1]) * v27_data);
            v301_acc += ((v305_data[2]) * v28_data);
            v301_acc += ((v305_data[3]) * v29_data);
            v301_acc += ((v305_data[4]) * v30_data);
            v301_acc += ((v305_data[5]) * v31_data);
            v301_acc += ((v305_data[6]) * v32_data);
            v301_acc += ((v305_data[7]) * v33_data);
            v301_acc += ((v305_data[8]) * v34_data);
            v301_acc += ((v305_data[9]) * v35_data);
            v301_acc += ((v305_data[10]) * v36_data);
            v301_acc += ((v305_data[11]) * v37_data);
            v301_acc += ((v305_data[12]) * v38_data);
            v301_acc += ((v305_data[13]) * v39_data);
            v301_acc += ((v305_data[14]) * v40_data);
            v301_acc += ((v305_data[15]) * v41_data);
            v301_acc.copy_to(ir1 + (112));
            tensorforge::intel_esimd::simd<float, 16> v338_acc{};
            tensorforge::intel_esimd::simd<float, 16> v342_data;
            v342_data.copy_from(s0 + (128_i32));
            v338_acc += ((v342_data[0]) * v26_data);
            v338_acc += ((v342_data[1]) * v27_data);
            v338_acc += ((v342_data[2]) * v28_data);
            v338_acc += ((v342_data[3]) * v29_data);
            v338_acc += ((v342_data[4]) * v30_data);
            v338_acc += ((v342_data[5]) * v31_data);
            v338_acc += ((v342_data[6]) * v32_data);
            v338_acc += ((v342_data[7]) * v33_data);
            v338_acc += ((v342_data[8]) * v34_data);
            v338_acc += ((v342_data[9]) * v35_data);
            v338_acc += ((v342_data[10]) * v36_data);
            v338_acc += ((v342_data[11]) * v37_data);
            v338_acc += ((v342_data[12]) * v38_data);
            v338_acc += ((v342_data[13]) * v39_data);
            v338_acc += ((v342_data[14]) * v40_data);
            v338_acc += ((v342_data[15]) * v41_data);
            v338_acc.copy_to(ir1 + (128));
            tensorforge::intel_esimd::simd<float, 16> v375_acc{};
            tensorforge::intel_esimd::simd<float, 16> v379_data;
            v379_data.copy_from(s0 + (144_i32));
            v375_acc += ((v379_data[0]) * v26_data);
            v375_acc += ((v379_data[1]) * v27_data);
            v375_acc += ((v379_data[2]) * v28_data);
            v375_acc += ((v379_data[3]) * v29_data);
            v375_acc += ((v379_data[4]) * v30_data);
            v375_acc += ((v379_data[5]) * v31_data);
            v375_acc += ((v379_data[6]) * v32_data);
            v375_acc += ((v379_data[7]) * v33_data);
            v375_acc += ((v379_data[8]) * v34_data);
            v375_acc += ((v379_data[9]) * v35_data);
            v375_acc += ((v379_data[10]) * v36_data);
            v375_acc += ((v379_data[11]) * v37_data);
            v375_acc += ((v379_data[12]) * v38_data);
            v375_acc += ((v379_data[13]) * v39_data);
            v375_acc += ((v379_data[14]) * v40_data);
            v375_acc += ((v379_data[15]) * v41_data);
            v375_acc.copy_to(ir1 + (144));
            tensorforge::intel_esimd::simd<float, 16> v412_acc{};
            tensorforge::intel_esimd::simd<float, 16> v416_data;
            v416_data.copy_from(s0 + (160_i32));
            v412_acc += ((v416_data[0]) * v26_data);
            v412_acc += ((v416_data[1]) * v27_data);
            v412_acc += ((v416_data[2]) * v28_data);
            v412_acc += ((v416_data[3]) * v29_data);
            v412_acc += ((v416_data[4]) * v30_data);
            v412_acc += ((v416_data[5]) * v31_data);
            v412_acc += ((v416_data[6]) * v32_data);
            v412_acc += ((v416_data[7]) * v33_data);
            v412_acc += ((v416_data[8]) * v34_data);
            v412_acc += ((v416_data[9]) * v35_data);
            v412_acc += ((v416_data[10]) * v36_data);
            v412_acc += ((v416_data[11]) * v37_data);
            v412_acc += ((v416_data[12]) * v38_data);
            v412_acc += ((v416_data[13]) * v39_data);
            v412_acc += ((v416_data[14]) * v40_data);
            v412_acc += ((v416_data[15]) * v41_data);
            v412_acc.copy_to(ir1 + (160));
            tensorforge::intel_esimd::simd<float, 16> v449_acc{};
            tensorforge::intel_esimd::simd<float, 16> v453_data;
            v453_data.copy_from(s0 + (176_i32));
            v449_acc += ((v453_data[0]) * v26_data);
            v449_acc += ((v453_data[1]) * v27_data);
            v449_acc += ((v453_data[2]) * v28_data);
            v449_acc += ((v453_data[3]) * v29_data);
            v449_acc += ((v453_data[4]) * v30_data);
            v449_acc += ((v453_data[5]) * v31_data);
            v449_acc += ((v453_data[6]) * v32_data);
            v449_acc += ((v453_data[7]) * v33_data);
            v449_acc += ((v453_data[8]) * v34_data);
            v449_acc += ((v453_data[9]) * v35_data);
            v449_acc += ((v453_data[10]) * v36_data);
            v449_acc += ((v453_data[11]) * v37_data);
            v449_acc += ((v453_data[12]) * v38_data);
            v449_acc += ((v453_data[13]) * v39_data);
            v449_acc += ((v453_data[14]) * v40_data);
            v449_acc += ((v453_data[15]) * v41_data);
            v449_acc.copy_to(ir1 + (176));
            tensorforge::intel_esimd::simd<float, 16> v486_acc{};
            tensorforge::intel_esimd::simd<float, 16> v490_data;
            v490_data.copy_from(s0 + (192_i32));
            v486_acc += ((v490_data[0]) * v26_data);
            v486_acc += ((v490_data[1]) * v27_data);
            v486_acc += ((v490_data[2]) * v28_data);
            v486_acc += ((v490_data[3]) * v29_data);
            v486_acc += ((v490_data[4]) * v30_data);
            v486_acc += ((v490_data[5]) * v31_data);
            v486_acc += ((v490_data[6]) * v32_data);
            v486_acc += ((v490_data[7]) * v33_data);
            v486_acc += ((v490_data[8]) * v34_data);
            v486_acc += ((v490_data[9]) * v35_data);
            v486_acc += ((v490_data[10]) * v36_data);
            v486_acc += ((v490_data[11]) * v37_data);
            v486_acc += ((v490_data[12]) * v38_data);
            v486_acc += ((v490_data[13]) * v39_data);
            v486_acc += ((v490_data[14]) * v40_data);
            v486_acc += ((v490_data[15]) * v41_data);
            v486_acc.copy_to(ir1 + (192));
            tensorforge::intel_esimd::simd<float, 16> v523_acc{};
            tensorforge::intel_esimd::simd<float, 16> v527_data;
            v527_data.copy_from(s0 + (208_i32));
            v523_acc += ((v527_data[0]) * v26_data);
            v523_acc += ((v527_data[1]) * v27_data);
            v523_acc += ((v527_data[2]) * v28_data);
            v523_acc += ((v527_data[3]) * v29_data);
            v523_acc += ((v527_data[4]) * v30_data);
            v523_acc += ((v527_data[5]) * v31_data);
            v523_acc += ((v527_data[6]) * v32_data);
            v523_acc += ((v527_data[7]) * v33_data);
            v523_acc += ((v527_data[8]) * v34_data);
            v523_acc += ((v527_data[9]) * v35_data);
            v523_acc += ((v527_data[10]) * v36_data);
            v523_acc += ((v527_data[11]) * v37_data);
            v523_acc += ((v527_data[12]) * v38_data);
            v523_acc += ((v527_data[13]) * v39_data);
            v523_acc += ((v527_data[14]) * v40_data);
            v523_acc += ((v527_data[15]) * v41_data);
            v523_acc.copy_to(ir1 + (208));
            tensorforge::intel_esimd::simd<float, 16> v560_acc{};
            tensorforge::intel_esimd::simd<float, 16> v564_data;
            v564_data.copy_from(s0 + (224_i32));
            v560_acc += ((v564_data[0]) * v26_data);
            v560_acc += ((v564_data[1]) * v27_data);
            v560_acc += ((v564_data[2]) * v28_data);
            v560_acc += ((v564_data[3]) * v29_data);
            v560_acc += ((v564_data[4]) * v30_data);
            v560_acc += ((v564_data[5]) * v31_data);
            v560_acc += ((v564_data[6]) * v32_data);
            v560_acc += ((v564_data[7]) * v33_data);
            v560_acc += ((v564_data[8]) * v34_data);
            v560_acc += ((v564_data[9]) * v35_data);
            v560_acc += ((v564_data[10]) * v36_data);
            v560_acc += ((v564_data[11]) * v37_data);
            v560_acc += ((v564_data[12]) * v38_data);
            v560_acc += ((v564_data[13]) * v39_data);
            v560_acc += ((v564_data[14]) * v40_data);
            v560_acc += ((v564_data[15]) * v41_data);
            v560_acc.copy_to(ir1 + (224));
            tensorforge::intel_esimd::simd<float, 16> v597_acc{};
            tensorforge::intel_esimd::simd<float, 16> v601_data;
            v601_data.copy_from(s0 + (240_i32));
            v597_acc += ((v601_data[0]) * v26_data);
            v597_acc += ((v601_data[1]) * v27_data);
            v597_acc += ((v601_data[2]) * v28_data);
            v597_acc += ((v601_data[3]) * v29_data);
            v597_acc += ((v601_data[4]) * v30_data);
            v597_acc += ((v601_data[5]) * v31_data);
            v597_acc += ((v601_data[6]) * v32_data);
            v597_acc += ((v601_data[7]) * v33_data);
            v597_acc += ((v601_data[8]) * v34_data);
            v597_acc += ((v601_data[9]) * v35_data);
            v597_acc += ((v601_data[10]) * v36_data);
            v597_acc += ((v601_data[11]) * v37_data);
            v597_acc += ((v601_data[12]) * v38_data);
            v597_acc += ((v601_data[13]) * v39_data);
            v597_acc += ((v601_data[14]) * v40_data);
            v597_acc += ((v601_data[15]) * v41_data);
            v597_acc.copy_to(ir1 + (240));
            #pragma unroll
            for (int32_t v634_n0 = 0; v634_n0 < 1; ++v634_n0) {
              int32_t v636_a = v634_n0 * 16;
              #pragma unroll
              for (int32_t v635_n1 = 0; v635_n1 < 16; ++v635_n1) {
                int32_t v638_a = v636_a + (v635_n1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v639_data;
                v639_data.copy_from(ir1 + (v638_a));
                v639_data.copy_to(r1 + (v638_a));
              }
            }
            // glb_m0 = store{r>g}(r1);
            #pragma unroll
            for (int32_t v643_i0 = 0; v643_i0 < 1; ++v643_i0) {
              int32_t v645_a = v643_i0 * 16;
              #pragma unroll
              for (int32_t v644_i1 = 0; v644_i1 < 16; ++v644_i1) {
                int32_t v647_a = v645_a + (v644_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v648_data;
                v648_data.copy_from(r1 + (v647_a));
                v648_data.copy_to(glb_m0 + (v647_a));
              }
            }
          }
        }
      });
    }
  });
}

