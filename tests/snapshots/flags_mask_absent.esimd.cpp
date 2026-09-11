// === base name ===
kernel_e05b830b4c5f0c45

// === header ===
void launcher_kernel_e05b830b4c5f0c45(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e05b830b4c5f0c45(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
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
  kernel_kernel_e05b830b4c5f0c45(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e05b830b4c5f0c45(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (4352, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[272 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[256];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 256 + 0 + m2_extraOffset];
            float r0[256]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v14_i0 = 0; v14_i0 < 1; ++v14_i0) {
              int32_t v16_lead = v14_i0 * 16;
              #pragma unroll
              for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
                int32_t v19_a = v16_lead + (v15_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v20_data;
                v20_data.copy_from(glb_m1 + (v19_a));
                v20_data.copy_to(r0 + (v19_a));
              }
            }
            // s0 = load{g>s}(glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v24_ld;
            v24_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
            v24_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
            tensorforge::intel_esimd::simd<float, 64> v25_ld;
            v25_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
            v25_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
            tensorforge::intel_esimd::simd<float, 64> v26_ld;
            v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 128));
            v26_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 128));
            tensorforge::intel_esimd::simd<float, 64> v27_ld;
            v27_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 192));
            v27_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 192));
            // wait(r0 = load{g>r}(glb_m1););
            // wait(s0 = load{g>s}(glb_m2[0, 1]));
            float r1[256]{};
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir1[256]{};
            tensorforge::intel_esimd::simd<float, 16> v30_data;
            v30_data.copy_from(r0 + (0));
            tensorforge::intel_esimd::simd<float, 16> v31_data;
            v31_data.copy_from(r0 + (16));
            tensorforge::intel_esimd::simd<float, 16> v32_data;
            v32_data.copy_from(r0 + (32));
            tensorforge::intel_esimd::simd<float, 16> v33_data;
            v33_data.copy_from(r0 + (48));
            tensorforge::intel_esimd::simd<float, 16> v34_data;
            v34_data.copy_from(r0 + (64));
            tensorforge::intel_esimd::simd<float, 16> v35_data;
            v35_data.copy_from(r0 + (80));
            tensorforge::intel_esimd::simd<float, 16> v36_data;
            v36_data.copy_from(r0 + (96));
            tensorforge::intel_esimd::simd<float, 16> v37_data;
            v37_data.copy_from(r0 + (112));
            tensorforge::intel_esimd::simd<float, 16> v38_data;
            v38_data.copy_from(r0 + (128));
            tensorforge::intel_esimd::simd<float, 16> v39_data;
            v39_data.copy_from(r0 + (144));
            tensorforge::intel_esimd::simd<float, 16> v40_data;
            v40_data.copy_from(r0 + (160));
            tensorforge::intel_esimd::simd<float, 16> v41_data;
            v41_data.copy_from(r0 + (176));
            tensorforge::intel_esimd::simd<float, 16> v42_data;
            v42_data.copy_from(r0 + (192));
            tensorforge::intel_esimd::simd<float, 16> v43_data;
            v43_data.copy_from(r0 + (208));
            tensorforge::intel_esimd::simd<float, 16> v44_data;
            v44_data.copy_from(r0 + (224));
            tensorforge::intel_esimd::simd<float, 16> v45_data;
            v45_data.copy_from(r0 + (240));
            tensorforge::intel_esimd::simd<float, 16> v46_acc{};
            tensorforge::intel_esimd::simd<float, 16> v50_data;
            v50_data.copy_from(s0 + (0_i32));
            v46_acc += ((static_cast<float>(v50_data[0])) * v30_data);
            v46_acc += ((static_cast<float>(v50_data[1])) * v31_data);
            v46_acc += ((static_cast<float>(v50_data[2])) * v32_data);
            v46_acc += ((static_cast<float>(v50_data[3])) * v33_data);
            v46_acc += ((static_cast<float>(v50_data[4])) * v34_data);
            v46_acc += ((static_cast<float>(v50_data[5])) * v35_data);
            v46_acc += ((static_cast<float>(v50_data[6])) * v36_data);
            v46_acc += ((static_cast<float>(v50_data[7])) * v37_data);
            v46_acc += ((static_cast<float>(v50_data[8])) * v38_data);
            v46_acc += ((static_cast<float>(v50_data[9])) * v39_data);
            v46_acc += ((static_cast<float>(v50_data[10])) * v40_data);
            v46_acc += ((static_cast<float>(v50_data[11])) * v41_data);
            v46_acc += ((static_cast<float>(v50_data[12])) * v42_data);
            v46_acc += ((static_cast<float>(v50_data[13])) * v43_data);
            v46_acc += ((static_cast<float>(v50_data[14])) * v44_data);
            v46_acc += ((static_cast<float>(v50_data[15])) * v45_data);
            v46_acc.copy_to(ir1 + (0));
            tensorforge::intel_esimd::simd<float, 16> v83_acc{};
            tensorforge::intel_esimd::simd<float, 16> v87_data;
            v87_data.copy_from(s0 + (16_i32));
            v83_acc += ((static_cast<float>(v87_data[0])) * v30_data);
            v83_acc += ((static_cast<float>(v87_data[1])) * v31_data);
            v83_acc += ((static_cast<float>(v87_data[2])) * v32_data);
            v83_acc += ((static_cast<float>(v87_data[3])) * v33_data);
            v83_acc += ((static_cast<float>(v87_data[4])) * v34_data);
            v83_acc += ((static_cast<float>(v87_data[5])) * v35_data);
            v83_acc += ((static_cast<float>(v87_data[6])) * v36_data);
            v83_acc += ((static_cast<float>(v87_data[7])) * v37_data);
            v83_acc += ((static_cast<float>(v87_data[8])) * v38_data);
            v83_acc += ((static_cast<float>(v87_data[9])) * v39_data);
            v83_acc += ((static_cast<float>(v87_data[10])) * v40_data);
            v83_acc += ((static_cast<float>(v87_data[11])) * v41_data);
            v83_acc += ((static_cast<float>(v87_data[12])) * v42_data);
            v83_acc += ((static_cast<float>(v87_data[13])) * v43_data);
            v83_acc += ((static_cast<float>(v87_data[14])) * v44_data);
            v83_acc += ((static_cast<float>(v87_data[15])) * v45_data);
            v83_acc.copy_to(ir1 + (16));
            tensorforge::intel_esimd::simd<float, 16> v120_acc{};
            tensorforge::intel_esimd::simd<float, 16> v124_data;
            v124_data.copy_from(s0 + (32_i32));
            v120_acc += ((static_cast<float>(v124_data[0])) * v30_data);
            v120_acc += ((static_cast<float>(v124_data[1])) * v31_data);
            v120_acc += ((static_cast<float>(v124_data[2])) * v32_data);
            v120_acc += ((static_cast<float>(v124_data[3])) * v33_data);
            v120_acc += ((static_cast<float>(v124_data[4])) * v34_data);
            v120_acc += ((static_cast<float>(v124_data[5])) * v35_data);
            v120_acc += ((static_cast<float>(v124_data[6])) * v36_data);
            v120_acc += ((static_cast<float>(v124_data[7])) * v37_data);
            v120_acc += ((static_cast<float>(v124_data[8])) * v38_data);
            v120_acc += ((static_cast<float>(v124_data[9])) * v39_data);
            v120_acc += ((static_cast<float>(v124_data[10])) * v40_data);
            v120_acc += ((static_cast<float>(v124_data[11])) * v41_data);
            v120_acc += ((static_cast<float>(v124_data[12])) * v42_data);
            v120_acc += ((static_cast<float>(v124_data[13])) * v43_data);
            v120_acc += ((static_cast<float>(v124_data[14])) * v44_data);
            v120_acc += ((static_cast<float>(v124_data[15])) * v45_data);
            v120_acc.copy_to(ir1 + (32));
            tensorforge::intel_esimd::simd<float, 16> v157_acc{};
            tensorforge::intel_esimd::simd<float, 16> v161_data;
            v161_data.copy_from(s0 + (48_i32));
            v157_acc += ((static_cast<float>(v161_data[0])) * v30_data);
            v157_acc += ((static_cast<float>(v161_data[1])) * v31_data);
            v157_acc += ((static_cast<float>(v161_data[2])) * v32_data);
            v157_acc += ((static_cast<float>(v161_data[3])) * v33_data);
            v157_acc += ((static_cast<float>(v161_data[4])) * v34_data);
            v157_acc += ((static_cast<float>(v161_data[5])) * v35_data);
            v157_acc += ((static_cast<float>(v161_data[6])) * v36_data);
            v157_acc += ((static_cast<float>(v161_data[7])) * v37_data);
            v157_acc += ((static_cast<float>(v161_data[8])) * v38_data);
            v157_acc += ((static_cast<float>(v161_data[9])) * v39_data);
            v157_acc += ((static_cast<float>(v161_data[10])) * v40_data);
            v157_acc += ((static_cast<float>(v161_data[11])) * v41_data);
            v157_acc += ((static_cast<float>(v161_data[12])) * v42_data);
            v157_acc += ((static_cast<float>(v161_data[13])) * v43_data);
            v157_acc += ((static_cast<float>(v161_data[14])) * v44_data);
            v157_acc += ((static_cast<float>(v161_data[15])) * v45_data);
            v157_acc.copy_to(ir1 + (48));
            tensorforge::intel_esimd::simd<float, 16> v194_acc{};
            tensorforge::intel_esimd::simd<float, 16> v198_data;
            v198_data.copy_from(s0 + (64_i32));
            v194_acc += ((static_cast<float>(v198_data[0])) * v30_data);
            v194_acc += ((static_cast<float>(v198_data[1])) * v31_data);
            v194_acc += ((static_cast<float>(v198_data[2])) * v32_data);
            v194_acc += ((static_cast<float>(v198_data[3])) * v33_data);
            v194_acc += ((static_cast<float>(v198_data[4])) * v34_data);
            v194_acc += ((static_cast<float>(v198_data[5])) * v35_data);
            v194_acc += ((static_cast<float>(v198_data[6])) * v36_data);
            v194_acc += ((static_cast<float>(v198_data[7])) * v37_data);
            v194_acc += ((static_cast<float>(v198_data[8])) * v38_data);
            v194_acc += ((static_cast<float>(v198_data[9])) * v39_data);
            v194_acc += ((static_cast<float>(v198_data[10])) * v40_data);
            v194_acc += ((static_cast<float>(v198_data[11])) * v41_data);
            v194_acc += ((static_cast<float>(v198_data[12])) * v42_data);
            v194_acc += ((static_cast<float>(v198_data[13])) * v43_data);
            v194_acc += ((static_cast<float>(v198_data[14])) * v44_data);
            v194_acc += ((static_cast<float>(v198_data[15])) * v45_data);
            v194_acc.copy_to(ir1 + (64));
            tensorforge::intel_esimd::simd<float, 16> v231_acc{};
            tensorforge::intel_esimd::simd<float, 16> v235_data;
            v235_data.copy_from(s0 + (80_i32));
            v231_acc += ((static_cast<float>(v235_data[0])) * v30_data);
            v231_acc += ((static_cast<float>(v235_data[1])) * v31_data);
            v231_acc += ((static_cast<float>(v235_data[2])) * v32_data);
            v231_acc += ((static_cast<float>(v235_data[3])) * v33_data);
            v231_acc += ((static_cast<float>(v235_data[4])) * v34_data);
            v231_acc += ((static_cast<float>(v235_data[5])) * v35_data);
            v231_acc += ((static_cast<float>(v235_data[6])) * v36_data);
            v231_acc += ((static_cast<float>(v235_data[7])) * v37_data);
            v231_acc += ((static_cast<float>(v235_data[8])) * v38_data);
            v231_acc += ((static_cast<float>(v235_data[9])) * v39_data);
            v231_acc += ((static_cast<float>(v235_data[10])) * v40_data);
            v231_acc += ((static_cast<float>(v235_data[11])) * v41_data);
            v231_acc += ((static_cast<float>(v235_data[12])) * v42_data);
            v231_acc += ((static_cast<float>(v235_data[13])) * v43_data);
            v231_acc += ((static_cast<float>(v235_data[14])) * v44_data);
            v231_acc += ((static_cast<float>(v235_data[15])) * v45_data);
            v231_acc.copy_to(ir1 + (80));
            tensorforge::intel_esimd::simd<float, 16> v268_acc{};
            tensorforge::intel_esimd::simd<float, 16> v272_data;
            v272_data.copy_from(s0 + (96_i32));
            v268_acc += ((static_cast<float>(v272_data[0])) * v30_data);
            v268_acc += ((static_cast<float>(v272_data[1])) * v31_data);
            v268_acc += ((static_cast<float>(v272_data[2])) * v32_data);
            v268_acc += ((static_cast<float>(v272_data[3])) * v33_data);
            v268_acc += ((static_cast<float>(v272_data[4])) * v34_data);
            v268_acc += ((static_cast<float>(v272_data[5])) * v35_data);
            v268_acc += ((static_cast<float>(v272_data[6])) * v36_data);
            v268_acc += ((static_cast<float>(v272_data[7])) * v37_data);
            v268_acc += ((static_cast<float>(v272_data[8])) * v38_data);
            v268_acc += ((static_cast<float>(v272_data[9])) * v39_data);
            v268_acc += ((static_cast<float>(v272_data[10])) * v40_data);
            v268_acc += ((static_cast<float>(v272_data[11])) * v41_data);
            v268_acc += ((static_cast<float>(v272_data[12])) * v42_data);
            v268_acc += ((static_cast<float>(v272_data[13])) * v43_data);
            v268_acc += ((static_cast<float>(v272_data[14])) * v44_data);
            v268_acc += ((static_cast<float>(v272_data[15])) * v45_data);
            v268_acc.copy_to(ir1 + (96));
            tensorforge::intel_esimd::simd<float, 16> v305_acc{};
            tensorforge::intel_esimd::simd<float, 16> v309_data;
            v309_data.copy_from(s0 + (112_i32));
            v305_acc += ((static_cast<float>(v309_data[0])) * v30_data);
            v305_acc += ((static_cast<float>(v309_data[1])) * v31_data);
            v305_acc += ((static_cast<float>(v309_data[2])) * v32_data);
            v305_acc += ((static_cast<float>(v309_data[3])) * v33_data);
            v305_acc += ((static_cast<float>(v309_data[4])) * v34_data);
            v305_acc += ((static_cast<float>(v309_data[5])) * v35_data);
            v305_acc += ((static_cast<float>(v309_data[6])) * v36_data);
            v305_acc += ((static_cast<float>(v309_data[7])) * v37_data);
            v305_acc += ((static_cast<float>(v309_data[8])) * v38_data);
            v305_acc += ((static_cast<float>(v309_data[9])) * v39_data);
            v305_acc += ((static_cast<float>(v309_data[10])) * v40_data);
            v305_acc += ((static_cast<float>(v309_data[11])) * v41_data);
            v305_acc += ((static_cast<float>(v309_data[12])) * v42_data);
            v305_acc += ((static_cast<float>(v309_data[13])) * v43_data);
            v305_acc += ((static_cast<float>(v309_data[14])) * v44_data);
            v305_acc += ((static_cast<float>(v309_data[15])) * v45_data);
            v305_acc.copy_to(ir1 + (112));
            tensorforge::intel_esimd::simd<float, 16> v342_acc{};
            tensorforge::intel_esimd::simd<float, 16> v346_data;
            v346_data.copy_from(s0 + (128_i32));
            v342_acc += ((static_cast<float>(v346_data[0])) * v30_data);
            v342_acc += ((static_cast<float>(v346_data[1])) * v31_data);
            v342_acc += ((static_cast<float>(v346_data[2])) * v32_data);
            v342_acc += ((static_cast<float>(v346_data[3])) * v33_data);
            v342_acc += ((static_cast<float>(v346_data[4])) * v34_data);
            v342_acc += ((static_cast<float>(v346_data[5])) * v35_data);
            v342_acc += ((static_cast<float>(v346_data[6])) * v36_data);
            v342_acc += ((static_cast<float>(v346_data[7])) * v37_data);
            v342_acc += ((static_cast<float>(v346_data[8])) * v38_data);
            v342_acc += ((static_cast<float>(v346_data[9])) * v39_data);
            v342_acc += ((static_cast<float>(v346_data[10])) * v40_data);
            v342_acc += ((static_cast<float>(v346_data[11])) * v41_data);
            v342_acc += ((static_cast<float>(v346_data[12])) * v42_data);
            v342_acc += ((static_cast<float>(v346_data[13])) * v43_data);
            v342_acc += ((static_cast<float>(v346_data[14])) * v44_data);
            v342_acc += ((static_cast<float>(v346_data[15])) * v45_data);
            v342_acc.copy_to(ir1 + (128));
            tensorforge::intel_esimd::simd<float, 16> v379_acc{};
            tensorforge::intel_esimd::simd<float, 16> v383_data;
            v383_data.copy_from(s0 + (144_i32));
            v379_acc += ((static_cast<float>(v383_data[0])) * v30_data);
            v379_acc += ((static_cast<float>(v383_data[1])) * v31_data);
            v379_acc += ((static_cast<float>(v383_data[2])) * v32_data);
            v379_acc += ((static_cast<float>(v383_data[3])) * v33_data);
            v379_acc += ((static_cast<float>(v383_data[4])) * v34_data);
            v379_acc += ((static_cast<float>(v383_data[5])) * v35_data);
            v379_acc += ((static_cast<float>(v383_data[6])) * v36_data);
            v379_acc += ((static_cast<float>(v383_data[7])) * v37_data);
            v379_acc += ((static_cast<float>(v383_data[8])) * v38_data);
            v379_acc += ((static_cast<float>(v383_data[9])) * v39_data);
            v379_acc += ((static_cast<float>(v383_data[10])) * v40_data);
            v379_acc += ((static_cast<float>(v383_data[11])) * v41_data);
            v379_acc += ((static_cast<float>(v383_data[12])) * v42_data);
            v379_acc += ((static_cast<float>(v383_data[13])) * v43_data);
            v379_acc += ((static_cast<float>(v383_data[14])) * v44_data);
            v379_acc += ((static_cast<float>(v383_data[15])) * v45_data);
            v379_acc.copy_to(ir1 + (144));
            tensorforge::intel_esimd::simd<float, 16> v416_acc{};
            tensorforge::intel_esimd::simd<float, 16> v420_data;
            v420_data.copy_from(s0 + (160_i32));
            v416_acc += ((static_cast<float>(v420_data[0])) * v30_data);
            v416_acc += ((static_cast<float>(v420_data[1])) * v31_data);
            v416_acc += ((static_cast<float>(v420_data[2])) * v32_data);
            v416_acc += ((static_cast<float>(v420_data[3])) * v33_data);
            v416_acc += ((static_cast<float>(v420_data[4])) * v34_data);
            v416_acc += ((static_cast<float>(v420_data[5])) * v35_data);
            v416_acc += ((static_cast<float>(v420_data[6])) * v36_data);
            v416_acc += ((static_cast<float>(v420_data[7])) * v37_data);
            v416_acc += ((static_cast<float>(v420_data[8])) * v38_data);
            v416_acc += ((static_cast<float>(v420_data[9])) * v39_data);
            v416_acc += ((static_cast<float>(v420_data[10])) * v40_data);
            v416_acc += ((static_cast<float>(v420_data[11])) * v41_data);
            v416_acc += ((static_cast<float>(v420_data[12])) * v42_data);
            v416_acc += ((static_cast<float>(v420_data[13])) * v43_data);
            v416_acc += ((static_cast<float>(v420_data[14])) * v44_data);
            v416_acc += ((static_cast<float>(v420_data[15])) * v45_data);
            v416_acc.copy_to(ir1 + (160));
            tensorforge::intel_esimd::simd<float, 16> v453_acc{};
            tensorforge::intel_esimd::simd<float, 16> v457_data;
            v457_data.copy_from(s0 + (176_i32));
            v453_acc += ((static_cast<float>(v457_data[0])) * v30_data);
            v453_acc += ((static_cast<float>(v457_data[1])) * v31_data);
            v453_acc += ((static_cast<float>(v457_data[2])) * v32_data);
            v453_acc += ((static_cast<float>(v457_data[3])) * v33_data);
            v453_acc += ((static_cast<float>(v457_data[4])) * v34_data);
            v453_acc += ((static_cast<float>(v457_data[5])) * v35_data);
            v453_acc += ((static_cast<float>(v457_data[6])) * v36_data);
            v453_acc += ((static_cast<float>(v457_data[7])) * v37_data);
            v453_acc += ((static_cast<float>(v457_data[8])) * v38_data);
            v453_acc += ((static_cast<float>(v457_data[9])) * v39_data);
            v453_acc += ((static_cast<float>(v457_data[10])) * v40_data);
            v453_acc += ((static_cast<float>(v457_data[11])) * v41_data);
            v453_acc += ((static_cast<float>(v457_data[12])) * v42_data);
            v453_acc += ((static_cast<float>(v457_data[13])) * v43_data);
            v453_acc += ((static_cast<float>(v457_data[14])) * v44_data);
            v453_acc += ((static_cast<float>(v457_data[15])) * v45_data);
            v453_acc.copy_to(ir1 + (176));
            tensorforge::intel_esimd::simd<float, 16> v490_acc{};
            tensorforge::intel_esimd::simd<float, 16> v494_data;
            v494_data.copy_from(s0 + (192_i32));
            v490_acc += ((static_cast<float>(v494_data[0])) * v30_data);
            v490_acc += ((static_cast<float>(v494_data[1])) * v31_data);
            v490_acc += ((static_cast<float>(v494_data[2])) * v32_data);
            v490_acc += ((static_cast<float>(v494_data[3])) * v33_data);
            v490_acc += ((static_cast<float>(v494_data[4])) * v34_data);
            v490_acc += ((static_cast<float>(v494_data[5])) * v35_data);
            v490_acc += ((static_cast<float>(v494_data[6])) * v36_data);
            v490_acc += ((static_cast<float>(v494_data[7])) * v37_data);
            v490_acc += ((static_cast<float>(v494_data[8])) * v38_data);
            v490_acc += ((static_cast<float>(v494_data[9])) * v39_data);
            v490_acc += ((static_cast<float>(v494_data[10])) * v40_data);
            v490_acc += ((static_cast<float>(v494_data[11])) * v41_data);
            v490_acc += ((static_cast<float>(v494_data[12])) * v42_data);
            v490_acc += ((static_cast<float>(v494_data[13])) * v43_data);
            v490_acc += ((static_cast<float>(v494_data[14])) * v44_data);
            v490_acc += ((static_cast<float>(v494_data[15])) * v45_data);
            v490_acc.copy_to(ir1 + (192));
            tensorforge::intel_esimd::simd<float, 16> v527_acc{};
            tensorforge::intel_esimd::simd<float, 16> v531_data;
            v531_data.copy_from(s0 + (208_i32));
            v527_acc += ((static_cast<float>(v531_data[0])) * v30_data);
            v527_acc += ((static_cast<float>(v531_data[1])) * v31_data);
            v527_acc += ((static_cast<float>(v531_data[2])) * v32_data);
            v527_acc += ((static_cast<float>(v531_data[3])) * v33_data);
            v527_acc += ((static_cast<float>(v531_data[4])) * v34_data);
            v527_acc += ((static_cast<float>(v531_data[5])) * v35_data);
            v527_acc += ((static_cast<float>(v531_data[6])) * v36_data);
            v527_acc += ((static_cast<float>(v531_data[7])) * v37_data);
            v527_acc += ((static_cast<float>(v531_data[8])) * v38_data);
            v527_acc += ((static_cast<float>(v531_data[9])) * v39_data);
            v527_acc += ((static_cast<float>(v531_data[10])) * v40_data);
            v527_acc += ((static_cast<float>(v531_data[11])) * v41_data);
            v527_acc += ((static_cast<float>(v531_data[12])) * v42_data);
            v527_acc += ((static_cast<float>(v531_data[13])) * v43_data);
            v527_acc += ((static_cast<float>(v531_data[14])) * v44_data);
            v527_acc += ((static_cast<float>(v531_data[15])) * v45_data);
            v527_acc.copy_to(ir1 + (208));
            tensorforge::intel_esimd::simd<float, 16> v564_acc{};
            tensorforge::intel_esimd::simd<float, 16> v568_data;
            v568_data.copy_from(s0 + (224_i32));
            v564_acc += ((static_cast<float>(v568_data[0])) * v30_data);
            v564_acc += ((static_cast<float>(v568_data[1])) * v31_data);
            v564_acc += ((static_cast<float>(v568_data[2])) * v32_data);
            v564_acc += ((static_cast<float>(v568_data[3])) * v33_data);
            v564_acc += ((static_cast<float>(v568_data[4])) * v34_data);
            v564_acc += ((static_cast<float>(v568_data[5])) * v35_data);
            v564_acc += ((static_cast<float>(v568_data[6])) * v36_data);
            v564_acc += ((static_cast<float>(v568_data[7])) * v37_data);
            v564_acc += ((static_cast<float>(v568_data[8])) * v38_data);
            v564_acc += ((static_cast<float>(v568_data[9])) * v39_data);
            v564_acc += ((static_cast<float>(v568_data[10])) * v40_data);
            v564_acc += ((static_cast<float>(v568_data[11])) * v41_data);
            v564_acc += ((static_cast<float>(v568_data[12])) * v42_data);
            v564_acc += ((static_cast<float>(v568_data[13])) * v43_data);
            v564_acc += ((static_cast<float>(v568_data[14])) * v44_data);
            v564_acc += ((static_cast<float>(v568_data[15])) * v45_data);
            v564_acc.copy_to(ir1 + (224));
            tensorforge::intel_esimd::simd<float, 16> v601_acc{};
            tensorforge::intel_esimd::simd<float, 16> v605_data;
            v605_data.copy_from(s0 + (240_i32));
            v601_acc += ((static_cast<float>(v605_data[0])) * v30_data);
            v601_acc += ((static_cast<float>(v605_data[1])) * v31_data);
            v601_acc += ((static_cast<float>(v605_data[2])) * v32_data);
            v601_acc += ((static_cast<float>(v605_data[3])) * v33_data);
            v601_acc += ((static_cast<float>(v605_data[4])) * v34_data);
            v601_acc += ((static_cast<float>(v605_data[5])) * v35_data);
            v601_acc += ((static_cast<float>(v605_data[6])) * v36_data);
            v601_acc += ((static_cast<float>(v605_data[7])) * v37_data);
            v601_acc += ((static_cast<float>(v605_data[8])) * v38_data);
            v601_acc += ((static_cast<float>(v605_data[9])) * v39_data);
            v601_acc += ((static_cast<float>(v605_data[10])) * v40_data);
            v601_acc += ((static_cast<float>(v605_data[11])) * v41_data);
            v601_acc += ((static_cast<float>(v605_data[12])) * v42_data);
            v601_acc += ((static_cast<float>(v605_data[13])) * v43_data);
            v601_acc += ((static_cast<float>(v605_data[14])) * v44_data);
            v601_acc += ((static_cast<float>(v605_data[15])) * v45_data);
            v601_acc.copy_to(ir1 + (240));
            #pragma unroll
            for (int32_t v638_n0 = 0; v638_n0 < 1; ++v638_n0) {
              int32_t v640_a = v638_n0 * 16;
              #pragma unroll
              for (int32_t v639_n1 = 0; v639_n1 < 16; ++v639_n1) {
                int32_t v642_a = v640_a + (v639_n1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v643_data;
                v643_data.copy_from(ir1 + (v642_a));
                v643_data.copy_to(r1 + (v642_a));
              }
            }
            // glb_m0 = store{r>g}(r1);
            #pragma unroll
            for (int32_t v647_i0 = 0; v647_i0 < 1; ++v647_i0) {
              int32_t v649_a = v647_i0 * 16;
              #pragma unroll
              for (int32_t v648_i1 = 0; v648_i1 < 16; ++v648_i1) {
                int32_t v651_a = v649_a + (v648_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v652_data;
                v652_data.copy_from(r1 + (v651_a));
                v652_data.copy_to(glb_m0 + (v651_a));
              }
            }
          }
        }
      });
    }
  });
}

