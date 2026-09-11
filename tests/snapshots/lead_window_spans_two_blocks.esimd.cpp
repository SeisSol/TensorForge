// === base name ===
kernel_fcdefc48366a8e67

// === header ===
void launcher_kernel_fcdefc48366a8e67(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_fcdefc48366a8e67(const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_fcdefc48366a8e67(stream, grid, block,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_fcdefc48366a8e67(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float* m1, float** m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 64×13(64×13) {0..64}×{0..13} pointer_based
        // m1 6(6) {0..6} none
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
        // t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..64}×{0..13}×{0..6})[0, 1, 2] = m0 64×13(64×13) {0..64}×{0..13} pointer_based({0..64}×{0..13})[0, 1]×m1 6(6) {0..6} none({0..6})[2]
        // m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based({0..15}×{0..1}×{0..6})[0, 1, 2] += t0 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} strided({0..15}×{0..1}×{0..6})[0, 1, 2]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v1_batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v1_batchId0][0 + m2_extraOffset];
              float r0[832]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v12_i0 = 0; v12_i0 < 2; ++v12_i0) {
                int32_t v14_lead = v12_i0 * 32;
                #pragma unroll
                for (int32_t v13_i1 = 0; v13_i1 < 13; ++v13_i1) {
                  int32_t v17_a = v14_lead + (v13_i1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v18_data;
                  v18_data.copy_from(glb_m0 + (v17_a));
                  v18_data.copy_to(r0 + (v17_a));
                }
              }
              float r2[384]{};
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 1; ++v23_i1) {
                int32_t v31_a = 20_i32 + ((v23_i1 + 12) * 64);
                int32_t v36_a = 20 + (v23_i1 * 64);
                #pragma unroll
                for (int32_t v24_i2 = 0; v24_i2 < 6; ++v24_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v33_data;
                  v33_data.copy_from(glb_m2 + ((v31_a + (v24_i2 * 832))));
                  v33_data.copy_to(r2 + ((v36_a + (v24_i2 * 64))));
                }
              }
              #pragma unroll
              for (int32_t v38_i1 = 0; v38_i1 < 1; ++v38_i1) {
                int32_t v46_a = 32_i32 + ((v38_i1 + 12) * 64);
                int32_t v51_a = 32 + (v38_i1 * 64);
                #pragma unroll
                for (int32_t v39_i2 = 0; v39_i2 < 6; ++v39_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v48_data;
                  v48_data.copy_from(glb_m2 + ((v46_a + (v39_i2 * 832))));
                  v48_data.copy_to(r2 + ((v51_a + (v39_i2 * 64))));
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[4992]{};
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 32> v54_data;
              v54_data.copy_from(r0 + (0));
              float v55_data = glb_m1[0];
              tensorforge::intel_esimd::simd<float, 32> v57_data;
              v57_data.copy_from(r1 + (0));
              (v57_data + (v54_data * v55_data)).copy_to(r1 + (0));
              float v60_data = glb_m1[1];
              tensorforge::intel_esimd::simd<float, 32> v62_data;
              v62_data.copy_from(r1 + (832));
              (v62_data + (v54_data * v60_data)).copy_to(r1 + (832));
              float v65_data = glb_m1[2];
              tensorforge::intel_esimd::simd<float, 32> v67_data;
              v67_data.copy_from(r1 + (1664));
              (v67_data + (v54_data * v65_data)).copy_to(r1 + (1664));
              float v70_data = glb_m1[3];
              tensorforge::intel_esimd::simd<float, 32> v72_data;
              v72_data.copy_from(r1 + (2496));
              (v72_data + (v54_data * v70_data)).copy_to(r1 + (2496));
              float v75_data = glb_m1[4];
              tensorforge::intel_esimd::simd<float, 32> v77_data;
              v77_data.copy_from(r1 + (3328));
              (v77_data + (v54_data * v75_data)).copy_to(r1 + (3328));
              float v80_data = glb_m1[5];
              tensorforge::intel_esimd::simd<float, 32> v82_data;
              v82_data.copy_from(r1 + (4160));
              (v82_data + (v54_data * v80_data)).copy_to(r1 + (4160));
              tensorforge::intel_esimd::simd<float, 32> v84_data;
              v84_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 32> v87_data;
              v87_data.copy_from(r1 + (64));
              (v87_data + (v84_data * v55_data)).copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v92_data;
              v92_data.copy_from(r1 + (896));
              (v92_data + (v84_data * v60_data)).copy_to(r1 + (896));
              tensorforge::intel_esimd::simd<float, 32> v97_data;
              v97_data.copy_from(r1 + (1728));
              (v97_data + (v84_data * v65_data)).copy_to(r1 + (1728));
              tensorforge::intel_esimd::simd<float, 32> v102_data;
              v102_data.copy_from(r1 + (2560));
              (v102_data + (v84_data * v70_data)).copy_to(r1 + (2560));
              tensorforge::intel_esimd::simd<float, 32> v107_data;
              v107_data.copy_from(r1 + (3392));
              (v107_data + (v84_data * v75_data)).copy_to(r1 + (3392));
              tensorforge::intel_esimd::simd<float, 32> v112_data;
              v112_data.copy_from(r1 + (4224));
              (v112_data + (v84_data * v80_data)).copy_to(r1 + (4224));
              tensorforge::intel_esimd::simd<float, 32> v114_data;
              v114_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 32> v117_data;
              v117_data.copy_from(r1 + (128));
              (v117_data + (v114_data * v55_data)).copy_to(r1 + (128));
              tensorforge::intel_esimd::simd<float, 32> v122_data;
              v122_data.copy_from(r1 + (960));
              (v122_data + (v114_data * v60_data)).copy_to(r1 + (960));
              tensorforge::intel_esimd::simd<float, 32> v127_data;
              v127_data.copy_from(r1 + (1792));
              (v127_data + (v114_data * v65_data)).copy_to(r1 + (1792));
              tensorforge::intel_esimd::simd<float, 32> v132_data;
              v132_data.copy_from(r1 + (2624));
              (v132_data + (v114_data * v70_data)).copy_to(r1 + (2624));
              tensorforge::intel_esimd::simd<float, 32> v137_data;
              v137_data.copy_from(r1 + (3456));
              (v137_data + (v114_data * v75_data)).copy_to(r1 + (3456));
              tensorforge::intel_esimd::simd<float, 32> v142_data;
              v142_data.copy_from(r1 + (4288));
              (v142_data + (v114_data * v80_data)).copy_to(r1 + (4288));
              tensorforge::intel_esimd::simd<float, 32> v144_data;
              v144_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 32> v147_data;
              v147_data.copy_from(r1 + (192));
              (v147_data + (v144_data * v55_data)).copy_to(r1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v152_data;
              v152_data.copy_from(r1 + (1024));
              (v152_data + (v144_data * v60_data)).copy_to(r1 + (1024));
              tensorforge::intel_esimd::simd<float, 32> v157_data;
              v157_data.copy_from(r1 + (1856));
              (v157_data + (v144_data * v65_data)).copy_to(r1 + (1856));
              tensorforge::intel_esimd::simd<float, 32> v162_data;
              v162_data.copy_from(r1 + (2688));
              (v162_data + (v144_data * v70_data)).copy_to(r1 + (2688));
              tensorforge::intel_esimd::simd<float, 32> v167_data;
              v167_data.copy_from(r1 + (3520));
              (v167_data + (v144_data * v75_data)).copy_to(r1 + (3520));
              tensorforge::intel_esimd::simd<float, 32> v172_data;
              v172_data.copy_from(r1 + (4352));
              (v172_data + (v144_data * v80_data)).copy_to(r1 + (4352));
              tensorforge::intel_esimd::simd<float, 32> v174_data;
              v174_data.copy_from(r0 + (256));
              tensorforge::intel_esimd::simd<float, 32> v177_data;
              v177_data.copy_from(r1 + (256));
              (v177_data + (v174_data * v55_data)).copy_to(r1 + (256));
              tensorforge::intel_esimd::simd<float, 32> v182_data;
              v182_data.copy_from(r1 + (1088));
              (v182_data + (v174_data * v60_data)).copy_to(r1 + (1088));
              tensorforge::intel_esimd::simd<float, 32> v187_data;
              v187_data.copy_from(r1 + (1920));
              (v187_data + (v174_data * v65_data)).copy_to(r1 + (1920));
              tensorforge::intel_esimd::simd<float, 32> v192_data;
              v192_data.copy_from(r1 + (2752));
              (v192_data + (v174_data * v70_data)).copy_to(r1 + (2752));
              tensorforge::intel_esimd::simd<float, 32> v197_data;
              v197_data.copy_from(r1 + (3584));
              (v197_data + (v174_data * v75_data)).copy_to(r1 + (3584));
              tensorforge::intel_esimd::simd<float, 32> v202_data;
              v202_data.copy_from(r1 + (4416));
              (v202_data + (v174_data * v80_data)).copy_to(r1 + (4416));
              tensorforge::intel_esimd::simd<float, 32> v204_data;
              v204_data.copy_from(r0 + (320));
              tensorforge::intel_esimd::simd<float, 32> v207_data;
              v207_data.copy_from(r1 + (320));
              (v207_data + (v204_data * v55_data)).copy_to(r1 + (320));
              tensorforge::intel_esimd::simd<float, 32> v212_data;
              v212_data.copy_from(r1 + (1152));
              (v212_data + (v204_data * v60_data)).copy_to(r1 + (1152));
              tensorforge::intel_esimd::simd<float, 32> v217_data;
              v217_data.copy_from(r1 + (1984));
              (v217_data + (v204_data * v65_data)).copy_to(r1 + (1984));
              tensorforge::intel_esimd::simd<float, 32> v222_data;
              v222_data.copy_from(r1 + (2816));
              (v222_data + (v204_data * v70_data)).copy_to(r1 + (2816));
              tensorforge::intel_esimd::simd<float, 32> v227_data;
              v227_data.copy_from(r1 + (3648));
              (v227_data + (v204_data * v75_data)).copy_to(r1 + (3648));
              tensorforge::intel_esimd::simd<float, 32> v232_data;
              v232_data.copy_from(r1 + (4480));
              (v232_data + (v204_data * v80_data)).copy_to(r1 + (4480));
              tensorforge::intel_esimd::simd<float, 32> v234_data;
              v234_data.copy_from(r0 + (384));
              tensorforge::intel_esimd::simd<float, 32> v237_data;
              v237_data.copy_from(r1 + (384));
              (v237_data + (v234_data * v55_data)).copy_to(r1 + (384));
              tensorforge::intel_esimd::simd<float, 32> v242_data;
              v242_data.copy_from(r1 + (1216));
              (v242_data + (v234_data * v60_data)).copy_to(r1 + (1216));
              tensorforge::intel_esimd::simd<float, 32> v247_data;
              v247_data.copy_from(r1 + (2048));
              (v247_data + (v234_data * v65_data)).copy_to(r1 + (2048));
              tensorforge::intel_esimd::simd<float, 32> v252_data;
              v252_data.copy_from(r1 + (2880));
              (v252_data + (v234_data * v70_data)).copy_to(r1 + (2880));
              tensorforge::intel_esimd::simd<float, 32> v257_data;
              v257_data.copy_from(r1 + (3712));
              (v257_data + (v234_data * v75_data)).copy_to(r1 + (3712));
              tensorforge::intel_esimd::simd<float, 32> v262_data;
              v262_data.copy_from(r1 + (4544));
              (v262_data + (v234_data * v80_data)).copy_to(r1 + (4544));
              tensorforge::intel_esimd::simd<float, 32> v264_data;
              v264_data.copy_from(r0 + (448));
              tensorforge::intel_esimd::simd<float, 32> v267_data;
              v267_data.copy_from(r1 + (448));
              (v267_data + (v264_data * v55_data)).copy_to(r1 + (448));
              tensorforge::intel_esimd::simd<float, 32> v272_data;
              v272_data.copy_from(r1 + (1280));
              (v272_data + (v264_data * v60_data)).copy_to(r1 + (1280));
              tensorforge::intel_esimd::simd<float, 32> v277_data;
              v277_data.copy_from(r1 + (2112));
              (v277_data + (v264_data * v65_data)).copy_to(r1 + (2112));
              tensorforge::intel_esimd::simd<float, 32> v282_data;
              v282_data.copy_from(r1 + (2944));
              (v282_data + (v264_data * v70_data)).copy_to(r1 + (2944));
              tensorforge::intel_esimd::simd<float, 32> v287_data;
              v287_data.copy_from(r1 + (3776));
              (v287_data + (v264_data * v75_data)).copy_to(r1 + (3776));
              tensorforge::intel_esimd::simd<float, 32> v292_data;
              v292_data.copy_from(r1 + (4608));
              (v292_data + (v264_data * v80_data)).copy_to(r1 + (4608));
              tensorforge::intel_esimd::simd<float, 32> v294_data;
              v294_data.copy_from(r0 + (512));
              tensorforge::intel_esimd::simd<float, 32> v297_data;
              v297_data.copy_from(r1 + (512));
              (v297_data + (v294_data * v55_data)).copy_to(r1 + (512));
              tensorforge::intel_esimd::simd<float, 32> v302_data;
              v302_data.copy_from(r1 + (1344));
              (v302_data + (v294_data * v60_data)).copy_to(r1 + (1344));
              tensorforge::intel_esimd::simd<float, 32> v307_data;
              v307_data.copy_from(r1 + (2176));
              (v307_data + (v294_data * v65_data)).copy_to(r1 + (2176));
              tensorforge::intel_esimd::simd<float, 32> v312_data;
              v312_data.copy_from(r1 + (3008));
              (v312_data + (v294_data * v70_data)).copy_to(r1 + (3008));
              tensorforge::intel_esimd::simd<float, 32> v317_data;
              v317_data.copy_from(r1 + (3840));
              (v317_data + (v294_data * v75_data)).copy_to(r1 + (3840));
              tensorforge::intel_esimd::simd<float, 32> v322_data;
              v322_data.copy_from(r1 + (4672));
              (v322_data + (v294_data * v80_data)).copy_to(r1 + (4672));
              tensorforge::intel_esimd::simd<float, 32> v324_data;
              v324_data.copy_from(r0 + (576));
              tensorforge::intel_esimd::simd<float, 32> v327_data;
              v327_data.copy_from(r1 + (576));
              (v327_data + (v324_data * v55_data)).copy_to(r1 + (576));
              tensorforge::intel_esimd::simd<float, 32> v332_data;
              v332_data.copy_from(r1 + (1408));
              (v332_data + (v324_data * v60_data)).copy_to(r1 + (1408));
              tensorforge::intel_esimd::simd<float, 32> v337_data;
              v337_data.copy_from(r1 + (2240));
              (v337_data + (v324_data * v65_data)).copy_to(r1 + (2240));
              tensorforge::intel_esimd::simd<float, 32> v342_data;
              v342_data.copy_from(r1 + (3072));
              (v342_data + (v324_data * v70_data)).copy_to(r1 + (3072));
              tensorforge::intel_esimd::simd<float, 32> v347_data;
              v347_data.copy_from(r1 + (3904));
              (v347_data + (v324_data * v75_data)).copy_to(r1 + (3904));
              tensorforge::intel_esimd::simd<float, 32> v352_data;
              v352_data.copy_from(r1 + (4736));
              (v352_data + (v324_data * v80_data)).copy_to(r1 + (4736));
              tensorforge::intel_esimd::simd<float, 32> v354_data;
              v354_data.copy_from(r0 + (640));
              tensorforge::intel_esimd::simd<float, 32> v357_data;
              v357_data.copy_from(r1 + (640));
              (v357_data + (v354_data * v55_data)).copy_to(r1 + (640));
              tensorforge::intel_esimd::simd<float, 32> v362_data;
              v362_data.copy_from(r1 + (1472));
              (v362_data + (v354_data * v60_data)).copy_to(r1 + (1472));
              tensorforge::intel_esimd::simd<float, 32> v367_data;
              v367_data.copy_from(r1 + (2304));
              (v367_data + (v354_data * v65_data)).copy_to(r1 + (2304));
              tensorforge::intel_esimd::simd<float, 32> v372_data;
              v372_data.copy_from(r1 + (3136));
              (v372_data + (v354_data * v70_data)).copy_to(r1 + (3136));
              tensorforge::intel_esimd::simd<float, 32> v377_data;
              v377_data.copy_from(r1 + (3968));
              (v377_data + (v354_data * v75_data)).copy_to(r1 + (3968));
              tensorforge::intel_esimd::simd<float, 32> v382_data;
              v382_data.copy_from(r1 + (4800));
              (v382_data + (v354_data * v80_data)).copy_to(r1 + (4800));
              tensorforge::intel_esimd::simd<float, 32> v384_data;
              v384_data.copy_from(r0 + (704));
              tensorforge::intel_esimd::simd<float, 32> v387_data;
              v387_data.copy_from(r1 + (704));
              (v387_data + (v384_data * v55_data)).copy_to(r1 + (704));
              tensorforge::intel_esimd::simd<float, 32> v392_data;
              v392_data.copy_from(r1 + (1536));
              (v392_data + (v384_data * v60_data)).copy_to(r1 + (1536));
              tensorforge::intel_esimd::simd<float, 32> v397_data;
              v397_data.copy_from(r1 + (2368));
              (v397_data + (v384_data * v65_data)).copy_to(r1 + (2368));
              tensorforge::intel_esimd::simd<float, 32> v402_data;
              v402_data.copy_from(r1 + (3200));
              (v402_data + (v384_data * v70_data)).copy_to(r1 + (3200));
              tensorforge::intel_esimd::simd<float, 32> v407_data;
              v407_data.copy_from(r1 + (4032));
              (v407_data + (v384_data * v75_data)).copy_to(r1 + (4032));
              tensorforge::intel_esimd::simd<float, 32> v412_data;
              v412_data.copy_from(r1 + (4864));
              (v412_data + (v384_data * v80_data)).copy_to(r1 + (4864));
              tensorforge::intel_esimd::simd<float, 32> v414_data;
              v414_data.copy_from(r0 + (768));
              tensorforge::intel_esimd::simd<float, 32> v417_data;
              v417_data.copy_from(r1 + (768));
              (v417_data + (v414_data * v55_data)).copy_to(r1 + (768));
              tensorforge::intel_esimd::simd<float, 32> v422_data;
              v422_data.copy_from(r1 + (1600));
              (v422_data + (v414_data * v60_data)).copy_to(r1 + (1600));
              tensorforge::intel_esimd::simd<float, 32> v427_data;
              v427_data.copy_from(r1 + (2432));
              (v427_data + (v414_data * v65_data)).copy_to(r1 + (2432));
              tensorforge::intel_esimd::simd<float, 32> v432_data;
              v432_data.copy_from(r1 + (3264));
              (v432_data + (v414_data * v70_data)).copy_to(r1 + (3264));
              tensorforge::intel_esimd::simd<float, 32> v437_data;
              v437_data.copy_from(r1 + (4096));
              (v437_data + (v414_data * v75_data)).copy_to(r1 + (4096));
              tensorforge::intel_esimd::simd<float, 32> v442_data;
              v442_data.copy_from(r1 + (4928));
              (v442_data + (v414_data * v80_data)).copy_to(r1 + (4928));
              tensorforge::intel_esimd::simd<float, 32> v444_data;
              v444_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 32> v447_data;
              v447_data.copy_from(r1 + (32));
              (v447_data + (v444_data * v55_data)).copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v452_data;
              v452_data.copy_from(r1 + (864));
              (v452_data + (v444_data * v60_data)).copy_to(r1 + (864));
              tensorforge::intel_esimd::simd<float, 32> v457_data;
              v457_data.copy_from(r1 + (1696));
              (v457_data + (v444_data * v65_data)).copy_to(r1 + (1696));
              tensorforge::intel_esimd::simd<float, 32> v462_data;
              v462_data.copy_from(r1 + (2528));
              (v462_data + (v444_data * v70_data)).copy_to(r1 + (2528));
              tensorforge::intel_esimd::simd<float, 32> v467_data;
              v467_data.copy_from(r1 + (3360));
              (v467_data + (v444_data * v75_data)).copy_to(r1 + (3360));
              tensorforge::intel_esimd::simd<float, 32> v472_data;
              v472_data.copy_from(r1 + (4192));
              (v472_data + (v444_data * v80_data)).copy_to(r1 + (4192));
              tensorforge::intel_esimd::simd<float, 32> v474_data;
              v474_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 32> v477_data;
              v477_data.copy_from(r1 + (96));
              (v477_data + (v474_data * v55_data)).copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 32> v482_data;
              v482_data.copy_from(r1 + (928));
              (v482_data + (v474_data * v60_data)).copy_to(r1 + (928));
              tensorforge::intel_esimd::simd<float, 32> v487_data;
              v487_data.copy_from(r1 + (1760));
              (v487_data + (v474_data * v65_data)).copy_to(r1 + (1760));
              tensorforge::intel_esimd::simd<float, 32> v492_data;
              v492_data.copy_from(r1 + (2592));
              (v492_data + (v474_data * v70_data)).copy_to(r1 + (2592));
              tensorforge::intel_esimd::simd<float, 32> v497_data;
              v497_data.copy_from(r1 + (3424));
              (v497_data + (v474_data * v75_data)).copy_to(r1 + (3424));
              tensorforge::intel_esimd::simd<float, 32> v502_data;
              v502_data.copy_from(r1 + (4256));
              (v502_data + (v474_data * v80_data)).copy_to(r1 + (4256));
              tensorforge::intel_esimd::simd<float, 32> v504_data;
              v504_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 32> v507_data;
              v507_data.copy_from(r1 + (160));
              (v507_data + (v504_data * v55_data)).copy_to(r1 + (160));
              tensorforge::intel_esimd::simd<float, 32> v512_data;
              v512_data.copy_from(r1 + (992));
              (v512_data + (v504_data * v60_data)).copy_to(r1 + (992));
              tensorforge::intel_esimd::simd<float, 32> v517_data;
              v517_data.copy_from(r1 + (1824));
              (v517_data + (v504_data * v65_data)).copy_to(r1 + (1824));
              tensorforge::intel_esimd::simd<float, 32> v522_data;
              v522_data.copy_from(r1 + (2656));
              (v522_data + (v504_data * v70_data)).copy_to(r1 + (2656));
              tensorforge::intel_esimd::simd<float, 32> v527_data;
              v527_data.copy_from(r1 + (3488));
              (v527_data + (v504_data * v75_data)).copy_to(r1 + (3488));
              tensorforge::intel_esimd::simd<float, 32> v532_data;
              v532_data.copy_from(r1 + (4320));
              (v532_data + (v504_data * v80_data)).copy_to(r1 + (4320));
              tensorforge::intel_esimd::simd<float, 32> v534_data;
              v534_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 32> v537_data;
              v537_data.copy_from(r1 + (224));
              (v537_data + (v534_data * v55_data)).copy_to(r1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v542_data;
              v542_data.copy_from(r1 + (1056));
              (v542_data + (v534_data * v60_data)).copy_to(r1 + (1056));
              tensorforge::intel_esimd::simd<float, 32> v547_data;
              v547_data.copy_from(r1 + (1888));
              (v547_data + (v534_data * v65_data)).copy_to(r1 + (1888));
              tensorforge::intel_esimd::simd<float, 32> v552_data;
              v552_data.copy_from(r1 + (2720));
              (v552_data + (v534_data * v70_data)).copy_to(r1 + (2720));
              tensorforge::intel_esimd::simd<float, 32> v557_data;
              v557_data.copy_from(r1 + (3552));
              (v557_data + (v534_data * v75_data)).copy_to(r1 + (3552));
              tensorforge::intel_esimd::simd<float, 32> v562_data;
              v562_data.copy_from(r1 + (4384));
              (v562_data + (v534_data * v80_data)).copy_to(r1 + (4384));
              tensorforge::intel_esimd::simd<float, 32> v564_data;
              v564_data.copy_from(r0 + (288));
              tensorforge::intel_esimd::simd<float, 32> v567_data;
              v567_data.copy_from(r1 + (288));
              (v567_data + (v564_data * v55_data)).copy_to(r1 + (288));
              tensorforge::intel_esimd::simd<float, 32> v572_data;
              v572_data.copy_from(r1 + (1120));
              (v572_data + (v564_data * v60_data)).copy_to(r1 + (1120));
              tensorforge::intel_esimd::simd<float, 32> v577_data;
              v577_data.copy_from(r1 + (1952));
              (v577_data + (v564_data * v65_data)).copy_to(r1 + (1952));
              tensorforge::intel_esimd::simd<float, 32> v582_data;
              v582_data.copy_from(r1 + (2784));
              (v582_data + (v564_data * v70_data)).copy_to(r1 + (2784));
              tensorforge::intel_esimd::simd<float, 32> v587_data;
              v587_data.copy_from(r1 + (3616));
              (v587_data + (v564_data * v75_data)).copy_to(r1 + (3616));
              tensorforge::intel_esimd::simd<float, 32> v592_data;
              v592_data.copy_from(r1 + (4448));
              (v592_data + (v564_data * v80_data)).copy_to(r1 + (4448));
              tensorforge::intel_esimd::simd<float, 32> v594_data;
              v594_data.copy_from(r0 + (352));
              tensorforge::intel_esimd::simd<float, 32> v597_data;
              v597_data.copy_from(r1 + (352));
              (v597_data + (v594_data * v55_data)).copy_to(r1 + (352));
              tensorforge::intel_esimd::simd<float, 32> v602_data;
              v602_data.copy_from(r1 + (1184));
              (v602_data + (v594_data * v60_data)).copy_to(r1 + (1184));
              tensorforge::intel_esimd::simd<float, 32> v607_data;
              v607_data.copy_from(r1 + (2016));
              (v607_data + (v594_data * v65_data)).copy_to(r1 + (2016));
              tensorforge::intel_esimd::simd<float, 32> v612_data;
              v612_data.copy_from(r1 + (2848));
              (v612_data + (v594_data * v70_data)).copy_to(r1 + (2848));
              tensorforge::intel_esimd::simd<float, 32> v617_data;
              v617_data.copy_from(r1 + (3680));
              (v617_data + (v594_data * v75_data)).copy_to(r1 + (3680));
              tensorforge::intel_esimd::simd<float, 32> v622_data;
              v622_data.copy_from(r1 + (4512));
              (v622_data + (v594_data * v80_data)).copy_to(r1 + (4512));
              tensorforge::intel_esimd::simd<float, 32> v624_data;
              v624_data.copy_from(r0 + (416));
              tensorforge::intel_esimd::simd<float, 32> v627_data;
              v627_data.copy_from(r1 + (416));
              (v627_data + (v624_data * v55_data)).copy_to(r1 + (416));
              tensorforge::intel_esimd::simd<float, 32> v632_data;
              v632_data.copy_from(r1 + (1248));
              (v632_data + (v624_data * v60_data)).copy_to(r1 + (1248));
              tensorforge::intel_esimd::simd<float, 32> v637_data;
              v637_data.copy_from(r1 + (2080));
              (v637_data + (v624_data * v65_data)).copy_to(r1 + (2080));
              tensorforge::intel_esimd::simd<float, 32> v642_data;
              v642_data.copy_from(r1 + (2912));
              (v642_data + (v624_data * v70_data)).copy_to(r1 + (2912));
              tensorforge::intel_esimd::simd<float, 32> v647_data;
              v647_data.copy_from(r1 + (3744));
              (v647_data + (v624_data * v75_data)).copy_to(r1 + (3744));
              tensorforge::intel_esimd::simd<float, 32> v652_data;
              v652_data.copy_from(r1 + (4576));
              (v652_data + (v624_data * v80_data)).copy_to(r1 + (4576));
              tensorforge::intel_esimd::simd<float, 32> v654_data;
              v654_data.copy_from(r0 + (480));
              tensorforge::intel_esimd::simd<float, 32> v657_data;
              v657_data.copy_from(r1 + (480));
              (v657_data + (v654_data * v55_data)).copy_to(r1 + (480));
              tensorforge::intel_esimd::simd<float, 32> v662_data;
              v662_data.copy_from(r1 + (1312));
              (v662_data + (v654_data * v60_data)).copy_to(r1 + (1312));
              tensorforge::intel_esimd::simd<float, 32> v667_data;
              v667_data.copy_from(r1 + (2144));
              (v667_data + (v654_data * v65_data)).copy_to(r1 + (2144));
              tensorforge::intel_esimd::simd<float, 32> v672_data;
              v672_data.copy_from(r1 + (2976));
              (v672_data + (v654_data * v70_data)).copy_to(r1 + (2976));
              tensorforge::intel_esimd::simd<float, 32> v677_data;
              v677_data.copy_from(r1 + (3808));
              (v677_data + (v654_data * v75_data)).copy_to(r1 + (3808));
              tensorforge::intel_esimd::simd<float, 32> v682_data;
              v682_data.copy_from(r1 + (4640));
              (v682_data + (v654_data * v80_data)).copy_to(r1 + (4640));
              tensorforge::intel_esimd::simd<float, 32> v684_data;
              v684_data.copy_from(r0 + (544));
              tensorforge::intel_esimd::simd<float, 32> v687_data;
              v687_data.copy_from(r1 + (544));
              (v687_data + (v684_data * v55_data)).copy_to(r1 + (544));
              tensorforge::intel_esimd::simd<float, 32> v692_data;
              v692_data.copy_from(r1 + (1376));
              (v692_data + (v684_data * v60_data)).copy_to(r1 + (1376));
              tensorforge::intel_esimd::simd<float, 32> v697_data;
              v697_data.copy_from(r1 + (2208));
              (v697_data + (v684_data * v65_data)).copy_to(r1 + (2208));
              tensorforge::intel_esimd::simd<float, 32> v702_data;
              v702_data.copy_from(r1 + (3040));
              (v702_data + (v684_data * v70_data)).copy_to(r1 + (3040));
              tensorforge::intel_esimd::simd<float, 32> v707_data;
              v707_data.copy_from(r1 + (3872));
              (v707_data + (v684_data * v75_data)).copy_to(r1 + (3872));
              tensorforge::intel_esimd::simd<float, 32> v712_data;
              v712_data.copy_from(r1 + (4704));
              (v712_data + (v684_data * v80_data)).copy_to(r1 + (4704));
              tensorforge::intel_esimd::simd<float, 32> v714_data;
              v714_data.copy_from(r0 + (608));
              tensorforge::intel_esimd::simd<float, 32> v717_data;
              v717_data.copy_from(r1 + (608));
              (v717_data + (v714_data * v55_data)).copy_to(r1 + (608));
              tensorforge::intel_esimd::simd<float, 32> v722_data;
              v722_data.copy_from(r1 + (1440));
              (v722_data + (v714_data * v60_data)).copy_to(r1 + (1440));
              tensorforge::intel_esimd::simd<float, 32> v727_data;
              v727_data.copy_from(r1 + (2272));
              (v727_data + (v714_data * v65_data)).copy_to(r1 + (2272));
              tensorforge::intel_esimd::simd<float, 32> v732_data;
              v732_data.copy_from(r1 + (3104));
              (v732_data + (v714_data * v70_data)).copy_to(r1 + (3104));
              tensorforge::intel_esimd::simd<float, 32> v737_data;
              v737_data.copy_from(r1 + (3936));
              (v737_data + (v714_data * v75_data)).copy_to(r1 + (3936));
              tensorforge::intel_esimd::simd<float, 32> v742_data;
              v742_data.copy_from(r1 + (4768));
              (v742_data + (v714_data * v80_data)).copy_to(r1 + (4768));
              tensorforge::intel_esimd::simd<float, 32> v744_data;
              v744_data.copy_from(r0 + (672));
              tensorforge::intel_esimd::simd<float, 32> v747_data;
              v747_data.copy_from(r1 + (672));
              (v747_data + (v744_data * v55_data)).copy_to(r1 + (672));
              tensorforge::intel_esimd::simd<float, 32> v752_data;
              v752_data.copy_from(r1 + (1504));
              (v752_data + (v744_data * v60_data)).copy_to(r1 + (1504));
              tensorforge::intel_esimd::simd<float, 32> v757_data;
              v757_data.copy_from(r1 + (2336));
              (v757_data + (v744_data * v65_data)).copy_to(r1 + (2336));
              tensorforge::intel_esimd::simd<float, 32> v762_data;
              v762_data.copy_from(r1 + (3168));
              (v762_data + (v744_data * v70_data)).copy_to(r1 + (3168));
              tensorforge::intel_esimd::simd<float, 32> v767_data;
              v767_data.copy_from(r1 + (4000));
              (v767_data + (v744_data * v75_data)).copy_to(r1 + (4000));
              tensorforge::intel_esimd::simd<float, 32> v772_data;
              v772_data.copy_from(r1 + (4832));
              (v772_data + (v744_data * v80_data)).copy_to(r1 + (4832));
              tensorforge::intel_esimd::simd<float, 32> v774_data;
              v774_data.copy_from(r0 + (736));
              tensorforge::intel_esimd::simd<float, 32> v777_data;
              v777_data.copy_from(r1 + (736));
              (v777_data + (v774_data * v55_data)).copy_to(r1 + (736));
              tensorforge::intel_esimd::simd<float, 32> v782_data;
              v782_data.copy_from(r1 + (1568));
              (v782_data + (v774_data * v60_data)).copy_to(r1 + (1568));
              tensorforge::intel_esimd::simd<float, 32> v787_data;
              v787_data.copy_from(r1 + (2400));
              (v787_data + (v774_data * v65_data)).copy_to(r1 + (2400));
              tensorforge::intel_esimd::simd<float, 32> v792_data;
              v792_data.copy_from(r1 + (3232));
              (v792_data + (v774_data * v70_data)).copy_to(r1 + (3232));
              tensorforge::intel_esimd::simd<float, 32> v797_data;
              v797_data.copy_from(r1 + (4064));
              (v797_data + (v774_data * v75_data)).copy_to(r1 + (4064));
              tensorforge::intel_esimd::simd<float, 32> v802_data;
              v802_data.copy_from(r1 + (4896));
              (v802_data + (v774_data * v80_data)).copy_to(r1 + (4896));
              tensorforge::intel_esimd::simd<float, 32> v804_data;
              v804_data.copy_from(r0 + (800));
              tensorforge::intel_esimd::simd<float, 32> v807_data;
              v807_data.copy_from(r1 + (800));
              (v807_data + (v804_data * v55_data)).copy_to(r1 + (800));
              tensorforge::intel_esimd::simd<float, 32> v812_data;
              v812_data.copy_from(r1 + (1632));
              (v812_data + (v804_data * v60_data)).copy_to(r1 + (1632));
              tensorforge::intel_esimd::simd<float, 32> v817_data;
              v817_data.copy_from(r1 + (2464));
              (v817_data + (v804_data * v65_data)).copy_to(r1 + (2464));
              tensorforge::intel_esimd::simd<float, 32> v822_data;
              v822_data.copy_from(r1 + (3296));
              (v822_data + (v804_data * v70_data)).copy_to(r1 + (3296));
              tensorforge::intel_esimd::simd<float, 32> v827_data;
              v827_data.copy_from(r1 + (4128));
              (v827_data + (v804_data * v75_data)).copy_to(r1 + (4128));
              tensorforge::intel_esimd::simd<float, 32> v832_data;
              v832_data.copy_from(r1 + (4960));
              (v832_data + (v804_data * v80_data)).copy_to(r1 + (4960));
              // wait(r2 = load{g>r}(glb_m2););
              float r3[384]{};
              // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(20, 35), (0, 1), (0, 6)] []
              float ir3[384]{};
              tensorforge::intel_esimd::simd<float, 12> v836_data;
              v836_data.copy_from(r1 + (788));
              tensorforge::intel_esimd::simd<float, 12> v837_data;
              v837_data.copy_from(ir3 + (20));
              (v837_data + v836_data).copy_to(ir3 + (20));
              tensorforge::intel_esimd::simd<float, 12> v839_data;
              v839_data.copy_from(r1 + (1620));
              tensorforge::intel_esimd::simd<float, 12> v840_data;
              v840_data.copy_from(ir3 + (84));
              (v840_data + v839_data).copy_to(ir3 + (84));
              tensorforge::intel_esimd::simd<float, 12> v842_data;
              v842_data.copy_from(r1 + (2452));
              tensorforge::intel_esimd::simd<float, 12> v843_data;
              v843_data.copy_from(ir3 + (148));
              (v843_data + v842_data).copy_to(ir3 + (148));
              tensorforge::intel_esimd::simd<float, 12> v845_data;
              v845_data.copy_from(r1 + (3284));
              tensorforge::intel_esimd::simd<float, 12> v846_data;
              v846_data.copy_from(ir3 + (212));
              (v846_data + v845_data).copy_to(ir3 + (212));
              tensorforge::intel_esimd::simd<float, 12> v848_data;
              v848_data.copy_from(r1 + (4116));
              tensorforge::intel_esimd::simd<float, 12> v849_data;
              v849_data.copy_from(ir3 + (276));
              (v849_data + v848_data).copy_to(ir3 + (276));
              tensorforge::intel_esimd::simd<float, 12> v851_data;
              v851_data.copy_from(r1 + (4948));
              tensorforge::intel_esimd::simd<float, 12> v852_data;
              v852_data.copy_from(ir3 + (340));
              (v852_data + v851_data).copy_to(ir3 + (340));
              tensorforge::intel_esimd::simd<float, 3> v854_data;
              v854_data.copy_from(r1 + (800));
              tensorforge::intel_esimd::simd<float, 3> v855_data;
              v855_data.copy_from(ir3 + (32));
              (v855_data + v854_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 3> v857_data;
              v857_data.copy_from(r1 + (1632));
              tensorforge::intel_esimd::simd<float, 3> v858_data;
              v858_data.copy_from(ir3 + (96));
              (v858_data + v857_data).copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 3> v860_data;
              v860_data.copy_from(r1 + (2464));
              tensorforge::intel_esimd::simd<float, 3> v861_data;
              v861_data.copy_from(ir3 + (160));
              (v861_data + v860_data).copy_to(ir3 + (160));
              tensorforge::intel_esimd::simd<float, 3> v863_data;
              v863_data.copy_from(r1 + (3296));
              tensorforge::intel_esimd::simd<float, 3> v864_data;
              v864_data.copy_from(ir3 + (224));
              (v864_data + v863_data).copy_to(ir3 + (224));
              tensorforge::intel_esimd::simd<float, 3> v866_data;
              v866_data.copy_from(r1 + (4128));
              tensorforge::intel_esimd::simd<float, 3> v867_data;
              v867_data.copy_from(ir3 + (288));
              (v867_data + v866_data).copy_to(ir3 + (288));
              tensorforge::intel_esimd::simd<float, 3> v869_data;
              v869_data.copy_from(r1 + (4960));
              tensorforge::intel_esimd::simd<float, 3> v870_data;
              v870_data.copy_from(ir3 + (352));
              (v870_data + v869_data).copy_to(ir3 + (352));
              #pragma unroll
              for (int32_t v872_n1 = 0; v872_n1 < 1; ++v872_n1) {
                int32_t v876_a = 20 + (v872_n1 * 64);
                #pragma unroll
                for (int32_t v873_n2 = 0; v873_n2 < 6; ++v873_n2) {
                  int32_t v875_a = v873_n2 * 64;
                  tensorforge::intel_esimd::simd<float, 12> v878_data;
                  v878_data.copy_from(ir3 + ((v876_a + v875_a)));
                  tensorforge::intel_esimd::simd<float, 12> v883_data;
                  v883_data.copy_from(r2 + ((v876_a + v875_a)));
                  (v883_data + v878_data).copy_to(r3 + ((v876_a + v875_a)));
                }
              }
              #pragma unroll
              for (int32_t v889_n1 = 0; v889_n1 < 1; ++v889_n1) {
                int32_t v893_a = 32 + (v889_n1 * 64);
                #pragma unroll
                for (int32_t v890_n2 = 0; v890_n2 < 6; ++v890_n2) {
                  int32_t v892_a = v890_n2 * 64;
                  tensorforge::intel_esimd::simd<float, 3> v895_data;
                  v895_data.copy_from(ir3 + ((v893_a + v892_a)));
                  tensorforge::intel_esimd::simd<float, 3> v900_data;
                  v900_data.copy_from(r2 + ((v893_a + v892_a)));
                  (v900_data + v895_data).copy_to(r3 + ((v893_a + v892_a)));
                }
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v906_i1 = 0; v906_i1 < 1; ++v906_i1) {
                int32_t v910_a = 20 + (v906_i1 * 64);
                int32_t v919_a = 20_i32 + ((v906_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v907_i2 = 0; v907_i2 < 6; ++v907_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v912_data;
                  v912_data.copy_from(r3 + ((v910_a + (v907_i2 * 64))));
                  v912_data.copy_to(glb_m2 + ((v919_a + (v907_i2 * 832))));
                }
              }
              #pragma unroll
              for (int32_t v921_i1 = 0; v921_i1 < 1; ++v921_i1) {
                int32_t v925_a = 32 + (v921_i1 * 64);
                int32_t v934_a = 32_i32 + ((v921_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v922_i2 = 0; v922_i2 < 6; ++v922_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v927_data;
                  v927_data.copy_from(r3 + ((v925_a + (v922_i2 * 64))));
                  v927_data.copy_to(glb_m2 + ((v934_a + (v922_i2 * 832))));
                }
              }
            }
          }
        }
      });
    }
  });
}

