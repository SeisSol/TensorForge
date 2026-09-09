// === base name ===
kernel_2b2b528d1715574e

// === header ===
void launcher_kernel_2b2b528d1715574e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_2b2b528d1715574e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2b2b528d1715574e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2b2b528d1715574e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1024, cgh); {
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
          float* localShrMem0 = &totalShrMem[64 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[48];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 46 + 0 + m2_extraOffset];
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
              tensorforge::intel_esimd::simd<float, 32> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v21_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 14) {
                tensorforge::intel_esimd::simd<float, 16> v22_ld;
                v22_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 32));
                v22_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 32));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v41_acc{};
              tensorforge::intel_esimd::simd<float, 16> v42_lin;
              v42_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v43_bc = v42_lin[0];
              v41_acc += (v43_bc * v25_data);
              float v45_bc = v42_lin[1];
              v41_acc += (v45_bc * v26_data);
              float v47_bc = v42_lin[2];
              v41_acc += (v47_bc * v27_data);
              float v49_bc = v42_lin[3];
              v41_acc += (v49_bc * v28_data);
              float v51_bc = v42_lin[4];
              v41_acc += (v51_bc * v29_data);
              float v53_bc = v42_lin[5];
              v41_acc += (v53_bc * v30_data);
              float v55_bc = v42_lin[6];
              v41_acc += (v55_bc * v31_data);
              float v57_bc = v42_lin[7];
              v41_acc += (v57_bc * v32_data);
              float v59_bc = v42_lin[8];
              v41_acc += (v59_bc * v33_data);
              float v61_bc = v42_lin[9];
              v41_acc += (v61_bc * v34_data);
              float v63_bc = v42_lin[10];
              v41_acc += (v63_bc * v35_data);
              float v65_bc = v42_lin[11];
              v41_acc += (v65_bc * v36_data);
              float v67_bc = v42_lin[12];
              v41_acc += (v67_bc * v37_data);
              float v69_bc = v42_lin[13];
              v41_acc += (v69_bc * v38_data);
              float v71_bc = v42_lin[14];
              v41_acc += (v71_bc * v39_data);
              float v73_bc = v42_lin[15];
              v41_acc += (v73_bc * v40_data);
              v41_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v75_acc{};
              v75_acc += (v43_bc * v25_data);
              v75_acc += (v45_bc * v26_data);
              v75_acc += (v47_bc * v27_data);
              v75_acc += (v49_bc * v28_data);
              v75_acc += (v51_bc * v29_data);
              v75_acc += (v53_bc * v30_data);
              v75_acc += (v55_bc * v31_data);
              v75_acc += (v57_bc * v32_data);
              v75_acc += (v59_bc * v33_data);
              v75_acc += (v61_bc * v34_data);
              v75_acc += (v63_bc * v35_data);
              v75_acc += (v65_bc * v36_data);
              v75_acc += (v67_bc * v37_data);
              v75_acc += (v69_bc * v38_data);
              v75_acc += (v71_bc * v39_data);
              v75_acc += (v73_bc * v40_data);
              v75_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v109_acc{};
              v109_acc += (v43_bc * v25_data);
              v109_acc += (v45_bc * v26_data);
              v109_acc += (v47_bc * v27_data);
              v109_acc += (v49_bc * v28_data);
              v109_acc += (v51_bc * v29_data);
              v109_acc += (v53_bc * v30_data);
              v109_acc += (v55_bc * v31_data);
              v109_acc += (v57_bc * v32_data);
              v109_acc += (v59_bc * v33_data);
              v109_acc += (v61_bc * v34_data);
              v109_acc += (v63_bc * v35_data);
              v109_acc += (v65_bc * v36_data);
              v109_acc += (v67_bc * v37_data);
              v109_acc += (v69_bc * v38_data);
              v109_acc += (v71_bc * v39_data);
              v109_acc += (v73_bc * v40_data);
              v109_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v143_acc{};
              v143_acc += (v43_bc * v25_data);
              v143_acc += (v45_bc * v26_data);
              v143_acc += (v47_bc * v27_data);
              v143_acc += (v49_bc * v28_data);
              v143_acc += (v51_bc * v29_data);
              v143_acc += (v53_bc * v30_data);
              v143_acc += (v55_bc * v31_data);
              v143_acc += (v57_bc * v32_data);
              v143_acc += (v59_bc * v33_data);
              v143_acc += (v61_bc * v34_data);
              v143_acc += (v63_bc * v35_data);
              v143_acc += (v65_bc * v36_data);
              v143_acc += (v67_bc * v37_data);
              v143_acc += (v69_bc * v38_data);
              v143_acc += (v71_bc * v39_data);
              v143_acc += (v73_bc * v40_data);
              v143_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v177_acc{};
              v177_acc += (v43_bc * v25_data);
              v177_acc += (v45_bc * v26_data);
              v177_acc += (v47_bc * v27_data);
              v177_acc += (v49_bc * v28_data);
              v177_acc += (v51_bc * v29_data);
              v177_acc += (v53_bc * v30_data);
              v177_acc += (v55_bc * v31_data);
              v177_acc += (v57_bc * v32_data);
              v177_acc += (v59_bc * v33_data);
              v177_acc += (v61_bc * v34_data);
              v177_acc += (v63_bc * v35_data);
              v177_acc += (v65_bc * v36_data);
              v177_acc += (v67_bc * v37_data);
              v177_acc += (v69_bc * v38_data);
              v177_acc += (v71_bc * v39_data);
              v177_acc += (v73_bc * v40_data);
              v177_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v211_acc{};
              v211_acc += (v43_bc * v25_data);
              v211_acc += (v45_bc * v26_data);
              v211_acc += (v47_bc * v27_data);
              v211_acc += (v49_bc * v28_data);
              v211_acc += (v51_bc * v29_data);
              v211_acc += (v53_bc * v30_data);
              v211_acc += (v55_bc * v31_data);
              v211_acc += (v57_bc * v32_data);
              v211_acc += (v59_bc * v33_data);
              v211_acc += (v61_bc * v34_data);
              v211_acc += (v63_bc * v35_data);
              v211_acc += (v65_bc * v36_data);
              v211_acc += (v67_bc * v37_data);
              v211_acc += (v69_bc * v38_data);
              v211_acc += (v71_bc * v39_data);
              v211_acc += (v73_bc * v40_data);
              v211_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v245_acc{};
              v245_acc += (v43_bc * v25_data);
              v245_acc += (v45_bc * v26_data);
              v245_acc += (v47_bc * v27_data);
              v245_acc += (v49_bc * v28_data);
              v245_acc += (v51_bc * v29_data);
              v245_acc += (v53_bc * v30_data);
              v245_acc += (v55_bc * v31_data);
              v245_acc += (v57_bc * v32_data);
              v245_acc += (v59_bc * v33_data);
              v245_acc += (v61_bc * v34_data);
              v245_acc += (v63_bc * v35_data);
              v245_acc += (v65_bc * v36_data);
              v245_acc += (v67_bc * v37_data);
              v245_acc += (v69_bc * v38_data);
              v245_acc += (v71_bc * v39_data);
              v245_acc += (v73_bc * v40_data);
              v245_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v279_acc{};
              v279_acc += (v43_bc * v25_data);
              v279_acc += (v45_bc * v26_data);
              v279_acc += (v47_bc * v27_data);
              v279_acc += (v49_bc * v28_data);
              v279_acc += (v51_bc * v29_data);
              v279_acc += (v53_bc * v30_data);
              v279_acc += (v55_bc * v31_data);
              v279_acc += (v57_bc * v32_data);
              v279_acc += (v59_bc * v33_data);
              v279_acc += (v61_bc * v34_data);
              v279_acc += (v63_bc * v35_data);
              v279_acc += (v65_bc * v36_data);
              v279_acc += (v67_bc * v37_data);
              v279_acc += (v69_bc * v38_data);
              v279_acc += (v71_bc * v39_data);
              v279_acc += (v73_bc * v40_data);
              v279_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v313_acc{};
              v313_acc += (v43_bc * v25_data);
              v313_acc += (v45_bc * v26_data);
              v313_acc += (v47_bc * v27_data);
              v313_acc += (v49_bc * v28_data);
              v313_acc += (v51_bc * v29_data);
              v313_acc += (v53_bc * v30_data);
              v313_acc += (v55_bc * v31_data);
              v313_acc += (v57_bc * v32_data);
              v313_acc += (v59_bc * v33_data);
              v313_acc += (v61_bc * v34_data);
              v313_acc += (v63_bc * v35_data);
              v313_acc += (v65_bc * v36_data);
              v313_acc += (v67_bc * v37_data);
              v313_acc += (v69_bc * v38_data);
              v313_acc += (v71_bc * v39_data);
              v313_acc += (v73_bc * v40_data);
              v313_acc.copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 16> v347_acc{};
              v347_acc += (v43_bc * v25_data);
              v347_acc += (v45_bc * v26_data);
              v347_acc += (v47_bc * v27_data);
              v347_acc += (v49_bc * v28_data);
              v347_acc += (v51_bc * v29_data);
              v347_acc += (v53_bc * v30_data);
              v347_acc += (v55_bc * v31_data);
              v347_acc += (v57_bc * v32_data);
              v347_acc += (v59_bc * v33_data);
              v347_acc += (v61_bc * v34_data);
              v347_acc += (v63_bc * v35_data);
              v347_acc += (v65_bc * v36_data);
              v347_acc += (v67_bc * v37_data);
              v347_acc += (v69_bc * v38_data);
              v347_acc += (v71_bc * v39_data);
              v347_acc += (v73_bc * v40_data);
              v347_acc.copy_to(ir1 + (144));
              tensorforge::intel_esimd::simd<float, 16> v381_acc{};
              v381_acc += (v43_bc * v25_data);
              v381_acc += (v45_bc * v26_data);
              v381_acc += (v47_bc * v27_data);
              v381_acc += (v49_bc * v28_data);
              v381_acc += (v51_bc * v29_data);
              v381_acc += (v53_bc * v30_data);
              v381_acc += (v55_bc * v31_data);
              v381_acc += (v57_bc * v32_data);
              v381_acc += (v59_bc * v33_data);
              v381_acc += (v61_bc * v34_data);
              v381_acc += (v63_bc * v35_data);
              v381_acc += (v65_bc * v36_data);
              v381_acc += (v67_bc * v37_data);
              v381_acc += (v69_bc * v38_data);
              v381_acc += (v71_bc * v39_data);
              v381_acc += (v73_bc * v40_data);
              v381_acc.copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 16> v415_acc{};
              v415_acc += (v43_bc * v25_data);
              v415_acc += (v45_bc * v26_data);
              v415_acc += (v47_bc * v27_data);
              v415_acc += (v49_bc * v28_data);
              v415_acc += (v51_bc * v29_data);
              v415_acc += (v53_bc * v30_data);
              v415_acc += (v55_bc * v31_data);
              v415_acc += (v57_bc * v32_data);
              v415_acc += (v59_bc * v33_data);
              v415_acc += (v61_bc * v34_data);
              v415_acc += (v63_bc * v35_data);
              v415_acc += (v65_bc * v36_data);
              v415_acc += (v67_bc * v37_data);
              v415_acc += (v69_bc * v38_data);
              v415_acc += (v71_bc * v39_data);
              v415_acc += (v73_bc * v40_data);
              v415_acc.copy_to(ir1 + (176));
              tensorforge::intel_esimd::simd<float, 16> v449_acc{};
              v449_acc += (v43_bc * v25_data);
              v449_acc += (v45_bc * v26_data);
              v449_acc += (v47_bc * v27_data);
              v449_acc += (v49_bc * v28_data);
              v449_acc += (v51_bc * v29_data);
              v449_acc += (v53_bc * v30_data);
              v449_acc += (v55_bc * v31_data);
              v449_acc += (v57_bc * v32_data);
              v449_acc += (v59_bc * v33_data);
              v449_acc += (v61_bc * v34_data);
              v449_acc += (v63_bc * v35_data);
              v449_acc += (v65_bc * v36_data);
              v449_acc += (v67_bc * v37_data);
              v449_acc += (v69_bc * v38_data);
              v449_acc += (v71_bc * v39_data);
              v449_acc += (v73_bc * v40_data);
              v449_acc.copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 16> v483_acc{};
              v483_acc += (v43_bc * v25_data);
              v483_acc += (v45_bc * v26_data);
              v483_acc += (v47_bc * v27_data);
              v483_acc += (v49_bc * v28_data);
              v483_acc += (v51_bc * v29_data);
              v483_acc += (v53_bc * v30_data);
              v483_acc += (v55_bc * v31_data);
              v483_acc += (v57_bc * v32_data);
              v483_acc += (v59_bc * v33_data);
              v483_acc += (v61_bc * v34_data);
              v483_acc += (v63_bc * v35_data);
              v483_acc += (v65_bc * v36_data);
              v483_acc += (v67_bc * v37_data);
              v483_acc += (v69_bc * v38_data);
              v483_acc += (v71_bc * v39_data);
              v483_acc += (v73_bc * v40_data);
              v483_acc.copy_to(ir1 + (208));
              tensorforge::intel_esimd::simd<float, 16> v517_acc{};
              v517_acc += (v43_bc * v25_data);
              v517_acc += (v45_bc * v26_data);
              v517_acc += (v47_bc * v27_data);
              v517_acc += (v49_bc * v28_data);
              v517_acc += (v51_bc * v29_data);
              v517_acc += (v53_bc * v30_data);
              v517_acc += (v55_bc * v31_data);
              v517_acc += (v57_bc * v32_data);
              v517_acc += (v59_bc * v33_data);
              v517_acc += (v61_bc * v34_data);
              v517_acc += (v63_bc * v35_data);
              v517_acc += (v65_bc * v36_data);
              v517_acc += (v67_bc * v37_data);
              v517_acc += (v69_bc * v38_data);
              v517_acc += (v71_bc * v39_data);
              v517_acc += (v73_bc * v40_data);
              v517_acc.copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 16> v551_acc{};
              v551_acc += (v43_bc * v25_data);
              v551_acc += (v45_bc * v26_data);
              v551_acc += (v47_bc * v27_data);
              v551_acc += (v49_bc * v28_data);
              v551_acc += (v51_bc * v29_data);
              v551_acc += (v53_bc * v30_data);
              v551_acc += (v55_bc * v31_data);
              v551_acc += (v57_bc * v32_data);
              v551_acc += (v59_bc * v33_data);
              v551_acc += (v61_bc * v34_data);
              v551_acc += (v63_bc * v35_data);
              v551_acc += (v65_bc * v36_data);
              v551_acc += (v67_bc * v37_data);
              v551_acc += (v69_bc * v38_data);
              v551_acc += (v71_bc * v39_data);
              v551_acc += (v73_bc * v40_data);
              v551_acc.copy_to(ir1 + (240));
              #pragma unroll
              for (int32_t v585_n0 = 0; v585_n0 < 1; ++v585_n0) {
                int32_t v587_a = v585_n0 * 16;
                #pragma unroll
                for (int32_t v586_n1 = 0; v586_n1 < 16; ++v586_n1) {
                  int32_t v589_a = v587_a + (v586_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v590_data;
                  v590_data.copy_from(ir1 + (v589_a));
                  v590_data.copy_to(r1 + (v589_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v594_i0 = 0; v594_i0 < 1; ++v594_i0) {
                int32_t v596_a = v594_i0 * 16;
                #pragma unroll
                for (int32_t v595_i1 = 0; v595_i1 < 16; ++v595_i1) {
                  int32_t v598_a = v596_a + (v595_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v599_data;
                  v599_data.copy_from(r1 + (v598_a));
                  v599_data.copy_to(glb_m0 + (v598_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

