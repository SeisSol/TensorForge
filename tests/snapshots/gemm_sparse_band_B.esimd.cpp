// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_30948bd44e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_30948bd44e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1024, cgh); {
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
          float* localShrMem0 = &totalShrMem[64 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[48];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 46 + 0 + m2_extraOffset];
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
              tensorforge::intel_esimd::simd<float, 32> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v17_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 14) {
                tensorforge::intel_esimd::simd<float, 16> v18_ld;
                v18_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 32));
                v18_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 32));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v37_acc{};
              tensorforge::intel_esimd::simd<float, 16> v38_lin;
              v38_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v39_bc = v38_lin[0];
              v37_acc += (v39_bc * v21_data);
              float v41_bc = v38_lin[1];
              v37_acc += (v41_bc * v22_data);
              float v43_bc = v38_lin[2];
              v37_acc += (v43_bc * v23_data);
              float v45_bc = v38_lin[3];
              v37_acc += (v45_bc * v24_data);
              float v47_bc = v38_lin[4];
              v37_acc += (v47_bc * v25_data);
              float v49_bc = v38_lin[5];
              v37_acc += (v49_bc * v26_data);
              float v51_bc = v38_lin[6];
              v37_acc += (v51_bc * v27_data);
              float v53_bc = v38_lin[7];
              v37_acc += (v53_bc * v28_data);
              float v55_bc = v38_lin[8];
              v37_acc += (v55_bc * v29_data);
              float v57_bc = v38_lin[9];
              v37_acc += (v57_bc * v30_data);
              float v59_bc = v38_lin[10];
              v37_acc += (v59_bc * v31_data);
              float v61_bc = v38_lin[11];
              v37_acc += (v61_bc * v32_data);
              float v63_bc = v38_lin[12];
              v37_acc += (v63_bc * v33_data);
              float v65_bc = v38_lin[13];
              v37_acc += (v65_bc * v34_data);
              float v67_bc = v38_lin[14];
              v37_acc += (v67_bc * v35_data);
              float v69_bc = v38_lin[15];
              v37_acc += (v69_bc * v36_data);
              v37_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v71_acc{};
              v71_acc += (v39_bc * v21_data);
              v71_acc += (v41_bc * v22_data);
              v71_acc += (v43_bc * v23_data);
              v71_acc += (v45_bc * v24_data);
              v71_acc += (v47_bc * v25_data);
              v71_acc += (v49_bc * v26_data);
              v71_acc += (v51_bc * v27_data);
              v71_acc += (v53_bc * v28_data);
              v71_acc += (v55_bc * v29_data);
              v71_acc += (v57_bc * v30_data);
              v71_acc += (v59_bc * v31_data);
              v71_acc += (v61_bc * v32_data);
              v71_acc += (v63_bc * v33_data);
              v71_acc += (v65_bc * v34_data);
              v71_acc += (v67_bc * v35_data);
              v71_acc += (v69_bc * v36_data);
              v71_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v105_acc{};
              v105_acc += (v39_bc * v21_data);
              v105_acc += (v41_bc * v22_data);
              v105_acc += (v43_bc * v23_data);
              v105_acc += (v45_bc * v24_data);
              v105_acc += (v47_bc * v25_data);
              v105_acc += (v49_bc * v26_data);
              v105_acc += (v51_bc * v27_data);
              v105_acc += (v53_bc * v28_data);
              v105_acc += (v55_bc * v29_data);
              v105_acc += (v57_bc * v30_data);
              v105_acc += (v59_bc * v31_data);
              v105_acc += (v61_bc * v32_data);
              v105_acc += (v63_bc * v33_data);
              v105_acc += (v65_bc * v34_data);
              v105_acc += (v67_bc * v35_data);
              v105_acc += (v69_bc * v36_data);
              v105_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v139_acc{};
              v139_acc += (v39_bc * v21_data);
              v139_acc += (v41_bc * v22_data);
              v139_acc += (v43_bc * v23_data);
              v139_acc += (v45_bc * v24_data);
              v139_acc += (v47_bc * v25_data);
              v139_acc += (v49_bc * v26_data);
              v139_acc += (v51_bc * v27_data);
              v139_acc += (v53_bc * v28_data);
              v139_acc += (v55_bc * v29_data);
              v139_acc += (v57_bc * v30_data);
              v139_acc += (v59_bc * v31_data);
              v139_acc += (v61_bc * v32_data);
              v139_acc += (v63_bc * v33_data);
              v139_acc += (v65_bc * v34_data);
              v139_acc += (v67_bc * v35_data);
              v139_acc += (v69_bc * v36_data);
              v139_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              v173_acc += (v39_bc * v21_data);
              v173_acc += (v41_bc * v22_data);
              v173_acc += (v43_bc * v23_data);
              v173_acc += (v45_bc * v24_data);
              v173_acc += (v47_bc * v25_data);
              v173_acc += (v49_bc * v26_data);
              v173_acc += (v51_bc * v27_data);
              v173_acc += (v53_bc * v28_data);
              v173_acc += (v55_bc * v29_data);
              v173_acc += (v57_bc * v30_data);
              v173_acc += (v59_bc * v31_data);
              v173_acc += (v61_bc * v32_data);
              v173_acc += (v63_bc * v33_data);
              v173_acc += (v65_bc * v34_data);
              v173_acc += (v67_bc * v35_data);
              v173_acc += (v69_bc * v36_data);
              v173_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v207_acc{};
              v207_acc += (v39_bc * v21_data);
              v207_acc += (v41_bc * v22_data);
              v207_acc += (v43_bc * v23_data);
              v207_acc += (v45_bc * v24_data);
              v207_acc += (v47_bc * v25_data);
              v207_acc += (v49_bc * v26_data);
              v207_acc += (v51_bc * v27_data);
              v207_acc += (v53_bc * v28_data);
              v207_acc += (v55_bc * v29_data);
              v207_acc += (v57_bc * v30_data);
              v207_acc += (v59_bc * v31_data);
              v207_acc += (v61_bc * v32_data);
              v207_acc += (v63_bc * v33_data);
              v207_acc += (v65_bc * v34_data);
              v207_acc += (v67_bc * v35_data);
              v207_acc += (v69_bc * v36_data);
              v207_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v241_acc{};
              v241_acc += (v39_bc * v21_data);
              v241_acc += (v41_bc * v22_data);
              v241_acc += (v43_bc * v23_data);
              v241_acc += (v45_bc * v24_data);
              v241_acc += (v47_bc * v25_data);
              v241_acc += (v49_bc * v26_data);
              v241_acc += (v51_bc * v27_data);
              v241_acc += (v53_bc * v28_data);
              v241_acc += (v55_bc * v29_data);
              v241_acc += (v57_bc * v30_data);
              v241_acc += (v59_bc * v31_data);
              v241_acc += (v61_bc * v32_data);
              v241_acc += (v63_bc * v33_data);
              v241_acc += (v65_bc * v34_data);
              v241_acc += (v67_bc * v35_data);
              v241_acc += (v69_bc * v36_data);
              v241_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v275_acc{};
              v275_acc += (v39_bc * v21_data);
              v275_acc += (v41_bc * v22_data);
              v275_acc += (v43_bc * v23_data);
              v275_acc += (v45_bc * v24_data);
              v275_acc += (v47_bc * v25_data);
              v275_acc += (v49_bc * v26_data);
              v275_acc += (v51_bc * v27_data);
              v275_acc += (v53_bc * v28_data);
              v275_acc += (v55_bc * v29_data);
              v275_acc += (v57_bc * v30_data);
              v275_acc += (v59_bc * v31_data);
              v275_acc += (v61_bc * v32_data);
              v275_acc += (v63_bc * v33_data);
              v275_acc += (v65_bc * v34_data);
              v275_acc += (v67_bc * v35_data);
              v275_acc += (v69_bc * v36_data);
              v275_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v309_acc{};
              v309_acc += (v39_bc * v21_data);
              v309_acc += (v41_bc * v22_data);
              v309_acc += (v43_bc * v23_data);
              v309_acc += (v45_bc * v24_data);
              v309_acc += (v47_bc * v25_data);
              v309_acc += (v49_bc * v26_data);
              v309_acc += (v51_bc * v27_data);
              v309_acc += (v53_bc * v28_data);
              v309_acc += (v55_bc * v29_data);
              v309_acc += (v57_bc * v30_data);
              v309_acc += (v59_bc * v31_data);
              v309_acc += (v61_bc * v32_data);
              v309_acc += (v63_bc * v33_data);
              v309_acc += (v65_bc * v34_data);
              v309_acc += (v67_bc * v35_data);
              v309_acc += (v69_bc * v36_data);
              v309_acc.copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 16> v343_acc{};
              v343_acc += (v39_bc * v21_data);
              v343_acc += (v41_bc * v22_data);
              v343_acc += (v43_bc * v23_data);
              v343_acc += (v45_bc * v24_data);
              v343_acc += (v47_bc * v25_data);
              v343_acc += (v49_bc * v26_data);
              v343_acc += (v51_bc * v27_data);
              v343_acc += (v53_bc * v28_data);
              v343_acc += (v55_bc * v29_data);
              v343_acc += (v57_bc * v30_data);
              v343_acc += (v59_bc * v31_data);
              v343_acc += (v61_bc * v32_data);
              v343_acc += (v63_bc * v33_data);
              v343_acc += (v65_bc * v34_data);
              v343_acc += (v67_bc * v35_data);
              v343_acc += (v69_bc * v36_data);
              v343_acc.copy_to(ir1 + (144));
              tensorforge::intel_esimd::simd<float, 16> v377_acc{};
              v377_acc += (v39_bc * v21_data);
              v377_acc += (v41_bc * v22_data);
              v377_acc += (v43_bc * v23_data);
              v377_acc += (v45_bc * v24_data);
              v377_acc += (v47_bc * v25_data);
              v377_acc += (v49_bc * v26_data);
              v377_acc += (v51_bc * v27_data);
              v377_acc += (v53_bc * v28_data);
              v377_acc += (v55_bc * v29_data);
              v377_acc += (v57_bc * v30_data);
              v377_acc += (v59_bc * v31_data);
              v377_acc += (v61_bc * v32_data);
              v377_acc += (v63_bc * v33_data);
              v377_acc += (v65_bc * v34_data);
              v377_acc += (v67_bc * v35_data);
              v377_acc += (v69_bc * v36_data);
              v377_acc.copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 16> v411_acc{};
              v411_acc += (v39_bc * v21_data);
              v411_acc += (v41_bc * v22_data);
              v411_acc += (v43_bc * v23_data);
              v411_acc += (v45_bc * v24_data);
              v411_acc += (v47_bc * v25_data);
              v411_acc += (v49_bc * v26_data);
              v411_acc += (v51_bc * v27_data);
              v411_acc += (v53_bc * v28_data);
              v411_acc += (v55_bc * v29_data);
              v411_acc += (v57_bc * v30_data);
              v411_acc += (v59_bc * v31_data);
              v411_acc += (v61_bc * v32_data);
              v411_acc += (v63_bc * v33_data);
              v411_acc += (v65_bc * v34_data);
              v411_acc += (v67_bc * v35_data);
              v411_acc += (v69_bc * v36_data);
              v411_acc.copy_to(ir1 + (176));
              tensorforge::intel_esimd::simd<float, 16> v445_acc{};
              v445_acc += (v39_bc * v21_data);
              v445_acc += (v41_bc * v22_data);
              v445_acc += (v43_bc * v23_data);
              v445_acc += (v45_bc * v24_data);
              v445_acc += (v47_bc * v25_data);
              v445_acc += (v49_bc * v26_data);
              v445_acc += (v51_bc * v27_data);
              v445_acc += (v53_bc * v28_data);
              v445_acc += (v55_bc * v29_data);
              v445_acc += (v57_bc * v30_data);
              v445_acc += (v59_bc * v31_data);
              v445_acc += (v61_bc * v32_data);
              v445_acc += (v63_bc * v33_data);
              v445_acc += (v65_bc * v34_data);
              v445_acc += (v67_bc * v35_data);
              v445_acc += (v69_bc * v36_data);
              v445_acc.copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 16> v479_acc{};
              v479_acc += (v39_bc * v21_data);
              v479_acc += (v41_bc * v22_data);
              v479_acc += (v43_bc * v23_data);
              v479_acc += (v45_bc * v24_data);
              v479_acc += (v47_bc * v25_data);
              v479_acc += (v49_bc * v26_data);
              v479_acc += (v51_bc * v27_data);
              v479_acc += (v53_bc * v28_data);
              v479_acc += (v55_bc * v29_data);
              v479_acc += (v57_bc * v30_data);
              v479_acc += (v59_bc * v31_data);
              v479_acc += (v61_bc * v32_data);
              v479_acc += (v63_bc * v33_data);
              v479_acc += (v65_bc * v34_data);
              v479_acc += (v67_bc * v35_data);
              v479_acc += (v69_bc * v36_data);
              v479_acc.copy_to(ir1 + (208));
              tensorforge::intel_esimd::simd<float, 16> v513_acc{};
              v513_acc += (v39_bc * v21_data);
              v513_acc += (v41_bc * v22_data);
              v513_acc += (v43_bc * v23_data);
              v513_acc += (v45_bc * v24_data);
              v513_acc += (v47_bc * v25_data);
              v513_acc += (v49_bc * v26_data);
              v513_acc += (v51_bc * v27_data);
              v513_acc += (v53_bc * v28_data);
              v513_acc += (v55_bc * v29_data);
              v513_acc += (v57_bc * v30_data);
              v513_acc += (v59_bc * v31_data);
              v513_acc += (v61_bc * v32_data);
              v513_acc += (v63_bc * v33_data);
              v513_acc += (v65_bc * v34_data);
              v513_acc += (v67_bc * v35_data);
              v513_acc += (v69_bc * v36_data);
              v513_acc.copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 16> v547_acc{};
              v547_acc += (v39_bc * v21_data);
              v547_acc += (v41_bc * v22_data);
              v547_acc += (v43_bc * v23_data);
              v547_acc += (v45_bc * v24_data);
              v547_acc += (v47_bc * v25_data);
              v547_acc += (v49_bc * v26_data);
              v547_acc += (v51_bc * v27_data);
              v547_acc += (v53_bc * v28_data);
              v547_acc += (v55_bc * v29_data);
              v547_acc += (v57_bc * v30_data);
              v547_acc += (v59_bc * v31_data);
              v547_acc += (v61_bc * v32_data);
              v547_acc += (v63_bc * v33_data);
              v547_acc += (v65_bc * v34_data);
              v547_acc += (v67_bc * v35_data);
              v547_acc += (v69_bc * v36_data);
              v547_acc.copy_to(ir1 + (240));
              #pragma unroll
              for (int32_t v581_n0 = 0; v581_n0 < 1; ++v581_n0) {
                int32_t v583_a = v581_n0 * 16;
                #pragma unroll
                for (int32_t v582_n1 = 0; v582_n1 < 16; ++v582_n1) {
                  int32_t v585_a = v583_a + (v582_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v586_data;
                  v586_data.copy_from(ir1 + (v585_a));
                  v586_data.copy_to(r1 + (v585_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v590_i0 = 0; v590_i0 < 1; ++v590_i0) {
                int32_t v592_a = v590_i0 * 16;
                #pragma unroll
                for (int32_t v591_i1 = 0; v591_i1 < 16; ++v591_i1) {
                  int32_t v594_a = v592_a + (v591_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v595_data;
                  v595_data.copy_from(r1 + (v594_a));
                  v595_data.copy_to(glb_m0 + (v594_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

