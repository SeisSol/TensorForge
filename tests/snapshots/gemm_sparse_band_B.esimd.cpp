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
              tensorforge::intel_esimd::simd<float, 16> v40_p = (v38_lin[0]) * v21_data;
              v37_acc += v40_p;
              tensorforge::intel_esimd::simd<float, 16> v42_p = (v38_lin[1]) * v22_data;
              v37_acc += v42_p;
              tensorforge::intel_esimd::simd<float, 16> v44_p = (v38_lin[2]) * v23_data;
              v37_acc += v44_p;
              tensorforge::intel_esimd::simd<float, 16> v46_p = (v38_lin[3]) * v24_data;
              v37_acc += v46_p;
              tensorforge::intel_esimd::simd<float, 16> v48_p = (v38_lin[4]) * v25_data;
              v37_acc += v48_p;
              tensorforge::intel_esimd::simd<float, 16> v50_p = (v38_lin[5]) * v26_data;
              v37_acc += v50_p;
              tensorforge::intel_esimd::simd<float, 16> v52_p = (v38_lin[6]) * v27_data;
              v37_acc += v52_p;
              tensorforge::intel_esimd::simd<float, 16> v54_p = (v38_lin[7]) * v28_data;
              v37_acc += v54_p;
              tensorforge::intel_esimd::simd<float, 16> v56_p = (v38_lin[8]) * v29_data;
              v37_acc += v56_p;
              tensorforge::intel_esimd::simd<float, 16> v58_p = (v38_lin[9]) * v30_data;
              v37_acc += v58_p;
              tensorforge::intel_esimd::simd<float, 16> v60_p = (v38_lin[10]) * v31_data;
              v37_acc += v60_p;
              tensorforge::intel_esimd::simd<float, 16> v62_p = (v38_lin[11]) * v32_data;
              v37_acc += v62_p;
              tensorforge::intel_esimd::simd<float, 16> v64_p = (v38_lin[12]) * v33_data;
              v37_acc += v64_p;
              tensorforge::intel_esimd::simd<float, 16> v66_p = (v38_lin[13]) * v34_data;
              v37_acc += v66_p;
              tensorforge::intel_esimd::simd<float, 16> v68_p = (v38_lin[14]) * v35_data;
              v37_acc += v68_p;
              tensorforge::intel_esimd::simd<float, 16> v70_p = (v38_lin[15]) * v36_data;
              v37_acc += v70_p;
              v37_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v71_acc{};
              v71_acc += v40_p;
              v71_acc += v42_p;
              v71_acc += v44_p;
              v71_acc += v46_p;
              v71_acc += v48_p;
              v71_acc += v50_p;
              v71_acc += v52_p;
              v71_acc += v54_p;
              v71_acc += v56_p;
              v71_acc += v58_p;
              v71_acc += v60_p;
              v71_acc += v62_p;
              v71_acc += v64_p;
              v71_acc += v66_p;
              v71_acc += v68_p;
              v71_acc += v70_p;
              v71_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v105_acc{};
              v105_acc += v40_p;
              v105_acc += v42_p;
              v105_acc += v44_p;
              v105_acc += v46_p;
              v105_acc += v48_p;
              v105_acc += v50_p;
              v105_acc += v52_p;
              v105_acc += v54_p;
              v105_acc += v56_p;
              v105_acc += v58_p;
              v105_acc += v60_p;
              v105_acc += v62_p;
              v105_acc += v64_p;
              v105_acc += v66_p;
              v105_acc += v68_p;
              v105_acc += v70_p;
              v105_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v139_acc{};
              v139_acc += v40_p;
              v139_acc += v42_p;
              v139_acc += v44_p;
              v139_acc += v46_p;
              v139_acc += v48_p;
              v139_acc += v50_p;
              v139_acc += v52_p;
              v139_acc += v54_p;
              v139_acc += v56_p;
              v139_acc += v58_p;
              v139_acc += v60_p;
              v139_acc += v62_p;
              v139_acc += v64_p;
              v139_acc += v66_p;
              v139_acc += v68_p;
              v139_acc += v70_p;
              v139_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              v173_acc += v40_p;
              v173_acc += v42_p;
              v173_acc += v44_p;
              v173_acc += v46_p;
              v173_acc += v48_p;
              v173_acc += v50_p;
              v173_acc += v52_p;
              v173_acc += v54_p;
              v173_acc += v56_p;
              v173_acc += v58_p;
              v173_acc += v60_p;
              v173_acc += v62_p;
              v173_acc += v64_p;
              v173_acc += v66_p;
              v173_acc += v68_p;
              v173_acc += v70_p;
              v173_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v207_acc{};
              v207_acc += v40_p;
              v207_acc += v42_p;
              v207_acc += v44_p;
              v207_acc += v46_p;
              v207_acc += v48_p;
              v207_acc += v50_p;
              v207_acc += v52_p;
              v207_acc += v54_p;
              v207_acc += v56_p;
              v207_acc += v58_p;
              v207_acc += v60_p;
              v207_acc += v62_p;
              v207_acc += v64_p;
              v207_acc += v66_p;
              v207_acc += v68_p;
              v207_acc += v70_p;
              v207_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v241_acc{};
              v241_acc += v40_p;
              v241_acc += v42_p;
              v241_acc += v44_p;
              v241_acc += v46_p;
              v241_acc += v48_p;
              v241_acc += v50_p;
              v241_acc += v52_p;
              v241_acc += v54_p;
              v241_acc += v56_p;
              v241_acc += v58_p;
              v241_acc += v60_p;
              v241_acc += v62_p;
              v241_acc += v64_p;
              v241_acc += v66_p;
              v241_acc += v68_p;
              v241_acc += v70_p;
              v241_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v275_acc{};
              v275_acc += v40_p;
              v275_acc += v42_p;
              v275_acc += v44_p;
              v275_acc += v46_p;
              v275_acc += v48_p;
              v275_acc += v50_p;
              v275_acc += v52_p;
              v275_acc += v54_p;
              v275_acc += v56_p;
              v275_acc += v58_p;
              v275_acc += v60_p;
              v275_acc += v62_p;
              v275_acc += v64_p;
              v275_acc += v66_p;
              v275_acc += v68_p;
              v275_acc += v70_p;
              v275_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v309_acc{};
              v309_acc += v40_p;
              v309_acc += v42_p;
              v309_acc += v44_p;
              v309_acc += v46_p;
              v309_acc += v48_p;
              v309_acc += v50_p;
              v309_acc += v52_p;
              v309_acc += v54_p;
              v309_acc += v56_p;
              v309_acc += v58_p;
              v309_acc += v60_p;
              v309_acc += v62_p;
              v309_acc += v64_p;
              v309_acc += v66_p;
              v309_acc += v68_p;
              v309_acc += v70_p;
              v309_acc.copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 16> v343_acc{};
              v343_acc += v40_p;
              v343_acc += v42_p;
              v343_acc += v44_p;
              v343_acc += v46_p;
              v343_acc += v48_p;
              v343_acc += v50_p;
              v343_acc += v52_p;
              v343_acc += v54_p;
              v343_acc += v56_p;
              v343_acc += v58_p;
              v343_acc += v60_p;
              v343_acc += v62_p;
              v343_acc += v64_p;
              v343_acc += v66_p;
              v343_acc += v68_p;
              v343_acc += v70_p;
              v343_acc.copy_to(ir1 + (144));
              tensorforge::intel_esimd::simd<float, 16> v377_acc{};
              v377_acc += v40_p;
              v377_acc += v42_p;
              v377_acc += v44_p;
              v377_acc += v46_p;
              v377_acc += v48_p;
              v377_acc += v50_p;
              v377_acc += v52_p;
              v377_acc += v54_p;
              v377_acc += v56_p;
              v377_acc += v58_p;
              v377_acc += v60_p;
              v377_acc += v62_p;
              v377_acc += v64_p;
              v377_acc += v66_p;
              v377_acc += v68_p;
              v377_acc += v70_p;
              v377_acc.copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 16> v411_acc{};
              v411_acc += v40_p;
              v411_acc += v42_p;
              v411_acc += v44_p;
              v411_acc += v46_p;
              v411_acc += v48_p;
              v411_acc += v50_p;
              v411_acc += v52_p;
              v411_acc += v54_p;
              v411_acc += v56_p;
              v411_acc += v58_p;
              v411_acc += v60_p;
              v411_acc += v62_p;
              v411_acc += v64_p;
              v411_acc += v66_p;
              v411_acc += v68_p;
              v411_acc += v70_p;
              v411_acc.copy_to(ir1 + (176));
              tensorforge::intel_esimd::simd<float, 16> v445_acc{};
              v445_acc += v40_p;
              v445_acc += v42_p;
              v445_acc += v44_p;
              v445_acc += v46_p;
              v445_acc += v48_p;
              v445_acc += v50_p;
              v445_acc += v52_p;
              v445_acc += v54_p;
              v445_acc += v56_p;
              v445_acc += v58_p;
              v445_acc += v60_p;
              v445_acc += v62_p;
              v445_acc += v64_p;
              v445_acc += v66_p;
              v445_acc += v68_p;
              v445_acc += v70_p;
              v445_acc.copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 16> v479_acc{};
              v479_acc += v40_p;
              v479_acc += v42_p;
              v479_acc += v44_p;
              v479_acc += v46_p;
              v479_acc += v48_p;
              v479_acc += v50_p;
              v479_acc += v52_p;
              v479_acc += v54_p;
              v479_acc += v56_p;
              v479_acc += v58_p;
              v479_acc += v60_p;
              v479_acc += v62_p;
              v479_acc += v64_p;
              v479_acc += v66_p;
              v479_acc += v68_p;
              v479_acc += v70_p;
              v479_acc.copy_to(ir1 + (208));
              tensorforge::intel_esimd::simd<float, 16> v513_acc{};
              v513_acc += v40_p;
              v513_acc += v42_p;
              v513_acc += v44_p;
              v513_acc += v46_p;
              v513_acc += v48_p;
              v513_acc += v50_p;
              v513_acc += v52_p;
              v513_acc += v54_p;
              v513_acc += v56_p;
              v513_acc += v58_p;
              v513_acc += v60_p;
              v513_acc += v62_p;
              v513_acc += v64_p;
              v513_acc += v66_p;
              v513_acc += v68_p;
              v513_acc += v70_p;
              v513_acc.copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 16> v547_acc{};
              v547_acc += v40_p;
              v547_acc += v42_p;
              v547_acc += v44_p;
              v547_acc += v46_p;
              v547_acc += v48_p;
              v547_acc += v50_p;
              v547_acc += v52_p;
              v547_acc += v54_p;
              v547_acc += v56_p;
              v547_acc += v58_p;
              v547_acc += v60_p;
              v547_acc += v62_p;
              v547_acc += v64_p;
              v547_acc += v66_p;
              v547_acc += v68_p;
              v547_acc += v70_p;
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

