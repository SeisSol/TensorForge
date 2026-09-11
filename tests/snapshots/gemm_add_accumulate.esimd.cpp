// === base name ===
kernel_908abafb74cff6d7

// === header ===
void launcher_kernel_908abafb74cff6d7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_908abafb74cff6d7(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_908abafb74cff6d7(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_908abafb74cff6d7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 12×16(12×16) {0..12}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] += m1 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
                tensorforge::intel_esimd::simd<float, 12> v20_data;
                v20_data.copy_from(glb_m1 + ((v15_i1 * 12)));
                v20_data.copy_to(r0 + ((v15_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v23_ld;
              v23_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v23_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v24_ld;
              v24_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v24_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              float r1[128]{};
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v26_i1 = 0; v26_i1 < 8; ++v26_i1) {
                tensorforge::intel_esimd::simd<float, 12> v31_data;
                v31_data.copy_from(glb_m0 + ((v26_i1 * 12)));
                v31_data.copy_to(r1 + ((v26_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[128]{};
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v52_acc{};
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(s0 + (0_i32));
              v52_acc += ((static_cast<float>(v56_data[0])) * v36_data);
              v52_acc += ((static_cast<float>(v56_data[1])) * v37_data);
              v52_acc += ((static_cast<float>(v56_data[2])) * v38_data);
              v52_acc += ((static_cast<float>(v56_data[3])) * v39_data);
              v52_acc += ((static_cast<float>(v56_data[4])) * v40_data);
              v52_acc += ((static_cast<float>(v56_data[5])) * v41_data);
              v52_acc += ((static_cast<float>(v56_data[6])) * v42_data);
              v52_acc += ((static_cast<float>(v56_data[7])) * v43_data);
              v52_acc += ((static_cast<float>(v56_data[8])) * v44_data);
              v52_acc += ((static_cast<float>(v56_data[9])) * v45_data);
              v52_acc += ((static_cast<float>(v56_data[10])) * v46_data);
              v52_acc += ((static_cast<float>(v56_data[11])) * v47_data);
              v52_acc += ((static_cast<float>(v56_data[12])) * v48_data);
              v52_acc += ((static_cast<float>(v56_data[13])) * v49_data);
              v52_acc += ((static_cast<float>(v56_data[14])) * v50_data);
              v52_acc += ((static_cast<float>(v56_data[15])) * v51_data);
              v52_acc.copy_to(ir2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(s0 + (16_i32));
              v89_acc += ((static_cast<float>(v93_data[0])) * v36_data);
              v89_acc += ((static_cast<float>(v93_data[1])) * v37_data);
              v89_acc += ((static_cast<float>(v93_data[2])) * v38_data);
              v89_acc += ((static_cast<float>(v93_data[3])) * v39_data);
              v89_acc += ((static_cast<float>(v93_data[4])) * v40_data);
              v89_acc += ((static_cast<float>(v93_data[5])) * v41_data);
              v89_acc += ((static_cast<float>(v93_data[6])) * v42_data);
              v89_acc += ((static_cast<float>(v93_data[7])) * v43_data);
              v89_acc += ((static_cast<float>(v93_data[8])) * v44_data);
              v89_acc += ((static_cast<float>(v93_data[9])) * v45_data);
              v89_acc += ((static_cast<float>(v93_data[10])) * v46_data);
              v89_acc += ((static_cast<float>(v93_data[11])) * v47_data);
              v89_acc += ((static_cast<float>(v93_data[12])) * v48_data);
              v89_acc += ((static_cast<float>(v93_data[13])) * v49_data);
              v89_acc += ((static_cast<float>(v93_data[14])) * v50_data);
              v89_acc += ((static_cast<float>(v93_data[15])) * v51_data);
              v89_acc.copy_to(ir2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v126_acc{};
              tensorforge::intel_esimd::simd<float, 16> v130_data;
              v130_data.copy_from(s0 + (32_i32));
              v126_acc += ((static_cast<float>(v130_data[0])) * v36_data);
              v126_acc += ((static_cast<float>(v130_data[1])) * v37_data);
              v126_acc += ((static_cast<float>(v130_data[2])) * v38_data);
              v126_acc += ((static_cast<float>(v130_data[3])) * v39_data);
              v126_acc += ((static_cast<float>(v130_data[4])) * v40_data);
              v126_acc += ((static_cast<float>(v130_data[5])) * v41_data);
              v126_acc += ((static_cast<float>(v130_data[6])) * v42_data);
              v126_acc += ((static_cast<float>(v130_data[7])) * v43_data);
              v126_acc += ((static_cast<float>(v130_data[8])) * v44_data);
              v126_acc += ((static_cast<float>(v130_data[9])) * v45_data);
              v126_acc += ((static_cast<float>(v130_data[10])) * v46_data);
              v126_acc += ((static_cast<float>(v130_data[11])) * v47_data);
              v126_acc += ((static_cast<float>(v130_data[12])) * v48_data);
              v126_acc += ((static_cast<float>(v130_data[13])) * v49_data);
              v126_acc += ((static_cast<float>(v130_data[14])) * v50_data);
              v126_acc += ((static_cast<float>(v130_data[15])) * v51_data);
              v126_acc.copy_to(ir2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v163_acc{};
              tensorforge::intel_esimd::simd<float, 16> v167_data;
              v167_data.copy_from(s0 + (48_i32));
              v163_acc += ((static_cast<float>(v167_data[0])) * v36_data);
              v163_acc += ((static_cast<float>(v167_data[1])) * v37_data);
              v163_acc += ((static_cast<float>(v167_data[2])) * v38_data);
              v163_acc += ((static_cast<float>(v167_data[3])) * v39_data);
              v163_acc += ((static_cast<float>(v167_data[4])) * v40_data);
              v163_acc += ((static_cast<float>(v167_data[5])) * v41_data);
              v163_acc += ((static_cast<float>(v167_data[6])) * v42_data);
              v163_acc += ((static_cast<float>(v167_data[7])) * v43_data);
              v163_acc += ((static_cast<float>(v167_data[8])) * v44_data);
              v163_acc += ((static_cast<float>(v167_data[9])) * v45_data);
              v163_acc += ((static_cast<float>(v167_data[10])) * v46_data);
              v163_acc += ((static_cast<float>(v167_data[11])) * v47_data);
              v163_acc += ((static_cast<float>(v167_data[12])) * v48_data);
              v163_acc += ((static_cast<float>(v167_data[13])) * v49_data);
              v163_acc += ((static_cast<float>(v167_data[14])) * v50_data);
              v163_acc += ((static_cast<float>(v167_data[15])) * v51_data);
              v163_acc.copy_to(ir2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v200_acc{};
              tensorforge::intel_esimd::simd<float, 16> v204_data;
              v204_data.copy_from(s0 + (64_i32));
              v200_acc += ((static_cast<float>(v204_data[0])) * v36_data);
              v200_acc += ((static_cast<float>(v204_data[1])) * v37_data);
              v200_acc += ((static_cast<float>(v204_data[2])) * v38_data);
              v200_acc += ((static_cast<float>(v204_data[3])) * v39_data);
              v200_acc += ((static_cast<float>(v204_data[4])) * v40_data);
              v200_acc += ((static_cast<float>(v204_data[5])) * v41_data);
              v200_acc += ((static_cast<float>(v204_data[6])) * v42_data);
              v200_acc += ((static_cast<float>(v204_data[7])) * v43_data);
              v200_acc += ((static_cast<float>(v204_data[8])) * v44_data);
              v200_acc += ((static_cast<float>(v204_data[9])) * v45_data);
              v200_acc += ((static_cast<float>(v204_data[10])) * v46_data);
              v200_acc += ((static_cast<float>(v204_data[11])) * v47_data);
              v200_acc += ((static_cast<float>(v204_data[12])) * v48_data);
              v200_acc += ((static_cast<float>(v204_data[13])) * v49_data);
              v200_acc += ((static_cast<float>(v204_data[14])) * v50_data);
              v200_acc += ((static_cast<float>(v204_data[15])) * v51_data);
              v200_acc.copy_to(ir2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v237_acc{};
              tensorforge::intel_esimd::simd<float, 16> v241_data;
              v241_data.copy_from(s0 + (80_i32));
              v237_acc += ((static_cast<float>(v241_data[0])) * v36_data);
              v237_acc += ((static_cast<float>(v241_data[1])) * v37_data);
              v237_acc += ((static_cast<float>(v241_data[2])) * v38_data);
              v237_acc += ((static_cast<float>(v241_data[3])) * v39_data);
              v237_acc += ((static_cast<float>(v241_data[4])) * v40_data);
              v237_acc += ((static_cast<float>(v241_data[5])) * v41_data);
              v237_acc += ((static_cast<float>(v241_data[6])) * v42_data);
              v237_acc += ((static_cast<float>(v241_data[7])) * v43_data);
              v237_acc += ((static_cast<float>(v241_data[8])) * v44_data);
              v237_acc += ((static_cast<float>(v241_data[9])) * v45_data);
              v237_acc += ((static_cast<float>(v241_data[10])) * v46_data);
              v237_acc += ((static_cast<float>(v241_data[11])) * v47_data);
              v237_acc += ((static_cast<float>(v241_data[12])) * v48_data);
              v237_acc += ((static_cast<float>(v241_data[13])) * v49_data);
              v237_acc += ((static_cast<float>(v241_data[14])) * v50_data);
              v237_acc += ((static_cast<float>(v241_data[15])) * v51_data);
              v237_acc.copy_to(ir2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v274_acc{};
              tensorforge::intel_esimd::simd<float, 16> v278_data;
              v278_data.copy_from(s0 + (96_i32));
              v274_acc += ((static_cast<float>(v278_data[0])) * v36_data);
              v274_acc += ((static_cast<float>(v278_data[1])) * v37_data);
              v274_acc += ((static_cast<float>(v278_data[2])) * v38_data);
              v274_acc += ((static_cast<float>(v278_data[3])) * v39_data);
              v274_acc += ((static_cast<float>(v278_data[4])) * v40_data);
              v274_acc += ((static_cast<float>(v278_data[5])) * v41_data);
              v274_acc += ((static_cast<float>(v278_data[6])) * v42_data);
              v274_acc += ((static_cast<float>(v278_data[7])) * v43_data);
              v274_acc += ((static_cast<float>(v278_data[8])) * v44_data);
              v274_acc += ((static_cast<float>(v278_data[9])) * v45_data);
              v274_acc += ((static_cast<float>(v278_data[10])) * v46_data);
              v274_acc += ((static_cast<float>(v278_data[11])) * v47_data);
              v274_acc += ((static_cast<float>(v278_data[12])) * v48_data);
              v274_acc += ((static_cast<float>(v278_data[13])) * v49_data);
              v274_acc += ((static_cast<float>(v278_data[14])) * v50_data);
              v274_acc += ((static_cast<float>(v278_data[15])) * v51_data);
              v274_acc.copy_to(ir2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v311_acc{};
              tensorforge::intel_esimd::simd<float, 16> v315_data;
              v315_data.copy_from(s0 + (112_i32));
              v311_acc += ((static_cast<float>(v315_data[0])) * v36_data);
              v311_acc += ((static_cast<float>(v315_data[1])) * v37_data);
              v311_acc += ((static_cast<float>(v315_data[2])) * v38_data);
              v311_acc += ((static_cast<float>(v315_data[3])) * v39_data);
              v311_acc += ((static_cast<float>(v315_data[4])) * v40_data);
              v311_acc += ((static_cast<float>(v315_data[5])) * v41_data);
              v311_acc += ((static_cast<float>(v315_data[6])) * v42_data);
              v311_acc += ((static_cast<float>(v315_data[7])) * v43_data);
              v311_acc += ((static_cast<float>(v315_data[8])) * v44_data);
              v311_acc += ((static_cast<float>(v315_data[9])) * v45_data);
              v311_acc += ((static_cast<float>(v315_data[10])) * v46_data);
              v311_acc += ((static_cast<float>(v315_data[11])) * v47_data);
              v311_acc += ((static_cast<float>(v315_data[12])) * v48_data);
              v311_acc += ((static_cast<float>(v315_data[13])) * v49_data);
              v311_acc += ((static_cast<float>(v315_data[14])) * v50_data);
              v311_acc += ((static_cast<float>(v315_data[15])) * v51_data);
              v311_acc.copy_to(ir2 + (112));
              #pragma unroll
              for (int32_t v348_n1 = 0; v348_n1 < 8; ++v348_n1) {
                int32_t v349_a = v348_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v351_data;
                v351_data.copy_from(ir2 + (v349_a));
                tensorforge::intel_esimd::simd<float, 12> v354_data;
                v354_data.copy_from(r1 + (v349_a));
                (v354_data + v351_data).copy_to(r2 + (v349_a));
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v358_i1 = 0; v358_i1 < 8; ++v358_i1) {
                tensorforge::intel_esimd::simd<float, 12> v361_data;
                v361_data.copy_from(r2 + ((v358_i1 * 16)));
                v361_data.copy_to(glb_m0 + ((v358_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

