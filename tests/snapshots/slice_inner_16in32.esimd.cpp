// === base name ===
kernel_b12d2baa38572b22

// === header ===
void launcher_kernel_b12d2baa38572b22(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_b12d2baa38572b22(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_b12d2baa38572b22(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b12d2baa38572b22(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 32×32(32×32) {0..32}×{0..32} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 1024 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v17_lead = v15_i0 * 16;
                int32_t v19_off = v17_lead + 8;
                #pragma unroll
                for (int32_t v16_i1 = 8; v16_i1 < 24; ++v16_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v22_data;
                  v22_data.copy_from(glb_m1 + ((v19_off + (v16_i1 * 32))));
                  v22_data.copy_to(r0 + ((v17_lead + ((v16_i1 - 8) * 16))));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v27_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v28_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v47_acc{};
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(s0 + (0_i32));
              v47_acc += ((static_cast<float>(v51_data[0])) * v31_data);
              v47_acc += ((static_cast<float>(v51_data[1])) * v32_data);
              v47_acc += ((static_cast<float>(v51_data[2])) * v33_data);
              v47_acc += ((static_cast<float>(v51_data[3])) * v34_data);
              v47_acc += ((static_cast<float>(v51_data[4])) * v35_data);
              v47_acc += ((static_cast<float>(v51_data[5])) * v36_data);
              v47_acc += ((static_cast<float>(v51_data[6])) * v37_data);
              v47_acc += ((static_cast<float>(v51_data[7])) * v38_data);
              v47_acc += ((static_cast<float>(v51_data[8])) * v39_data);
              v47_acc += ((static_cast<float>(v51_data[9])) * v40_data);
              v47_acc += ((static_cast<float>(v51_data[10])) * v41_data);
              v47_acc += ((static_cast<float>(v51_data[11])) * v42_data);
              v47_acc += ((static_cast<float>(v51_data[12])) * v43_data);
              v47_acc += ((static_cast<float>(v51_data[13])) * v44_data);
              v47_acc += ((static_cast<float>(v51_data[14])) * v45_data);
              v47_acc += ((static_cast<float>(v51_data[15])) * v46_data);
              v47_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v84_acc{};
              tensorforge::intel_esimd::simd<float, 16> v88_data;
              v88_data.copy_from(s0 + (16_i32));
              v84_acc += ((static_cast<float>(v88_data[0])) * v31_data);
              v84_acc += ((static_cast<float>(v88_data[1])) * v32_data);
              v84_acc += ((static_cast<float>(v88_data[2])) * v33_data);
              v84_acc += ((static_cast<float>(v88_data[3])) * v34_data);
              v84_acc += ((static_cast<float>(v88_data[4])) * v35_data);
              v84_acc += ((static_cast<float>(v88_data[5])) * v36_data);
              v84_acc += ((static_cast<float>(v88_data[6])) * v37_data);
              v84_acc += ((static_cast<float>(v88_data[7])) * v38_data);
              v84_acc += ((static_cast<float>(v88_data[8])) * v39_data);
              v84_acc += ((static_cast<float>(v88_data[9])) * v40_data);
              v84_acc += ((static_cast<float>(v88_data[10])) * v41_data);
              v84_acc += ((static_cast<float>(v88_data[11])) * v42_data);
              v84_acc += ((static_cast<float>(v88_data[12])) * v43_data);
              v84_acc += ((static_cast<float>(v88_data[13])) * v44_data);
              v84_acc += ((static_cast<float>(v88_data[14])) * v45_data);
              v84_acc += ((static_cast<float>(v88_data[15])) * v46_data);
              v84_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v121_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data;
              v125_data.copy_from(s0 + (32_i32));
              v121_acc += ((static_cast<float>(v125_data[0])) * v31_data);
              v121_acc += ((static_cast<float>(v125_data[1])) * v32_data);
              v121_acc += ((static_cast<float>(v125_data[2])) * v33_data);
              v121_acc += ((static_cast<float>(v125_data[3])) * v34_data);
              v121_acc += ((static_cast<float>(v125_data[4])) * v35_data);
              v121_acc += ((static_cast<float>(v125_data[5])) * v36_data);
              v121_acc += ((static_cast<float>(v125_data[6])) * v37_data);
              v121_acc += ((static_cast<float>(v125_data[7])) * v38_data);
              v121_acc += ((static_cast<float>(v125_data[8])) * v39_data);
              v121_acc += ((static_cast<float>(v125_data[9])) * v40_data);
              v121_acc += ((static_cast<float>(v125_data[10])) * v41_data);
              v121_acc += ((static_cast<float>(v125_data[11])) * v42_data);
              v121_acc += ((static_cast<float>(v125_data[12])) * v43_data);
              v121_acc += ((static_cast<float>(v125_data[13])) * v44_data);
              v121_acc += ((static_cast<float>(v125_data[14])) * v45_data);
              v121_acc += ((static_cast<float>(v125_data[15])) * v46_data);
              v121_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v162_data;
              v162_data.copy_from(s0 + (48_i32));
              v158_acc += ((static_cast<float>(v162_data[0])) * v31_data);
              v158_acc += ((static_cast<float>(v162_data[1])) * v32_data);
              v158_acc += ((static_cast<float>(v162_data[2])) * v33_data);
              v158_acc += ((static_cast<float>(v162_data[3])) * v34_data);
              v158_acc += ((static_cast<float>(v162_data[4])) * v35_data);
              v158_acc += ((static_cast<float>(v162_data[5])) * v36_data);
              v158_acc += ((static_cast<float>(v162_data[6])) * v37_data);
              v158_acc += ((static_cast<float>(v162_data[7])) * v38_data);
              v158_acc += ((static_cast<float>(v162_data[8])) * v39_data);
              v158_acc += ((static_cast<float>(v162_data[9])) * v40_data);
              v158_acc += ((static_cast<float>(v162_data[10])) * v41_data);
              v158_acc += ((static_cast<float>(v162_data[11])) * v42_data);
              v158_acc += ((static_cast<float>(v162_data[12])) * v43_data);
              v158_acc += ((static_cast<float>(v162_data[13])) * v44_data);
              v158_acc += ((static_cast<float>(v162_data[14])) * v45_data);
              v158_acc += ((static_cast<float>(v162_data[15])) * v46_data);
              v158_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v195_acc{};
              tensorforge::intel_esimd::simd<float, 16> v199_data;
              v199_data.copy_from(s0 + (64_i32));
              v195_acc += ((static_cast<float>(v199_data[0])) * v31_data);
              v195_acc += ((static_cast<float>(v199_data[1])) * v32_data);
              v195_acc += ((static_cast<float>(v199_data[2])) * v33_data);
              v195_acc += ((static_cast<float>(v199_data[3])) * v34_data);
              v195_acc += ((static_cast<float>(v199_data[4])) * v35_data);
              v195_acc += ((static_cast<float>(v199_data[5])) * v36_data);
              v195_acc += ((static_cast<float>(v199_data[6])) * v37_data);
              v195_acc += ((static_cast<float>(v199_data[7])) * v38_data);
              v195_acc += ((static_cast<float>(v199_data[8])) * v39_data);
              v195_acc += ((static_cast<float>(v199_data[9])) * v40_data);
              v195_acc += ((static_cast<float>(v199_data[10])) * v41_data);
              v195_acc += ((static_cast<float>(v199_data[11])) * v42_data);
              v195_acc += ((static_cast<float>(v199_data[12])) * v43_data);
              v195_acc += ((static_cast<float>(v199_data[13])) * v44_data);
              v195_acc += ((static_cast<float>(v199_data[14])) * v45_data);
              v195_acc += ((static_cast<float>(v199_data[15])) * v46_data);
              v195_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v232_acc{};
              tensorforge::intel_esimd::simd<float, 16> v236_data;
              v236_data.copy_from(s0 + (80_i32));
              v232_acc += ((static_cast<float>(v236_data[0])) * v31_data);
              v232_acc += ((static_cast<float>(v236_data[1])) * v32_data);
              v232_acc += ((static_cast<float>(v236_data[2])) * v33_data);
              v232_acc += ((static_cast<float>(v236_data[3])) * v34_data);
              v232_acc += ((static_cast<float>(v236_data[4])) * v35_data);
              v232_acc += ((static_cast<float>(v236_data[5])) * v36_data);
              v232_acc += ((static_cast<float>(v236_data[6])) * v37_data);
              v232_acc += ((static_cast<float>(v236_data[7])) * v38_data);
              v232_acc += ((static_cast<float>(v236_data[8])) * v39_data);
              v232_acc += ((static_cast<float>(v236_data[9])) * v40_data);
              v232_acc += ((static_cast<float>(v236_data[10])) * v41_data);
              v232_acc += ((static_cast<float>(v236_data[11])) * v42_data);
              v232_acc += ((static_cast<float>(v236_data[12])) * v43_data);
              v232_acc += ((static_cast<float>(v236_data[13])) * v44_data);
              v232_acc += ((static_cast<float>(v236_data[14])) * v45_data);
              v232_acc += ((static_cast<float>(v236_data[15])) * v46_data);
              v232_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v269_acc{};
              tensorforge::intel_esimd::simd<float, 16> v273_data;
              v273_data.copy_from(s0 + (96_i32));
              v269_acc += ((static_cast<float>(v273_data[0])) * v31_data);
              v269_acc += ((static_cast<float>(v273_data[1])) * v32_data);
              v269_acc += ((static_cast<float>(v273_data[2])) * v33_data);
              v269_acc += ((static_cast<float>(v273_data[3])) * v34_data);
              v269_acc += ((static_cast<float>(v273_data[4])) * v35_data);
              v269_acc += ((static_cast<float>(v273_data[5])) * v36_data);
              v269_acc += ((static_cast<float>(v273_data[6])) * v37_data);
              v269_acc += ((static_cast<float>(v273_data[7])) * v38_data);
              v269_acc += ((static_cast<float>(v273_data[8])) * v39_data);
              v269_acc += ((static_cast<float>(v273_data[9])) * v40_data);
              v269_acc += ((static_cast<float>(v273_data[10])) * v41_data);
              v269_acc += ((static_cast<float>(v273_data[11])) * v42_data);
              v269_acc += ((static_cast<float>(v273_data[12])) * v43_data);
              v269_acc += ((static_cast<float>(v273_data[13])) * v44_data);
              v269_acc += ((static_cast<float>(v273_data[14])) * v45_data);
              v269_acc += ((static_cast<float>(v273_data[15])) * v46_data);
              v269_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v306_acc{};
              tensorforge::intel_esimd::simd<float, 16> v310_data;
              v310_data.copy_from(s0 + (112_i32));
              v306_acc += ((static_cast<float>(v310_data[0])) * v31_data);
              v306_acc += ((static_cast<float>(v310_data[1])) * v32_data);
              v306_acc += ((static_cast<float>(v310_data[2])) * v33_data);
              v306_acc += ((static_cast<float>(v310_data[3])) * v34_data);
              v306_acc += ((static_cast<float>(v310_data[4])) * v35_data);
              v306_acc += ((static_cast<float>(v310_data[5])) * v36_data);
              v306_acc += ((static_cast<float>(v310_data[6])) * v37_data);
              v306_acc += ((static_cast<float>(v310_data[7])) * v38_data);
              v306_acc += ((static_cast<float>(v310_data[8])) * v39_data);
              v306_acc += ((static_cast<float>(v310_data[9])) * v40_data);
              v306_acc += ((static_cast<float>(v310_data[10])) * v41_data);
              v306_acc += ((static_cast<float>(v310_data[11])) * v42_data);
              v306_acc += ((static_cast<float>(v310_data[12])) * v43_data);
              v306_acc += ((static_cast<float>(v310_data[13])) * v44_data);
              v306_acc += ((static_cast<float>(v310_data[14])) * v45_data);
              v306_acc += ((static_cast<float>(v310_data[15])) * v46_data);
              v306_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v343_n0 = 0; v343_n0 < 1; ++v343_n0) {
                int32_t v345_a = v343_n0 * 16;
                #pragma unroll
                for (int32_t v344_n1 = 0; v344_n1 < 8; ++v344_n1) {
                  int32_t v347_a = v345_a + (v344_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v348_data;
                  v348_data.copy_from(ir1 + (v347_a));
                  v348_data.copy_to(r1 + (v347_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v352_i0 = 0; v352_i0 < 1; ++v352_i0) {
                int32_t v354_a = v352_i0 * 16;
                #pragma unroll
                for (int32_t v353_i1 = 0; v353_i1 < 8; ++v353_i1) {
                  int32_t v356_a = v354_a + (v353_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v357_data;
                  v357_data.copy_from(r1 + (v356_a));
                  v357_data.copy_to(glb_m0 + (v356_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

