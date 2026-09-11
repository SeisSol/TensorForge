// === base name ===
kernel_38a8333b9b835615

// === header ===
void launcher_kernel_38a8333b9b835615(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_38a8333b9b835615(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_38a8333b9b835615(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_38a8333b9b835615(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1024, cgh); {
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
          float* localShrMem0 = &totalShrMem[64 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[48];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 46 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v17_lead = v15_i0 * 16;
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 16; ++v16_i1) {
                  int32_t v20_a = v17_lead + (v16_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v21_data;
                  v21_data.copy_from(glb_m1 + (v20_a));
                  v21_data.copy_to(r0 + (v20_a));
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v25_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 14) {
                tensorforge::intel_esimd::simd<float, 16> v26_ld;
                v26_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 32));
                v26_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 32));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v45_acc{};
              tensorforge::intel_esimd::simd<float, 16> v46_lin;
              v46_lin.copy_from(s0 + (0 + item.get_local_id(0) * 1));
              float v47_bc = static_cast<float>(v46_lin[0]);
              v45_acc += (v47_bc * v29_data);
              float v49_bc = static_cast<float>(v46_lin[1]);
              v45_acc += (v49_bc * v30_data);
              float v51_bc = static_cast<float>(v46_lin[2]);
              v45_acc += (v51_bc * v31_data);
              float v53_bc = static_cast<float>(v46_lin[3]);
              v45_acc += (v53_bc * v32_data);
              float v55_bc = static_cast<float>(v46_lin[4]);
              v45_acc += (v55_bc * v33_data);
              float v57_bc = static_cast<float>(v46_lin[5]);
              v45_acc += (v57_bc * v34_data);
              float v59_bc = static_cast<float>(v46_lin[6]);
              v45_acc += (v59_bc * v35_data);
              float v61_bc = static_cast<float>(v46_lin[7]);
              v45_acc += (v61_bc * v36_data);
              float v63_bc = static_cast<float>(v46_lin[8]);
              v45_acc += (v63_bc * v37_data);
              float v65_bc = static_cast<float>(v46_lin[9]);
              v45_acc += (v65_bc * v38_data);
              float v67_bc = static_cast<float>(v46_lin[10]);
              v45_acc += (v67_bc * v39_data);
              float v69_bc = static_cast<float>(v46_lin[11]);
              v45_acc += (v69_bc * v40_data);
              float v71_bc = static_cast<float>(v46_lin[12]);
              v45_acc += (v71_bc * v41_data);
              float v73_bc = static_cast<float>(v46_lin[13]);
              v45_acc += (v73_bc * v42_data);
              float v75_bc = static_cast<float>(v46_lin[14]);
              v45_acc += (v75_bc * v43_data);
              float v77_bc = static_cast<float>(v46_lin[15]);
              v45_acc += (v77_bc * v44_data);
              v45_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              v79_acc += (v47_bc * v29_data);
              v79_acc += (v49_bc * v30_data);
              v79_acc += (v51_bc * v31_data);
              v79_acc += (v53_bc * v32_data);
              v79_acc += (v55_bc * v33_data);
              v79_acc += (v57_bc * v34_data);
              v79_acc += (v59_bc * v35_data);
              v79_acc += (v61_bc * v36_data);
              v79_acc += (v63_bc * v37_data);
              v79_acc += (v65_bc * v38_data);
              v79_acc += (v67_bc * v39_data);
              v79_acc += (v69_bc * v40_data);
              v79_acc += (v71_bc * v41_data);
              v79_acc += (v73_bc * v42_data);
              v79_acc += (v75_bc * v43_data);
              v79_acc += (v77_bc * v44_data);
              v79_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              v113_acc += (v47_bc * v29_data);
              v113_acc += (v49_bc * v30_data);
              v113_acc += (v51_bc * v31_data);
              v113_acc += (v53_bc * v32_data);
              v113_acc += (v55_bc * v33_data);
              v113_acc += (v57_bc * v34_data);
              v113_acc += (v59_bc * v35_data);
              v113_acc += (v61_bc * v36_data);
              v113_acc += (v63_bc * v37_data);
              v113_acc += (v65_bc * v38_data);
              v113_acc += (v67_bc * v39_data);
              v113_acc += (v69_bc * v40_data);
              v113_acc += (v71_bc * v41_data);
              v113_acc += (v73_bc * v42_data);
              v113_acc += (v75_bc * v43_data);
              v113_acc += (v77_bc * v44_data);
              v113_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v147_acc{};
              v147_acc += (v47_bc * v29_data);
              v147_acc += (v49_bc * v30_data);
              v147_acc += (v51_bc * v31_data);
              v147_acc += (v53_bc * v32_data);
              v147_acc += (v55_bc * v33_data);
              v147_acc += (v57_bc * v34_data);
              v147_acc += (v59_bc * v35_data);
              v147_acc += (v61_bc * v36_data);
              v147_acc += (v63_bc * v37_data);
              v147_acc += (v65_bc * v38_data);
              v147_acc += (v67_bc * v39_data);
              v147_acc += (v69_bc * v40_data);
              v147_acc += (v71_bc * v41_data);
              v147_acc += (v73_bc * v42_data);
              v147_acc += (v75_bc * v43_data);
              v147_acc += (v77_bc * v44_data);
              v147_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v181_acc{};
              v181_acc += (v47_bc * v29_data);
              v181_acc += (v49_bc * v30_data);
              v181_acc += (v51_bc * v31_data);
              v181_acc += (v53_bc * v32_data);
              v181_acc += (v55_bc * v33_data);
              v181_acc += (v57_bc * v34_data);
              v181_acc += (v59_bc * v35_data);
              v181_acc += (v61_bc * v36_data);
              v181_acc += (v63_bc * v37_data);
              v181_acc += (v65_bc * v38_data);
              v181_acc += (v67_bc * v39_data);
              v181_acc += (v69_bc * v40_data);
              v181_acc += (v71_bc * v41_data);
              v181_acc += (v73_bc * v42_data);
              v181_acc += (v75_bc * v43_data);
              v181_acc += (v77_bc * v44_data);
              v181_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v215_acc{};
              v215_acc += (v47_bc * v29_data);
              v215_acc += (v49_bc * v30_data);
              v215_acc += (v51_bc * v31_data);
              v215_acc += (v53_bc * v32_data);
              v215_acc += (v55_bc * v33_data);
              v215_acc += (v57_bc * v34_data);
              v215_acc += (v59_bc * v35_data);
              v215_acc += (v61_bc * v36_data);
              v215_acc += (v63_bc * v37_data);
              v215_acc += (v65_bc * v38_data);
              v215_acc += (v67_bc * v39_data);
              v215_acc += (v69_bc * v40_data);
              v215_acc += (v71_bc * v41_data);
              v215_acc += (v73_bc * v42_data);
              v215_acc += (v75_bc * v43_data);
              v215_acc += (v77_bc * v44_data);
              v215_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v249_acc{};
              v249_acc += (v47_bc * v29_data);
              v249_acc += (v49_bc * v30_data);
              v249_acc += (v51_bc * v31_data);
              v249_acc += (v53_bc * v32_data);
              v249_acc += (v55_bc * v33_data);
              v249_acc += (v57_bc * v34_data);
              v249_acc += (v59_bc * v35_data);
              v249_acc += (v61_bc * v36_data);
              v249_acc += (v63_bc * v37_data);
              v249_acc += (v65_bc * v38_data);
              v249_acc += (v67_bc * v39_data);
              v249_acc += (v69_bc * v40_data);
              v249_acc += (v71_bc * v41_data);
              v249_acc += (v73_bc * v42_data);
              v249_acc += (v75_bc * v43_data);
              v249_acc += (v77_bc * v44_data);
              v249_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v283_acc{};
              v283_acc += (v47_bc * v29_data);
              v283_acc += (v49_bc * v30_data);
              v283_acc += (v51_bc * v31_data);
              v283_acc += (v53_bc * v32_data);
              v283_acc += (v55_bc * v33_data);
              v283_acc += (v57_bc * v34_data);
              v283_acc += (v59_bc * v35_data);
              v283_acc += (v61_bc * v36_data);
              v283_acc += (v63_bc * v37_data);
              v283_acc += (v65_bc * v38_data);
              v283_acc += (v67_bc * v39_data);
              v283_acc += (v69_bc * v40_data);
              v283_acc += (v71_bc * v41_data);
              v283_acc += (v73_bc * v42_data);
              v283_acc += (v75_bc * v43_data);
              v283_acc += (v77_bc * v44_data);
              v283_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v317_acc{};
              v317_acc += (v47_bc * v29_data);
              v317_acc += (v49_bc * v30_data);
              v317_acc += (v51_bc * v31_data);
              v317_acc += (v53_bc * v32_data);
              v317_acc += (v55_bc * v33_data);
              v317_acc += (v57_bc * v34_data);
              v317_acc += (v59_bc * v35_data);
              v317_acc += (v61_bc * v36_data);
              v317_acc += (v63_bc * v37_data);
              v317_acc += (v65_bc * v38_data);
              v317_acc += (v67_bc * v39_data);
              v317_acc += (v69_bc * v40_data);
              v317_acc += (v71_bc * v41_data);
              v317_acc += (v73_bc * v42_data);
              v317_acc += (v75_bc * v43_data);
              v317_acc += (v77_bc * v44_data);
              v317_acc.copy_to(ir1 + (128));
              tensorforge::intel_esimd::simd<float, 16> v351_acc{};
              v351_acc += (v47_bc * v29_data);
              v351_acc += (v49_bc * v30_data);
              v351_acc += (v51_bc * v31_data);
              v351_acc += (v53_bc * v32_data);
              v351_acc += (v55_bc * v33_data);
              v351_acc += (v57_bc * v34_data);
              v351_acc += (v59_bc * v35_data);
              v351_acc += (v61_bc * v36_data);
              v351_acc += (v63_bc * v37_data);
              v351_acc += (v65_bc * v38_data);
              v351_acc += (v67_bc * v39_data);
              v351_acc += (v69_bc * v40_data);
              v351_acc += (v71_bc * v41_data);
              v351_acc += (v73_bc * v42_data);
              v351_acc += (v75_bc * v43_data);
              v351_acc += (v77_bc * v44_data);
              v351_acc.copy_to(ir1 + (144));
              tensorforge::intel_esimd::simd<float, 16> v385_acc{};
              v385_acc += (v47_bc * v29_data);
              v385_acc += (v49_bc * v30_data);
              v385_acc += (v51_bc * v31_data);
              v385_acc += (v53_bc * v32_data);
              v385_acc += (v55_bc * v33_data);
              v385_acc += (v57_bc * v34_data);
              v385_acc += (v59_bc * v35_data);
              v385_acc += (v61_bc * v36_data);
              v385_acc += (v63_bc * v37_data);
              v385_acc += (v65_bc * v38_data);
              v385_acc += (v67_bc * v39_data);
              v385_acc += (v69_bc * v40_data);
              v385_acc += (v71_bc * v41_data);
              v385_acc += (v73_bc * v42_data);
              v385_acc += (v75_bc * v43_data);
              v385_acc += (v77_bc * v44_data);
              v385_acc.copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 16> v419_acc{};
              v419_acc += (v47_bc * v29_data);
              v419_acc += (v49_bc * v30_data);
              v419_acc += (v51_bc * v31_data);
              v419_acc += (v53_bc * v32_data);
              v419_acc += (v55_bc * v33_data);
              v419_acc += (v57_bc * v34_data);
              v419_acc += (v59_bc * v35_data);
              v419_acc += (v61_bc * v36_data);
              v419_acc += (v63_bc * v37_data);
              v419_acc += (v65_bc * v38_data);
              v419_acc += (v67_bc * v39_data);
              v419_acc += (v69_bc * v40_data);
              v419_acc += (v71_bc * v41_data);
              v419_acc += (v73_bc * v42_data);
              v419_acc += (v75_bc * v43_data);
              v419_acc += (v77_bc * v44_data);
              v419_acc.copy_to(ir1 + (176));
              tensorforge::intel_esimd::simd<float, 16> v453_acc{};
              v453_acc += (v47_bc * v29_data);
              v453_acc += (v49_bc * v30_data);
              v453_acc += (v51_bc * v31_data);
              v453_acc += (v53_bc * v32_data);
              v453_acc += (v55_bc * v33_data);
              v453_acc += (v57_bc * v34_data);
              v453_acc += (v59_bc * v35_data);
              v453_acc += (v61_bc * v36_data);
              v453_acc += (v63_bc * v37_data);
              v453_acc += (v65_bc * v38_data);
              v453_acc += (v67_bc * v39_data);
              v453_acc += (v69_bc * v40_data);
              v453_acc += (v71_bc * v41_data);
              v453_acc += (v73_bc * v42_data);
              v453_acc += (v75_bc * v43_data);
              v453_acc += (v77_bc * v44_data);
              v453_acc.copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              v487_acc += (v47_bc * v29_data);
              v487_acc += (v49_bc * v30_data);
              v487_acc += (v51_bc * v31_data);
              v487_acc += (v53_bc * v32_data);
              v487_acc += (v55_bc * v33_data);
              v487_acc += (v57_bc * v34_data);
              v487_acc += (v59_bc * v35_data);
              v487_acc += (v61_bc * v36_data);
              v487_acc += (v63_bc * v37_data);
              v487_acc += (v65_bc * v38_data);
              v487_acc += (v67_bc * v39_data);
              v487_acc += (v69_bc * v40_data);
              v487_acc += (v71_bc * v41_data);
              v487_acc += (v73_bc * v42_data);
              v487_acc += (v75_bc * v43_data);
              v487_acc += (v77_bc * v44_data);
              v487_acc.copy_to(ir1 + (208));
              tensorforge::intel_esimd::simd<float, 16> v521_acc{};
              v521_acc += (v47_bc * v29_data);
              v521_acc += (v49_bc * v30_data);
              v521_acc += (v51_bc * v31_data);
              v521_acc += (v53_bc * v32_data);
              v521_acc += (v55_bc * v33_data);
              v521_acc += (v57_bc * v34_data);
              v521_acc += (v59_bc * v35_data);
              v521_acc += (v61_bc * v36_data);
              v521_acc += (v63_bc * v37_data);
              v521_acc += (v65_bc * v38_data);
              v521_acc += (v67_bc * v39_data);
              v521_acc += (v69_bc * v40_data);
              v521_acc += (v71_bc * v41_data);
              v521_acc += (v73_bc * v42_data);
              v521_acc += (v75_bc * v43_data);
              v521_acc += (v77_bc * v44_data);
              v521_acc.copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 16> v555_acc{};
              v555_acc += (v47_bc * v29_data);
              v555_acc += (v49_bc * v30_data);
              v555_acc += (v51_bc * v31_data);
              v555_acc += (v53_bc * v32_data);
              v555_acc += (v55_bc * v33_data);
              v555_acc += (v57_bc * v34_data);
              v555_acc += (v59_bc * v35_data);
              v555_acc += (v61_bc * v36_data);
              v555_acc += (v63_bc * v37_data);
              v555_acc += (v65_bc * v38_data);
              v555_acc += (v67_bc * v39_data);
              v555_acc += (v69_bc * v40_data);
              v555_acc += (v71_bc * v41_data);
              v555_acc += (v73_bc * v42_data);
              v555_acc += (v75_bc * v43_data);
              v555_acc += (v77_bc * v44_data);
              v555_acc.copy_to(ir1 + (240));
              #pragma unroll
              for (int32_t v589_n0 = 0; v589_n0 < 1; ++v589_n0) {
                int32_t v591_a = v589_n0 * 16;
                #pragma unroll
                for (int32_t v590_n1 = 0; v590_n1 < 16; ++v590_n1) {
                  int32_t v593_a = v591_a + (v590_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v594_data;
                  v594_data.copy_from(ir1 + (v593_a));
                  v594_data.copy_to(r1 + (v593_a));
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v598_i0 = 0; v598_i0 < 1; ++v598_i0) {
                int32_t v600_a = v598_i0 * 16;
                #pragma unroll
                for (int32_t v599_i1 = 0; v599_i1 < 16; ++v599_i1) {
                  int32_t v602_a = v600_a + (v599_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v603_data;
                  v603_data.copy_from(r1 + (v602_a));
                  v603_data.copy_to(glb_m0 + (v602_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

