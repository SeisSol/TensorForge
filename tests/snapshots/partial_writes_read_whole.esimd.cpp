// === base name ===
kernel_a3dea5d554e8bccb

// === header ===
void launcher_kernel_a3dea5d554e8bccb(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_a3dea5d554e8bccb(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_a3dea5d554e8bccb(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a3dea5d554e8bccb(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (3072, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×9(32×9) {0..32}×{0..9} pointer_based
        // m1 16×9(16×9) {0..16}×{0..9} pointer_based
        // m2 16×9(16×9) {0..16}×{0..9} pointer_based
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based
        // m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
        // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[384 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[384];
          float * __restrict__ s0 = &localShrMem0[96];
          float * __restrict__ s1 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v4_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v4_batchId0][0 + m4_extraOffset];
              float r0[288]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v20_lead = v18_i0 * 32;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                  int32_t v23_a = v20_lead + (v19_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v24_data;
                  v24_data.copy_from(glb_m0 + (v23_a));
                  v24_data.copy_to(r0 + (v23_a));
                }
              }
              float r2[288]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
                tensorforge::intel_esimd::simd<float, 16> v34_data;
                v34_data.copy_from(glb_m1 + ((v29_i1 * 16)));
                v34_data.copy_to(r2 + ((v29_i1 * 32)));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[288]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v38_data;
              v38_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 32> v39_data;
              v39_data.copy_from(r1 + (0));
              (v39_data + v38_data).copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 32> v41_data;
              v41_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 32> v42_data;
              v42_data.copy_from(r1 + (32));
              (v42_data + v41_data).copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 32> v44_data;
              v44_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 32> v45_data;
              v45_data.copy_from(r1 + (64));
              (v45_data + v44_data).copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 32> v47_data;
              v47_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 32> v48_data;
              v48_data.copy_from(r1 + (96));
              (v48_data + v47_data).copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 32> v51_data;
              v51_data.copy_from(r1 + (128));
              (v51_data + v50_data).copy_to(r1 + (128));
              tensorforge::intel_esimd::simd<float, 32> v53_data;
              v53_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 32> v54_data;
              v54_data.copy_from(r1 + (160));
              (v54_data + v53_data).copy_to(r1 + (160));
              tensorforge::intel_esimd::simd<float, 32> v56_data;
              v56_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 32> v57_data;
              v57_data.copy_from(r1 + (192));
              (v57_data + v56_data).copy_to(r1 + (192));
              tensorforge::intel_esimd::simd<float, 32> v59_data;
              v59_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 32> v60_data;
              v60_data.copy_from(r1 + (224));
              (v60_data + v59_data).copy_to(r1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v62_data;
              v62_data.copy_from(r0 + (256));
              tensorforge::intel_esimd::simd<float, 32> v63_data;
              v63_data.copy_from(r1 + (256));
              (v63_data + v62_data).copy_to(r1 + (256));
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v65_i0 = 0; v65_i0 < 1; ++v65_i0) {
                int32_t v67_a = v65_i0 * 32;
                #pragma unroll
                for (int32_t v66_i1 = 0; v66_i1 < 9; ++v66_i1) {
                  int32_t v69_a = v67_a + (v66_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v70_data;
                  v70_data.copy_from(r1 + (v69_a));
                  v70_data.copy_to(s0 + (v69_a));
                }
              }
              float r4[288]{};
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v76_i1 = 0; v76_i1 < 9; ++v76_i1) {
                tensorforge::intel_esimd::simd<float, 16> v81_data;
                v81_data.copy_from(glb_m2 + ((v76_i1 * 16)));
                v81_data.copy_to(r4 + ((v76_i1 * 32)));
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[288]{};
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir3[288]{};
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v87_data;
              v87_data.copy_from(ir3 + (0));
              (v87_data + v86_data).copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v90_data;
              v90_data.copy_from(ir3 + (32));
              (v90_data + v89_data).copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(ir3 + (64));
              (v93_data + v92_data).copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v95_data;
              v95_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v96_data;
              v96_data.copy_from(ir3 + (96));
              (v96_data + v95_data).copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v98_data;
              v98_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v99_data;
              v99_data.copy_from(ir3 + (128));
              (v99_data + v98_data).copy_to(ir3 + (128));
              tensorforge::intel_esimd::simd<float, 16> v101_data;
              v101_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v102_data;
              v102_data.copy_from(ir3 + (160));
              (v102_data + v101_data).copy_to(ir3 + (160));
              tensorforge::intel_esimd::simd<float, 16> v104_data;
              v104_data.copy_from(r2 + (192));
              tensorforge::intel_esimd::simd<float, 16> v105_data;
              v105_data.copy_from(ir3 + (192));
              (v105_data + v104_data).copy_to(ir3 + (192));
              tensorforge::intel_esimd::simd<float, 16> v107_data;
              v107_data.copy_from(r2 + (224));
              tensorforge::intel_esimd::simd<float, 16> v108_data;
              v108_data.copy_from(ir3 + (224));
              (v108_data + v107_data).copy_to(ir3 + (224));
              tensorforge::intel_esimd::simd<float, 16> v110_data;
              v110_data.copy_from(r2 + (256));
              tensorforge::intel_esimd::simd<float, 16> v111_data;
              v111_data.copy_from(ir3 + (256));
              (v111_data + v110_data).copy_to(ir3 + (256));
              #pragma unroll
              for (int32_t v113_n1 = 0; v113_n1 < 9; ++v113_n1) {
                int32_t v114_a = v113_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v116_data;
                v116_data.copy_from(ir3 + (v114_a));
                tensorforge::intel_esimd::simd<float, 16> v121_data;
                v121_data.copy_from(s0 + (v114_a));
                (v121_data + v116_data).copy_to(r3 + (v114_a));
              }
              // s0 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v125_i1 = 0; v125_i1 < 9; ++v125_i1) {
                int32_t v126_a = v125_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v128_data;
                v128_data.copy_from(r3 + (v126_a));
                v128_data.copy_to(s0 + (v126_a));
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v133_ld;
              v133_ld.copy_from(glb_m4 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v133_ld.copy_to(s1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 17) {
                tensorforge::intel_esimd::simd<float, 32> v134_ld;
                v134_ld.copy_from(glb_m4 + (0 + 0 + 1 * item.get_local_id(0) + 64));
                v134_ld.copy_to(s1 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[288]{};
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              float ir5[288]{};
              tensorforge::intel_esimd::simd<float, 16> v137_data;
              v137_data.copy_from(r4 + (0));
              tensorforge::intel_esimd::simd<float, 16> v138_data;
              v138_data.copy_from(ir5 + (0));
              (v138_data + v137_data).copy_to(ir5 + (0));
              tensorforge::intel_esimd::simd<float, 16> v140_data;
              v140_data.copy_from(r4 + (32));
              tensorforge::intel_esimd::simd<float, 16> v141_data;
              v141_data.copy_from(ir5 + (32));
              (v141_data + v140_data).copy_to(ir5 + (32));
              tensorforge::intel_esimd::simd<float, 16> v143_data;
              v143_data.copy_from(r4 + (64));
              tensorforge::intel_esimd::simd<float, 16> v144_data;
              v144_data.copy_from(ir5 + (64));
              (v144_data + v143_data).copy_to(ir5 + (64));
              tensorforge::intel_esimd::simd<float, 16> v146_data;
              v146_data.copy_from(r4 + (96));
              tensorforge::intel_esimd::simd<float, 16> v147_data;
              v147_data.copy_from(ir5 + (96));
              (v147_data + v146_data).copy_to(ir5 + (96));
              tensorforge::intel_esimd::simd<float, 16> v149_data;
              v149_data.copy_from(r4 + (128));
              tensorforge::intel_esimd::simd<float, 16> v150_data;
              v150_data.copy_from(ir5 + (128));
              (v150_data + v149_data).copy_to(ir5 + (128));
              tensorforge::intel_esimd::simd<float, 16> v152_data;
              v152_data.copy_from(r4 + (160));
              tensorforge::intel_esimd::simd<float, 16> v153_data;
              v153_data.copy_from(ir5 + (160));
              (v153_data + v152_data).copy_to(ir5 + (160));
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(r4 + (192));
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(ir5 + (192));
              (v156_data + v155_data).copy_to(ir5 + (192));
              tensorforge::intel_esimd::simd<float, 16> v158_data;
              v158_data.copy_from(r4 + (224));
              tensorforge::intel_esimd::simd<float, 16> v159_data;
              v159_data.copy_from(ir5 + (224));
              (v159_data + v158_data).copy_to(ir5 + (224));
              tensorforge::intel_esimd::simd<float, 16> v161_data;
              v161_data.copy_from(r4 + (256));
              tensorforge::intel_esimd::simd<float, 16> v162_data;
              v162_data.copy_from(ir5 + (256));
              (v162_data + v161_data).copy_to(ir5 + (256));
              #pragma unroll
              for (int32_t v164_n1 = 0; v164_n1 < 9; ++v164_n1) {
                int32_t v165_a = v164_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v167_data;
                v167_data.copy_from(ir5 + (v165_a));
                tensorforge::intel_esimd::simd<float, 16> v172_data;
                v172_data.copy_from(s0 + (v165_a));
                (v172_data + v167_data).copy_to(r5 + (v165_a));
              }
              // s0 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v176_i1 = 0; v176_i1 < 9; ++v176_i1) {
                int32_t v177_a = v176_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v179_data;
                v179_data.copy_from(r5 + (v177_a));
                v179_data.copy_to(s0 + (v177_a));
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              float r6[288]{};
              // r6 = +(s0 * s1) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir6[288]{};
              tensorforge::intel_esimd::simd<float, 32> v189_data;
              v189_data.copy_from(s0 + (0_i32));
              float v190_data = s1[0];
              tensorforge::intel_esimd::simd<float, 32> v192_data;
              v192_data.copy_from(ir6 + (0));
              (v192_data + (v189_data * v190_data)).copy_to(ir6 + (0));
              float v198_data = s1[9];
              tensorforge::intel_esimd::simd<float, 32> v200_data;
              v200_data.copy_from(ir6 + (32));
              (v200_data + (v189_data * v198_data)).copy_to(ir6 + (32));
              float v206_data = s1[18];
              tensorforge::intel_esimd::simd<float, 32> v208_data;
              v208_data.copy_from(ir6 + (64));
              (v208_data + (v189_data * v206_data)).copy_to(ir6 + (64));
              float v214_data = s1[27];
              tensorforge::intel_esimd::simd<float, 32> v216_data;
              v216_data.copy_from(ir6 + (96));
              (v216_data + (v189_data * v214_data)).copy_to(ir6 + (96));
              float v222_data = s1[36];
              tensorforge::intel_esimd::simd<float, 32> v224_data;
              v224_data.copy_from(ir6 + (128));
              (v224_data + (v189_data * v222_data)).copy_to(ir6 + (128));
              float v230_data = s1[45];
              tensorforge::intel_esimd::simd<float, 32> v232_data;
              v232_data.copy_from(ir6 + (160));
              (v232_data + (v189_data * v230_data)).copy_to(ir6 + (160));
              float v238_data = s1[54];
              tensorforge::intel_esimd::simd<float, 32> v240_data;
              v240_data.copy_from(ir6 + (192));
              (v240_data + (v189_data * v238_data)).copy_to(ir6 + (192));
              float v246_data = s1[63];
              tensorforge::intel_esimd::simd<float, 32> v248_data;
              v248_data.copy_from(ir6 + (224));
              (v248_data + (v189_data * v246_data)).copy_to(ir6 + (224));
              float v254_data = s1[72];
              tensorforge::intel_esimd::simd<float, 32> v256_data;
              v256_data.copy_from(ir6 + (256));
              (v256_data + (v189_data * v254_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v261_data;
              v261_data.copy_from(s0 + (32_i32));
              float v262_data = s1[1];
              tensorforge::intel_esimd::simd<float, 32> v264_data;
              v264_data.copy_from(ir6 + (0));
              (v264_data + (v261_data * v262_data)).copy_to(ir6 + (0));
              float v270_data = s1[10];
              tensorforge::intel_esimd::simd<float, 32> v272_data;
              v272_data.copy_from(ir6 + (32));
              (v272_data + (v261_data * v270_data)).copy_to(ir6 + (32));
              float v278_data = s1[19];
              tensorforge::intel_esimd::simd<float, 32> v280_data;
              v280_data.copy_from(ir6 + (64));
              (v280_data + (v261_data * v278_data)).copy_to(ir6 + (64));
              float v286_data = s1[28];
              tensorforge::intel_esimd::simd<float, 32> v288_data;
              v288_data.copy_from(ir6 + (96));
              (v288_data + (v261_data * v286_data)).copy_to(ir6 + (96));
              float v294_data = s1[37];
              tensorforge::intel_esimd::simd<float, 32> v296_data;
              v296_data.copy_from(ir6 + (128));
              (v296_data + (v261_data * v294_data)).copy_to(ir6 + (128));
              float v302_data = s1[46];
              tensorforge::intel_esimd::simd<float, 32> v304_data;
              v304_data.copy_from(ir6 + (160));
              (v304_data + (v261_data * v302_data)).copy_to(ir6 + (160));
              float v310_data = s1[55];
              tensorforge::intel_esimd::simd<float, 32> v312_data;
              v312_data.copy_from(ir6 + (192));
              (v312_data + (v261_data * v310_data)).copy_to(ir6 + (192));
              float v318_data = s1[64];
              tensorforge::intel_esimd::simd<float, 32> v320_data;
              v320_data.copy_from(ir6 + (224));
              (v320_data + (v261_data * v318_data)).copy_to(ir6 + (224));
              float v326_data = s1[73];
              tensorforge::intel_esimd::simd<float, 32> v328_data;
              v328_data.copy_from(ir6 + (256));
              (v328_data + (v261_data * v326_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v333_data;
              v333_data.copy_from(s0 + (64_i32));
              float v334_data = s1[2];
              tensorforge::intel_esimd::simd<float, 32> v336_data;
              v336_data.copy_from(ir6 + (0));
              (v336_data + (v333_data * v334_data)).copy_to(ir6 + (0));
              float v342_data = s1[11];
              tensorforge::intel_esimd::simd<float, 32> v344_data;
              v344_data.copy_from(ir6 + (32));
              (v344_data + (v333_data * v342_data)).copy_to(ir6 + (32));
              float v350_data = s1[20];
              tensorforge::intel_esimd::simd<float, 32> v352_data;
              v352_data.copy_from(ir6 + (64));
              (v352_data + (v333_data * v350_data)).copy_to(ir6 + (64));
              float v358_data = s1[29];
              tensorforge::intel_esimd::simd<float, 32> v360_data;
              v360_data.copy_from(ir6 + (96));
              (v360_data + (v333_data * v358_data)).copy_to(ir6 + (96));
              float v366_data = s1[38];
              tensorforge::intel_esimd::simd<float, 32> v368_data;
              v368_data.copy_from(ir6 + (128));
              (v368_data + (v333_data * v366_data)).copy_to(ir6 + (128));
              float v374_data = s1[47];
              tensorforge::intel_esimd::simd<float, 32> v376_data;
              v376_data.copy_from(ir6 + (160));
              (v376_data + (v333_data * v374_data)).copy_to(ir6 + (160));
              float v382_data = s1[56];
              tensorforge::intel_esimd::simd<float, 32> v384_data;
              v384_data.copy_from(ir6 + (192));
              (v384_data + (v333_data * v382_data)).copy_to(ir6 + (192));
              float v390_data = s1[65];
              tensorforge::intel_esimd::simd<float, 32> v392_data;
              v392_data.copy_from(ir6 + (224));
              (v392_data + (v333_data * v390_data)).copy_to(ir6 + (224));
              float v398_data = s1[74];
              tensorforge::intel_esimd::simd<float, 32> v400_data;
              v400_data.copy_from(ir6 + (256));
              (v400_data + (v333_data * v398_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v405_data;
              v405_data.copy_from(s0 + (96_i32));
              float v406_data = s1[3];
              tensorforge::intel_esimd::simd<float, 32> v408_data;
              v408_data.copy_from(ir6 + (0));
              (v408_data + (v405_data * v406_data)).copy_to(ir6 + (0));
              float v414_data = s1[12];
              tensorforge::intel_esimd::simd<float, 32> v416_data;
              v416_data.copy_from(ir6 + (32));
              (v416_data + (v405_data * v414_data)).copy_to(ir6 + (32));
              float v422_data = s1[21];
              tensorforge::intel_esimd::simd<float, 32> v424_data;
              v424_data.copy_from(ir6 + (64));
              (v424_data + (v405_data * v422_data)).copy_to(ir6 + (64));
              float v430_data = s1[30];
              tensorforge::intel_esimd::simd<float, 32> v432_data;
              v432_data.copy_from(ir6 + (96));
              (v432_data + (v405_data * v430_data)).copy_to(ir6 + (96));
              float v438_data = s1[39];
              tensorforge::intel_esimd::simd<float, 32> v440_data;
              v440_data.copy_from(ir6 + (128));
              (v440_data + (v405_data * v438_data)).copy_to(ir6 + (128));
              float v446_data = s1[48];
              tensorforge::intel_esimd::simd<float, 32> v448_data;
              v448_data.copy_from(ir6 + (160));
              (v448_data + (v405_data * v446_data)).copy_to(ir6 + (160));
              float v454_data = s1[57];
              tensorforge::intel_esimd::simd<float, 32> v456_data;
              v456_data.copy_from(ir6 + (192));
              (v456_data + (v405_data * v454_data)).copy_to(ir6 + (192));
              float v462_data = s1[66];
              tensorforge::intel_esimd::simd<float, 32> v464_data;
              v464_data.copy_from(ir6 + (224));
              (v464_data + (v405_data * v462_data)).copy_to(ir6 + (224));
              float v470_data = s1[75];
              tensorforge::intel_esimd::simd<float, 32> v472_data;
              v472_data.copy_from(ir6 + (256));
              (v472_data + (v405_data * v470_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v477_data;
              v477_data.copy_from(s0 + (128_i32));
              float v478_data = s1[4];
              tensorforge::intel_esimd::simd<float, 32> v480_data;
              v480_data.copy_from(ir6 + (0));
              (v480_data + (v477_data * v478_data)).copy_to(ir6 + (0));
              float v486_data = s1[13];
              tensorforge::intel_esimd::simd<float, 32> v488_data;
              v488_data.copy_from(ir6 + (32));
              (v488_data + (v477_data * v486_data)).copy_to(ir6 + (32));
              float v494_data = s1[22];
              tensorforge::intel_esimd::simd<float, 32> v496_data;
              v496_data.copy_from(ir6 + (64));
              (v496_data + (v477_data * v494_data)).copy_to(ir6 + (64));
              float v502_data = s1[31];
              tensorforge::intel_esimd::simd<float, 32> v504_data;
              v504_data.copy_from(ir6 + (96));
              (v504_data + (v477_data * v502_data)).copy_to(ir6 + (96));
              float v510_data = s1[40];
              tensorforge::intel_esimd::simd<float, 32> v512_data;
              v512_data.copy_from(ir6 + (128));
              (v512_data + (v477_data * v510_data)).copy_to(ir6 + (128));
              float v518_data = s1[49];
              tensorforge::intel_esimd::simd<float, 32> v520_data;
              v520_data.copy_from(ir6 + (160));
              (v520_data + (v477_data * v518_data)).copy_to(ir6 + (160));
              float v526_data = s1[58];
              tensorforge::intel_esimd::simd<float, 32> v528_data;
              v528_data.copy_from(ir6 + (192));
              (v528_data + (v477_data * v526_data)).copy_to(ir6 + (192));
              float v534_data = s1[67];
              tensorforge::intel_esimd::simd<float, 32> v536_data;
              v536_data.copy_from(ir6 + (224));
              (v536_data + (v477_data * v534_data)).copy_to(ir6 + (224));
              float v542_data = s1[76];
              tensorforge::intel_esimd::simd<float, 32> v544_data;
              v544_data.copy_from(ir6 + (256));
              (v544_data + (v477_data * v542_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v549_data;
              v549_data.copy_from(s0 + (160_i32));
              float v550_data = s1[5];
              tensorforge::intel_esimd::simd<float, 32> v552_data;
              v552_data.copy_from(ir6 + (0));
              (v552_data + (v549_data * v550_data)).copy_to(ir6 + (0));
              float v558_data = s1[14];
              tensorforge::intel_esimd::simd<float, 32> v560_data;
              v560_data.copy_from(ir6 + (32));
              (v560_data + (v549_data * v558_data)).copy_to(ir6 + (32));
              float v566_data = s1[23];
              tensorforge::intel_esimd::simd<float, 32> v568_data;
              v568_data.copy_from(ir6 + (64));
              (v568_data + (v549_data * v566_data)).copy_to(ir6 + (64));
              float v574_data = s1[32];
              tensorforge::intel_esimd::simd<float, 32> v576_data;
              v576_data.copy_from(ir6 + (96));
              (v576_data + (v549_data * v574_data)).copy_to(ir6 + (96));
              float v582_data = s1[41];
              tensorforge::intel_esimd::simd<float, 32> v584_data;
              v584_data.copy_from(ir6 + (128));
              (v584_data + (v549_data * v582_data)).copy_to(ir6 + (128));
              float v590_data = s1[50];
              tensorforge::intel_esimd::simd<float, 32> v592_data;
              v592_data.copy_from(ir6 + (160));
              (v592_data + (v549_data * v590_data)).copy_to(ir6 + (160));
              float v598_data = s1[59];
              tensorforge::intel_esimd::simd<float, 32> v600_data;
              v600_data.copy_from(ir6 + (192));
              (v600_data + (v549_data * v598_data)).copy_to(ir6 + (192));
              float v606_data = s1[68];
              tensorforge::intel_esimd::simd<float, 32> v608_data;
              v608_data.copy_from(ir6 + (224));
              (v608_data + (v549_data * v606_data)).copy_to(ir6 + (224));
              float v614_data = s1[77];
              tensorforge::intel_esimd::simd<float, 32> v616_data;
              v616_data.copy_from(ir6 + (256));
              (v616_data + (v549_data * v614_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v621_data;
              v621_data.copy_from(s0 + (192_i32));
              float v622_data = s1[6];
              tensorforge::intel_esimd::simd<float, 32> v624_data;
              v624_data.copy_from(ir6 + (0));
              (v624_data + (v621_data * v622_data)).copy_to(ir6 + (0));
              float v630_data = s1[15];
              tensorforge::intel_esimd::simd<float, 32> v632_data;
              v632_data.copy_from(ir6 + (32));
              (v632_data + (v621_data * v630_data)).copy_to(ir6 + (32));
              float v638_data = s1[24];
              tensorforge::intel_esimd::simd<float, 32> v640_data;
              v640_data.copy_from(ir6 + (64));
              (v640_data + (v621_data * v638_data)).copy_to(ir6 + (64));
              float v646_data = s1[33];
              tensorforge::intel_esimd::simd<float, 32> v648_data;
              v648_data.copy_from(ir6 + (96));
              (v648_data + (v621_data * v646_data)).copy_to(ir6 + (96));
              float v654_data = s1[42];
              tensorforge::intel_esimd::simd<float, 32> v656_data;
              v656_data.copy_from(ir6 + (128));
              (v656_data + (v621_data * v654_data)).copy_to(ir6 + (128));
              float v662_data = s1[51];
              tensorforge::intel_esimd::simd<float, 32> v664_data;
              v664_data.copy_from(ir6 + (160));
              (v664_data + (v621_data * v662_data)).copy_to(ir6 + (160));
              float v670_data = s1[60];
              tensorforge::intel_esimd::simd<float, 32> v672_data;
              v672_data.copy_from(ir6 + (192));
              (v672_data + (v621_data * v670_data)).copy_to(ir6 + (192));
              float v678_data = s1[69];
              tensorforge::intel_esimd::simd<float, 32> v680_data;
              v680_data.copy_from(ir6 + (224));
              (v680_data + (v621_data * v678_data)).copy_to(ir6 + (224));
              float v686_data = s1[78];
              tensorforge::intel_esimd::simd<float, 32> v688_data;
              v688_data.copy_from(ir6 + (256));
              (v688_data + (v621_data * v686_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v693_data;
              v693_data.copy_from(s0 + (224_i32));
              float v694_data = s1[7];
              tensorforge::intel_esimd::simd<float, 32> v696_data;
              v696_data.copy_from(ir6 + (0));
              (v696_data + (v693_data * v694_data)).copy_to(ir6 + (0));
              float v702_data = s1[16];
              tensorforge::intel_esimd::simd<float, 32> v704_data;
              v704_data.copy_from(ir6 + (32));
              (v704_data + (v693_data * v702_data)).copy_to(ir6 + (32));
              float v710_data = s1[25];
              tensorforge::intel_esimd::simd<float, 32> v712_data;
              v712_data.copy_from(ir6 + (64));
              (v712_data + (v693_data * v710_data)).copy_to(ir6 + (64));
              float v718_data = s1[34];
              tensorforge::intel_esimd::simd<float, 32> v720_data;
              v720_data.copy_from(ir6 + (96));
              (v720_data + (v693_data * v718_data)).copy_to(ir6 + (96));
              float v726_data = s1[43];
              tensorforge::intel_esimd::simd<float, 32> v728_data;
              v728_data.copy_from(ir6 + (128));
              (v728_data + (v693_data * v726_data)).copy_to(ir6 + (128));
              float v734_data = s1[52];
              tensorforge::intel_esimd::simd<float, 32> v736_data;
              v736_data.copy_from(ir6 + (160));
              (v736_data + (v693_data * v734_data)).copy_to(ir6 + (160));
              float v742_data = s1[61];
              tensorforge::intel_esimd::simd<float, 32> v744_data;
              v744_data.copy_from(ir6 + (192));
              (v744_data + (v693_data * v742_data)).copy_to(ir6 + (192));
              float v750_data = s1[70];
              tensorforge::intel_esimd::simd<float, 32> v752_data;
              v752_data.copy_from(ir6 + (224));
              (v752_data + (v693_data * v750_data)).copy_to(ir6 + (224));
              float v758_data = s1[79];
              tensorforge::intel_esimd::simd<float, 32> v760_data;
              v760_data.copy_from(ir6 + (256));
              (v760_data + (v693_data * v758_data)).copy_to(ir6 + (256));
              tensorforge::intel_esimd::simd<float, 32> v765_data;
              v765_data.copy_from(s0 + (256_i32));
              float v766_data = s1[8];
              tensorforge::intel_esimd::simd<float, 32> v768_data;
              v768_data.copy_from(ir6 + (0));
              (v768_data + (v765_data * v766_data)).copy_to(ir6 + (0));
              float v774_data = s1[17];
              tensorforge::intel_esimd::simd<float, 32> v776_data;
              v776_data.copy_from(ir6 + (32));
              (v776_data + (v765_data * v774_data)).copy_to(ir6 + (32));
              float v782_data = s1[26];
              tensorforge::intel_esimd::simd<float, 32> v784_data;
              v784_data.copy_from(ir6 + (64));
              (v784_data + (v765_data * v782_data)).copy_to(ir6 + (64));
              float v790_data = s1[35];
              tensorforge::intel_esimd::simd<float, 32> v792_data;
              v792_data.copy_from(ir6 + (96));
              (v792_data + (v765_data * v790_data)).copy_to(ir6 + (96));
              float v798_data = s1[44];
              tensorforge::intel_esimd::simd<float, 32> v800_data;
              v800_data.copy_from(ir6 + (128));
              (v800_data + (v765_data * v798_data)).copy_to(ir6 + (128));
              float v806_data = s1[53];
              tensorforge::intel_esimd::simd<float, 32> v808_data;
              v808_data.copy_from(ir6 + (160));
              (v808_data + (v765_data * v806_data)).copy_to(ir6 + (160));
              float v814_data = s1[62];
              tensorforge::intel_esimd::simd<float, 32> v816_data;
              v816_data.copy_from(ir6 + (192));
              (v816_data + (v765_data * v814_data)).copy_to(ir6 + (192));
              float v822_data = s1[71];
              tensorforge::intel_esimd::simd<float, 32> v824_data;
              v824_data.copy_from(ir6 + (224));
              (v824_data + (v765_data * v822_data)).copy_to(ir6 + (224));
              float v830_data = s1[80];
              tensorforge::intel_esimd::simd<float, 32> v832_data;
              v832_data.copy_from(ir6 + (256));
              (v832_data + (v765_data * v830_data)).copy_to(ir6 + (256));
              #pragma unroll
              for (int32_t v834_n0 = 0; v834_n0 < 1; ++v834_n0) {
                int32_t v836_a = v834_n0 * 32;
                #pragma unroll
                for (int32_t v835_n1 = 0; v835_n1 < 9; ++v835_n1) {
                  int32_t v838_a = v836_a + (v835_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v839_data;
                  v839_data.copy_from(ir6 + (v838_a));
                  v839_data.copy_to(r6 + (v838_a));
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v843_i0 = 0; v843_i0 < 1; ++v843_i0) {
                int32_t v845_a = v843_i0 * 32;
                #pragma unroll
                for (int32_t v844_i1 = 0; v844_i1 < 9; ++v844_i1) {
                  int32_t v847_a = v845_a + (v844_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v848_data;
                  v848_data.copy_from(r6 + (v847_a));
                  v848_data.copy_to(glb_m3 + (v847_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

