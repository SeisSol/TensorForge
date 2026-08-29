// === base name ===
kernel_f94e030d8c

// === header ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f94e030d8c(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f94e030d8c(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f94e030d8c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (9472, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 20×12(20×12) {0..20}×{0..12} strided
        // m2 20×16(20×16) {0..20}×{0..16} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[592 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[576];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[320];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v4_i0 = 0; v4_i0 < 1; ++v4_i0) {
                int32_t v6_lead = v4_i0 * 16;
                #pragma unroll
                for (int32_t v5_i1 = 0; v5_i1 < 12; ++v5_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v10_data;
                  v10_data.copy_from(glb_m1 + ((v6_lead + (v5_i1 * 20))));
                  int32_t v14_a = v6_lead + (v5_i1 * 21);
                  v10_data.copy_to(s0 + ((v14_a ^ ((v14_a >> 2) & 3))));
                }
              }
              #pragma unroll
              for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
                tensorforge::intel_esimd::simd<float, 4> v24_data;
                v24_data.copy_from(glb_m1 + ((16_i32 + (v18_i1 * 20))));
                int32_t v29_a = 16_i32 + (v18_i1 * 21);
                v24_data.copy_to(s0 + ((v29_a ^ ((v29_a >> 2) & 3))));
              }
              float* __restrict__ s1 = &localShrMem0[0];
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                *(sycl::vec<float, 4>*)&s1[0 + 0 + 4 * item.get_local_id(0) + i * 16] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + i * 16];
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(s0 + ((0_i32 ^ ((0_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(s0 + ((1_i32 ^ ((1_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(s0 + ((2_i32 ^ ((2_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(s0 + ((3_i32 ^ ((3_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(s0 + ((4_i32 ^ ((4_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v83_data;
              v83_data.copy_from(s0 + ((5_i32 ^ ((5_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v91_data;
              v91_data.copy_from(s0 + ((6_i32 ^ ((6_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v99_data;
              v99_data.copy_from(s0 + ((7_i32 ^ ((7_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v107_data;
              v107_data.copy_from(s0 + ((8_i32 ^ ((8_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v115_data;
              v115_data.copy_from(s0 + ((9_i32 ^ ((9_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v123_data;
              v123_data.copy_from(s0 + ((10_i32 ^ ((10_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v131_data;
              v131_data.copy_from(s0 + ((11_i32 ^ ((11_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v139_data;
              v139_data.copy_from(s0 + ((12_i32 ^ ((12_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v147_data;
              v147_data.copy_from(s0 + ((13_i32 ^ ((13_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(s0 + ((14_i32 ^ ((14_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v163_data;
              v163_data.copy_from(s0 + ((15_i32 ^ ((15_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v171_data;
              v171_data.copy_from(s0 + ((16_i32 ^ ((16_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v179_data;
              v179_data.copy_from(s0 + ((17_i32 ^ ((17_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v187_data;
              v187_data.copy_from(s0 + ((18_i32 ^ ((18_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v195_data;
              v195_data.copy_from(s0 + ((19_i32 ^ ((19_i32 >> 2) & 3))));
              tensorforge::intel_esimd::simd<float, 16> v196_acc{};
              tensorforge::intel_esimd::simd<float, 16> v203_data;
              v203_data.copy_from(s1 + ((0_i32 ^ ((0_i32 >> 5) & 31))));
              v196_acc += ((v203_data[0]) * v43_data);
              v196_acc += ((v203_data[1]) * v51_data);
              v196_acc += ((v203_data[2]) * v59_data);
              v196_acc += ((v203_data[3]) * v67_data);
              v196_acc += ((v203_data[4]) * v75_data);
              v196_acc += ((v203_data[5]) * v83_data);
              v196_acc += ((v203_data[6]) * v91_data);
              v196_acc += ((v203_data[7]) * v99_data);
              v196_acc += ((v203_data[8]) * v107_data);
              v196_acc += ((v203_data[9]) * v115_data);
              v196_acc += ((v203_data[10]) * v123_data);
              v196_acc += ((v203_data[11]) * v131_data);
              v196_acc += ((v203_data[12]) * v139_data);
              v196_acc += ((v203_data[13]) * v147_data);
              v196_acc += ((v203_data[14]) * v155_data);
              v196_acc += ((v203_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v242_data;
              v242_data.copy_from(s1 + ((16_i32 ^ ((16_i32 >> 5) & 31))));
              v196_acc += ((v242_data[0]) * v171_data);
              v196_acc += ((v242_data[1]) * v179_data);
              v196_acc += ((v242_data[2]) * v187_data);
              v196_acc += ((v242_data[3]) * v195_data);
              v196_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v251_acc{};
              tensorforge::intel_esimd::simd<float, 16> v258_data;
              v258_data.copy_from(s1 + ((20_i32 ^ ((20_i32 >> 5) & 31))));
              v251_acc += ((v258_data[0]) * v43_data);
              v251_acc += ((v258_data[1]) * v51_data);
              v251_acc += ((v258_data[2]) * v59_data);
              v251_acc += ((v258_data[3]) * v67_data);
              v251_acc += ((v258_data[4]) * v75_data);
              v251_acc += ((v258_data[5]) * v83_data);
              v251_acc += ((v258_data[6]) * v91_data);
              v251_acc += ((v258_data[7]) * v99_data);
              v251_acc += ((v258_data[8]) * v107_data);
              v251_acc += ((v258_data[9]) * v115_data);
              v251_acc += ((v258_data[10]) * v123_data);
              v251_acc += ((v258_data[11]) * v131_data);
              v251_acc += ((v258_data[12]) * v139_data);
              v251_acc += ((v258_data[13]) * v147_data);
              v251_acc += ((v258_data[14]) * v155_data);
              v251_acc += ((v258_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v297_data;
              v297_data.copy_from(s1 + ((36_i32 ^ ((36_i32 >> 5) & 31))));
              v251_acc += ((v297_data[0]) * v171_data);
              v251_acc += ((v297_data[1]) * v179_data);
              v251_acc += ((v297_data[2]) * v187_data);
              v251_acc += ((v297_data[3]) * v195_data);
              v251_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v306_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data;
              v313_data.copy_from(s1 + ((40_i32 ^ ((40_i32 >> 5) & 31))));
              v306_acc += ((v313_data[0]) * v43_data);
              v306_acc += ((v313_data[1]) * v51_data);
              v306_acc += ((v313_data[2]) * v59_data);
              v306_acc += ((v313_data[3]) * v67_data);
              v306_acc += ((v313_data[4]) * v75_data);
              v306_acc += ((v313_data[5]) * v83_data);
              v306_acc += ((v313_data[6]) * v91_data);
              v306_acc += ((v313_data[7]) * v99_data);
              v306_acc += ((v313_data[8]) * v107_data);
              v306_acc += ((v313_data[9]) * v115_data);
              v306_acc += ((v313_data[10]) * v123_data);
              v306_acc += ((v313_data[11]) * v131_data);
              v306_acc += ((v313_data[12]) * v139_data);
              v306_acc += ((v313_data[13]) * v147_data);
              v306_acc += ((v313_data[14]) * v155_data);
              v306_acc += ((v313_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v352_data;
              v352_data.copy_from(s1 + ((56_i32 ^ ((56_i32 >> 5) & 31))));
              v306_acc += ((v352_data[0]) * v171_data);
              v306_acc += ((v352_data[1]) * v179_data);
              v306_acc += ((v352_data[2]) * v187_data);
              v306_acc += ((v352_data[3]) * v195_data);
              v306_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v361_acc{};
              tensorforge::intel_esimd::simd<float, 16> v368_data;
              v368_data.copy_from(s1 + ((60_i32 ^ ((60_i32 >> 5) & 31))));
              v361_acc += ((v368_data[0]) * v43_data);
              v361_acc += ((v368_data[1]) * v51_data);
              v361_acc += ((v368_data[2]) * v59_data);
              v361_acc += ((v368_data[3]) * v67_data);
              v361_acc += ((v368_data[4]) * v75_data);
              v361_acc += ((v368_data[5]) * v83_data);
              v361_acc += ((v368_data[6]) * v91_data);
              v361_acc += ((v368_data[7]) * v99_data);
              v361_acc += ((v368_data[8]) * v107_data);
              v361_acc += ((v368_data[9]) * v115_data);
              v361_acc += ((v368_data[10]) * v123_data);
              v361_acc += ((v368_data[11]) * v131_data);
              v361_acc += ((v368_data[12]) * v139_data);
              v361_acc += ((v368_data[13]) * v147_data);
              v361_acc += ((v368_data[14]) * v155_data);
              v361_acc += ((v368_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v407_data;
              v407_data.copy_from(s1 + ((76_i32 ^ ((76_i32 >> 5) & 31))));
              v361_acc += ((v407_data[0]) * v171_data);
              v361_acc += ((v407_data[1]) * v179_data);
              v361_acc += ((v407_data[2]) * v187_data);
              v361_acc += ((v407_data[3]) * v195_data);
              v361_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v416_acc{};
              tensorforge::intel_esimd::simd<float, 16> v423_data;
              v423_data.copy_from(s1 + ((80_i32 ^ ((80_i32 >> 5) & 31))));
              v416_acc += ((v423_data[0]) * v43_data);
              v416_acc += ((v423_data[1]) * v51_data);
              v416_acc += ((v423_data[2]) * v59_data);
              v416_acc += ((v423_data[3]) * v67_data);
              v416_acc += ((v423_data[4]) * v75_data);
              v416_acc += ((v423_data[5]) * v83_data);
              v416_acc += ((v423_data[6]) * v91_data);
              v416_acc += ((v423_data[7]) * v99_data);
              v416_acc += ((v423_data[8]) * v107_data);
              v416_acc += ((v423_data[9]) * v115_data);
              v416_acc += ((v423_data[10]) * v123_data);
              v416_acc += ((v423_data[11]) * v131_data);
              v416_acc += ((v423_data[12]) * v139_data);
              v416_acc += ((v423_data[13]) * v147_data);
              v416_acc += ((v423_data[14]) * v155_data);
              v416_acc += ((v423_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v462_data;
              v462_data.copy_from(s1 + ((96_i32 ^ ((96_i32 >> 5) & 31))));
              v416_acc += ((v462_data[0]) * v171_data);
              v416_acc += ((v462_data[1]) * v179_data);
              v416_acc += ((v462_data[2]) * v187_data);
              v416_acc += ((v462_data[3]) * v195_data);
              v416_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v471_acc{};
              tensorforge::intel_esimd::simd<float, 16> v478_data;
              v478_data.copy_from(s1 + ((100_i32 ^ ((100_i32 >> 5) & 31))));
              v471_acc += ((v478_data[0]) * v43_data);
              v471_acc += ((v478_data[1]) * v51_data);
              v471_acc += ((v478_data[2]) * v59_data);
              v471_acc += ((v478_data[3]) * v67_data);
              v471_acc += ((v478_data[4]) * v75_data);
              v471_acc += ((v478_data[5]) * v83_data);
              v471_acc += ((v478_data[6]) * v91_data);
              v471_acc += ((v478_data[7]) * v99_data);
              v471_acc += ((v478_data[8]) * v107_data);
              v471_acc += ((v478_data[9]) * v115_data);
              v471_acc += ((v478_data[10]) * v123_data);
              v471_acc += ((v478_data[11]) * v131_data);
              v471_acc += ((v478_data[12]) * v139_data);
              v471_acc += ((v478_data[13]) * v147_data);
              v471_acc += ((v478_data[14]) * v155_data);
              v471_acc += ((v478_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v517_data;
              v517_data.copy_from(s1 + ((116_i32 ^ ((116_i32 >> 5) & 31))));
              v471_acc += ((v517_data[0]) * v171_data);
              v471_acc += ((v517_data[1]) * v179_data);
              v471_acc += ((v517_data[2]) * v187_data);
              v471_acc += ((v517_data[3]) * v195_data);
              v471_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v526_acc{};
              tensorforge::intel_esimd::simd<float, 16> v533_data;
              v533_data.copy_from(s1 + ((120_i32 ^ ((120_i32 >> 5) & 31))));
              v526_acc += ((v533_data[0]) * v43_data);
              v526_acc += ((v533_data[1]) * v51_data);
              v526_acc += ((v533_data[2]) * v59_data);
              v526_acc += ((v533_data[3]) * v67_data);
              v526_acc += ((v533_data[4]) * v75_data);
              v526_acc += ((v533_data[5]) * v83_data);
              v526_acc += ((v533_data[6]) * v91_data);
              v526_acc += ((v533_data[7]) * v99_data);
              v526_acc += ((v533_data[8]) * v107_data);
              v526_acc += ((v533_data[9]) * v115_data);
              v526_acc += ((v533_data[10]) * v123_data);
              v526_acc += ((v533_data[11]) * v131_data);
              v526_acc += ((v533_data[12]) * v139_data);
              v526_acc += ((v533_data[13]) * v147_data);
              v526_acc += ((v533_data[14]) * v155_data);
              v526_acc += ((v533_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v572_data;
              v572_data.copy_from(s1 + ((136_i32 ^ ((136_i32 >> 5) & 31))));
              v526_acc += ((v572_data[0]) * v171_data);
              v526_acc += ((v572_data[1]) * v179_data);
              v526_acc += ((v572_data[2]) * v187_data);
              v526_acc += ((v572_data[3]) * v195_data);
              v526_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v581_acc{};
              tensorforge::intel_esimd::simd<float, 16> v588_data;
              v588_data.copy_from(s1 + ((140_i32 ^ ((140_i32 >> 5) & 31))));
              v581_acc += ((v588_data[0]) * v43_data);
              v581_acc += ((v588_data[1]) * v51_data);
              v581_acc += ((v588_data[2]) * v59_data);
              v581_acc += ((v588_data[3]) * v67_data);
              v581_acc += ((v588_data[4]) * v75_data);
              v581_acc += ((v588_data[5]) * v83_data);
              v581_acc += ((v588_data[6]) * v91_data);
              v581_acc += ((v588_data[7]) * v99_data);
              v581_acc += ((v588_data[8]) * v107_data);
              v581_acc += ((v588_data[9]) * v115_data);
              v581_acc += ((v588_data[10]) * v123_data);
              v581_acc += ((v588_data[11]) * v131_data);
              v581_acc += ((v588_data[12]) * v139_data);
              v581_acc += ((v588_data[13]) * v147_data);
              v581_acc += ((v588_data[14]) * v155_data);
              v581_acc += ((v588_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v627_data;
              v627_data.copy_from(s1 + ((156_i32 ^ ((156_i32 >> 5) & 31))));
              v581_acc += ((v627_data[0]) * v171_data);
              v581_acc += ((v627_data[1]) * v179_data);
              v581_acc += ((v627_data[2]) * v187_data);
              v581_acc += ((v627_data[3]) * v195_data);
              v581_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v636_acc{};
              tensorforge::intel_esimd::simd<float, 16> v643_data;
              v643_data.copy_from(s1 + ((160_i32 ^ ((160_i32 >> 5) & 31))));
              v636_acc += ((v643_data[0]) * v43_data);
              v636_acc += ((v643_data[1]) * v51_data);
              v636_acc += ((v643_data[2]) * v59_data);
              v636_acc += ((v643_data[3]) * v67_data);
              v636_acc += ((v643_data[4]) * v75_data);
              v636_acc += ((v643_data[5]) * v83_data);
              v636_acc += ((v643_data[6]) * v91_data);
              v636_acc += ((v643_data[7]) * v99_data);
              v636_acc += ((v643_data[8]) * v107_data);
              v636_acc += ((v643_data[9]) * v115_data);
              v636_acc += ((v643_data[10]) * v123_data);
              v636_acc += ((v643_data[11]) * v131_data);
              v636_acc += ((v643_data[12]) * v139_data);
              v636_acc += ((v643_data[13]) * v147_data);
              v636_acc += ((v643_data[14]) * v155_data);
              v636_acc += ((v643_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v682_data;
              v682_data.copy_from(s1 + ((176_i32 ^ ((176_i32 >> 5) & 31))));
              v636_acc += ((v682_data[0]) * v171_data);
              v636_acc += ((v682_data[1]) * v179_data);
              v636_acc += ((v682_data[2]) * v187_data);
              v636_acc += ((v682_data[3]) * v195_data);
              v636_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v691_acc{};
              tensorforge::intel_esimd::simd<float, 16> v698_data;
              v698_data.copy_from(s1 + ((180_i32 ^ ((180_i32 >> 5) & 31))));
              v691_acc += ((v698_data[0]) * v43_data);
              v691_acc += ((v698_data[1]) * v51_data);
              v691_acc += ((v698_data[2]) * v59_data);
              v691_acc += ((v698_data[3]) * v67_data);
              v691_acc += ((v698_data[4]) * v75_data);
              v691_acc += ((v698_data[5]) * v83_data);
              v691_acc += ((v698_data[6]) * v91_data);
              v691_acc += ((v698_data[7]) * v99_data);
              v691_acc += ((v698_data[8]) * v107_data);
              v691_acc += ((v698_data[9]) * v115_data);
              v691_acc += ((v698_data[10]) * v123_data);
              v691_acc += ((v698_data[11]) * v131_data);
              v691_acc += ((v698_data[12]) * v139_data);
              v691_acc += ((v698_data[13]) * v147_data);
              v691_acc += ((v698_data[14]) * v155_data);
              v691_acc += ((v698_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v737_data;
              v737_data.copy_from(s1 + ((196_i32 ^ ((196_i32 >> 5) & 31))));
              v691_acc += ((v737_data[0]) * v171_data);
              v691_acc += ((v737_data[1]) * v179_data);
              v691_acc += ((v737_data[2]) * v187_data);
              v691_acc += ((v737_data[3]) * v195_data);
              v691_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v746_acc{};
              tensorforge::intel_esimd::simd<float, 16> v753_data;
              v753_data.copy_from(s1 + ((200_i32 ^ ((200_i32 >> 5) & 31))));
              v746_acc += ((v753_data[0]) * v43_data);
              v746_acc += ((v753_data[1]) * v51_data);
              v746_acc += ((v753_data[2]) * v59_data);
              v746_acc += ((v753_data[3]) * v67_data);
              v746_acc += ((v753_data[4]) * v75_data);
              v746_acc += ((v753_data[5]) * v83_data);
              v746_acc += ((v753_data[6]) * v91_data);
              v746_acc += ((v753_data[7]) * v99_data);
              v746_acc += ((v753_data[8]) * v107_data);
              v746_acc += ((v753_data[9]) * v115_data);
              v746_acc += ((v753_data[10]) * v123_data);
              v746_acc += ((v753_data[11]) * v131_data);
              v746_acc += ((v753_data[12]) * v139_data);
              v746_acc += ((v753_data[13]) * v147_data);
              v746_acc += ((v753_data[14]) * v155_data);
              v746_acc += ((v753_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v792_data;
              v792_data.copy_from(s1 + ((216_i32 ^ ((216_i32 >> 5) & 31))));
              v746_acc += ((v792_data[0]) * v171_data);
              v746_acc += ((v792_data[1]) * v179_data);
              v746_acc += ((v792_data[2]) * v187_data);
              v746_acc += ((v792_data[3]) * v195_data);
              v746_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v801_acc{};
              tensorforge::intel_esimd::simd<float, 16> v808_data;
              v808_data.copy_from(s1 + ((220_i32 ^ ((220_i32 >> 5) & 31))));
              v801_acc += ((v808_data[0]) * v43_data);
              v801_acc += ((v808_data[1]) * v51_data);
              v801_acc += ((v808_data[2]) * v59_data);
              v801_acc += ((v808_data[3]) * v67_data);
              v801_acc += ((v808_data[4]) * v75_data);
              v801_acc += ((v808_data[5]) * v83_data);
              v801_acc += ((v808_data[6]) * v91_data);
              v801_acc += ((v808_data[7]) * v99_data);
              v801_acc += ((v808_data[8]) * v107_data);
              v801_acc += ((v808_data[9]) * v115_data);
              v801_acc += ((v808_data[10]) * v123_data);
              v801_acc += ((v808_data[11]) * v131_data);
              v801_acc += ((v808_data[12]) * v139_data);
              v801_acc += ((v808_data[13]) * v147_data);
              v801_acc += ((v808_data[14]) * v155_data);
              v801_acc += ((v808_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v847_data;
              v847_data.copy_from(s1 + ((236_i32 ^ ((236_i32 >> 5) & 31))));
              v801_acc += ((v847_data[0]) * v171_data);
              v801_acc += ((v847_data[1]) * v179_data);
              v801_acc += ((v847_data[2]) * v187_data);
              v801_acc += ((v847_data[3]) * v195_data);
              v801_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v856_acc{};
              tensorforge::intel_esimd::simd<float, 16> v863_data;
              v863_data.copy_from(s1 + ((240_i32 ^ ((240_i32 >> 5) & 31))));
              v856_acc += ((v863_data[0]) * v43_data);
              v856_acc += ((v863_data[1]) * v51_data);
              v856_acc += ((v863_data[2]) * v59_data);
              v856_acc += ((v863_data[3]) * v67_data);
              v856_acc += ((v863_data[4]) * v75_data);
              v856_acc += ((v863_data[5]) * v83_data);
              v856_acc += ((v863_data[6]) * v91_data);
              v856_acc += ((v863_data[7]) * v99_data);
              v856_acc += ((v863_data[8]) * v107_data);
              v856_acc += ((v863_data[9]) * v115_data);
              v856_acc += ((v863_data[10]) * v123_data);
              v856_acc += ((v863_data[11]) * v131_data);
              v856_acc += ((v863_data[12]) * v139_data);
              v856_acc += ((v863_data[13]) * v147_data);
              v856_acc += ((v863_data[14]) * v155_data);
              v856_acc += ((v863_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v902_data;
              v902_data.copy_from(s1 + ((256_i32 ^ ((256_i32 >> 5) & 31))));
              v856_acc += ((v902_data[0]) * v171_data);
              v856_acc += ((v902_data[1]) * v179_data);
              v856_acc += ((v902_data[2]) * v187_data);
              v856_acc += ((v902_data[3]) * v195_data);
              v856_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v911_acc{};
              tensorforge::intel_esimd::simd<float, 16> v918_data;
              v918_data.copy_from(s1 + ((260_i32 ^ ((260_i32 >> 5) & 31))));
              v911_acc += ((v918_data[0]) * v43_data);
              v911_acc += ((v918_data[1]) * v51_data);
              v911_acc += ((v918_data[2]) * v59_data);
              v911_acc += ((v918_data[3]) * v67_data);
              v911_acc += ((v918_data[4]) * v75_data);
              v911_acc += ((v918_data[5]) * v83_data);
              v911_acc += ((v918_data[6]) * v91_data);
              v911_acc += ((v918_data[7]) * v99_data);
              v911_acc += ((v918_data[8]) * v107_data);
              v911_acc += ((v918_data[9]) * v115_data);
              v911_acc += ((v918_data[10]) * v123_data);
              v911_acc += ((v918_data[11]) * v131_data);
              v911_acc += ((v918_data[12]) * v139_data);
              v911_acc += ((v918_data[13]) * v147_data);
              v911_acc += ((v918_data[14]) * v155_data);
              v911_acc += ((v918_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v957_data;
              v957_data.copy_from(s1 + ((276_i32 ^ ((276_i32 >> 5) & 31))));
              v911_acc += ((v957_data[0]) * v171_data);
              v911_acc += ((v957_data[1]) * v179_data);
              v911_acc += ((v957_data[2]) * v187_data);
              v911_acc += ((v957_data[3]) * v195_data);
              v911_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v966_acc{};
              tensorforge::intel_esimd::simd<float, 16> v973_data;
              v973_data.copy_from(s1 + ((280_i32 ^ ((280_i32 >> 5) & 31))));
              v966_acc += ((v973_data[0]) * v43_data);
              v966_acc += ((v973_data[1]) * v51_data);
              v966_acc += ((v973_data[2]) * v59_data);
              v966_acc += ((v973_data[3]) * v67_data);
              v966_acc += ((v973_data[4]) * v75_data);
              v966_acc += ((v973_data[5]) * v83_data);
              v966_acc += ((v973_data[6]) * v91_data);
              v966_acc += ((v973_data[7]) * v99_data);
              v966_acc += ((v973_data[8]) * v107_data);
              v966_acc += ((v973_data[9]) * v115_data);
              v966_acc += ((v973_data[10]) * v123_data);
              v966_acc += ((v973_data[11]) * v131_data);
              v966_acc += ((v973_data[12]) * v139_data);
              v966_acc += ((v973_data[13]) * v147_data);
              v966_acc += ((v973_data[14]) * v155_data);
              v966_acc += ((v973_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v1012_data;
              v1012_data.copy_from(s1 + ((296_i32 ^ ((296_i32 >> 5) & 31))));
              v966_acc += ((v1012_data[0]) * v171_data);
              v966_acc += ((v1012_data[1]) * v179_data);
              v966_acc += ((v1012_data[2]) * v187_data);
              v966_acc += ((v1012_data[3]) * v195_data);
              v966_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v1021_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1028_data;
              v1028_data.copy_from(s1 + ((300_i32 ^ ((300_i32 >> 5) & 31))));
              v1021_acc += ((v1028_data[0]) * v43_data);
              v1021_acc += ((v1028_data[1]) * v51_data);
              v1021_acc += ((v1028_data[2]) * v59_data);
              v1021_acc += ((v1028_data[3]) * v67_data);
              v1021_acc += ((v1028_data[4]) * v75_data);
              v1021_acc += ((v1028_data[5]) * v83_data);
              v1021_acc += ((v1028_data[6]) * v91_data);
              v1021_acc += ((v1028_data[7]) * v99_data);
              v1021_acc += ((v1028_data[8]) * v107_data);
              v1021_acc += ((v1028_data[9]) * v115_data);
              v1021_acc += ((v1028_data[10]) * v123_data);
              v1021_acc += ((v1028_data[11]) * v131_data);
              v1021_acc += ((v1028_data[12]) * v139_data);
              v1021_acc += ((v1028_data[13]) * v147_data);
              v1021_acc += ((v1028_data[14]) * v155_data);
              v1021_acc += ((v1028_data[15]) * v163_data);
              tensorforge::intel_esimd::simd<float, 16> v1067_data;
              v1067_data.copy_from(s1 + ((316_i32 ^ ((316_i32 >> 5) & 31))));
              v1021_acc += ((v1067_data[0]) * v171_data);
              v1021_acc += ((v1067_data[1]) * v179_data);
              v1021_acc += ((v1067_data[2]) * v187_data);
              v1021_acc += ((v1067_data[3]) * v195_data);
              v1021_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v1076_n1 = 0; v1076_n1 < 16; ++v1076_n1) {
                int32_t v1077_a = v1076_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1079_data;
                v1079_data.copy_from(ir0 + (v1077_a));
                v1079_data.copy_to(r0 + (v1077_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v1082_i1 = 0; v1082_i1 < 16; ++v1082_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1085_data;
                v1085_data.copy_from(r0 + ((v1082_i1 * 16)));
                v1085_data.copy_to(glb_m0 + ((v1082_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

