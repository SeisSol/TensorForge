// === base name ===
kernel_f4d83c3c68c41dff

// === header ===
void launcher_kernel_f4d83c3c68c41dff(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f4d83c3c68c41dff(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f4d83c3c68c41dff(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f4d83c3c68c41dff(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×16(32×16) {0..32}×{0..16} strided
        // m1 32×12(32×12) {0..32}×{0..12} strided
        // m2 12×16(12×16) {0..12}×{0..16} strided
        // m3 32×12(32×12) {0..32}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 32×12(32×12) {0..32}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..16})[0, 1] = m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[-1, 1]
        // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m3 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m5 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 512 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 192 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[batchId0 * 384 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[batchId0 * 384 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[batchId0 * 96 + 0 + m6_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v14_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v21_lead = v14_lead + (v15_i0 * 32);
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
                  float v24_data = glb_m1[(v21_lead + (v16_i1 * 32))];
                  r0[(v15_i0 + v16_i1)] = v24_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 16; ++v31_i1) {
                  float v39_data = glb_m2[(v14_lead + (v31_i1 * 12))];
                  r1[v31_i1] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
                int32_t v51_lead = v14_lead + (v45_i0 * 32);
                #pragma unroll
                for (int32_t v46_i1 = 0; v46_i1 < 12; ++v46_i1) {
                  float v54_data = glb_m3[(v51_lead + (v46_i1 * 32))];
                  r3[(v45_i0 + v46_i1)] = v54_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 16)] [(0, 12)]
              float ir2[16]{};
              float v61_data = r0[0];
              float v62_data = r1[0];
              float v65_data = ir2[0];
              ir2[0] = (v65_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 0))));
              float v68_data = r1[1];
              float v71_data = ir2[1];
              ir2[1] = (v71_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 0))));
              float v74_data = r1[2];
              float v77_data = ir2[2];
              ir2[2] = (v77_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 0))));
              float v80_data = r1[3];
              float v83_data = ir2[3];
              ir2[3] = (v83_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 0))));
              float v86_data = r1[4];
              float v89_data = ir2[4];
              ir2[4] = (v89_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 0))));
              float v92_data = r1[5];
              float v95_data = ir2[5];
              ir2[5] = (v95_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 0))));
              float v98_data = r1[6];
              float v101_data = ir2[6];
              ir2[6] = (v101_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 0))));
              float v104_data = r1[7];
              float v107_data = ir2[7];
              ir2[7] = (v107_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 0))));
              float v110_data = r1[8];
              float v113_data = ir2[8];
              ir2[8] = (v113_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 0))));
              float v116_data = r1[9];
              float v119_data = ir2[9];
              ir2[9] = (v119_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 0))));
              float v122_data = r1[10];
              float v125_data = ir2[10];
              ir2[10] = (v125_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 0))));
              float v128_data = r1[11];
              float v131_data = ir2[11];
              ir2[11] = (v131_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 0))));
              float v134_data = r1[12];
              float v137_data = ir2[12];
              ir2[12] = (v137_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 0))));
              float v140_data = r1[13];
              float v143_data = ir2[13];
              ir2[13] = (v143_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 0))));
              float v146_data = r1[14];
              float v149_data = ir2[14];
              ir2[14] = (v149_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 0))));
              float v152_data = r1[15];
              float v155_data = ir2[15];
              ir2[15] = (v155_data + (v61_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 0))));
              float v160_data = r0[1];
              float v164_data = ir2[0];
              ir2[0] = (v164_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 1))));
              float v170_data = ir2[1];
              ir2[1] = (v170_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 1))));
              float v176_data = ir2[2];
              ir2[2] = (v176_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 1))));
              float v182_data = ir2[3];
              ir2[3] = (v182_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 1))));
              float v188_data = ir2[4];
              ir2[4] = (v188_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 1))));
              float v194_data = ir2[5];
              ir2[5] = (v194_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 1))));
              float v200_data = ir2[6];
              ir2[6] = (v200_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 1))));
              float v206_data = ir2[7];
              ir2[7] = (v206_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 1))));
              float v212_data = ir2[8];
              ir2[8] = (v212_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 1))));
              float v218_data = ir2[9];
              ir2[9] = (v218_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 1))));
              float v224_data = ir2[10];
              ir2[10] = (v224_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 1))));
              float v230_data = ir2[11];
              ir2[11] = (v230_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 1))));
              float v236_data = ir2[12];
              ir2[12] = (v236_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 1))));
              float v242_data = ir2[13];
              ir2[13] = (v242_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 1))));
              float v248_data = ir2[14];
              ir2[14] = (v248_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 1))));
              float v254_data = ir2[15];
              ir2[15] = (v254_data + (v160_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 1))));
              float v259_data = r0[2];
              float v263_data = ir2[0];
              ir2[0] = (v263_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 2))));
              float v269_data = ir2[1];
              ir2[1] = (v269_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 2))));
              float v275_data = ir2[2];
              ir2[2] = (v275_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 2))));
              float v281_data = ir2[3];
              ir2[3] = (v281_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 2))));
              float v287_data = ir2[4];
              ir2[4] = (v287_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 2))));
              float v293_data = ir2[5];
              ir2[5] = (v293_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 2))));
              float v299_data = ir2[6];
              ir2[6] = (v299_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 2))));
              float v305_data = ir2[7];
              ir2[7] = (v305_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 2))));
              float v311_data = ir2[8];
              ir2[8] = (v311_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 2))));
              float v317_data = ir2[9];
              ir2[9] = (v317_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 2))));
              float v323_data = ir2[10];
              ir2[10] = (v323_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 2))));
              float v329_data = ir2[11];
              ir2[11] = (v329_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 2))));
              float v335_data = ir2[12];
              ir2[12] = (v335_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 2))));
              float v341_data = ir2[13];
              ir2[13] = (v341_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 2))));
              float v347_data = ir2[14];
              ir2[14] = (v347_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 2))));
              float v353_data = ir2[15];
              ir2[15] = (v353_data + (v259_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 2))));
              float v358_data = r0[3];
              float v362_data = ir2[0];
              ir2[0] = (v362_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 3))));
              float v368_data = ir2[1];
              ir2[1] = (v368_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 3))));
              float v374_data = ir2[2];
              ir2[2] = (v374_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 3))));
              float v380_data = ir2[3];
              ir2[3] = (v380_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 3))));
              float v386_data = ir2[4];
              ir2[4] = (v386_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 3))));
              float v392_data = ir2[5];
              ir2[5] = (v392_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 3))));
              float v398_data = ir2[6];
              ir2[6] = (v398_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 3))));
              float v404_data = ir2[7];
              ir2[7] = (v404_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 3))));
              float v410_data = ir2[8];
              ir2[8] = (v410_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 3))));
              float v416_data = ir2[9];
              ir2[9] = (v416_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 3))));
              float v422_data = ir2[10];
              ir2[10] = (v422_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 3))));
              float v428_data = ir2[11];
              ir2[11] = (v428_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 3))));
              float v434_data = ir2[12];
              ir2[12] = (v434_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 3))));
              float v440_data = ir2[13];
              ir2[13] = (v440_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 3))));
              float v446_data = ir2[14];
              ir2[14] = (v446_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 3))));
              float v452_data = ir2[15];
              ir2[15] = (v452_data + (v358_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 3))));
              float v457_data = r0[4];
              float v461_data = ir2[0];
              ir2[0] = (v461_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 4))));
              float v467_data = ir2[1];
              ir2[1] = (v467_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 4))));
              float v473_data = ir2[2];
              ir2[2] = (v473_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 4))));
              float v479_data = ir2[3];
              ir2[3] = (v479_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 4))));
              float v485_data = ir2[4];
              ir2[4] = (v485_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 4))));
              float v491_data = ir2[5];
              ir2[5] = (v491_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 4))));
              float v497_data = ir2[6];
              ir2[6] = (v497_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 4))));
              float v503_data = ir2[7];
              ir2[7] = (v503_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 4))));
              float v509_data = ir2[8];
              ir2[8] = (v509_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 4))));
              float v515_data = ir2[9];
              ir2[9] = (v515_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 4))));
              float v521_data = ir2[10];
              ir2[10] = (v521_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 4))));
              float v527_data = ir2[11];
              ir2[11] = (v527_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 4))));
              float v533_data = ir2[12];
              ir2[12] = (v533_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 4))));
              float v539_data = ir2[13];
              ir2[13] = (v539_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 4))));
              float v545_data = ir2[14];
              ir2[14] = (v545_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 4))));
              float v551_data = ir2[15];
              ir2[15] = (v551_data + (v457_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 4))));
              float v556_data = r0[5];
              float v560_data = ir2[0];
              ir2[0] = (v560_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 5))));
              float v566_data = ir2[1];
              ir2[1] = (v566_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 5))));
              float v572_data = ir2[2];
              ir2[2] = (v572_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 5))));
              float v578_data = ir2[3];
              ir2[3] = (v578_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 5))));
              float v584_data = ir2[4];
              ir2[4] = (v584_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 5))));
              float v590_data = ir2[5];
              ir2[5] = (v590_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 5))));
              float v596_data = ir2[6];
              ir2[6] = (v596_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 5))));
              float v602_data = ir2[7];
              ir2[7] = (v602_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 5))));
              float v608_data = ir2[8];
              ir2[8] = (v608_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 5))));
              float v614_data = ir2[9];
              ir2[9] = (v614_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 5))));
              float v620_data = ir2[10];
              ir2[10] = (v620_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 5))));
              float v626_data = ir2[11];
              ir2[11] = (v626_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 5))));
              float v632_data = ir2[12];
              ir2[12] = (v632_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 5))));
              float v638_data = ir2[13];
              ir2[13] = (v638_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 5))));
              float v644_data = ir2[14];
              ir2[14] = (v644_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 5))));
              float v650_data = ir2[15];
              ir2[15] = (v650_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 5))));
              float v655_data = r0[6];
              float v659_data = ir2[0];
              ir2[0] = (v659_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 6))));
              float v665_data = ir2[1];
              ir2[1] = (v665_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 6))));
              float v671_data = ir2[2];
              ir2[2] = (v671_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 6))));
              float v677_data = ir2[3];
              ir2[3] = (v677_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 6))));
              float v683_data = ir2[4];
              ir2[4] = (v683_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 6))));
              float v689_data = ir2[5];
              ir2[5] = (v689_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 6))));
              float v695_data = ir2[6];
              ir2[6] = (v695_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 6))));
              float v701_data = ir2[7];
              ir2[7] = (v701_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 6))));
              float v707_data = ir2[8];
              ir2[8] = (v707_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 6))));
              float v713_data = ir2[9];
              ir2[9] = (v713_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 6))));
              float v719_data = ir2[10];
              ir2[10] = (v719_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 6))));
              float v725_data = ir2[11];
              ir2[11] = (v725_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 6))));
              float v731_data = ir2[12];
              ir2[12] = (v731_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 6))));
              float v737_data = ir2[13];
              ir2[13] = (v737_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 6))));
              float v743_data = ir2[14];
              ir2[14] = (v743_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 6))));
              float v749_data = ir2[15];
              ir2[15] = (v749_data + (v655_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 6))));
              float v754_data = r0[7];
              float v758_data = ir2[0];
              ir2[0] = (v758_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 7))));
              float v764_data = ir2[1];
              ir2[1] = (v764_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 7))));
              float v770_data = ir2[2];
              ir2[2] = (v770_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 7))));
              float v776_data = ir2[3];
              ir2[3] = (v776_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 7))));
              float v782_data = ir2[4];
              ir2[4] = (v782_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 7))));
              float v788_data = ir2[5];
              ir2[5] = (v788_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 7))));
              float v794_data = ir2[6];
              ir2[6] = (v794_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 7))));
              float v800_data = ir2[7];
              ir2[7] = (v800_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 7))));
              float v806_data = ir2[8];
              ir2[8] = (v806_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 7))));
              float v812_data = ir2[9];
              ir2[9] = (v812_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 7))));
              float v818_data = ir2[10];
              ir2[10] = (v818_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 7))));
              float v824_data = ir2[11];
              ir2[11] = (v824_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 7))));
              float v830_data = ir2[12];
              ir2[12] = (v830_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 7))));
              float v836_data = ir2[13];
              ir2[13] = (v836_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 7))));
              float v842_data = ir2[14];
              ir2[14] = (v842_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 7))));
              float v848_data = ir2[15];
              ir2[15] = (v848_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 7))));
              float v853_data = r0[8];
              float v857_data = ir2[0];
              ir2[0] = (v857_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 8))));
              float v863_data = ir2[1];
              ir2[1] = (v863_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 8))));
              float v869_data = ir2[2];
              ir2[2] = (v869_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 8))));
              float v875_data = ir2[3];
              ir2[3] = (v875_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 8))));
              float v881_data = ir2[4];
              ir2[4] = (v881_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 8))));
              float v887_data = ir2[5];
              ir2[5] = (v887_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 8))));
              float v893_data = ir2[6];
              ir2[6] = (v893_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 8))));
              float v899_data = ir2[7];
              ir2[7] = (v899_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 8))));
              float v905_data = ir2[8];
              ir2[8] = (v905_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 8))));
              float v911_data = ir2[9];
              ir2[9] = (v911_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 8))));
              float v917_data = ir2[10];
              ir2[10] = (v917_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 8))));
              float v923_data = ir2[11];
              ir2[11] = (v923_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 8))));
              float v929_data = ir2[12];
              ir2[12] = (v929_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 8))));
              float v935_data = ir2[13];
              ir2[13] = (v935_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 8))));
              float v941_data = ir2[14];
              ir2[14] = (v941_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 8))));
              float v947_data = ir2[15];
              ir2[15] = (v947_data + (v853_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 8))));
              float v952_data = r0[9];
              float v956_data = ir2[0];
              ir2[0] = (v956_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 9))));
              float v962_data = ir2[1];
              ir2[1] = (v962_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 9))));
              float v968_data = ir2[2];
              ir2[2] = (v968_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 9))));
              float v974_data = ir2[3];
              ir2[3] = (v974_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 9))));
              float v980_data = ir2[4];
              ir2[4] = (v980_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 9))));
              float v986_data = ir2[5];
              ir2[5] = (v986_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 9))));
              float v992_data = ir2[6];
              ir2[6] = (v992_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 9))));
              float v998_data = ir2[7];
              ir2[7] = (v998_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 9))));
              float v1004_data = ir2[8];
              ir2[8] = (v1004_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 9))));
              float v1010_data = ir2[9];
              ir2[9] = (v1010_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 9))));
              float v1016_data = ir2[10];
              ir2[10] = (v1016_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 9))));
              float v1022_data = ir2[11];
              ir2[11] = (v1022_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 9))));
              float v1028_data = ir2[12];
              ir2[12] = (v1028_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 9))));
              float v1034_data = ir2[13];
              ir2[13] = (v1034_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 9))));
              float v1040_data = ir2[14];
              ir2[14] = (v1040_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 9))));
              float v1046_data = ir2[15];
              ir2[15] = (v1046_data + (v952_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 9))));
              float v1051_data = r0[10];
              float v1055_data = ir2[0];
              ir2[0] = (v1055_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 10))));
              float v1061_data = ir2[1];
              ir2[1] = (v1061_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 10))));
              float v1067_data = ir2[2];
              ir2[2] = (v1067_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 10))));
              float v1073_data = ir2[3];
              ir2[3] = (v1073_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 10))));
              float v1079_data = ir2[4];
              ir2[4] = (v1079_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 10))));
              float v1085_data = ir2[5];
              ir2[5] = (v1085_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 10))));
              float v1091_data = ir2[6];
              ir2[6] = (v1091_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 10))));
              float v1097_data = ir2[7];
              ir2[7] = (v1097_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 10))));
              float v1103_data = ir2[8];
              ir2[8] = (v1103_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 10))));
              float v1109_data = ir2[9];
              ir2[9] = (v1109_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 10))));
              float v1115_data = ir2[10];
              ir2[10] = (v1115_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 10))));
              float v1121_data = ir2[11];
              ir2[11] = (v1121_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 10))));
              float v1127_data = ir2[12];
              ir2[12] = (v1127_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 10))));
              float v1133_data = ir2[13];
              ir2[13] = (v1133_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 10))));
              float v1139_data = ir2[14];
              ir2[14] = (v1139_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 10))));
              float v1145_data = ir2[15];
              ir2[15] = (v1145_data + (v1051_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 10))));
              float v1150_data = r0[11];
              float v1154_data = ir2[0];
              ir2[0] = (v1154_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 11))));
              float v1160_data = ir2[1];
              ir2[1] = (v1160_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 11))));
              float v1166_data = ir2[2];
              ir2[2] = (v1166_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 11))));
              float v1172_data = ir2[3];
              ir2[3] = (v1172_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 11))));
              float v1178_data = ir2[4];
              ir2[4] = (v1178_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 11))));
              float v1184_data = ir2[5];
              ir2[5] = (v1184_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 11))));
              float v1190_data = ir2[6];
              ir2[6] = (v1190_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 11))));
              float v1196_data = ir2[7];
              ir2[7] = (v1196_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 11))));
              float v1202_data = ir2[8];
              ir2[8] = (v1202_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 11))));
              float v1208_data = ir2[9];
              ir2[9] = (v1208_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 11))));
              float v1214_data = ir2[10];
              ir2[10] = (v1214_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 11))));
              float v1220_data = ir2[11];
              ir2[11] = (v1220_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 11))));
              float v1226_data = ir2[12];
              ir2[12] = (v1226_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 11))));
              float v1232_data = ir2[13];
              ir2[13] = (v1232_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 11))));
              float v1238_data = ir2[14];
              ir2[14] = (v1238_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 11))));
              float v1244_data = ir2[15];
              ir2[15] = (v1244_data + (v1150_data * (sycl::group_broadcast(item.get_sub_group(), v152_data, 11))));
              #pragma unroll
              for (int32_t v1249_n0 = 0; v1249_n0 < 1; ++v1249_n0) {
                #pragma unroll
                for (int32_t v1250_n1 = 0; v1250_n1 < 16; ++v1250_n1) {
                  int32_t v1251_a = v1249_n0 + v1250_n1;
                  float v1252_data = ir2[v1251_a];
                  r2[v1251_a] = v1252_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1257_i0 = 0; v1257_i0 < 1; ++v1257_i0) {
                int32_t v1265_lead = v14_lead + (v1257_i0 * 32);
                #pragma unroll
                for (int32_t v1258_i1 = 0; v1258_i1 < 16; ++v1258_i1) {
                  float v1260_data = r2[(v1257_i0 + v1258_i1)];
                  glb_m0[(v1265_lead + (v1258_i1 * 32))] = v1260_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1273_i1 = 0; v1273_i1 < 8; ++v1273_i1) {
                  float v1281_data = glb_m4[(v14_lead + (v1273_i1 * 12))];
                  r4[v1273_i1] = v1281_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1287_i0 = 0; v1287_i0 < 1; ++v1287_i0) {
                int32_t v1293_lead = v14_lead + (v1287_i0 * 32);
                #pragma unroll
                for (int32_t v1288_i1 = 0; v1288_i1 < 8; ++v1288_i1) {
                  float v1296_data = glb_m0[(v1293_lead + (v1288_i1 * 32))];
                  r5[(v1287_i0 + v1288_i1)] = v1296_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r7[12]{};
              // r7 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v1302_i0 = 0; v1302_i0 < 1; ++v1302_i0) {
                int32_t v1308_lead = v14_lead + (v1302_i0 * 32);
                #pragma unroll
                for (int32_t v1303_i1 = 0; v1303_i1 < 12; ++v1303_i1) {
                  float v1311_data = glb_m5[(v1308_lead + (v1303_i1 * 32))];
                  r7[(v1302_i0 + v1303_i1)] = v1311_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m0););
              float r6[8]{};
              // r6 = +(r3 * r4) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir6[8]{};
              float v1318_data = r3[0];
              float v1319_data = r4[0];
              float v1322_data = ir6[0];
              ir6[0] = (v1322_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 0))));
              float v1325_data = r4[1];
              float v1328_data = ir6[1];
              ir6[1] = (v1328_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 0))));
              float v1331_data = r4[2];
              float v1334_data = ir6[2];
              ir6[2] = (v1334_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 0))));
              float v1337_data = r4[3];
              float v1340_data = ir6[3];
              ir6[3] = (v1340_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 0))));
              float v1343_data = r4[4];
              float v1346_data = ir6[4];
              ir6[4] = (v1346_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 0))));
              float v1349_data = r4[5];
              float v1352_data = ir6[5];
              ir6[5] = (v1352_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 0))));
              float v1355_data = r4[6];
              float v1358_data = ir6[6];
              ir6[6] = (v1358_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 0))));
              float v1361_data = r4[7];
              float v1364_data = ir6[7];
              ir6[7] = (v1364_data + (v1318_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 0))));
              float v1369_data = r3[1];
              float v1373_data = ir6[0];
              ir6[0] = (v1373_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 1))));
              float v1379_data = ir6[1];
              ir6[1] = (v1379_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 1))));
              float v1385_data = ir6[2];
              ir6[2] = (v1385_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 1))));
              float v1391_data = ir6[3];
              ir6[3] = (v1391_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 1))));
              float v1397_data = ir6[4];
              ir6[4] = (v1397_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 1))));
              float v1403_data = ir6[5];
              ir6[5] = (v1403_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 1))));
              float v1409_data = ir6[6];
              ir6[6] = (v1409_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 1))));
              float v1415_data = ir6[7];
              ir6[7] = (v1415_data + (v1369_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 1))));
              float v1420_data = r3[2];
              float v1424_data = ir6[0];
              ir6[0] = (v1424_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 2))));
              float v1430_data = ir6[1];
              ir6[1] = (v1430_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 2))));
              float v1436_data = ir6[2];
              ir6[2] = (v1436_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 2))));
              float v1442_data = ir6[3];
              ir6[3] = (v1442_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 2))));
              float v1448_data = ir6[4];
              ir6[4] = (v1448_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 2))));
              float v1454_data = ir6[5];
              ir6[5] = (v1454_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 2))));
              float v1460_data = ir6[6];
              ir6[6] = (v1460_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 2))));
              float v1466_data = ir6[7];
              ir6[7] = (v1466_data + (v1420_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 2))));
              float v1471_data = r3[3];
              float v1475_data = ir6[0];
              ir6[0] = (v1475_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 3))));
              float v1481_data = ir6[1];
              ir6[1] = (v1481_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 3))));
              float v1487_data = ir6[2];
              ir6[2] = (v1487_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 3))));
              float v1493_data = ir6[3];
              ir6[3] = (v1493_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 3))));
              float v1499_data = ir6[4];
              ir6[4] = (v1499_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 3))));
              float v1505_data = ir6[5];
              ir6[5] = (v1505_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 3))));
              float v1511_data = ir6[6];
              ir6[6] = (v1511_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 3))));
              float v1517_data = ir6[7];
              ir6[7] = (v1517_data + (v1471_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 3))));
              float v1522_data = r3[4];
              float v1526_data = ir6[0];
              ir6[0] = (v1526_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 4))));
              float v1532_data = ir6[1];
              ir6[1] = (v1532_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 4))));
              float v1538_data = ir6[2];
              ir6[2] = (v1538_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 4))));
              float v1544_data = ir6[3];
              ir6[3] = (v1544_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 4))));
              float v1550_data = ir6[4];
              ir6[4] = (v1550_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 4))));
              float v1556_data = ir6[5];
              ir6[5] = (v1556_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 4))));
              float v1562_data = ir6[6];
              ir6[6] = (v1562_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 4))));
              float v1568_data = ir6[7];
              ir6[7] = (v1568_data + (v1522_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 4))));
              float v1573_data = r3[5];
              float v1577_data = ir6[0];
              ir6[0] = (v1577_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 5))));
              float v1583_data = ir6[1];
              ir6[1] = (v1583_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 5))));
              float v1589_data = ir6[2];
              ir6[2] = (v1589_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 5))));
              float v1595_data = ir6[3];
              ir6[3] = (v1595_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 5))));
              float v1601_data = ir6[4];
              ir6[4] = (v1601_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 5))));
              float v1607_data = ir6[5];
              ir6[5] = (v1607_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 5))));
              float v1613_data = ir6[6];
              ir6[6] = (v1613_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 5))));
              float v1619_data = ir6[7];
              ir6[7] = (v1619_data + (v1573_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 5))));
              float v1624_data = r3[6];
              float v1628_data = ir6[0];
              ir6[0] = (v1628_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 6))));
              float v1634_data = ir6[1];
              ir6[1] = (v1634_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 6))));
              float v1640_data = ir6[2];
              ir6[2] = (v1640_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 6))));
              float v1646_data = ir6[3];
              ir6[3] = (v1646_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 6))));
              float v1652_data = ir6[4];
              ir6[4] = (v1652_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 6))));
              float v1658_data = ir6[5];
              ir6[5] = (v1658_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 6))));
              float v1664_data = ir6[6];
              ir6[6] = (v1664_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 6))));
              float v1670_data = ir6[7];
              ir6[7] = (v1670_data + (v1624_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 6))));
              float v1675_data = r3[7];
              float v1679_data = ir6[0];
              ir6[0] = (v1679_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 7))));
              float v1685_data = ir6[1];
              ir6[1] = (v1685_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 7))));
              float v1691_data = ir6[2];
              ir6[2] = (v1691_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 7))));
              float v1697_data = ir6[3];
              ir6[3] = (v1697_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 7))));
              float v1703_data = ir6[4];
              ir6[4] = (v1703_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 7))));
              float v1709_data = ir6[5];
              ir6[5] = (v1709_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 7))));
              float v1715_data = ir6[6];
              ir6[6] = (v1715_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 7))));
              float v1721_data = ir6[7];
              ir6[7] = (v1721_data + (v1675_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 7))));
              float v1726_data = r3[8];
              float v1730_data = ir6[0];
              ir6[0] = (v1730_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 8))));
              float v1736_data = ir6[1];
              ir6[1] = (v1736_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 8))));
              float v1742_data = ir6[2];
              ir6[2] = (v1742_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 8))));
              float v1748_data = ir6[3];
              ir6[3] = (v1748_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 8))));
              float v1754_data = ir6[4];
              ir6[4] = (v1754_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 8))));
              float v1760_data = ir6[5];
              ir6[5] = (v1760_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 8))));
              float v1766_data = ir6[6];
              ir6[6] = (v1766_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 8))));
              float v1772_data = ir6[7];
              ir6[7] = (v1772_data + (v1726_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 8))));
              float v1777_data = r3[9];
              float v1781_data = ir6[0];
              ir6[0] = (v1781_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 9))));
              float v1787_data = ir6[1];
              ir6[1] = (v1787_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 9))));
              float v1793_data = ir6[2];
              ir6[2] = (v1793_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 9))));
              float v1799_data = ir6[3];
              ir6[3] = (v1799_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 9))));
              float v1805_data = ir6[4];
              ir6[4] = (v1805_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 9))));
              float v1811_data = ir6[5];
              ir6[5] = (v1811_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 9))));
              float v1817_data = ir6[6];
              ir6[6] = (v1817_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 9))));
              float v1823_data = ir6[7];
              ir6[7] = (v1823_data + (v1777_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 9))));
              float v1828_data = r3[10];
              float v1832_data = ir6[0];
              ir6[0] = (v1832_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 10))));
              float v1838_data = ir6[1];
              ir6[1] = (v1838_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 10))));
              float v1844_data = ir6[2];
              ir6[2] = (v1844_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 10))));
              float v1850_data = ir6[3];
              ir6[3] = (v1850_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 10))));
              float v1856_data = ir6[4];
              ir6[4] = (v1856_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 10))));
              float v1862_data = ir6[5];
              ir6[5] = (v1862_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 10))));
              float v1868_data = ir6[6];
              ir6[6] = (v1868_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 10))));
              float v1874_data = ir6[7];
              ir6[7] = (v1874_data + (v1828_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 10))));
              float v1879_data = r3[11];
              float v1883_data = ir6[0];
              ir6[0] = (v1883_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1319_data, 11))));
              float v1889_data = ir6[1];
              ir6[1] = (v1889_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1325_data, 11))));
              float v1895_data = ir6[2];
              ir6[2] = (v1895_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1331_data, 11))));
              float v1901_data = ir6[3];
              ir6[3] = (v1901_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1337_data, 11))));
              float v1907_data = ir6[4];
              ir6[4] = (v1907_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1343_data, 11))));
              float v1913_data = ir6[5];
              ir6[5] = (v1913_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1349_data, 11))));
              float v1919_data = ir6[6];
              ir6[6] = (v1919_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1355_data, 11))));
              float v1925_data = ir6[7];
              ir6[7] = (v1925_data + (v1879_data * (sycl::group_broadcast(item.get_sub_group(), v1361_data, 11))));
              #pragma unroll
              for (int32_t v1930_n0 = 0; v1930_n0 < 1; ++v1930_n0) {
                #pragma unroll
                for (int32_t v1931_n1 = 0; v1931_n1 < 8; ++v1931_n1) {
                  int32_t v1932_a = v1930_n0 + v1931_n1;
                  float v1933_data = ir6[v1932_a];
                  float v1935_data = r5[v1932_a];
                  r6[v1932_a] = (v1935_data + v1933_data);
                }
              }
              // glb_m0 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v1941_i0 = 0; v1941_i0 < 1; ++v1941_i0) {
                int32_t v1949_lead = v14_lead + (v1941_i0 * 32);
                #pragma unroll
                for (int32_t v1942_i1 = 0; v1942_i1 < 8; ++v1942_i1) {
                  float v1944_data = r6[(v1941_i0 + v1942_i1)];
                  glb_m0[(v1949_lead + (v1942_i1 * 32))] = v1944_data;
                }
              }
              float r8[8]{};
              // r8 = load{g>r}(glb_m6);
              if (v14_lead < 12) {
                #pragma unroll
                for (int32_t v1957_i1 = 0; v1957_i1 < 8; ++v1957_i1) {
                  float v1965_data = glb_m6[(v14_lead + (v1957_i1 * 12))];
                  r8[v1957_i1] = v1965_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m5););
              float r9[8]{};
              // r9 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1971_i0 = 0; v1971_i0 < 1; ++v1971_i0) {
                int32_t v1977_lead = v14_lead + (v1971_i0 * 32);
                #pragma unroll
                for (int32_t v1972_i1 = 0; v1972_i1 < 8; ++v1972_i1) {
                  float v1981_data = glb_m0[(v1977_lead + ((v1972_i1 + 8) * 32))];
                  r9[(v1971_i0 + v1972_i1)] = v1981_data;
                }
              }
              // wait(r8 = load{g>r}(glb_m6););
              // wait(r9 = load{g>r}(glb_m0););
              float r10[8]{};
              // r10 = +(r7 * r8) + name: r9, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir10[8]{};
              float v1988_data = r7[0];
              float v1989_data = r8[0];
              float v1992_data = ir10[0];
              ir10[0] = (v1992_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 0))));
              float v1995_data = r8[1];
              float v1998_data = ir10[1];
              ir10[1] = (v1998_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 0))));
              float v2001_data = r8[2];
              float v2004_data = ir10[2];
              ir10[2] = (v2004_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 0))));
              float v2007_data = r8[3];
              float v2010_data = ir10[3];
              ir10[3] = (v2010_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 0))));
              float v2013_data = r8[4];
              float v2016_data = ir10[4];
              ir10[4] = (v2016_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 0))));
              float v2019_data = r8[5];
              float v2022_data = ir10[5];
              ir10[5] = (v2022_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 0))));
              float v2025_data = r8[6];
              float v2028_data = ir10[6];
              ir10[6] = (v2028_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 0))));
              float v2031_data = r8[7];
              float v2034_data = ir10[7];
              ir10[7] = (v2034_data + (v1988_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 0))));
              float v2039_data = r7[1];
              float v2043_data = ir10[0];
              ir10[0] = (v2043_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 1))));
              float v2049_data = ir10[1];
              ir10[1] = (v2049_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 1))));
              float v2055_data = ir10[2];
              ir10[2] = (v2055_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 1))));
              float v2061_data = ir10[3];
              ir10[3] = (v2061_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 1))));
              float v2067_data = ir10[4];
              ir10[4] = (v2067_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 1))));
              float v2073_data = ir10[5];
              ir10[5] = (v2073_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 1))));
              float v2079_data = ir10[6];
              ir10[6] = (v2079_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 1))));
              float v2085_data = ir10[7];
              ir10[7] = (v2085_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 1))));
              float v2090_data = r7[2];
              float v2094_data = ir10[0];
              ir10[0] = (v2094_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 2))));
              float v2100_data = ir10[1];
              ir10[1] = (v2100_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 2))));
              float v2106_data = ir10[2];
              ir10[2] = (v2106_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 2))));
              float v2112_data = ir10[3];
              ir10[3] = (v2112_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 2))));
              float v2118_data = ir10[4];
              ir10[4] = (v2118_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 2))));
              float v2124_data = ir10[5];
              ir10[5] = (v2124_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 2))));
              float v2130_data = ir10[6];
              ir10[6] = (v2130_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 2))));
              float v2136_data = ir10[7];
              ir10[7] = (v2136_data + (v2090_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 2))));
              float v2141_data = r7[3];
              float v2145_data = ir10[0];
              ir10[0] = (v2145_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 3))));
              float v2151_data = ir10[1];
              ir10[1] = (v2151_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 3))));
              float v2157_data = ir10[2];
              ir10[2] = (v2157_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 3))));
              float v2163_data = ir10[3];
              ir10[3] = (v2163_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 3))));
              float v2169_data = ir10[4];
              ir10[4] = (v2169_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 3))));
              float v2175_data = ir10[5];
              ir10[5] = (v2175_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 3))));
              float v2181_data = ir10[6];
              ir10[6] = (v2181_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 3))));
              float v2187_data = ir10[7];
              ir10[7] = (v2187_data + (v2141_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 3))));
              float v2192_data = r7[4];
              float v2196_data = ir10[0];
              ir10[0] = (v2196_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 4))));
              float v2202_data = ir10[1];
              ir10[1] = (v2202_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 4))));
              float v2208_data = ir10[2];
              ir10[2] = (v2208_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 4))));
              float v2214_data = ir10[3];
              ir10[3] = (v2214_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 4))));
              float v2220_data = ir10[4];
              ir10[4] = (v2220_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 4))));
              float v2226_data = ir10[5];
              ir10[5] = (v2226_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 4))));
              float v2232_data = ir10[6];
              ir10[6] = (v2232_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 4))));
              float v2238_data = ir10[7];
              ir10[7] = (v2238_data + (v2192_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 4))));
              float v2243_data = r7[5];
              float v2247_data = ir10[0];
              ir10[0] = (v2247_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 5))));
              float v2253_data = ir10[1];
              ir10[1] = (v2253_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 5))));
              float v2259_data = ir10[2];
              ir10[2] = (v2259_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 5))));
              float v2265_data = ir10[3];
              ir10[3] = (v2265_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 5))));
              float v2271_data = ir10[4];
              ir10[4] = (v2271_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 5))));
              float v2277_data = ir10[5];
              ir10[5] = (v2277_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 5))));
              float v2283_data = ir10[6];
              ir10[6] = (v2283_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 5))));
              float v2289_data = ir10[7];
              ir10[7] = (v2289_data + (v2243_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 5))));
              float v2294_data = r7[6];
              float v2298_data = ir10[0];
              ir10[0] = (v2298_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 6))));
              float v2304_data = ir10[1];
              ir10[1] = (v2304_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 6))));
              float v2310_data = ir10[2];
              ir10[2] = (v2310_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 6))));
              float v2316_data = ir10[3];
              ir10[3] = (v2316_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 6))));
              float v2322_data = ir10[4];
              ir10[4] = (v2322_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 6))));
              float v2328_data = ir10[5];
              ir10[5] = (v2328_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 6))));
              float v2334_data = ir10[6];
              ir10[6] = (v2334_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 6))));
              float v2340_data = ir10[7];
              ir10[7] = (v2340_data + (v2294_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 6))));
              float v2345_data = r7[7];
              float v2349_data = ir10[0];
              ir10[0] = (v2349_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 7))));
              float v2355_data = ir10[1];
              ir10[1] = (v2355_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 7))));
              float v2361_data = ir10[2];
              ir10[2] = (v2361_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 7))));
              float v2367_data = ir10[3];
              ir10[3] = (v2367_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 7))));
              float v2373_data = ir10[4];
              ir10[4] = (v2373_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 7))));
              float v2379_data = ir10[5];
              ir10[5] = (v2379_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 7))));
              float v2385_data = ir10[6];
              ir10[6] = (v2385_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 7))));
              float v2391_data = ir10[7];
              ir10[7] = (v2391_data + (v2345_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 7))));
              float v2396_data = r7[8];
              float v2400_data = ir10[0];
              ir10[0] = (v2400_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 8))));
              float v2406_data = ir10[1];
              ir10[1] = (v2406_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 8))));
              float v2412_data = ir10[2];
              ir10[2] = (v2412_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 8))));
              float v2418_data = ir10[3];
              ir10[3] = (v2418_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 8))));
              float v2424_data = ir10[4];
              ir10[4] = (v2424_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 8))));
              float v2430_data = ir10[5];
              ir10[5] = (v2430_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 8))));
              float v2436_data = ir10[6];
              ir10[6] = (v2436_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 8))));
              float v2442_data = ir10[7];
              ir10[7] = (v2442_data + (v2396_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 8))));
              float v2447_data = r7[9];
              float v2451_data = ir10[0];
              ir10[0] = (v2451_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 9))));
              float v2457_data = ir10[1];
              ir10[1] = (v2457_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 9))));
              float v2463_data = ir10[2];
              ir10[2] = (v2463_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 9))));
              float v2469_data = ir10[3];
              ir10[3] = (v2469_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 9))));
              float v2475_data = ir10[4];
              ir10[4] = (v2475_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 9))));
              float v2481_data = ir10[5];
              ir10[5] = (v2481_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 9))));
              float v2487_data = ir10[6];
              ir10[6] = (v2487_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 9))));
              float v2493_data = ir10[7];
              ir10[7] = (v2493_data + (v2447_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 9))));
              float v2498_data = r7[10];
              float v2502_data = ir10[0];
              ir10[0] = (v2502_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 10))));
              float v2508_data = ir10[1];
              ir10[1] = (v2508_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 10))));
              float v2514_data = ir10[2];
              ir10[2] = (v2514_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 10))));
              float v2520_data = ir10[3];
              ir10[3] = (v2520_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 10))));
              float v2526_data = ir10[4];
              ir10[4] = (v2526_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 10))));
              float v2532_data = ir10[5];
              ir10[5] = (v2532_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 10))));
              float v2538_data = ir10[6];
              ir10[6] = (v2538_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 10))));
              float v2544_data = ir10[7];
              ir10[7] = (v2544_data + (v2498_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 10))));
              float v2549_data = r7[11];
              float v2553_data = ir10[0];
              ir10[0] = (v2553_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v1989_data, 11))));
              float v2559_data = ir10[1];
              ir10[1] = (v2559_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v1995_data, 11))));
              float v2565_data = ir10[2];
              ir10[2] = (v2565_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v2001_data, 11))));
              float v2571_data = ir10[3];
              ir10[3] = (v2571_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v2007_data, 11))));
              float v2577_data = ir10[4];
              ir10[4] = (v2577_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v2013_data, 11))));
              float v2583_data = ir10[5];
              ir10[5] = (v2583_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v2019_data, 11))));
              float v2589_data = ir10[6];
              ir10[6] = (v2589_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v2025_data, 11))));
              float v2595_data = ir10[7];
              ir10[7] = (v2595_data + (v2549_data * (sycl::group_broadcast(item.get_sub_group(), v2031_data, 11))));
              #pragma unroll
              for (int32_t v2600_n0 = 0; v2600_n0 < 1; ++v2600_n0) {
                #pragma unroll
                for (int32_t v2601_n1 = 0; v2601_n1 < 8; ++v2601_n1) {
                  int32_t v2602_a = v2600_n0 + v2601_n1;
                  float v2603_data = ir10[v2602_a];
                  float v2605_data = r9[v2602_a];
                  r10[v2602_a] = (v2605_data + v2603_data);
                }
              }
              // glb_m0 = store{r>g}(r10);
              #pragma unroll
              for (int32_t v2611_i0 = 0; v2611_i0 < 1; ++v2611_i0) {
                int32_t v2619_lead = v14_lead + (v2611_i0 * 32);
                #pragma unroll
                for (int32_t v2612_i1 = 0; v2612_i1 < 8; ++v2612_i1) {
                  float v2614_data = r10[(v2611_i0 + v2612_i1)];
                  glb_m0[(v2619_lead + ((v2612_i1 + 8) * 32))] = v2614_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

