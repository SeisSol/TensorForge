// === base name ===
kernel_0e79984cfef4d081

// === header ===
void launcher_kernel_0e79984cfef4d081(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_0e79984cfef4d081(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0e79984cfef4d081(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0e79984cfef4d081(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 32×32(32×32) {0..32}×{0..32} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
                int32_t v20_off = (v12_lead + (v13_i0 * 16)) + 8;
                #pragma unroll
                for (int32_t v14_i1 = 8; v14_i1 < 24; ++v14_i1) {
                  float v23_data = glb_m1[(v20_off + (v14_i1 * 32))];
                  r0[(v13_i0 + (v14_i1 - 8))] = v23_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
                int32_t v36_lead = v12_lead + (v30_i0 * 16);
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
                  float v39_data = glb_m2[(v36_lead + (v31_i1 * 16))];
                  r1[(v30_i0 + v31_i1)] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v46_data = r0[0];
              float v47_data = r1[0];
              float v50_data = ir2[0];
              ir2[0] = (v50_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 0))));
              float v53_data = r1[1];
              float v56_data = ir2[1];
              ir2[1] = (v56_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 0))));
              float v59_data = r1[2];
              float v62_data = ir2[2];
              ir2[2] = (v62_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 0))));
              float v65_data = r1[3];
              float v68_data = ir2[3];
              ir2[3] = (v68_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
              float v71_data = r1[4];
              float v74_data = ir2[4];
              ir2[4] = (v74_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
              float v77_data = r1[5];
              float v80_data = ir2[5];
              ir2[5] = (v80_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
              float v83_data = r1[6];
              float v86_data = ir2[6];
              ir2[6] = (v86_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
              float v89_data = r1[7];
              float v92_data = ir2[7];
              ir2[7] = (v92_data + (v46_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 0))));
              float v97_data = r0[1];
              float v101_data = ir2[0];
              ir2[0] = (v101_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 1))));
              float v107_data = ir2[1];
              ir2[1] = (v107_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 1))));
              float v113_data = ir2[2];
              ir2[2] = (v113_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 1))));
              float v119_data = ir2[3];
              ir2[3] = (v119_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 1))));
              float v125_data = ir2[4];
              ir2[4] = (v125_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 1))));
              float v131_data = ir2[5];
              ir2[5] = (v131_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 1))));
              float v137_data = ir2[6];
              ir2[6] = (v137_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 1))));
              float v143_data = ir2[7];
              ir2[7] = (v143_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 1))));
              float v148_data = r0[2];
              float v152_data = ir2[0];
              ir2[0] = (v152_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 2))));
              float v158_data = ir2[1];
              ir2[1] = (v158_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 2))));
              float v164_data = ir2[2];
              ir2[2] = (v164_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 2))));
              float v170_data = ir2[3];
              ir2[3] = (v170_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 2))));
              float v176_data = ir2[4];
              ir2[4] = (v176_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 2))));
              float v182_data = ir2[5];
              ir2[5] = (v182_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 2))));
              float v188_data = ir2[6];
              ir2[6] = (v188_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 2))));
              float v194_data = ir2[7];
              ir2[7] = (v194_data + (v148_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 2))));
              float v199_data = r0[3];
              float v203_data = ir2[0];
              ir2[0] = (v203_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 3))));
              float v209_data = ir2[1];
              ir2[1] = (v209_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 3))));
              float v215_data = ir2[2];
              ir2[2] = (v215_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 3))));
              float v221_data = ir2[3];
              ir2[3] = (v221_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 3))));
              float v227_data = ir2[4];
              ir2[4] = (v227_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 3))));
              float v233_data = ir2[5];
              ir2[5] = (v233_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 3))));
              float v239_data = ir2[6];
              ir2[6] = (v239_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 3))));
              float v245_data = ir2[7];
              ir2[7] = (v245_data + (v199_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 3))));
              float v250_data = r0[4];
              float v254_data = ir2[0];
              ir2[0] = (v254_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 4))));
              float v260_data = ir2[1];
              ir2[1] = (v260_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 4))));
              float v266_data = ir2[2];
              ir2[2] = (v266_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 4))));
              float v272_data = ir2[3];
              ir2[3] = (v272_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 4))));
              float v278_data = ir2[4];
              ir2[4] = (v278_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 4))));
              float v284_data = ir2[5];
              ir2[5] = (v284_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 4))));
              float v290_data = ir2[6];
              ir2[6] = (v290_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 4))));
              float v296_data = ir2[7];
              ir2[7] = (v296_data + (v250_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 4))));
              float v301_data = r0[5];
              float v305_data = ir2[0];
              ir2[0] = (v305_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 5))));
              float v311_data = ir2[1];
              ir2[1] = (v311_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 5))));
              float v317_data = ir2[2];
              ir2[2] = (v317_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 5))));
              float v323_data = ir2[3];
              ir2[3] = (v323_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 5))));
              float v329_data = ir2[4];
              ir2[4] = (v329_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 5))));
              float v335_data = ir2[5];
              ir2[5] = (v335_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 5))));
              float v341_data = ir2[6];
              ir2[6] = (v341_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 5))));
              float v347_data = ir2[7];
              ir2[7] = (v347_data + (v301_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 5))));
              float v352_data = r0[6];
              float v356_data = ir2[0];
              ir2[0] = (v356_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 6))));
              float v362_data = ir2[1];
              ir2[1] = (v362_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 6))));
              float v368_data = ir2[2];
              ir2[2] = (v368_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 6))));
              float v374_data = ir2[3];
              ir2[3] = (v374_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 6))));
              float v380_data = ir2[4];
              ir2[4] = (v380_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 6))));
              float v386_data = ir2[5];
              ir2[5] = (v386_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 6))));
              float v392_data = ir2[6];
              ir2[6] = (v392_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 6))));
              float v398_data = ir2[7];
              ir2[7] = (v398_data + (v352_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 6))));
              float v403_data = r0[7];
              float v407_data = ir2[0];
              ir2[0] = (v407_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 7))));
              float v413_data = ir2[1];
              ir2[1] = (v413_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 7))));
              float v419_data = ir2[2];
              ir2[2] = (v419_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 7))));
              float v425_data = ir2[3];
              ir2[3] = (v425_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 7))));
              float v431_data = ir2[4];
              ir2[4] = (v431_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 7))));
              float v437_data = ir2[5];
              ir2[5] = (v437_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 7))));
              float v443_data = ir2[6];
              ir2[6] = (v443_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 7))));
              float v449_data = ir2[7];
              ir2[7] = (v449_data + (v403_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 7))));
              float v454_data = r0[8];
              float v458_data = ir2[0];
              ir2[0] = (v458_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 8))));
              float v464_data = ir2[1];
              ir2[1] = (v464_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 8))));
              float v470_data = ir2[2];
              ir2[2] = (v470_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 8))));
              float v476_data = ir2[3];
              ir2[3] = (v476_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 8))));
              float v482_data = ir2[4];
              ir2[4] = (v482_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 8))));
              float v488_data = ir2[5];
              ir2[5] = (v488_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 8))));
              float v494_data = ir2[6];
              ir2[6] = (v494_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 8))));
              float v500_data = ir2[7];
              ir2[7] = (v500_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 8))));
              float v505_data = r0[9];
              float v509_data = ir2[0];
              ir2[0] = (v509_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 9))));
              float v515_data = ir2[1];
              ir2[1] = (v515_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 9))));
              float v521_data = ir2[2];
              ir2[2] = (v521_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 9))));
              float v527_data = ir2[3];
              ir2[3] = (v527_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 9))));
              float v533_data = ir2[4];
              ir2[4] = (v533_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 9))));
              float v539_data = ir2[5];
              ir2[5] = (v539_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 9))));
              float v545_data = ir2[6];
              ir2[6] = (v545_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 9))));
              float v551_data = ir2[7];
              ir2[7] = (v551_data + (v505_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 9))));
              float v556_data = r0[10];
              float v560_data = ir2[0];
              ir2[0] = (v560_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 10))));
              float v566_data = ir2[1];
              ir2[1] = (v566_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 10))));
              float v572_data = ir2[2];
              ir2[2] = (v572_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 10))));
              float v578_data = ir2[3];
              ir2[3] = (v578_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 10))));
              float v584_data = ir2[4];
              ir2[4] = (v584_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 10))));
              float v590_data = ir2[5];
              ir2[5] = (v590_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 10))));
              float v596_data = ir2[6];
              ir2[6] = (v596_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 10))));
              float v602_data = ir2[7];
              ir2[7] = (v602_data + (v556_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 10))));
              float v607_data = r0[11];
              float v611_data = ir2[0];
              ir2[0] = (v611_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 11))));
              float v617_data = ir2[1];
              ir2[1] = (v617_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 11))));
              float v623_data = ir2[2];
              ir2[2] = (v623_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 11))));
              float v629_data = ir2[3];
              ir2[3] = (v629_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 11))));
              float v635_data = ir2[4];
              ir2[4] = (v635_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 11))));
              float v641_data = ir2[5];
              ir2[5] = (v641_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 11))));
              float v647_data = ir2[6];
              ir2[6] = (v647_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 11))));
              float v653_data = ir2[7];
              ir2[7] = (v653_data + (v607_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 11))));
              float v658_data = r0[12];
              float v662_data = ir2[0];
              ir2[0] = (v662_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 12))));
              float v668_data = ir2[1];
              ir2[1] = (v668_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 12))));
              float v674_data = ir2[2];
              ir2[2] = (v674_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 12))));
              float v680_data = ir2[3];
              ir2[3] = (v680_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 12))));
              float v686_data = ir2[4];
              ir2[4] = (v686_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 12))));
              float v692_data = ir2[5];
              ir2[5] = (v692_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 12))));
              float v698_data = ir2[6];
              ir2[6] = (v698_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 12))));
              float v704_data = ir2[7];
              ir2[7] = (v704_data + (v658_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 12))));
              float v709_data = r0[13];
              float v713_data = ir2[0];
              ir2[0] = (v713_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 13))));
              float v719_data = ir2[1];
              ir2[1] = (v719_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 13))));
              float v725_data = ir2[2];
              ir2[2] = (v725_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 13))));
              float v731_data = ir2[3];
              ir2[3] = (v731_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 13))));
              float v737_data = ir2[4];
              ir2[4] = (v737_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 13))));
              float v743_data = ir2[5];
              ir2[5] = (v743_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 13))));
              float v749_data = ir2[6];
              ir2[6] = (v749_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 13))));
              float v755_data = ir2[7];
              ir2[7] = (v755_data + (v709_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 13))));
              float v760_data = r0[14];
              float v764_data = ir2[0];
              ir2[0] = (v764_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 14))));
              float v770_data = ir2[1];
              ir2[1] = (v770_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 14))));
              float v776_data = ir2[2];
              ir2[2] = (v776_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 14))));
              float v782_data = ir2[3];
              ir2[3] = (v782_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 14))));
              float v788_data = ir2[4];
              ir2[4] = (v788_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 14))));
              float v794_data = ir2[5];
              ir2[5] = (v794_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 14))));
              float v800_data = ir2[6];
              ir2[6] = (v800_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 14))));
              float v806_data = ir2[7];
              ir2[7] = (v806_data + (v760_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 14))));
              float v811_data = r0[15];
              float v815_data = ir2[0];
              ir2[0] = (v815_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 15))));
              float v821_data = ir2[1];
              ir2[1] = (v821_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 15))));
              float v827_data = ir2[2];
              ir2[2] = (v827_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 15))));
              float v833_data = ir2[3];
              ir2[3] = (v833_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 15))));
              float v839_data = ir2[4];
              ir2[4] = (v839_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 15))));
              float v845_data = ir2[5];
              ir2[5] = (v845_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 15))));
              float v851_data = ir2[6];
              ir2[6] = (v851_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 15))));
              float v857_data = ir2[7];
              ir2[7] = (v857_data + (v811_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 15))));
              #pragma unroll
              for (int32_t v862_n0 = 0; v862_n0 < 1; ++v862_n0) {
                #pragma unroll
                for (int32_t v863_n1 = 0; v863_n1 < 8; ++v863_n1) {
                  int32_t v864_a = v862_n0 + v863_n1;
                  float v865_data = ir2[v864_a];
                  r2[v864_a] = v865_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v870_i0 = 0; v870_i0 < 1; ++v870_i0) {
                int32_t v878_lead = v12_lead + (v870_i0 * 16);
                #pragma unroll
                for (int32_t v871_i1 = 0; v871_i1 < 8; ++v871_i1) {
                  float v873_data = r2[(v870_i0 + v871_i1)];
                  glb_m0[(v878_lead + (v871_i1 * 16))] = v873_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

