// === base name ===
kernel_21138a3fa2

// === header ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_21138a3fa2(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_21138a3fa2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v15_lead = v8_lead + (v9_i0 * 16);
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  float v18_data = glb_m1[(v15_lead + (v10_i1 * 16))];
                  r0[(v9_i0 + v10_i1)] = v18_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v30_lead = v8_lead + (v24_i0 * 16);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                  float v33_data = glb_m2[(v30_lead + (v25_i1 * 16))];
                  r1[(v24_i0 + v25_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v40_data = r0[0];
              float v41_data = r1[0];
              float v44_data = ir2[0];
              ir2[0] = (v44_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 0))));
              float v47_data = r1[1];
              float v50_data = ir2[1];
              ir2[1] = (v50_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 0))));
              float v53_data = r1[2];
              float v56_data = ir2[2];
              ir2[2] = (v56_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 0))));
              float v59_data = r1[3];
              float v62_data = ir2[3];
              ir2[3] = (v62_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 0))));
              float v65_data = r1[4];
              float v68_data = ir2[4];
              ir2[4] = (v68_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
              float v71_data = r1[5];
              float v74_data = ir2[5];
              ir2[5] = (v74_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
              float v77_data = r1[6];
              float v80_data = ir2[6];
              ir2[6] = (v80_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
              float v83_data = r1[7];
              float v86_data = ir2[7];
              ir2[7] = (v86_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
              float v91_data = r0[1];
              float v95_data = ir2[0];
              ir2[0] = (v95_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 1))));
              float v101_data = ir2[1];
              ir2[1] = (v101_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 1))));
              float v107_data = ir2[2];
              ir2[2] = (v107_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 1))));
              float v113_data = ir2[3];
              ir2[3] = (v113_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 1))));
              float v119_data = ir2[4];
              ir2[4] = (v119_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 1))));
              float v125_data = ir2[5];
              ir2[5] = (v125_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 1))));
              float v131_data = ir2[6];
              ir2[6] = (v131_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 1))));
              float v137_data = ir2[7];
              ir2[7] = (v137_data + (v91_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 1))));
              float v142_data = r0[2];
              float v146_data = ir2[0];
              ir2[0] = (v146_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 2))));
              float v152_data = ir2[1];
              ir2[1] = (v152_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 2))));
              float v158_data = ir2[2];
              ir2[2] = (v158_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 2))));
              float v164_data = ir2[3];
              ir2[3] = (v164_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 2))));
              float v170_data = ir2[4];
              ir2[4] = (v170_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 2))));
              float v176_data = ir2[5];
              ir2[5] = (v176_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 2))));
              float v182_data = ir2[6];
              ir2[6] = (v182_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 2))));
              float v188_data = ir2[7];
              ir2[7] = (v188_data + (v142_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 2))));
              float v193_data = r0[3];
              float v197_data = ir2[0];
              ir2[0] = (v197_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 3))));
              float v203_data = ir2[1];
              ir2[1] = (v203_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 3))));
              float v209_data = ir2[2];
              ir2[2] = (v209_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 3))));
              float v215_data = ir2[3];
              ir2[3] = (v215_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 3))));
              float v221_data = ir2[4];
              ir2[4] = (v221_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 3))));
              float v227_data = ir2[5];
              ir2[5] = (v227_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 3))));
              float v233_data = ir2[6];
              ir2[6] = (v233_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 3))));
              float v239_data = ir2[7];
              ir2[7] = (v239_data + (v193_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 3))));
              float v244_data = r0[4];
              float v248_data = ir2[0];
              ir2[0] = (v248_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 4))));
              float v254_data = ir2[1];
              ir2[1] = (v254_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 4))));
              float v260_data = ir2[2];
              ir2[2] = (v260_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 4))));
              float v266_data = ir2[3];
              ir2[3] = (v266_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 4))));
              float v272_data = ir2[4];
              ir2[4] = (v272_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 4))));
              float v278_data = ir2[5];
              ir2[5] = (v278_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 4))));
              float v284_data = ir2[6];
              ir2[6] = (v284_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 4))));
              float v290_data = ir2[7];
              ir2[7] = (v290_data + (v244_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 4))));
              float v295_data = r0[5];
              float v299_data = ir2[0];
              ir2[0] = (v299_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 5))));
              float v305_data = ir2[1];
              ir2[1] = (v305_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 5))));
              float v311_data = ir2[2];
              ir2[2] = (v311_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 5))));
              float v317_data = ir2[3];
              ir2[3] = (v317_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 5))));
              float v323_data = ir2[4];
              ir2[4] = (v323_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 5))));
              float v329_data = ir2[5];
              ir2[5] = (v329_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 5))));
              float v335_data = ir2[6];
              ir2[6] = (v335_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 5))));
              float v341_data = ir2[7];
              ir2[7] = (v341_data + (v295_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 5))));
              float v346_data = r0[6];
              float v350_data = ir2[0];
              ir2[0] = (v350_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 6))));
              float v356_data = ir2[1];
              ir2[1] = (v356_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 6))));
              float v362_data = ir2[2];
              ir2[2] = (v362_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 6))));
              float v368_data = ir2[3];
              ir2[3] = (v368_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 6))));
              float v374_data = ir2[4];
              ir2[4] = (v374_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 6))));
              float v380_data = ir2[5];
              ir2[5] = (v380_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 6))));
              float v386_data = ir2[6];
              ir2[6] = (v386_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 6))));
              float v392_data = ir2[7];
              ir2[7] = (v392_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 6))));
              float v397_data = r0[7];
              float v401_data = ir2[0];
              ir2[0] = (v401_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 7))));
              float v407_data = ir2[1];
              ir2[1] = (v407_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 7))));
              float v413_data = ir2[2];
              ir2[2] = (v413_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 7))));
              float v419_data = ir2[3];
              ir2[3] = (v419_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 7))));
              float v425_data = ir2[4];
              ir2[4] = (v425_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 7))));
              float v431_data = ir2[5];
              ir2[5] = (v431_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 7))));
              float v437_data = ir2[6];
              ir2[6] = (v437_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 7))));
              float v443_data = ir2[7];
              ir2[7] = (v443_data + (v397_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 7))));
              float v448_data = r0[8];
              float v452_data = ir2[0];
              ir2[0] = (v452_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 8))));
              float v458_data = ir2[1];
              ir2[1] = (v458_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 8))));
              float v464_data = ir2[2];
              ir2[2] = (v464_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 8))));
              float v470_data = ir2[3];
              ir2[3] = (v470_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 8))));
              float v476_data = ir2[4];
              ir2[4] = (v476_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 8))));
              float v482_data = ir2[5];
              ir2[5] = (v482_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 8))));
              float v488_data = ir2[6];
              ir2[6] = (v488_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 8))));
              float v494_data = ir2[7];
              ir2[7] = (v494_data + (v448_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 8))));
              float v499_data = r0[9];
              float v503_data = ir2[0];
              ir2[0] = (v503_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 9))));
              float v509_data = ir2[1];
              ir2[1] = (v509_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 9))));
              float v515_data = ir2[2];
              ir2[2] = (v515_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 9))));
              float v521_data = ir2[3];
              ir2[3] = (v521_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 9))));
              float v527_data = ir2[4];
              ir2[4] = (v527_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 9))));
              float v533_data = ir2[5];
              ir2[5] = (v533_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 9))));
              float v539_data = ir2[6];
              ir2[6] = (v539_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 9))));
              float v545_data = ir2[7];
              ir2[7] = (v545_data + (v499_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 9))));
              float v550_data = r0[10];
              float v554_data = ir2[0];
              ir2[0] = (v554_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 10))));
              float v560_data = ir2[1];
              ir2[1] = (v560_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 10))));
              float v566_data = ir2[2];
              ir2[2] = (v566_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 10))));
              float v572_data = ir2[3];
              ir2[3] = (v572_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 10))));
              float v578_data = ir2[4];
              ir2[4] = (v578_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 10))));
              float v584_data = ir2[5];
              ir2[5] = (v584_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 10))));
              float v590_data = ir2[6];
              ir2[6] = (v590_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 10))));
              float v596_data = ir2[7];
              ir2[7] = (v596_data + (v550_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 10))));
              float v601_data = r0[11];
              float v605_data = ir2[0];
              ir2[0] = (v605_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 11))));
              float v611_data = ir2[1];
              ir2[1] = (v611_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 11))));
              float v617_data = ir2[2];
              ir2[2] = (v617_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 11))));
              float v623_data = ir2[3];
              ir2[3] = (v623_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 11))));
              float v629_data = ir2[4];
              ir2[4] = (v629_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 11))));
              float v635_data = ir2[5];
              ir2[5] = (v635_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 11))));
              float v641_data = ir2[6];
              ir2[6] = (v641_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 11))));
              float v647_data = ir2[7];
              ir2[7] = (v647_data + (v601_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 11))));
              float v652_data = r0[12];
              float v656_data = ir2[0];
              ir2[0] = (v656_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 12))));
              float v662_data = ir2[1];
              ir2[1] = (v662_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 12))));
              float v668_data = ir2[2];
              ir2[2] = (v668_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 12))));
              float v674_data = ir2[3];
              ir2[3] = (v674_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 12))));
              float v680_data = ir2[4];
              ir2[4] = (v680_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 12))));
              float v686_data = ir2[5];
              ir2[5] = (v686_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 12))));
              float v692_data = ir2[6];
              ir2[6] = (v692_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 12))));
              float v698_data = ir2[7];
              ir2[7] = (v698_data + (v652_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 12))));
              float v703_data = r0[13];
              float v707_data = ir2[0];
              ir2[0] = (v707_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 13))));
              float v713_data = ir2[1];
              ir2[1] = (v713_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 13))));
              float v719_data = ir2[2];
              ir2[2] = (v719_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 13))));
              float v725_data = ir2[3];
              ir2[3] = (v725_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 13))));
              float v731_data = ir2[4];
              ir2[4] = (v731_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 13))));
              float v737_data = ir2[5];
              ir2[5] = (v737_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 13))));
              float v743_data = ir2[6];
              ir2[6] = (v743_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 13))));
              float v749_data = ir2[7];
              ir2[7] = (v749_data + (v703_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 13))));
              float v754_data = r0[14];
              float v758_data = ir2[0];
              ir2[0] = (v758_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 14))));
              float v764_data = ir2[1];
              ir2[1] = (v764_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 14))));
              float v770_data = ir2[2];
              ir2[2] = (v770_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 14))));
              float v776_data = ir2[3];
              ir2[3] = (v776_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 14))));
              float v782_data = ir2[4];
              ir2[4] = (v782_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 14))));
              float v788_data = ir2[5];
              ir2[5] = (v788_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 14))));
              float v794_data = ir2[6];
              ir2[6] = (v794_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 14))));
              float v800_data = ir2[7];
              ir2[7] = (v800_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 14))));
              float v805_data = r0[15];
              float v809_data = ir2[0];
              ir2[0] = (v809_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 15))));
              float v815_data = ir2[1];
              ir2[1] = (v815_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 15))));
              float v821_data = ir2[2];
              ir2[2] = (v821_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 15))));
              float v827_data = ir2[3];
              ir2[3] = (v827_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 15))));
              float v833_data = ir2[4];
              ir2[4] = (v833_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 15))));
              float v839_data = ir2[5];
              ir2[5] = (v839_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 15))));
              float v845_data = ir2[6];
              ir2[6] = (v845_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 15))));
              float v851_data = ir2[7];
              ir2[7] = (v851_data + (v805_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 15))));
              #pragma unroll
              for (int32_t v856_n0 = 0; v856_n0 < 1; ++v856_n0) {
                #pragma unroll
                for (int32_t v857_n1 = 0; v857_n1 < 8; ++v857_n1) {
                  int32_t v858_a = v856_n0 + v857_n1;
                  float v859_data = ir2[v858_a];
                  r2[v858_a] = v859_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v864_i0 = 0; v864_i0 < 1; ++v864_i0) {
                int32_t v872_lead = v8_lead + (v864_i0 * 16);
                #pragma unroll
                for (int32_t v865_i1 = 0; v865_i1 < 8; ++v865_i1) {
                  float v867_data = r2[(v864_i0 + v865_i1)];
                  glb_m0[(v872_lead + (v865_i1 * 16))] = v867_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

