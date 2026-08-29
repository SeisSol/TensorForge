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
              sycl::vec<float, 4> v21_lin = *(sycl::vec<float, 4>*)&glb_m2[0 + item.get_local_id(0) * 4];
              *(sycl::vec<float, 4>*)&r1[0] = v21_lin;
              sycl::vec<float, 4> v22_lin = *(sycl::vec<float, 4>*)&glb_m2[64 + item.get_local_id(0) * 4];
              *(sycl::vec<float, 4>*)&r1[4] = v22_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v28_data = r0[0];
              float v29_data = r1[0];
              float v32_data = ir2[0];
              ir2[0] = (v32_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 0))));
              float v35_data = r1[1];
              float v38_data = ir2[1];
              ir2[1] = (v38_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 0))));
              float v41_data = r1[2];
              float v44_data = ir2[2];
              ir2[2] = (v44_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 0))));
              float v47_data = r1[3];
              float v50_data = ir2[3];
              ir2[3] = (v50_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 0))));
              float v53_data = r1[4];
              float v56_data = ir2[4];
              ir2[4] = (v56_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 0))));
              float v59_data = r1[5];
              float v62_data = ir2[5];
              ir2[5] = (v62_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 0))));
              float v65_data = r1[6];
              float v68_data = ir2[6];
              ir2[6] = (v68_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
              float v71_data = r1[7];
              float v74_data = ir2[7];
              ir2[7] = (v74_data + (v28_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
              float v79_data = r0[1];
              float v83_data = ir2[0];
              ir2[0] = (v83_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 1))));
              float v89_data = ir2[1];
              ir2[1] = (v89_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 1))));
              float v95_data = ir2[2];
              ir2[2] = (v95_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 1))));
              float v101_data = ir2[3];
              ir2[3] = (v101_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 1))));
              float v107_data = ir2[4];
              ir2[4] = (v107_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 1))));
              float v113_data = ir2[5];
              ir2[5] = (v113_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 1))));
              float v119_data = ir2[6];
              ir2[6] = (v119_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 1))));
              float v125_data = ir2[7];
              ir2[7] = (v125_data + (v79_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 1))));
              float v130_data = r0[2];
              float v134_data = ir2[0];
              ir2[0] = (v134_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 2))));
              float v140_data = ir2[1];
              ir2[1] = (v140_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 2))));
              float v146_data = ir2[2];
              ir2[2] = (v146_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 2))));
              float v152_data = ir2[3];
              ir2[3] = (v152_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 2))));
              float v158_data = ir2[4];
              ir2[4] = (v158_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 2))));
              float v164_data = ir2[5];
              ir2[5] = (v164_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 2))));
              float v170_data = ir2[6];
              ir2[6] = (v170_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 2))));
              float v176_data = ir2[7];
              ir2[7] = (v176_data + (v130_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 2))));
              float v181_data = r0[3];
              float v185_data = ir2[0];
              ir2[0] = (v185_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 3))));
              float v191_data = ir2[1];
              ir2[1] = (v191_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 3))));
              float v197_data = ir2[2];
              ir2[2] = (v197_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 3))));
              float v203_data = ir2[3];
              ir2[3] = (v203_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 3))));
              float v209_data = ir2[4];
              ir2[4] = (v209_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 3))));
              float v215_data = ir2[5];
              ir2[5] = (v215_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 3))));
              float v221_data = ir2[6];
              ir2[6] = (v221_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 3))));
              float v227_data = ir2[7];
              ir2[7] = (v227_data + (v181_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 3))));
              float v232_data = r0[4];
              float v236_data = ir2[0];
              ir2[0] = (v236_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 4))));
              float v242_data = ir2[1];
              ir2[1] = (v242_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 4))));
              float v248_data = ir2[2];
              ir2[2] = (v248_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 4))));
              float v254_data = ir2[3];
              ir2[3] = (v254_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 4))));
              float v260_data = ir2[4];
              ir2[4] = (v260_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 4))));
              float v266_data = ir2[5];
              ir2[5] = (v266_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 4))));
              float v272_data = ir2[6];
              ir2[6] = (v272_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 4))));
              float v278_data = ir2[7];
              ir2[7] = (v278_data + (v232_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 4))));
              float v283_data = r0[5];
              float v287_data = ir2[0];
              ir2[0] = (v287_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 5))));
              float v293_data = ir2[1];
              ir2[1] = (v293_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 5))));
              float v299_data = ir2[2];
              ir2[2] = (v299_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 5))));
              float v305_data = ir2[3];
              ir2[3] = (v305_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 5))));
              float v311_data = ir2[4];
              ir2[4] = (v311_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 5))));
              float v317_data = ir2[5];
              ir2[5] = (v317_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 5))));
              float v323_data = ir2[6];
              ir2[6] = (v323_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 5))));
              float v329_data = ir2[7];
              ir2[7] = (v329_data + (v283_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 5))));
              float v334_data = r0[6];
              float v338_data = ir2[0];
              ir2[0] = (v338_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 6))));
              float v344_data = ir2[1];
              ir2[1] = (v344_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 6))));
              float v350_data = ir2[2];
              ir2[2] = (v350_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 6))));
              float v356_data = ir2[3];
              ir2[3] = (v356_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 6))));
              float v362_data = ir2[4];
              ir2[4] = (v362_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 6))));
              float v368_data = ir2[5];
              ir2[5] = (v368_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 6))));
              float v374_data = ir2[6];
              ir2[6] = (v374_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 6))));
              float v380_data = ir2[7];
              ir2[7] = (v380_data + (v334_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 6))));
              float v385_data = r0[7];
              float v389_data = ir2[0];
              ir2[0] = (v389_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 7))));
              float v395_data = ir2[1];
              ir2[1] = (v395_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 7))));
              float v401_data = ir2[2];
              ir2[2] = (v401_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 7))));
              float v407_data = ir2[3];
              ir2[3] = (v407_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 7))));
              float v413_data = ir2[4];
              ir2[4] = (v413_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 7))));
              float v419_data = ir2[5];
              ir2[5] = (v419_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 7))));
              float v425_data = ir2[6];
              ir2[6] = (v425_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 7))));
              float v431_data = ir2[7];
              ir2[7] = (v431_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 7))));
              float v436_data = r0[8];
              float v440_data = ir2[0];
              ir2[0] = (v440_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 8))));
              float v446_data = ir2[1];
              ir2[1] = (v446_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 8))));
              float v452_data = ir2[2];
              ir2[2] = (v452_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 8))));
              float v458_data = ir2[3];
              ir2[3] = (v458_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 8))));
              float v464_data = ir2[4];
              ir2[4] = (v464_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 8))));
              float v470_data = ir2[5];
              ir2[5] = (v470_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 8))));
              float v476_data = ir2[6];
              ir2[6] = (v476_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 8))));
              float v482_data = ir2[7];
              ir2[7] = (v482_data + (v436_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 8))));
              float v487_data = r0[9];
              float v491_data = ir2[0];
              ir2[0] = (v491_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 9))));
              float v497_data = ir2[1];
              ir2[1] = (v497_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 9))));
              float v503_data = ir2[2];
              ir2[2] = (v503_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 9))));
              float v509_data = ir2[3];
              ir2[3] = (v509_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 9))));
              float v515_data = ir2[4];
              ir2[4] = (v515_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 9))));
              float v521_data = ir2[5];
              ir2[5] = (v521_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 9))));
              float v527_data = ir2[6];
              ir2[6] = (v527_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 9))));
              float v533_data = ir2[7];
              ir2[7] = (v533_data + (v487_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 9))));
              float v538_data = r0[10];
              float v542_data = ir2[0];
              ir2[0] = (v542_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 10))));
              float v548_data = ir2[1];
              ir2[1] = (v548_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 10))));
              float v554_data = ir2[2];
              ir2[2] = (v554_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 10))));
              float v560_data = ir2[3];
              ir2[3] = (v560_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 10))));
              float v566_data = ir2[4];
              ir2[4] = (v566_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 10))));
              float v572_data = ir2[5];
              ir2[5] = (v572_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 10))));
              float v578_data = ir2[6];
              ir2[6] = (v578_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 10))));
              float v584_data = ir2[7];
              ir2[7] = (v584_data + (v538_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 10))));
              float v589_data = r0[11];
              float v593_data = ir2[0];
              ir2[0] = (v593_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 11))));
              float v599_data = ir2[1];
              ir2[1] = (v599_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 11))));
              float v605_data = ir2[2];
              ir2[2] = (v605_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 11))));
              float v611_data = ir2[3];
              ir2[3] = (v611_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 11))));
              float v617_data = ir2[4];
              ir2[4] = (v617_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 11))));
              float v623_data = ir2[5];
              ir2[5] = (v623_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 11))));
              float v629_data = ir2[6];
              ir2[6] = (v629_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 11))));
              float v635_data = ir2[7];
              ir2[7] = (v635_data + (v589_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 11))));
              float v640_data = r0[12];
              float v644_data = ir2[0];
              ir2[0] = (v644_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 12))));
              float v650_data = ir2[1];
              ir2[1] = (v650_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 12))));
              float v656_data = ir2[2];
              ir2[2] = (v656_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 12))));
              float v662_data = ir2[3];
              ir2[3] = (v662_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 12))));
              float v668_data = ir2[4];
              ir2[4] = (v668_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 12))));
              float v674_data = ir2[5];
              ir2[5] = (v674_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 12))));
              float v680_data = ir2[6];
              ir2[6] = (v680_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 12))));
              float v686_data = ir2[7];
              ir2[7] = (v686_data + (v640_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 12))));
              float v691_data = r0[13];
              float v695_data = ir2[0];
              ir2[0] = (v695_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 13))));
              float v701_data = ir2[1];
              ir2[1] = (v701_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 13))));
              float v707_data = ir2[2];
              ir2[2] = (v707_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 13))));
              float v713_data = ir2[3];
              ir2[3] = (v713_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 13))));
              float v719_data = ir2[4];
              ir2[4] = (v719_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 13))));
              float v725_data = ir2[5];
              ir2[5] = (v725_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 13))));
              float v731_data = ir2[6];
              ir2[6] = (v731_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 13))));
              float v737_data = ir2[7];
              ir2[7] = (v737_data + (v691_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 13))));
              float v742_data = r0[14];
              float v746_data = ir2[0];
              ir2[0] = (v746_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 14))));
              float v752_data = ir2[1];
              ir2[1] = (v752_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 14))));
              float v758_data = ir2[2];
              ir2[2] = (v758_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 14))));
              float v764_data = ir2[3];
              ir2[3] = (v764_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 14))));
              float v770_data = ir2[4];
              ir2[4] = (v770_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 14))));
              float v776_data = ir2[5];
              ir2[5] = (v776_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 14))));
              float v782_data = ir2[6];
              ir2[6] = (v782_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 14))));
              float v788_data = ir2[7];
              ir2[7] = (v788_data + (v742_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 14))));
              float v793_data = r0[15];
              float v797_data = ir2[0];
              ir2[0] = (v797_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v29_data, 15))));
              float v803_data = ir2[1];
              ir2[1] = (v803_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v35_data, 15))));
              float v809_data = ir2[2];
              ir2[2] = (v809_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 15))));
              float v815_data = ir2[3];
              ir2[3] = (v815_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 15))));
              float v821_data = ir2[4];
              ir2[4] = (v821_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 15))));
              float v827_data = ir2[5];
              ir2[5] = (v827_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 15))));
              float v833_data = ir2[6];
              ir2[6] = (v833_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 15))));
              float v839_data = ir2[7];
              ir2[7] = (v839_data + (v793_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 15))));
              #pragma unroll
              for (int32_t v844_n0 = 0; v844_n0 < 1; ++v844_n0) {
                #pragma unroll
                for (int32_t v845_n1 = 0; v845_n1 < 8; ++v845_n1) {
                  int32_t v846_a = v844_n0 + v845_n1;
                  float v847_data = ir2[v846_a];
                  r2[v846_a] = v847_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v852_i0 = 0; v852_i0 < 1; ++v852_i0) {
                int32_t v860_lead = v8_lead + (v852_i0 * 16);
                #pragma unroll
                for (int32_t v853_i1 = 0; v853_i1 < 8; ++v853_i1) {
                  float v855_data = r2[(v852_i0 + v853_i1)];
                  glb_m0[(v860_lead + (v853_i1 * 16))] = v855_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

