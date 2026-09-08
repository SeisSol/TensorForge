// === base name ===
kernel_5c7e4b9ce8

// === header ===
void launcher_kernel_5c7e4b9ce8(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5c7e4b9ce8(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5c7e4b9ce8(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5c7e4b9ce8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×11(16×11) {0..16}×{0..11} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×11(16×11) {0..16}×{0..11} strided
        // m0 16×11(16×11) {0..16}×{0..11} strided({0..16}×{0..11})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×11(16×11) {0..16}×{0..11} strided({0..16}×{0..11})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 176 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 176 + 0 + m2_extraOffset];
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
              float r1[11]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v30_lead = v8_lead + (v24_i0 * 16);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 11; ++v25_i1) {
                  float v33_data = glb_m2[(v30_lead + (v25_i1 * 16))];
                  r1[(v24_i0 + v25_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[11]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 11)] [(0, 16)]
              float ir2[11]{};
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
              float v89_data = r1[8];
              float v92_data = ir2[8];
              ir2[8] = (v92_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 0))));
              float v95_data = r1[9];
              float v98_data = ir2[9];
              ir2[9] = (v98_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 0))));
              float v101_data = r1[10];
              float v104_data = ir2[10];
              ir2[10] = (v104_data + (v40_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 0))));
              float v109_data = r0[1];
              float v113_data = ir2[0];
              ir2[0] = (v113_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 1))));
              float v119_data = ir2[1];
              ir2[1] = (v119_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 1))));
              float v125_data = ir2[2];
              ir2[2] = (v125_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 1))));
              float v131_data = ir2[3];
              ir2[3] = (v131_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 1))));
              float v137_data = ir2[4];
              ir2[4] = (v137_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 1))));
              float v143_data = ir2[5];
              ir2[5] = (v143_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 1))));
              float v149_data = ir2[6];
              ir2[6] = (v149_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 1))));
              float v155_data = ir2[7];
              ir2[7] = (v155_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 1))));
              float v161_data = ir2[8];
              ir2[8] = (v161_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 1))));
              float v167_data = ir2[9];
              ir2[9] = (v167_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 1))));
              float v173_data = ir2[10];
              ir2[10] = (v173_data + (v109_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 1))));
              float v178_data = r0[2];
              float v182_data = ir2[0];
              ir2[0] = (v182_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 2))));
              float v188_data = ir2[1];
              ir2[1] = (v188_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 2))));
              float v194_data = ir2[2];
              ir2[2] = (v194_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 2))));
              float v200_data = ir2[3];
              ir2[3] = (v200_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 2))));
              float v206_data = ir2[4];
              ir2[4] = (v206_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 2))));
              float v212_data = ir2[5];
              ir2[5] = (v212_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 2))));
              float v218_data = ir2[6];
              ir2[6] = (v218_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 2))));
              float v224_data = ir2[7];
              ir2[7] = (v224_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 2))));
              float v230_data = ir2[8];
              ir2[8] = (v230_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 2))));
              float v236_data = ir2[9];
              ir2[9] = (v236_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 2))));
              float v242_data = ir2[10];
              ir2[10] = (v242_data + (v178_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 2))));
              float v247_data = r0[3];
              float v251_data = ir2[0];
              ir2[0] = (v251_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 3))));
              float v257_data = ir2[1];
              ir2[1] = (v257_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 3))));
              float v263_data = ir2[2];
              ir2[2] = (v263_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 3))));
              float v269_data = ir2[3];
              ir2[3] = (v269_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 3))));
              float v275_data = ir2[4];
              ir2[4] = (v275_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 3))));
              float v281_data = ir2[5];
              ir2[5] = (v281_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 3))));
              float v287_data = ir2[6];
              ir2[6] = (v287_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 3))));
              float v293_data = ir2[7];
              ir2[7] = (v293_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 3))));
              float v299_data = ir2[8];
              ir2[8] = (v299_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 3))));
              float v305_data = ir2[9];
              ir2[9] = (v305_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 3))));
              float v311_data = ir2[10];
              ir2[10] = (v311_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 3))));
              float v316_data = r0[4];
              float v320_data = ir2[0];
              ir2[0] = (v320_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 4))));
              float v326_data = ir2[1];
              ir2[1] = (v326_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 4))));
              float v332_data = ir2[2];
              ir2[2] = (v332_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 4))));
              float v338_data = ir2[3];
              ir2[3] = (v338_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 4))));
              float v344_data = ir2[4];
              ir2[4] = (v344_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 4))));
              float v350_data = ir2[5];
              ir2[5] = (v350_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 4))));
              float v356_data = ir2[6];
              ir2[6] = (v356_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 4))));
              float v362_data = ir2[7];
              ir2[7] = (v362_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 4))));
              float v368_data = ir2[8];
              ir2[8] = (v368_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 4))));
              float v374_data = ir2[9];
              ir2[9] = (v374_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 4))));
              float v380_data = ir2[10];
              ir2[10] = (v380_data + (v316_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 4))));
              float v385_data = r0[5];
              float v389_data = ir2[0];
              ir2[0] = (v389_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 5))));
              float v395_data = ir2[1];
              ir2[1] = (v395_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 5))));
              float v401_data = ir2[2];
              ir2[2] = (v401_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 5))));
              float v407_data = ir2[3];
              ir2[3] = (v407_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 5))));
              float v413_data = ir2[4];
              ir2[4] = (v413_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 5))));
              float v419_data = ir2[5];
              ir2[5] = (v419_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 5))));
              float v425_data = ir2[6];
              ir2[6] = (v425_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 5))));
              float v431_data = ir2[7];
              ir2[7] = (v431_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 5))));
              float v437_data = ir2[8];
              ir2[8] = (v437_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 5))));
              float v443_data = ir2[9];
              ir2[9] = (v443_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 5))));
              float v449_data = ir2[10];
              ir2[10] = (v449_data + (v385_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 5))));
              float v454_data = r0[6];
              float v458_data = ir2[0];
              ir2[0] = (v458_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 6))));
              float v464_data = ir2[1];
              ir2[1] = (v464_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 6))));
              float v470_data = ir2[2];
              ir2[2] = (v470_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 6))));
              float v476_data = ir2[3];
              ir2[3] = (v476_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 6))));
              float v482_data = ir2[4];
              ir2[4] = (v482_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 6))));
              float v488_data = ir2[5];
              ir2[5] = (v488_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 6))));
              float v494_data = ir2[6];
              ir2[6] = (v494_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 6))));
              float v500_data = ir2[7];
              ir2[7] = (v500_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 6))));
              float v506_data = ir2[8];
              ir2[8] = (v506_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 6))));
              float v512_data = ir2[9];
              ir2[9] = (v512_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 6))));
              float v518_data = ir2[10];
              ir2[10] = (v518_data + (v454_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 6))));
              float v523_data = r0[7];
              float v527_data = ir2[0];
              ir2[0] = (v527_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 7))));
              float v533_data = ir2[1];
              ir2[1] = (v533_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 7))));
              float v539_data = ir2[2];
              ir2[2] = (v539_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 7))));
              float v545_data = ir2[3];
              ir2[3] = (v545_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 7))));
              float v551_data = ir2[4];
              ir2[4] = (v551_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 7))));
              float v557_data = ir2[5];
              ir2[5] = (v557_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 7))));
              float v563_data = ir2[6];
              ir2[6] = (v563_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 7))));
              float v569_data = ir2[7];
              ir2[7] = (v569_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 7))));
              float v575_data = ir2[8];
              ir2[8] = (v575_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 7))));
              float v581_data = ir2[9];
              ir2[9] = (v581_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 7))));
              float v587_data = ir2[10];
              ir2[10] = (v587_data + (v523_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 7))));
              float v592_data = r0[8];
              float v596_data = ir2[0];
              ir2[0] = (v596_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 8))));
              float v602_data = ir2[1];
              ir2[1] = (v602_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 8))));
              float v608_data = ir2[2];
              ir2[2] = (v608_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 8))));
              float v614_data = ir2[3];
              ir2[3] = (v614_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 8))));
              float v620_data = ir2[4];
              ir2[4] = (v620_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 8))));
              float v626_data = ir2[5];
              ir2[5] = (v626_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 8))));
              float v632_data = ir2[6];
              ir2[6] = (v632_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 8))));
              float v638_data = ir2[7];
              ir2[7] = (v638_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 8))));
              float v644_data = ir2[8];
              ir2[8] = (v644_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 8))));
              float v650_data = ir2[9];
              ir2[9] = (v650_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 8))));
              float v656_data = ir2[10];
              ir2[10] = (v656_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 8))));
              float v661_data = r0[9];
              float v665_data = ir2[0];
              ir2[0] = (v665_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 9))));
              float v671_data = ir2[1];
              ir2[1] = (v671_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 9))));
              float v677_data = ir2[2];
              ir2[2] = (v677_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 9))));
              float v683_data = ir2[3];
              ir2[3] = (v683_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 9))));
              float v689_data = ir2[4];
              ir2[4] = (v689_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 9))));
              float v695_data = ir2[5];
              ir2[5] = (v695_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 9))));
              float v701_data = ir2[6];
              ir2[6] = (v701_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 9))));
              float v707_data = ir2[7];
              ir2[7] = (v707_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 9))));
              float v713_data = ir2[8];
              ir2[8] = (v713_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 9))));
              float v719_data = ir2[9];
              ir2[9] = (v719_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 9))));
              float v725_data = ir2[10];
              ir2[10] = (v725_data + (v661_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 9))));
              float v730_data = r0[10];
              float v734_data = ir2[0];
              ir2[0] = (v734_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 10))));
              float v740_data = ir2[1];
              ir2[1] = (v740_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 10))));
              float v746_data = ir2[2];
              ir2[2] = (v746_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 10))));
              float v752_data = ir2[3];
              ir2[3] = (v752_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 10))));
              float v758_data = ir2[4];
              ir2[4] = (v758_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 10))));
              float v764_data = ir2[5];
              ir2[5] = (v764_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 10))));
              float v770_data = ir2[6];
              ir2[6] = (v770_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 10))));
              float v776_data = ir2[7];
              ir2[7] = (v776_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 10))));
              float v782_data = ir2[8];
              ir2[8] = (v782_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 10))));
              float v788_data = ir2[9];
              ir2[9] = (v788_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 10))));
              float v794_data = ir2[10];
              ir2[10] = (v794_data + (v730_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 10))));
              float v799_data = r0[11];
              float v803_data = ir2[0];
              ir2[0] = (v803_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 11))));
              float v809_data = ir2[1];
              ir2[1] = (v809_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 11))));
              float v815_data = ir2[2];
              ir2[2] = (v815_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 11))));
              float v821_data = ir2[3];
              ir2[3] = (v821_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 11))));
              float v827_data = ir2[4];
              ir2[4] = (v827_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 11))));
              float v833_data = ir2[5];
              ir2[5] = (v833_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 11))));
              float v839_data = ir2[6];
              ir2[6] = (v839_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 11))));
              float v845_data = ir2[7];
              ir2[7] = (v845_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 11))));
              float v851_data = ir2[8];
              ir2[8] = (v851_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 11))));
              float v857_data = ir2[9];
              ir2[9] = (v857_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 11))));
              float v863_data = ir2[10];
              ir2[10] = (v863_data + (v799_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 11))));
              float v868_data = r0[12];
              float v872_data = ir2[0];
              ir2[0] = (v872_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 12))));
              float v878_data = ir2[1];
              ir2[1] = (v878_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 12))));
              float v884_data = ir2[2];
              ir2[2] = (v884_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 12))));
              float v890_data = ir2[3];
              ir2[3] = (v890_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 12))));
              float v896_data = ir2[4];
              ir2[4] = (v896_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 12))));
              float v902_data = ir2[5];
              ir2[5] = (v902_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 12))));
              float v908_data = ir2[6];
              ir2[6] = (v908_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 12))));
              float v914_data = ir2[7];
              ir2[7] = (v914_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 12))));
              float v920_data = ir2[8];
              ir2[8] = (v920_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 12))));
              float v926_data = ir2[9];
              ir2[9] = (v926_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 12))));
              float v932_data = ir2[10];
              ir2[10] = (v932_data + (v868_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 12))));
              float v937_data = r0[13];
              float v941_data = ir2[0];
              ir2[0] = (v941_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 13))));
              float v947_data = ir2[1];
              ir2[1] = (v947_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 13))));
              float v953_data = ir2[2];
              ir2[2] = (v953_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 13))));
              float v959_data = ir2[3];
              ir2[3] = (v959_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 13))));
              float v965_data = ir2[4];
              ir2[4] = (v965_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 13))));
              float v971_data = ir2[5];
              ir2[5] = (v971_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 13))));
              float v977_data = ir2[6];
              ir2[6] = (v977_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 13))));
              float v983_data = ir2[7];
              ir2[7] = (v983_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 13))));
              float v989_data = ir2[8];
              ir2[8] = (v989_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 13))));
              float v995_data = ir2[9];
              ir2[9] = (v995_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 13))));
              float v1001_data = ir2[10];
              ir2[10] = (v1001_data + (v937_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 13))));
              float v1006_data = r0[14];
              float v1010_data = ir2[0];
              ir2[0] = (v1010_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 14))));
              float v1016_data = ir2[1];
              ir2[1] = (v1016_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 14))));
              float v1022_data = ir2[2];
              ir2[2] = (v1022_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 14))));
              float v1028_data = ir2[3];
              ir2[3] = (v1028_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 14))));
              float v1034_data = ir2[4];
              ir2[4] = (v1034_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 14))));
              float v1040_data = ir2[5];
              ir2[5] = (v1040_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 14))));
              float v1046_data = ir2[6];
              ir2[6] = (v1046_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 14))));
              float v1052_data = ir2[7];
              ir2[7] = (v1052_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 14))));
              float v1058_data = ir2[8];
              ir2[8] = (v1058_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 14))));
              float v1064_data = ir2[9];
              ir2[9] = (v1064_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 14))));
              float v1070_data = ir2[10];
              ir2[10] = (v1070_data + (v1006_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 14))));
              float v1075_data = r0[15];
              float v1079_data = ir2[0];
              ir2[0] = (v1079_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v41_data, 15))));
              float v1085_data = ir2[1];
              ir2[1] = (v1085_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v47_data, 15))));
              float v1091_data = ir2[2];
              ir2[2] = (v1091_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v53_data, 15))));
              float v1097_data = ir2[3];
              ir2[3] = (v1097_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v59_data, 15))));
              float v1103_data = ir2[4];
              ir2[4] = (v1103_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 15))));
              float v1109_data = ir2[5];
              ir2[5] = (v1109_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 15))));
              float v1115_data = ir2[6];
              ir2[6] = (v1115_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 15))));
              float v1121_data = ir2[7];
              ir2[7] = (v1121_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 15))));
              float v1127_data = ir2[8];
              ir2[8] = (v1127_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 15))));
              float v1133_data = ir2[9];
              ir2[9] = (v1133_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 15))));
              float v1139_data = ir2[10];
              ir2[10] = (v1139_data + (v1075_data * (sycl::group_broadcast(item.get_sub_group(), v101_data, 15))));
              #pragma unroll
              for (int32_t v1144_n0 = 0; v1144_n0 < 1; ++v1144_n0) {
                #pragma unroll
                for (int32_t v1145_n1 = 0; v1145_n1 < 11; ++v1145_n1) {
                  int32_t v1146_a = v1144_n0 + v1145_n1;
                  float v1147_data = ir2[v1146_a];
                  r2[v1146_a] = v1147_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1152_i0 = 0; v1152_i0 < 1; ++v1152_i0) {
                int32_t v1160_lead = v8_lead + (v1152_i0 * 16);
                #pragma unroll
                for (int32_t v1153_i1 = 0; v1153_i1 < 11; ++v1153_i1) {
                  float v1155_data = r2[(v1152_i0 + v1153_i1)];
                  glb_m0[(v1160_lead + (v1153_i1 * 16))] = v1155_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

