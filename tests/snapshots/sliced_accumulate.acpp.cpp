// === base name ===
kernel_82283a2aa0

// === header ===
void launcher_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_82283a2aa0(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  sycl::range<3> grid ((numElements0 + 8 - 1) / 8, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_82283a2aa0(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_82283a2aa0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, const float* m3, unsigned m3_extraOffset, const float* m4, unsigned m4_extraOffset, const float* m5, unsigned m5_extraOffset, const float* m6, unsigned m6_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
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
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
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
              int32_t v10_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
                int32_t v17_lead = v10_lead + (v11_i0 * 32);
                #pragma unroll
                for (int32_t v12_i1 = 0; v12_i1 < 12; ++v12_i1) {
                  float v20_data = glb_m1[(v17_lead + (v12_i1 * 32))];
                  r0[(v11_i0 + v12_i1)] = v20_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              if (v10_lead < 12) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 16; ++v27_i1) {
                  float v35_data = glb_m2[(v10_lead + (v27_i1 * 12))];
                  r1[v27_i1] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v41_i0 = 0; v41_i0 < 1; ++v41_i0) {
                int32_t v47_lead = v10_lead + (v41_i0 * 32);
                #pragma unroll
                for (int32_t v42_i1 = 0; v42_i1 < 12; ++v42_i1) {
                  float v50_data = glb_m3[(v47_lead + (v42_i1 * 32))];
                  r3[(v41_i0 + v42_i1)] = v50_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 16)] [(0, 12)]
              float ir2[16]{};
              float v57_data = r0[0];
              float v58_data = r1[0];
              float v61_data = ir2[0];
              ir2[0] = (v61_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
              float v64_data = r1[1];
              float v67_data = ir2[1];
              ir2[1] = (v67_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
              float v70_data = r1[2];
              float v73_data = ir2[2];
              ir2[2] = (v73_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
              float v76_data = r1[3];
              float v79_data = ir2[3];
              ir2[3] = (v79_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
              float v82_data = r1[4];
              float v85_data = ir2[4];
              ir2[4] = (v85_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
              float v88_data = r1[5];
              float v91_data = ir2[5];
              ir2[5] = (v91_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
              float v94_data = r1[6];
              float v97_data = ir2[6];
              ir2[6] = (v97_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 0))));
              float v100_data = r1[7];
              float v103_data = ir2[7];
              ir2[7] = (v103_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 0))));
              float v106_data = r1[8];
              float v109_data = ir2[8];
              ir2[8] = (v109_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 0))));
              float v112_data = r1[9];
              float v115_data = ir2[9];
              ir2[9] = (v115_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 0))));
              float v118_data = r1[10];
              float v121_data = ir2[10];
              ir2[10] = (v121_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 0))));
              float v124_data = r1[11];
              float v127_data = ir2[11];
              ir2[11] = (v127_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 0))));
              float v130_data = r1[12];
              float v133_data = ir2[12];
              ir2[12] = (v133_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 0))));
              float v136_data = r1[13];
              float v139_data = ir2[13];
              ir2[13] = (v139_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 0))));
              float v142_data = r1[14];
              float v145_data = ir2[14];
              ir2[14] = (v145_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 0))));
              float v148_data = r1[15];
              float v151_data = ir2[15];
              ir2[15] = (v151_data + (v57_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 0))));
              float v156_data = r0[1];
              float v160_data = ir2[0];
              ir2[0] = (v160_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 1))));
              float v166_data = ir2[1];
              ir2[1] = (v166_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 1))));
              float v172_data = ir2[2];
              ir2[2] = (v172_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 1))));
              float v178_data = ir2[3];
              ir2[3] = (v178_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 1))));
              float v184_data = ir2[4];
              ir2[4] = (v184_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 1))));
              float v190_data = ir2[5];
              ir2[5] = (v190_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 1))));
              float v196_data = ir2[6];
              ir2[6] = (v196_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 1))));
              float v202_data = ir2[7];
              ir2[7] = (v202_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 1))));
              float v208_data = ir2[8];
              ir2[8] = (v208_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 1))));
              float v214_data = ir2[9];
              ir2[9] = (v214_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 1))));
              float v220_data = ir2[10];
              ir2[10] = (v220_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 1))));
              float v226_data = ir2[11];
              ir2[11] = (v226_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 1))));
              float v232_data = ir2[12];
              ir2[12] = (v232_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 1))));
              float v238_data = ir2[13];
              ir2[13] = (v238_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 1))));
              float v244_data = ir2[14];
              ir2[14] = (v244_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 1))));
              float v250_data = ir2[15];
              ir2[15] = (v250_data + (v156_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 1))));
              float v255_data = r0[2];
              float v259_data = ir2[0];
              ir2[0] = (v259_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 2))));
              float v265_data = ir2[1];
              ir2[1] = (v265_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 2))));
              float v271_data = ir2[2];
              ir2[2] = (v271_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 2))));
              float v277_data = ir2[3];
              ir2[3] = (v277_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 2))));
              float v283_data = ir2[4];
              ir2[4] = (v283_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 2))));
              float v289_data = ir2[5];
              ir2[5] = (v289_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 2))));
              float v295_data = ir2[6];
              ir2[6] = (v295_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 2))));
              float v301_data = ir2[7];
              ir2[7] = (v301_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 2))));
              float v307_data = ir2[8];
              ir2[8] = (v307_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 2))));
              float v313_data = ir2[9];
              ir2[9] = (v313_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 2))));
              float v319_data = ir2[10];
              ir2[10] = (v319_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 2))));
              float v325_data = ir2[11];
              ir2[11] = (v325_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 2))));
              float v331_data = ir2[12];
              ir2[12] = (v331_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 2))));
              float v337_data = ir2[13];
              ir2[13] = (v337_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 2))));
              float v343_data = ir2[14];
              ir2[14] = (v343_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 2))));
              float v349_data = ir2[15];
              ir2[15] = (v349_data + (v255_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 2))));
              float v354_data = r0[3];
              float v358_data = ir2[0];
              ir2[0] = (v358_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 3))));
              float v364_data = ir2[1];
              ir2[1] = (v364_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 3))));
              float v370_data = ir2[2];
              ir2[2] = (v370_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 3))));
              float v376_data = ir2[3];
              ir2[3] = (v376_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 3))));
              float v382_data = ir2[4];
              ir2[4] = (v382_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 3))));
              float v388_data = ir2[5];
              ir2[5] = (v388_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 3))));
              float v394_data = ir2[6];
              ir2[6] = (v394_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 3))));
              float v400_data = ir2[7];
              ir2[7] = (v400_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 3))));
              float v406_data = ir2[8];
              ir2[8] = (v406_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 3))));
              float v412_data = ir2[9];
              ir2[9] = (v412_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 3))));
              float v418_data = ir2[10];
              ir2[10] = (v418_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 3))));
              float v424_data = ir2[11];
              ir2[11] = (v424_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 3))));
              float v430_data = ir2[12];
              ir2[12] = (v430_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 3))));
              float v436_data = ir2[13];
              ir2[13] = (v436_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 3))));
              float v442_data = ir2[14];
              ir2[14] = (v442_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 3))));
              float v448_data = ir2[15];
              ir2[15] = (v448_data + (v354_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 3))));
              float v453_data = r0[4];
              float v457_data = ir2[0];
              ir2[0] = (v457_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 4))));
              float v463_data = ir2[1];
              ir2[1] = (v463_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 4))));
              float v469_data = ir2[2];
              ir2[2] = (v469_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 4))));
              float v475_data = ir2[3];
              ir2[3] = (v475_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 4))));
              float v481_data = ir2[4];
              ir2[4] = (v481_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 4))));
              float v487_data = ir2[5];
              ir2[5] = (v487_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 4))));
              float v493_data = ir2[6];
              ir2[6] = (v493_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 4))));
              float v499_data = ir2[7];
              ir2[7] = (v499_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 4))));
              float v505_data = ir2[8];
              ir2[8] = (v505_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 4))));
              float v511_data = ir2[9];
              ir2[9] = (v511_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 4))));
              float v517_data = ir2[10];
              ir2[10] = (v517_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 4))));
              float v523_data = ir2[11];
              ir2[11] = (v523_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 4))));
              float v529_data = ir2[12];
              ir2[12] = (v529_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 4))));
              float v535_data = ir2[13];
              ir2[13] = (v535_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 4))));
              float v541_data = ir2[14];
              ir2[14] = (v541_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 4))));
              float v547_data = ir2[15];
              ir2[15] = (v547_data + (v453_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 4))));
              float v552_data = r0[5];
              float v556_data = ir2[0];
              ir2[0] = (v556_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 5))));
              float v562_data = ir2[1];
              ir2[1] = (v562_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 5))));
              float v568_data = ir2[2];
              ir2[2] = (v568_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 5))));
              float v574_data = ir2[3];
              ir2[3] = (v574_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 5))));
              float v580_data = ir2[4];
              ir2[4] = (v580_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 5))));
              float v586_data = ir2[5];
              ir2[5] = (v586_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 5))));
              float v592_data = ir2[6];
              ir2[6] = (v592_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 5))));
              float v598_data = ir2[7];
              ir2[7] = (v598_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 5))));
              float v604_data = ir2[8];
              ir2[8] = (v604_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 5))));
              float v610_data = ir2[9];
              ir2[9] = (v610_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 5))));
              float v616_data = ir2[10];
              ir2[10] = (v616_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 5))));
              float v622_data = ir2[11];
              ir2[11] = (v622_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 5))));
              float v628_data = ir2[12];
              ir2[12] = (v628_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 5))));
              float v634_data = ir2[13];
              ir2[13] = (v634_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 5))));
              float v640_data = ir2[14];
              ir2[14] = (v640_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 5))));
              float v646_data = ir2[15];
              ir2[15] = (v646_data + (v552_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 5))));
              float v651_data = r0[6];
              float v655_data = ir2[0];
              ir2[0] = (v655_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 6))));
              float v661_data = ir2[1];
              ir2[1] = (v661_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 6))));
              float v667_data = ir2[2];
              ir2[2] = (v667_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 6))));
              float v673_data = ir2[3];
              ir2[3] = (v673_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 6))));
              float v679_data = ir2[4];
              ir2[4] = (v679_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 6))));
              float v685_data = ir2[5];
              ir2[5] = (v685_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 6))));
              float v691_data = ir2[6];
              ir2[6] = (v691_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 6))));
              float v697_data = ir2[7];
              ir2[7] = (v697_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 6))));
              float v703_data = ir2[8];
              ir2[8] = (v703_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 6))));
              float v709_data = ir2[9];
              ir2[9] = (v709_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 6))));
              float v715_data = ir2[10];
              ir2[10] = (v715_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 6))));
              float v721_data = ir2[11];
              ir2[11] = (v721_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 6))));
              float v727_data = ir2[12];
              ir2[12] = (v727_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 6))));
              float v733_data = ir2[13];
              ir2[13] = (v733_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 6))));
              float v739_data = ir2[14];
              ir2[14] = (v739_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 6))));
              float v745_data = ir2[15];
              ir2[15] = (v745_data + (v651_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 6))));
              float v750_data = r0[7];
              float v754_data = ir2[0];
              ir2[0] = (v754_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 7))));
              float v760_data = ir2[1];
              ir2[1] = (v760_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 7))));
              float v766_data = ir2[2];
              ir2[2] = (v766_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 7))));
              float v772_data = ir2[3];
              ir2[3] = (v772_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 7))));
              float v778_data = ir2[4];
              ir2[4] = (v778_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 7))));
              float v784_data = ir2[5];
              ir2[5] = (v784_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 7))));
              float v790_data = ir2[6];
              ir2[6] = (v790_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 7))));
              float v796_data = ir2[7];
              ir2[7] = (v796_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 7))));
              float v802_data = ir2[8];
              ir2[8] = (v802_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 7))));
              float v808_data = ir2[9];
              ir2[9] = (v808_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 7))));
              float v814_data = ir2[10];
              ir2[10] = (v814_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 7))));
              float v820_data = ir2[11];
              ir2[11] = (v820_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 7))));
              float v826_data = ir2[12];
              ir2[12] = (v826_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 7))));
              float v832_data = ir2[13];
              ir2[13] = (v832_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 7))));
              float v838_data = ir2[14];
              ir2[14] = (v838_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 7))));
              float v844_data = ir2[15];
              ir2[15] = (v844_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 7))));
              float v849_data = r0[8];
              float v853_data = ir2[0];
              ir2[0] = (v853_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 8))));
              float v859_data = ir2[1];
              ir2[1] = (v859_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 8))));
              float v865_data = ir2[2];
              ir2[2] = (v865_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 8))));
              float v871_data = ir2[3];
              ir2[3] = (v871_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 8))));
              float v877_data = ir2[4];
              ir2[4] = (v877_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 8))));
              float v883_data = ir2[5];
              ir2[5] = (v883_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 8))));
              float v889_data = ir2[6];
              ir2[6] = (v889_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 8))));
              float v895_data = ir2[7];
              ir2[7] = (v895_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 8))));
              float v901_data = ir2[8];
              ir2[8] = (v901_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 8))));
              float v907_data = ir2[9];
              ir2[9] = (v907_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 8))));
              float v913_data = ir2[10];
              ir2[10] = (v913_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 8))));
              float v919_data = ir2[11];
              ir2[11] = (v919_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 8))));
              float v925_data = ir2[12];
              ir2[12] = (v925_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 8))));
              float v931_data = ir2[13];
              ir2[13] = (v931_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 8))));
              float v937_data = ir2[14];
              ir2[14] = (v937_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 8))));
              float v943_data = ir2[15];
              ir2[15] = (v943_data + (v849_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 8))));
              float v948_data = r0[9];
              float v952_data = ir2[0];
              ir2[0] = (v952_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 9))));
              float v958_data = ir2[1];
              ir2[1] = (v958_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 9))));
              float v964_data = ir2[2];
              ir2[2] = (v964_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 9))));
              float v970_data = ir2[3];
              ir2[3] = (v970_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 9))));
              float v976_data = ir2[4];
              ir2[4] = (v976_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 9))));
              float v982_data = ir2[5];
              ir2[5] = (v982_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 9))));
              float v988_data = ir2[6];
              ir2[6] = (v988_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 9))));
              float v994_data = ir2[7];
              ir2[7] = (v994_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 9))));
              float v1000_data = ir2[8];
              ir2[8] = (v1000_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 9))));
              float v1006_data = ir2[9];
              ir2[9] = (v1006_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 9))));
              float v1012_data = ir2[10];
              ir2[10] = (v1012_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 9))));
              float v1018_data = ir2[11];
              ir2[11] = (v1018_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 9))));
              float v1024_data = ir2[12];
              ir2[12] = (v1024_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 9))));
              float v1030_data = ir2[13];
              ir2[13] = (v1030_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 9))));
              float v1036_data = ir2[14];
              ir2[14] = (v1036_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 9))));
              float v1042_data = ir2[15];
              ir2[15] = (v1042_data + (v948_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 9))));
              float v1047_data = r0[10];
              float v1051_data = ir2[0];
              ir2[0] = (v1051_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 10))));
              float v1057_data = ir2[1];
              ir2[1] = (v1057_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 10))));
              float v1063_data = ir2[2];
              ir2[2] = (v1063_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 10))));
              float v1069_data = ir2[3];
              ir2[3] = (v1069_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 10))));
              float v1075_data = ir2[4];
              ir2[4] = (v1075_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 10))));
              float v1081_data = ir2[5];
              ir2[5] = (v1081_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 10))));
              float v1087_data = ir2[6];
              ir2[6] = (v1087_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 10))));
              float v1093_data = ir2[7];
              ir2[7] = (v1093_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 10))));
              float v1099_data = ir2[8];
              ir2[8] = (v1099_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 10))));
              float v1105_data = ir2[9];
              ir2[9] = (v1105_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 10))));
              float v1111_data = ir2[10];
              ir2[10] = (v1111_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 10))));
              float v1117_data = ir2[11];
              ir2[11] = (v1117_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 10))));
              float v1123_data = ir2[12];
              ir2[12] = (v1123_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 10))));
              float v1129_data = ir2[13];
              ir2[13] = (v1129_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 10))));
              float v1135_data = ir2[14];
              ir2[14] = (v1135_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 10))));
              float v1141_data = ir2[15];
              ir2[15] = (v1141_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 10))));
              float v1146_data = r0[11];
              float v1150_data = ir2[0];
              ir2[0] = (v1150_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 11))));
              float v1156_data = ir2[1];
              ir2[1] = (v1156_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 11))));
              float v1162_data = ir2[2];
              ir2[2] = (v1162_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 11))));
              float v1168_data = ir2[3];
              ir2[3] = (v1168_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 11))));
              float v1174_data = ir2[4];
              ir2[4] = (v1174_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 11))));
              float v1180_data = ir2[5];
              ir2[5] = (v1180_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 11))));
              float v1186_data = ir2[6];
              ir2[6] = (v1186_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v94_data, 11))));
              float v1192_data = ir2[7];
              ir2[7] = (v1192_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v100_data, 11))));
              float v1198_data = ir2[8];
              ir2[8] = (v1198_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v106_data, 11))));
              float v1204_data = ir2[9];
              ir2[9] = (v1204_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v112_data, 11))));
              float v1210_data = ir2[10];
              ir2[10] = (v1210_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v118_data, 11))));
              float v1216_data = ir2[11];
              ir2[11] = (v1216_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v124_data, 11))));
              float v1222_data = ir2[12];
              ir2[12] = (v1222_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v130_data, 11))));
              float v1228_data = ir2[13];
              ir2[13] = (v1228_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v136_data, 11))));
              float v1234_data = ir2[14];
              ir2[14] = (v1234_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 11))));
              float v1240_data = ir2[15];
              ir2[15] = (v1240_data + (v1146_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 11))));
              #pragma unroll
              for (int32_t v1245_n0 = 0; v1245_n0 < 1; ++v1245_n0) {
                #pragma unroll
                for (int32_t v1246_n1 = 0; v1246_n1 < 16; ++v1246_n1) {
                  int32_t v1247_a = v1245_n0 + v1246_n1;
                  float v1248_data = ir2[v1247_a];
                  r2[v1247_a] = v1248_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1253_i0 = 0; v1253_i0 < 1; ++v1253_i0) {
                int32_t v1261_lead = v10_lead + (v1253_i0 * 32);
                #pragma unroll
                for (int32_t v1254_i1 = 0; v1254_i1 < 16; ++v1254_i1) {
                  float v1256_data = r2[(v1253_i0 + v1254_i1)];
                  glb_m0[(v1261_lead + (v1254_i1 * 32))] = v1256_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v10_lead < 12) {
                #pragma unroll
                for (int32_t v1269_i1 = 0; v1269_i1 < 8; ++v1269_i1) {
                  float v1277_data = glb_m4[(v10_lead + (v1269_i1 * 12))];
                  r4[v1269_i1] = v1277_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1283_i0 = 0; v1283_i0 < 1; ++v1283_i0) {
                int32_t v1289_lead = v10_lead + (v1283_i0 * 32);
                #pragma unroll
                for (int32_t v1284_i1 = 0; v1284_i1 < 8; ++v1284_i1) {
                  float v1292_data = glb_m0[(v1289_lead + (v1284_i1 * 32))];
                  r5[(v1283_i0 + v1284_i1)] = v1292_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r7[12]{};
              // r7 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v1298_i0 = 0; v1298_i0 < 1; ++v1298_i0) {
                int32_t v1304_lead = v10_lead + (v1298_i0 * 32);
                #pragma unroll
                for (int32_t v1299_i1 = 0; v1299_i1 < 12; ++v1299_i1) {
                  float v1307_data = glb_m5[(v1304_lead + (v1299_i1 * 32))];
                  r7[(v1298_i0 + v1299_i1)] = v1307_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m0););
              float r6[8]{};
              // r6 = +(r3 * r4) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir6[8]{};
              float v1314_data = r3[0];
              float v1315_data = r4[0];
              float v1318_data = ir6[0];
              ir6[0] = (v1318_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 0))));
              float v1321_data = r4[1];
              float v1324_data = ir6[1];
              ir6[1] = (v1324_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 0))));
              float v1327_data = r4[2];
              float v1330_data = ir6[2];
              ir6[2] = (v1330_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 0))));
              float v1333_data = r4[3];
              float v1336_data = ir6[3];
              ir6[3] = (v1336_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 0))));
              float v1339_data = r4[4];
              float v1342_data = ir6[4];
              ir6[4] = (v1342_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 0))));
              float v1345_data = r4[5];
              float v1348_data = ir6[5];
              ir6[5] = (v1348_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 0))));
              float v1351_data = r4[6];
              float v1354_data = ir6[6];
              ir6[6] = (v1354_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 0))));
              float v1357_data = r4[7];
              float v1360_data = ir6[7];
              ir6[7] = (v1360_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 0))));
              float v1365_data = r3[1];
              float v1369_data = ir6[0];
              ir6[0] = (v1369_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 1))));
              float v1375_data = ir6[1];
              ir6[1] = (v1375_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 1))));
              float v1381_data = ir6[2];
              ir6[2] = (v1381_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 1))));
              float v1387_data = ir6[3];
              ir6[3] = (v1387_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 1))));
              float v1393_data = ir6[4];
              ir6[4] = (v1393_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 1))));
              float v1399_data = ir6[5];
              ir6[5] = (v1399_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 1))));
              float v1405_data = ir6[6];
              ir6[6] = (v1405_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 1))));
              float v1411_data = ir6[7];
              ir6[7] = (v1411_data + (v1365_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 1))));
              float v1416_data = r3[2];
              float v1420_data = ir6[0];
              ir6[0] = (v1420_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 2))));
              float v1426_data = ir6[1];
              ir6[1] = (v1426_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 2))));
              float v1432_data = ir6[2];
              ir6[2] = (v1432_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 2))));
              float v1438_data = ir6[3];
              ir6[3] = (v1438_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 2))));
              float v1444_data = ir6[4];
              ir6[4] = (v1444_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 2))));
              float v1450_data = ir6[5];
              ir6[5] = (v1450_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 2))));
              float v1456_data = ir6[6];
              ir6[6] = (v1456_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 2))));
              float v1462_data = ir6[7];
              ir6[7] = (v1462_data + (v1416_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 2))));
              float v1467_data = r3[3];
              float v1471_data = ir6[0];
              ir6[0] = (v1471_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 3))));
              float v1477_data = ir6[1];
              ir6[1] = (v1477_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 3))));
              float v1483_data = ir6[2];
              ir6[2] = (v1483_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 3))));
              float v1489_data = ir6[3];
              ir6[3] = (v1489_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 3))));
              float v1495_data = ir6[4];
              ir6[4] = (v1495_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 3))));
              float v1501_data = ir6[5];
              ir6[5] = (v1501_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 3))));
              float v1507_data = ir6[6];
              ir6[6] = (v1507_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 3))));
              float v1513_data = ir6[7];
              ir6[7] = (v1513_data + (v1467_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 3))));
              float v1518_data = r3[4];
              float v1522_data = ir6[0];
              ir6[0] = (v1522_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 4))));
              float v1528_data = ir6[1];
              ir6[1] = (v1528_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 4))));
              float v1534_data = ir6[2];
              ir6[2] = (v1534_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 4))));
              float v1540_data = ir6[3];
              ir6[3] = (v1540_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 4))));
              float v1546_data = ir6[4];
              ir6[4] = (v1546_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 4))));
              float v1552_data = ir6[5];
              ir6[5] = (v1552_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 4))));
              float v1558_data = ir6[6];
              ir6[6] = (v1558_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 4))));
              float v1564_data = ir6[7];
              ir6[7] = (v1564_data + (v1518_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 4))));
              float v1569_data = r3[5];
              float v1573_data = ir6[0];
              ir6[0] = (v1573_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 5))));
              float v1579_data = ir6[1];
              ir6[1] = (v1579_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 5))));
              float v1585_data = ir6[2];
              ir6[2] = (v1585_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 5))));
              float v1591_data = ir6[3];
              ir6[3] = (v1591_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 5))));
              float v1597_data = ir6[4];
              ir6[4] = (v1597_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 5))));
              float v1603_data = ir6[5];
              ir6[5] = (v1603_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 5))));
              float v1609_data = ir6[6];
              ir6[6] = (v1609_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 5))));
              float v1615_data = ir6[7];
              ir6[7] = (v1615_data + (v1569_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 5))));
              float v1620_data = r3[6];
              float v1624_data = ir6[0];
              ir6[0] = (v1624_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 6))));
              float v1630_data = ir6[1];
              ir6[1] = (v1630_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 6))));
              float v1636_data = ir6[2];
              ir6[2] = (v1636_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 6))));
              float v1642_data = ir6[3];
              ir6[3] = (v1642_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 6))));
              float v1648_data = ir6[4];
              ir6[4] = (v1648_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 6))));
              float v1654_data = ir6[5];
              ir6[5] = (v1654_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 6))));
              float v1660_data = ir6[6];
              ir6[6] = (v1660_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 6))));
              float v1666_data = ir6[7];
              ir6[7] = (v1666_data + (v1620_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 6))));
              float v1671_data = r3[7];
              float v1675_data = ir6[0];
              ir6[0] = (v1675_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 7))));
              float v1681_data = ir6[1];
              ir6[1] = (v1681_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 7))));
              float v1687_data = ir6[2];
              ir6[2] = (v1687_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 7))));
              float v1693_data = ir6[3];
              ir6[3] = (v1693_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 7))));
              float v1699_data = ir6[4];
              ir6[4] = (v1699_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 7))));
              float v1705_data = ir6[5];
              ir6[5] = (v1705_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 7))));
              float v1711_data = ir6[6];
              ir6[6] = (v1711_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 7))));
              float v1717_data = ir6[7];
              ir6[7] = (v1717_data + (v1671_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 7))));
              float v1722_data = r3[8];
              float v1726_data = ir6[0];
              ir6[0] = (v1726_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 8))));
              float v1732_data = ir6[1];
              ir6[1] = (v1732_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 8))));
              float v1738_data = ir6[2];
              ir6[2] = (v1738_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 8))));
              float v1744_data = ir6[3];
              ir6[3] = (v1744_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 8))));
              float v1750_data = ir6[4];
              ir6[4] = (v1750_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 8))));
              float v1756_data = ir6[5];
              ir6[5] = (v1756_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 8))));
              float v1762_data = ir6[6];
              ir6[6] = (v1762_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 8))));
              float v1768_data = ir6[7];
              ir6[7] = (v1768_data + (v1722_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 8))));
              float v1773_data = r3[9];
              float v1777_data = ir6[0];
              ir6[0] = (v1777_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 9))));
              float v1783_data = ir6[1];
              ir6[1] = (v1783_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 9))));
              float v1789_data = ir6[2];
              ir6[2] = (v1789_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 9))));
              float v1795_data = ir6[3];
              ir6[3] = (v1795_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 9))));
              float v1801_data = ir6[4];
              ir6[4] = (v1801_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 9))));
              float v1807_data = ir6[5];
              ir6[5] = (v1807_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 9))));
              float v1813_data = ir6[6];
              ir6[6] = (v1813_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 9))));
              float v1819_data = ir6[7];
              ir6[7] = (v1819_data + (v1773_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 9))));
              float v1824_data = r3[10];
              float v1828_data = ir6[0];
              ir6[0] = (v1828_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 10))));
              float v1834_data = ir6[1];
              ir6[1] = (v1834_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 10))));
              float v1840_data = ir6[2];
              ir6[2] = (v1840_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 10))));
              float v1846_data = ir6[3];
              ir6[3] = (v1846_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 10))));
              float v1852_data = ir6[4];
              ir6[4] = (v1852_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 10))));
              float v1858_data = ir6[5];
              ir6[5] = (v1858_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 10))));
              float v1864_data = ir6[6];
              ir6[6] = (v1864_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 10))));
              float v1870_data = ir6[7];
              ir6[7] = (v1870_data + (v1824_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 10))));
              float v1875_data = r3[11];
              float v1879_data = ir6[0];
              ir6[0] = (v1879_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1315_data, 11))));
              float v1885_data = ir6[1];
              ir6[1] = (v1885_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1321_data, 11))));
              float v1891_data = ir6[2];
              ir6[2] = (v1891_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1327_data, 11))));
              float v1897_data = ir6[3];
              ir6[3] = (v1897_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1333_data, 11))));
              float v1903_data = ir6[4];
              ir6[4] = (v1903_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1339_data, 11))));
              float v1909_data = ir6[5];
              ir6[5] = (v1909_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1345_data, 11))));
              float v1915_data = ir6[6];
              ir6[6] = (v1915_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1351_data, 11))));
              float v1921_data = ir6[7];
              ir6[7] = (v1921_data + (v1875_data * (sycl::group_broadcast(item.get_sub_group(), v1357_data, 11))));
              #pragma unroll
              for (int32_t v1926_n0 = 0; v1926_n0 < 1; ++v1926_n0) {
                #pragma unroll
                for (int32_t v1927_n1 = 0; v1927_n1 < 8; ++v1927_n1) {
                  int32_t v1928_a = v1926_n0 + v1927_n1;
                  float v1929_data = ir6[v1928_a];
                  float v1931_data = r5[v1928_a];
                  r6[v1928_a] = (v1931_data + v1929_data);
                }
              }
              // glb_m0 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v1937_i0 = 0; v1937_i0 < 1; ++v1937_i0) {
                int32_t v1945_lead = v10_lead + (v1937_i0 * 32);
                #pragma unroll
                for (int32_t v1938_i1 = 0; v1938_i1 < 8; ++v1938_i1) {
                  float v1940_data = r6[(v1937_i0 + v1938_i1)];
                  glb_m0[(v1945_lead + (v1938_i1 * 32))] = v1940_data;
                }
              }
              float r8[8]{};
              // r8 = load{g>r}(glb_m6);
              if (v10_lead < 12) {
                #pragma unroll
                for (int32_t v1953_i1 = 0; v1953_i1 < 8; ++v1953_i1) {
                  float v1961_data = glb_m6[(v10_lead + (v1953_i1 * 12))];
                  r8[v1953_i1] = v1961_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m5););
              float r9[8]{};
              // r9 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1967_i0 = 0; v1967_i0 < 1; ++v1967_i0) {
                int32_t v1973_lead = v10_lead + (v1967_i0 * 32);
                #pragma unroll
                for (int32_t v1968_i1 = 0; v1968_i1 < 8; ++v1968_i1) {
                  float v1977_data = glb_m0[(v1973_lead + ((v1968_i1 + 8) * 32))];
                  r9[(v1967_i0 + v1968_i1)] = v1977_data;
                }
              }
              // wait(r8 = load{g>r}(glb_m6););
              // wait(r9 = load{g>r}(glb_m0););
              float r10[8]{};
              // r10 = +(r7 * r8) + name: r9, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir10[8]{};
              float v1984_data = r7[0];
              float v1985_data = r8[0];
              float v1988_data = ir10[0];
              ir10[0] = (v1988_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 0))));
              float v1991_data = r8[1];
              float v1994_data = ir10[1];
              ir10[1] = (v1994_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 0))));
              float v1997_data = r8[2];
              float v2000_data = ir10[2];
              ir10[2] = (v2000_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 0))));
              float v2003_data = r8[3];
              float v2006_data = ir10[3];
              ir10[3] = (v2006_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 0))));
              float v2009_data = r8[4];
              float v2012_data = ir10[4];
              ir10[4] = (v2012_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 0))));
              float v2015_data = r8[5];
              float v2018_data = ir10[5];
              ir10[5] = (v2018_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 0))));
              float v2021_data = r8[6];
              float v2024_data = ir10[6];
              ir10[6] = (v2024_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 0))));
              float v2027_data = r8[7];
              float v2030_data = ir10[7];
              ir10[7] = (v2030_data + (v1984_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 0))));
              float v2035_data = r7[1];
              float v2039_data = ir10[0];
              ir10[0] = (v2039_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 1))));
              float v2045_data = ir10[1];
              ir10[1] = (v2045_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 1))));
              float v2051_data = ir10[2];
              ir10[2] = (v2051_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 1))));
              float v2057_data = ir10[3];
              ir10[3] = (v2057_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 1))));
              float v2063_data = ir10[4];
              ir10[4] = (v2063_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 1))));
              float v2069_data = ir10[5];
              ir10[5] = (v2069_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 1))));
              float v2075_data = ir10[6];
              ir10[6] = (v2075_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 1))));
              float v2081_data = ir10[7];
              ir10[7] = (v2081_data + (v2035_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 1))));
              float v2086_data = r7[2];
              float v2090_data = ir10[0];
              ir10[0] = (v2090_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 2))));
              float v2096_data = ir10[1];
              ir10[1] = (v2096_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 2))));
              float v2102_data = ir10[2];
              ir10[2] = (v2102_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 2))));
              float v2108_data = ir10[3];
              ir10[3] = (v2108_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 2))));
              float v2114_data = ir10[4];
              ir10[4] = (v2114_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 2))));
              float v2120_data = ir10[5];
              ir10[5] = (v2120_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 2))));
              float v2126_data = ir10[6];
              ir10[6] = (v2126_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 2))));
              float v2132_data = ir10[7];
              ir10[7] = (v2132_data + (v2086_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 2))));
              float v2137_data = r7[3];
              float v2141_data = ir10[0];
              ir10[0] = (v2141_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 3))));
              float v2147_data = ir10[1];
              ir10[1] = (v2147_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 3))));
              float v2153_data = ir10[2];
              ir10[2] = (v2153_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 3))));
              float v2159_data = ir10[3];
              ir10[3] = (v2159_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 3))));
              float v2165_data = ir10[4];
              ir10[4] = (v2165_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 3))));
              float v2171_data = ir10[5];
              ir10[5] = (v2171_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 3))));
              float v2177_data = ir10[6];
              ir10[6] = (v2177_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 3))));
              float v2183_data = ir10[7];
              ir10[7] = (v2183_data + (v2137_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 3))));
              float v2188_data = r7[4];
              float v2192_data = ir10[0];
              ir10[0] = (v2192_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 4))));
              float v2198_data = ir10[1];
              ir10[1] = (v2198_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 4))));
              float v2204_data = ir10[2];
              ir10[2] = (v2204_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 4))));
              float v2210_data = ir10[3];
              ir10[3] = (v2210_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 4))));
              float v2216_data = ir10[4];
              ir10[4] = (v2216_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 4))));
              float v2222_data = ir10[5];
              ir10[5] = (v2222_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 4))));
              float v2228_data = ir10[6];
              ir10[6] = (v2228_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 4))));
              float v2234_data = ir10[7];
              ir10[7] = (v2234_data + (v2188_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 4))));
              float v2239_data = r7[5];
              float v2243_data = ir10[0];
              ir10[0] = (v2243_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 5))));
              float v2249_data = ir10[1];
              ir10[1] = (v2249_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 5))));
              float v2255_data = ir10[2];
              ir10[2] = (v2255_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 5))));
              float v2261_data = ir10[3];
              ir10[3] = (v2261_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 5))));
              float v2267_data = ir10[4];
              ir10[4] = (v2267_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 5))));
              float v2273_data = ir10[5];
              ir10[5] = (v2273_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 5))));
              float v2279_data = ir10[6];
              ir10[6] = (v2279_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 5))));
              float v2285_data = ir10[7];
              ir10[7] = (v2285_data + (v2239_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 5))));
              float v2290_data = r7[6];
              float v2294_data = ir10[0];
              ir10[0] = (v2294_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 6))));
              float v2300_data = ir10[1];
              ir10[1] = (v2300_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 6))));
              float v2306_data = ir10[2];
              ir10[2] = (v2306_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 6))));
              float v2312_data = ir10[3];
              ir10[3] = (v2312_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 6))));
              float v2318_data = ir10[4];
              ir10[4] = (v2318_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 6))));
              float v2324_data = ir10[5];
              ir10[5] = (v2324_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 6))));
              float v2330_data = ir10[6];
              ir10[6] = (v2330_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 6))));
              float v2336_data = ir10[7];
              ir10[7] = (v2336_data + (v2290_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 6))));
              float v2341_data = r7[7];
              float v2345_data = ir10[0];
              ir10[0] = (v2345_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 7))));
              float v2351_data = ir10[1];
              ir10[1] = (v2351_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 7))));
              float v2357_data = ir10[2];
              ir10[2] = (v2357_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 7))));
              float v2363_data = ir10[3];
              ir10[3] = (v2363_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 7))));
              float v2369_data = ir10[4];
              ir10[4] = (v2369_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 7))));
              float v2375_data = ir10[5];
              ir10[5] = (v2375_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 7))));
              float v2381_data = ir10[6];
              ir10[6] = (v2381_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 7))));
              float v2387_data = ir10[7];
              ir10[7] = (v2387_data + (v2341_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 7))));
              float v2392_data = r7[8];
              float v2396_data = ir10[0];
              ir10[0] = (v2396_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 8))));
              float v2402_data = ir10[1];
              ir10[1] = (v2402_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 8))));
              float v2408_data = ir10[2];
              ir10[2] = (v2408_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 8))));
              float v2414_data = ir10[3];
              ir10[3] = (v2414_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 8))));
              float v2420_data = ir10[4];
              ir10[4] = (v2420_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 8))));
              float v2426_data = ir10[5];
              ir10[5] = (v2426_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 8))));
              float v2432_data = ir10[6];
              ir10[6] = (v2432_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 8))));
              float v2438_data = ir10[7];
              ir10[7] = (v2438_data + (v2392_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 8))));
              float v2443_data = r7[9];
              float v2447_data = ir10[0];
              ir10[0] = (v2447_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 9))));
              float v2453_data = ir10[1];
              ir10[1] = (v2453_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 9))));
              float v2459_data = ir10[2];
              ir10[2] = (v2459_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 9))));
              float v2465_data = ir10[3];
              ir10[3] = (v2465_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 9))));
              float v2471_data = ir10[4];
              ir10[4] = (v2471_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 9))));
              float v2477_data = ir10[5];
              ir10[5] = (v2477_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 9))));
              float v2483_data = ir10[6];
              ir10[6] = (v2483_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 9))));
              float v2489_data = ir10[7];
              ir10[7] = (v2489_data + (v2443_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 9))));
              float v2494_data = r7[10];
              float v2498_data = ir10[0];
              ir10[0] = (v2498_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 10))));
              float v2504_data = ir10[1];
              ir10[1] = (v2504_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 10))));
              float v2510_data = ir10[2];
              ir10[2] = (v2510_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 10))));
              float v2516_data = ir10[3];
              ir10[3] = (v2516_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 10))));
              float v2522_data = ir10[4];
              ir10[4] = (v2522_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 10))));
              float v2528_data = ir10[5];
              ir10[5] = (v2528_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 10))));
              float v2534_data = ir10[6];
              ir10[6] = (v2534_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 10))));
              float v2540_data = ir10[7];
              ir10[7] = (v2540_data + (v2494_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 10))));
              float v2545_data = r7[11];
              float v2549_data = ir10[0];
              ir10[0] = (v2549_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v1985_data, 11))));
              float v2555_data = ir10[1];
              ir10[1] = (v2555_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v1991_data, 11))));
              float v2561_data = ir10[2];
              ir10[2] = (v2561_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v1997_data, 11))));
              float v2567_data = ir10[3];
              ir10[3] = (v2567_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v2003_data, 11))));
              float v2573_data = ir10[4];
              ir10[4] = (v2573_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v2009_data, 11))));
              float v2579_data = ir10[5];
              ir10[5] = (v2579_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v2015_data, 11))));
              float v2585_data = ir10[6];
              ir10[6] = (v2585_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v2021_data, 11))));
              float v2591_data = ir10[7];
              ir10[7] = (v2591_data + (v2545_data * (sycl::group_broadcast(item.get_sub_group(), v2027_data, 11))));
              #pragma unroll
              for (int32_t v2596_n0 = 0; v2596_n0 < 1; ++v2596_n0) {
                #pragma unroll
                for (int32_t v2597_n1 = 0; v2597_n1 < 8; ++v2597_n1) {
                  int32_t v2598_a = v2596_n0 + v2597_n1;
                  float v2599_data = ir10[v2598_a];
                  float v2601_data = r9[v2598_a];
                  r10[v2598_a] = (v2601_data + v2599_data);
                }
              }
              // glb_m0 = store{r>g}(r10);
              #pragma unroll
              for (int32_t v2607_i0 = 0; v2607_i0 < 1; ++v2607_i0) {
                int32_t v2615_lead = v10_lead + (v2607_i0 * 32);
                #pragma unroll
                for (int32_t v2608_i1 = 0; v2608_i1 < 8; ++v2608_i1) {
                  float v2610_data = r10[(v2607_i0 + v2608_i1)];
                  glb_m0[(v2615_lead + ((v2608_i1 + 8) * 32))] = v2610_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

