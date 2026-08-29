// === base name ===
kernel_e7f2438624

// === header ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e7f2438624(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e7f2438624(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 12×20(12×20) {0..12}×{0..20} strided
        // m2 16×20(16×20) {0..16}×{0..20} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float r0[20]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 20; ++v10_i1) {
                  float v18_data = glb_m1[(v8_lead + (v10_i1 * 12))];
                  r0[v10_i1] = v18_data;
                }
              }
              float r1[20]{};
              // r1 = load{g>r}(glb_m2);
              float v21_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v21_lin;
              float v22_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v22_lin;
              float v23_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v23_lin;
              float v24_lin = glb_m2[48 + item.get_local_id(0) * 1];
              r1[3] = v24_lin;
              float v25_lin = glb_m2[64 + item.get_local_id(0) * 1];
              r1[4] = v25_lin;
              float v26_lin = glb_m2[80 + item.get_local_id(0) * 1];
              r1[5] = v26_lin;
              float v27_lin = glb_m2[96 + item.get_local_id(0) * 1];
              r1[6] = v27_lin;
              float v28_lin = glb_m2[112 + item.get_local_id(0) * 1];
              r1[7] = v28_lin;
              float v29_lin = glb_m2[128 + item.get_local_id(0) * 1];
              r1[8] = v29_lin;
              float v30_lin = glb_m2[144 + item.get_local_id(0) * 1];
              r1[9] = v30_lin;
              float v31_lin = glb_m2[160 + item.get_local_id(0) * 1];
              r1[10] = v31_lin;
              float v32_lin = glb_m2[176 + item.get_local_id(0) * 1];
              r1[11] = v32_lin;
              float v33_lin = glb_m2[192 + item.get_local_id(0) * 1];
              r1[12] = v33_lin;
              float v34_lin = glb_m2[208 + item.get_local_id(0) * 1];
              r1[13] = v34_lin;
              float v35_lin = glb_m2[224 + item.get_local_id(0) * 1];
              r1[14] = v35_lin;
              float v36_lin = glb_m2[240 + item.get_local_id(0) * 1];
              r1[15] = v36_lin;
              float v37_lin = glb_m2[256 + item.get_local_id(0) * 1];
              r1[16] = v37_lin;
              float v38_lin = glb_m2[272 + item.get_local_id(0) * 1];
              r1[17] = v38_lin;
              float v39_lin = glb_m2[288 + item.get_local_id(0) * 1];
              r1[18] = v39_lin;
              float v40_lin = glb_m2[304 + item.get_local_id(0) * 1];
              r1[19] = v40_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir2[16]{};
              if (v8_lead < 12) {
                float v47_data = r0[0];
                float v48_data = r1[0];
                float v51_data = ir2[0];
                ir2[0] = (v51_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 0))));
                float v57_data = ir2[1];
                ir2[1] = (v57_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 1))));
                float v63_data = ir2[2];
                ir2[2] = (v63_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 2))));
                float v69_data = ir2[3];
                ir2[3] = (v69_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 3))));
                float v75_data = ir2[4];
                ir2[4] = (v75_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 4))));
                float v81_data = ir2[5];
                ir2[5] = (v81_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 5))));
                float v87_data = ir2[6];
                ir2[6] = (v87_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 6))));
                float v93_data = ir2[7];
                ir2[7] = (v93_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 7))));
                float v99_data = ir2[8];
                ir2[8] = (v99_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 8))));
                float v105_data = ir2[9];
                ir2[9] = (v105_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 9))));
                float v111_data = ir2[10];
                ir2[10] = (v111_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 10))));
                float v117_data = ir2[11];
                ir2[11] = (v117_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 11))));
                float v123_data = ir2[12];
                ir2[12] = (v123_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 12))));
                float v129_data = ir2[13];
                ir2[13] = (v129_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 13))));
                float v135_data = ir2[14];
                ir2[14] = (v135_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 14))));
                float v141_data = ir2[15];
                ir2[15] = (v141_data + (v47_data * (sycl::group_broadcast(item.get_sub_group(), v48_data, 15))));
              }
              if (v8_lead < 12) {
                float v147_data = r0[1];
                float v148_data = r1[1];
                float v151_data = ir2[0];
                ir2[0] = (v151_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 0))));
                float v157_data = ir2[1];
                ir2[1] = (v157_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 1))));
                float v163_data = ir2[2];
                ir2[2] = (v163_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 2))));
                float v169_data = ir2[3];
                ir2[3] = (v169_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 3))));
                float v175_data = ir2[4];
                ir2[4] = (v175_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 4))));
                float v181_data = ir2[5];
                ir2[5] = (v181_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 5))));
                float v187_data = ir2[6];
                ir2[6] = (v187_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 6))));
                float v193_data = ir2[7];
                ir2[7] = (v193_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 7))));
                float v199_data = ir2[8];
                ir2[8] = (v199_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 8))));
                float v205_data = ir2[9];
                ir2[9] = (v205_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 9))));
                float v211_data = ir2[10];
                ir2[10] = (v211_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 10))));
                float v217_data = ir2[11];
                ir2[11] = (v217_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 11))));
                float v223_data = ir2[12];
                ir2[12] = (v223_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 12))));
                float v229_data = ir2[13];
                ir2[13] = (v229_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 13))));
                float v235_data = ir2[14];
                ir2[14] = (v235_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 14))));
                float v241_data = ir2[15];
                ir2[15] = (v241_data + (v147_data * (sycl::group_broadcast(item.get_sub_group(), v148_data, 15))));
              }
              if (v8_lead < 12) {
                float v247_data = r0[2];
                float v248_data = r1[2];
                float v251_data = ir2[0];
                ir2[0] = (v251_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 0))));
                float v257_data = ir2[1];
                ir2[1] = (v257_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 1))));
                float v263_data = ir2[2];
                ir2[2] = (v263_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 2))));
                float v269_data = ir2[3];
                ir2[3] = (v269_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 3))));
                float v275_data = ir2[4];
                ir2[4] = (v275_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 4))));
                float v281_data = ir2[5];
                ir2[5] = (v281_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 5))));
                float v287_data = ir2[6];
                ir2[6] = (v287_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 6))));
                float v293_data = ir2[7];
                ir2[7] = (v293_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 7))));
                float v299_data = ir2[8];
                ir2[8] = (v299_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 8))));
                float v305_data = ir2[9];
                ir2[9] = (v305_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 9))));
                float v311_data = ir2[10];
                ir2[10] = (v311_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 10))));
                float v317_data = ir2[11];
                ir2[11] = (v317_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 11))));
                float v323_data = ir2[12];
                ir2[12] = (v323_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 12))));
                float v329_data = ir2[13];
                ir2[13] = (v329_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 13))));
                float v335_data = ir2[14];
                ir2[14] = (v335_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 14))));
                float v341_data = ir2[15];
                ir2[15] = (v341_data + (v247_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 15))));
              }
              if (v8_lead < 12) {
                float v347_data = r0[3];
                float v348_data = r1[3];
                float v351_data = ir2[0];
                ir2[0] = (v351_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 0))));
                float v357_data = ir2[1];
                ir2[1] = (v357_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 1))));
                float v363_data = ir2[2];
                ir2[2] = (v363_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 2))));
                float v369_data = ir2[3];
                ir2[3] = (v369_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 3))));
                float v375_data = ir2[4];
                ir2[4] = (v375_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 4))));
                float v381_data = ir2[5];
                ir2[5] = (v381_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 5))));
                float v387_data = ir2[6];
                ir2[6] = (v387_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 6))));
                float v393_data = ir2[7];
                ir2[7] = (v393_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 7))));
                float v399_data = ir2[8];
                ir2[8] = (v399_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 8))));
                float v405_data = ir2[9];
                ir2[9] = (v405_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 9))));
                float v411_data = ir2[10];
                ir2[10] = (v411_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 10))));
                float v417_data = ir2[11];
                ir2[11] = (v417_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 11))));
                float v423_data = ir2[12];
                ir2[12] = (v423_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 12))));
                float v429_data = ir2[13];
                ir2[13] = (v429_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 13))));
                float v435_data = ir2[14];
                ir2[14] = (v435_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 14))));
                float v441_data = ir2[15];
                ir2[15] = (v441_data + (v347_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 15))));
              }
              if (v8_lead < 12) {
                float v447_data = r0[4];
                float v448_data = r1[4];
                float v451_data = ir2[0];
                ir2[0] = (v451_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 0))));
                float v457_data = ir2[1];
                ir2[1] = (v457_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 1))));
                float v463_data = ir2[2];
                ir2[2] = (v463_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 2))));
                float v469_data = ir2[3];
                ir2[3] = (v469_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 3))));
                float v475_data = ir2[4];
                ir2[4] = (v475_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 4))));
                float v481_data = ir2[5];
                ir2[5] = (v481_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 5))));
                float v487_data = ir2[6];
                ir2[6] = (v487_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 6))));
                float v493_data = ir2[7];
                ir2[7] = (v493_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 7))));
                float v499_data = ir2[8];
                ir2[8] = (v499_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 8))));
                float v505_data = ir2[9];
                ir2[9] = (v505_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 9))));
                float v511_data = ir2[10];
                ir2[10] = (v511_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 10))));
                float v517_data = ir2[11];
                ir2[11] = (v517_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 11))));
                float v523_data = ir2[12];
                ir2[12] = (v523_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 12))));
                float v529_data = ir2[13];
                ir2[13] = (v529_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 13))));
                float v535_data = ir2[14];
                ir2[14] = (v535_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 14))));
                float v541_data = ir2[15];
                ir2[15] = (v541_data + (v447_data * (sycl::group_broadcast(item.get_sub_group(), v448_data, 15))));
              }
              if (v8_lead < 12) {
                float v547_data = r0[5];
                float v548_data = r1[5];
                float v551_data = ir2[0];
                ir2[0] = (v551_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 0))));
                float v557_data = ir2[1];
                ir2[1] = (v557_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 1))));
                float v563_data = ir2[2];
                ir2[2] = (v563_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 2))));
                float v569_data = ir2[3];
                ir2[3] = (v569_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 3))));
                float v575_data = ir2[4];
                ir2[4] = (v575_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 4))));
                float v581_data = ir2[5];
                ir2[5] = (v581_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 5))));
                float v587_data = ir2[6];
                ir2[6] = (v587_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 6))));
                float v593_data = ir2[7];
                ir2[7] = (v593_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 7))));
                float v599_data = ir2[8];
                ir2[8] = (v599_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 8))));
                float v605_data = ir2[9];
                ir2[9] = (v605_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 9))));
                float v611_data = ir2[10];
                ir2[10] = (v611_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 10))));
                float v617_data = ir2[11];
                ir2[11] = (v617_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 11))));
                float v623_data = ir2[12];
                ir2[12] = (v623_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 12))));
                float v629_data = ir2[13];
                ir2[13] = (v629_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 13))));
                float v635_data = ir2[14];
                ir2[14] = (v635_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 14))));
                float v641_data = ir2[15];
                ir2[15] = (v641_data + (v547_data * (sycl::group_broadcast(item.get_sub_group(), v548_data, 15))));
              }
              if (v8_lead < 12) {
                float v647_data = r0[6];
                float v648_data = r1[6];
                float v651_data = ir2[0];
                ir2[0] = (v651_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 0))));
                float v657_data = ir2[1];
                ir2[1] = (v657_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 1))));
                float v663_data = ir2[2];
                ir2[2] = (v663_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 2))));
                float v669_data = ir2[3];
                ir2[3] = (v669_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 3))));
                float v675_data = ir2[4];
                ir2[4] = (v675_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 4))));
                float v681_data = ir2[5];
                ir2[5] = (v681_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 5))));
                float v687_data = ir2[6];
                ir2[6] = (v687_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 6))));
                float v693_data = ir2[7];
                ir2[7] = (v693_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 7))));
                float v699_data = ir2[8];
                ir2[8] = (v699_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 8))));
                float v705_data = ir2[9];
                ir2[9] = (v705_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 9))));
                float v711_data = ir2[10];
                ir2[10] = (v711_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 10))));
                float v717_data = ir2[11];
                ir2[11] = (v717_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 11))));
                float v723_data = ir2[12];
                ir2[12] = (v723_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 12))));
                float v729_data = ir2[13];
                ir2[13] = (v729_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 13))));
                float v735_data = ir2[14];
                ir2[14] = (v735_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 14))));
                float v741_data = ir2[15];
                ir2[15] = (v741_data + (v647_data * (sycl::group_broadcast(item.get_sub_group(), v648_data, 15))));
              }
              if (v8_lead < 12) {
                float v747_data = r0[7];
                float v748_data = r1[7];
                float v751_data = ir2[0];
                ir2[0] = (v751_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 0))));
                float v757_data = ir2[1];
                ir2[1] = (v757_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 1))));
                float v763_data = ir2[2];
                ir2[2] = (v763_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 2))));
                float v769_data = ir2[3];
                ir2[3] = (v769_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 3))));
                float v775_data = ir2[4];
                ir2[4] = (v775_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 4))));
                float v781_data = ir2[5];
                ir2[5] = (v781_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 5))));
                float v787_data = ir2[6];
                ir2[6] = (v787_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 6))));
                float v793_data = ir2[7];
                ir2[7] = (v793_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 7))));
                float v799_data = ir2[8];
                ir2[8] = (v799_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 8))));
                float v805_data = ir2[9];
                ir2[9] = (v805_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 9))));
                float v811_data = ir2[10];
                ir2[10] = (v811_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 10))));
                float v817_data = ir2[11];
                ir2[11] = (v817_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 11))));
                float v823_data = ir2[12];
                ir2[12] = (v823_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 12))));
                float v829_data = ir2[13];
                ir2[13] = (v829_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 13))));
                float v835_data = ir2[14];
                ir2[14] = (v835_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 14))));
                float v841_data = ir2[15];
                ir2[15] = (v841_data + (v747_data * (sycl::group_broadcast(item.get_sub_group(), v748_data, 15))));
              }
              if (v8_lead < 12) {
                float v847_data = r0[8];
                float v848_data = r1[8];
                float v851_data = ir2[0];
                ir2[0] = (v851_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 0))));
                float v857_data = ir2[1];
                ir2[1] = (v857_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 1))));
                float v863_data = ir2[2];
                ir2[2] = (v863_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 2))));
                float v869_data = ir2[3];
                ir2[3] = (v869_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 3))));
                float v875_data = ir2[4];
                ir2[4] = (v875_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 4))));
                float v881_data = ir2[5];
                ir2[5] = (v881_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 5))));
                float v887_data = ir2[6];
                ir2[6] = (v887_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 6))));
                float v893_data = ir2[7];
                ir2[7] = (v893_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 7))));
                float v899_data = ir2[8];
                ir2[8] = (v899_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 8))));
                float v905_data = ir2[9];
                ir2[9] = (v905_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 9))));
                float v911_data = ir2[10];
                ir2[10] = (v911_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 10))));
                float v917_data = ir2[11];
                ir2[11] = (v917_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 11))));
                float v923_data = ir2[12];
                ir2[12] = (v923_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 12))));
                float v929_data = ir2[13];
                ir2[13] = (v929_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 13))));
                float v935_data = ir2[14];
                ir2[14] = (v935_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 14))));
                float v941_data = ir2[15];
                ir2[15] = (v941_data + (v847_data * (sycl::group_broadcast(item.get_sub_group(), v848_data, 15))));
              }
              if (v8_lead < 12) {
                float v947_data = r0[9];
                float v948_data = r1[9];
                float v951_data = ir2[0];
                ir2[0] = (v951_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 0))));
                float v957_data = ir2[1];
                ir2[1] = (v957_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 1))));
                float v963_data = ir2[2];
                ir2[2] = (v963_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 2))));
                float v969_data = ir2[3];
                ir2[3] = (v969_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 3))));
                float v975_data = ir2[4];
                ir2[4] = (v975_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 4))));
                float v981_data = ir2[5];
                ir2[5] = (v981_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 5))));
                float v987_data = ir2[6];
                ir2[6] = (v987_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 6))));
                float v993_data = ir2[7];
                ir2[7] = (v993_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 7))));
                float v999_data = ir2[8];
                ir2[8] = (v999_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 8))));
                float v1005_data = ir2[9];
                ir2[9] = (v1005_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 9))));
                float v1011_data = ir2[10];
                ir2[10] = (v1011_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 10))));
                float v1017_data = ir2[11];
                ir2[11] = (v1017_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 11))));
                float v1023_data = ir2[12];
                ir2[12] = (v1023_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 12))));
                float v1029_data = ir2[13];
                ir2[13] = (v1029_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 13))));
                float v1035_data = ir2[14];
                ir2[14] = (v1035_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 14))));
                float v1041_data = ir2[15];
                ir2[15] = (v1041_data + (v947_data * (sycl::group_broadcast(item.get_sub_group(), v948_data, 15))));
              }
              if (v8_lead < 12) {
                float v1047_data = r0[10];
                float v1048_data = r1[10];
                float v1051_data = ir2[0];
                ir2[0] = (v1051_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 0))));
                float v1057_data = ir2[1];
                ir2[1] = (v1057_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 1))));
                float v1063_data = ir2[2];
                ir2[2] = (v1063_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 2))));
                float v1069_data = ir2[3];
                ir2[3] = (v1069_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 3))));
                float v1075_data = ir2[4];
                ir2[4] = (v1075_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 4))));
                float v1081_data = ir2[5];
                ir2[5] = (v1081_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 5))));
                float v1087_data = ir2[6];
                ir2[6] = (v1087_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 6))));
                float v1093_data = ir2[7];
                ir2[7] = (v1093_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 7))));
                float v1099_data = ir2[8];
                ir2[8] = (v1099_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 8))));
                float v1105_data = ir2[9];
                ir2[9] = (v1105_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 9))));
                float v1111_data = ir2[10];
                ir2[10] = (v1111_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 10))));
                float v1117_data = ir2[11];
                ir2[11] = (v1117_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 11))));
                float v1123_data = ir2[12];
                ir2[12] = (v1123_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 12))));
                float v1129_data = ir2[13];
                ir2[13] = (v1129_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 13))));
                float v1135_data = ir2[14];
                ir2[14] = (v1135_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 14))));
                float v1141_data = ir2[15];
                ir2[15] = (v1141_data + (v1047_data * (sycl::group_broadcast(item.get_sub_group(), v1048_data, 15))));
              }
              if (v8_lead < 12) {
                float v1147_data = r0[11];
                float v1148_data = r1[11];
                float v1151_data = ir2[0];
                ir2[0] = (v1151_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 0))));
                float v1157_data = ir2[1];
                ir2[1] = (v1157_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 1))));
                float v1163_data = ir2[2];
                ir2[2] = (v1163_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 2))));
                float v1169_data = ir2[3];
                ir2[3] = (v1169_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 3))));
                float v1175_data = ir2[4];
                ir2[4] = (v1175_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 4))));
                float v1181_data = ir2[5];
                ir2[5] = (v1181_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 5))));
                float v1187_data = ir2[6];
                ir2[6] = (v1187_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 6))));
                float v1193_data = ir2[7];
                ir2[7] = (v1193_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 7))));
                float v1199_data = ir2[8];
                ir2[8] = (v1199_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 8))));
                float v1205_data = ir2[9];
                ir2[9] = (v1205_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 9))));
                float v1211_data = ir2[10];
                ir2[10] = (v1211_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 10))));
                float v1217_data = ir2[11];
                ir2[11] = (v1217_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 11))));
                float v1223_data = ir2[12];
                ir2[12] = (v1223_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 12))));
                float v1229_data = ir2[13];
                ir2[13] = (v1229_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 13))));
                float v1235_data = ir2[14];
                ir2[14] = (v1235_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 14))));
                float v1241_data = ir2[15];
                ir2[15] = (v1241_data + (v1147_data * (sycl::group_broadcast(item.get_sub_group(), v1148_data, 15))));
              }
              if (v8_lead < 12) {
                float v1247_data = r0[12];
                float v1248_data = r1[12];
                float v1251_data = ir2[0];
                ir2[0] = (v1251_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 0))));
                float v1257_data = ir2[1];
                ir2[1] = (v1257_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 1))));
                float v1263_data = ir2[2];
                ir2[2] = (v1263_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 2))));
                float v1269_data = ir2[3];
                ir2[3] = (v1269_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 3))));
                float v1275_data = ir2[4];
                ir2[4] = (v1275_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 4))));
                float v1281_data = ir2[5];
                ir2[5] = (v1281_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 5))));
                float v1287_data = ir2[6];
                ir2[6] = (v1287_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 6))));
                float v1293_data = ir2[7];
                ir2[7] = (v1293_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 7))));
                float v1299_data = ir2[8];
                ir2[8] = (v1299_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 8))));
                float v1305_data = ir2[9];
                ir2[9] = (v1305_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 9))));
                float v1311_data = ir2[10];
                ir2[10] = (v1311_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 10))));
                float v1317_data = ir2[11];
                ir2[11] = (v1317_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 11))));
                float v1323_data = ir2[12];
                ir2[12] = (v1323_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 12))));
                float v1329_data = ir2[13];
                ir2[13] = (v1329_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 13))));
                float v1335_data = ir2[14];
                ir2[14] = (v1335_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 14))));
                float v1341_data = ir2[15];
                ir2[15] = (v1341_data + (v1247_data * (sycl::group_broadcast(item.get_sub_group(), v1248_data, 15))));
              }
              if (v8_lead < 12) {
                float v1347_data = r0[13];
                float v1348_data = r1[13];
                float v1351_data = ir2[0];
                ir2[0] = (v1351_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 0))));
                float v1357_data = ir2[1];
                ir2[1] = (v1357_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 1))));
                float v1363_data = ir2[2];
                ir2[2] = (v1363_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 2))));
                float v1369_data = ir2[3];
                ir2[3] = (v1369_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 3))));
                float v1375_data = ir2[4];
                ir2[4] = (v1375_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 4))));
                float v1381_data = ir2[5];
                ir2[5] = (v1381_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 5))));
                float v1387_data = ir2[6];
                ir2[6] = (v1387_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 6))));
                float v1393_data = ir2[7];
                ir2[7] = (v1393_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 7))));
                float v1399_data = ir2[8];
                ir2[8] = (v1399_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 8))));
                float v1405_data = ir2[9];
                ir2[9] = (v1405_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 9))));
                float v1411_data = ir2[10];
                ir2[10] = (v1411_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 10))));
                float v1417_data = ir2[11];
                ir2[11] = (v1417_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 11))));
                float v1423_data = ir2[12];
                ir2[12] = (v1423_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 12))));
                float v1429_data = ir2[13];
                ir2[13] = (v1429_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 13))));
                float v1435_data = ir2[14];
                ir2[14] = (v1435_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 14))));
                float v1441_data = ir2[15];
                ir2[15] = (v1441_data + (v1347_data * (sycl::group_broadcast(item.get_sub_group(), v1348_data, 15))));
              }
              if (v8_lead < 12) {
                float v1447_data = r0[14];
                float v1448_data = r1[14];
                float v1451_data = ir2[0];
                ir2[0] = (v1451_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 0))));
                float v1457_data = ir2[1];
                ir2[1] = (v1457_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 1))));
                float v1463_data = ir2[2];
                ir2[2] = (v1463_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 2))));
                float v1469_data = ir2[3];
                ir2[3] = (v1469_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 3))));
                float v1475_data = ir2[4];
                ir2[4] = (v1475_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 4))));
                float v1481_data = ir2[5];
                ir2[5] = (v1481_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 5))));
                float v1487_data = ir2[6];
                ir2[6] = (v1487_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 6))));
                float v1493_data = ir2[7];
                ir2[7] = (v1493_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 7))));
                float v1499_data = ir2[8];
                ir2[8] = (v1499_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 8))));
                float v1505_data = ir2[9];
                ir2[9] = (v1505_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 9))));
                float v1511_data = ir2[10];
                ir2[10] = (v1511_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 10))));
                float v1517_data = ir2[11];
                ir2[11] = (v1517_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 11))));
                float v1523_data = ir2[12];
                ir2[12] = (v1523_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 12))));
                float v1529_data = ir2[13];
                ir2[13] = (v1529_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 13))));
                float v1535_data = ir2[14];
                ir2[14] = (v1535_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 14))));
                float v1541_data = ir2[15];
                ir2[15] = (v1541_data + (v1447_data * (sycl::group_broadcast(item.get_sub_group(), v1448_data, 15))));
              }
              if (v8_lead < 12) {
                float v1547_data = r0[15];
                float v1548_data = r1[15];
                float v1551_data = ir2[0];
                ir2[0] = (v1551_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 0))));
                float v1557_data = ir2[1];
                ir2[1] = (v1557_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 1))));
                float v1563_data = ir2[2];
                ir2[2] = (v1563_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 2))));
                float v1569_data = ir2[3];
                ir2[3] = (v1569_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 3))));
                float v1575_data = ir2[4];
                ir2[4] = (v1575_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 4))));
                float v1581_data = ir2[5];
                ir2[5] = (v1581_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 5))));
                float v1587_data = ir2[6];
                ir2[6] = (v1587_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 6))));
                float v1593_data = ir2[7];
                ir2[7] = (v1593_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 7))));
                float v1599_data = ir2[8];
                ir2[8] = (v1599_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 8))));
                float v1605_data = ir2[9];
                ir2[9] = (v1605_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 9))));
                float v1611_data = ir2[10];
                ir2[10] = (v1611_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 10))));
                float v1617_data = ir2[11];
                ir2[11] = (v1617_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 11))));
                float v1623_data = ir2[12];
                ir2[12] = (v1623_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 12))));
                float v1629_data = ir2[13];
                ir2[13] = (v1629_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 13))));
                float v1635_data = ir2[14];
                ir2[14] = (v1635_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 14))));
                float v1641_data = ir2[15];
                ir2[15] = (v1641_data + (v1547_data * (sycl::group_broadcast(item.get_sub_group(), v1548_data, 15))));
              }
              if (v8_lead < 12) {
                float v1647_data = r0[16];
                float v1648_data = r1[16];
                float v1651_data = ir2[0];
                ir2[0] = (v1651_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 0))));
                float v1657_data = ir2[1];
                ir2[1] = (v1657_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 1))));
                float v1663_data = ir2[2];
                ir2[2] = (v1663_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 2))));
                float v1669_data = ir2[3];
                ir2[3] = (v1669_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 3))));
                float v1675_data = ir2[4];
                ir2[4] = (v1675_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 4))));
                float v1681_data = ir2[5];
                ir2[5] = (v1681_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 5))));
                float v1687_data = ir2[6];
                ir2[6] = (v1687_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 6))));
                float v1693_data = ir2[7];
                ir2[7] = (v1693_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 7))));
                float v1699_data = ir2[8];
                ir2[8] = (v1699_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 8))));
                float v1705_data = ir2[9];
                ir2[9] = (v1705_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 9))));
                float v1711_data = ir2[10];
                ir2[10] = (v1711_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 10))));
                float v1717_data = ir2[11];
                ir2[11] = (v1717_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 11))));
                float v1723_data = ir2[12];
                ir2[12] = (v1723_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 12))));
                float v1729_data = ir2[13];
                ir2[13] = (v1729_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 13))));
                float v1735_data = ir2[14];
                ir2[14] = (v1735_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 14))));
                float v1741_data = ir2[15];
                ir2[15] = (v1741_data + (v1647_data * (sycl::group_broadcast(item.get_sub_group(), v1648_data, 15))));
              }
              if (v8_lead < 12) {
                float v1747_data = r0[17];
                float v1748_data = r1[17];
                float v1751_data = ir2[0];
                ir2[0] = (v1751_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 0))));
                float v1757_data = ir2[1];
                ir2[1] = (v1757_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 1))));
                float v1763_data = ir2[2];
                ir2[2] = (v1763_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 2))));
                float v1769_data = ir2[3];
                ir2[3] = (v1769_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 3))));
                float v1775_data = ir2[4];
                ir2[4] = (v1775_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 4))));
                float v1781_data = ir2[5];
                ir2[5] = (v1781_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 5))));
                float v1787_data = ir2[6];
                ir2[6] = (v1787_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 6))));
                float v1793_data = ir2[7];
                ir2[7] = (v1793_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 7))));
                float v1799_data = ir2[8];
                ir2[8] = (v1799_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 8))));
                float v1805_data = ir2[9];
                ir2[9] = (v1805_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 9))));
                float v1811_data = ir2[10];
                ir2[10] = (v1811_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 10))));
                float v1817_data = ir2[11];
                ir2[11] = (v1817_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 11))));
                float v1823_data = ir2[12];
                ir2[12] = (v1823_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 12))));
                float v1829_data = ir2[13];
                ir2[13] = (v1829_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 13))));
                float v1835_data = ir2[14];
                ir2[14] = (v1835_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 14))));
                float v1841_data = ir2[15];
                ir2[15] = (v1841_data + (v1747_data * (sycl::group_broadcast(item.get_sub_group(), v1748_data, 15))));
              }
              if (v8_lead < 12) {
                float v1847_data = r0[18];
                float v1848_data = r1[18];
                float v1851_data = ir2[0];
                ir2[0] = (v1851_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 0))));
                float v1857_data = ir2[1];
                ir2[1] = (v1857_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 1))));
                float v1863_data = ir2[2];
                ir2[2] = (v1863_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 2))));
                float v1869_data = ir2[3];
                ir2[3] = (v1869_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 3))));
                float v1875_data = ir2[4];
                ir2[4] = (v1875_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 4))));
                float v1881_data = ir2[5];
                ir2[5] = (v1881_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 5))));
                float v1887_data = ir2[6];
                ir2[6] = (v1887_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 6))));
                float v1893_data = ir2[7];
                ir2[7] = (v1893_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 7))));
                float v1899_data = ir2[8];
                ir2[8] = (v1899_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 8))));
                float v1905_data = ir2[9];
                ir2[9] = (v1905_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 9))));
                float v1911_data = ir2[10];
                ir2[10] = (v1911_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 10))));
                float v1917_data = ir2[11];
                ir2[11] = (v1917_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 11))));
                float v1923_data = ir2[12];
                ir2[12] = (v1923_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 12))));
                float v1929_data = ir2[13];
                ir2[13] = (v1929_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 13))));
                float v1935_data = ir2[14];
                ir2[14] = (v1935_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 14))));
                float v1941_data = ir2[15];
                ir2[15] = (v1941_data + (v1847_data * (sycl::group_broadcast(item.get_sub_group(), v1848_data, 15))));
              }
              if (v8_lead < 12) {
                float v1947_data = r0[19];
                float v1948_data = r1[19];
                float v1951_data = ir2[0];
                ir2[0] = (v1951_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 0))));
                float v1957_data = ir2[1];
                ir2[1] = (v1957_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 1))));
                float v1963_data = ir2[2];
                ir2[2] = (v1963_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 2))));
                float v1969_data = ir2[3];
                ir2[3] = (v1969_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 3))));
                float v1975_data = ir2[4];
                ir2[4] = (v1975_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 4))));
                float v1981_data = ir2[5];
                ir2[5] = (v1981_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 5))));
                float v1987_data = ir2[6];
                ir2[6] = (v1987_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 6))));
                float v1993_data = ir2[7];
                ir2[7] = (v1993_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 7))));
                float v1999_data = ir2[8];
                ir2[8] = (v1999_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 8))));
                float v2005_data = ir2[9];
                ir2[9] = (v2005_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 9))));
                float v2011_data = ir2[10];
                ir2[10] = (v2011_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 10))));
                float v2017_data = ir2[11];
                ir2[11] = (v2017_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 11))));
                float v2023_data = ir2[12];
                ir2[12] = (v2023_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 12))));
                float v2029_data = ir2[13];
                ir2[13] = (v2029_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 13))));
                float v2035_data = ir2[14];
                ir2[14] = (v2035_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 14))));
                float v2041_data = ir2[15];
                ir2[15] = (v2041_data + (v1947_data * (sycl::group_broadcast(item.get_sub_group(), v1948_data, 15))));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2047_n1 = 0; v2047_n1 < 16; ++v2047_n1) {
                  float v2049_data = ir2[v2047_n1];
                  r2[v2047_n1] = v2049_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v2055_i1 = 0; v2055_i1 < 16; ++v2055_i1) {
                  float v2057_data = r2[v2055_i1];
                  glb_m0[(v8_lead + (v2055_i1 * 12))] = v2057_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

