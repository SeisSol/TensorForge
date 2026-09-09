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
        // options: default
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
              bool v9_g = v8_lead < 12;
              if (v9_g) {
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 20; ++v10_i1) {
                  float v18_data = glb_m1[(v8_lead + (v10_i1 * 12))];
                  r0[v10_i1] = v18_data;
                }
              }
              float r1[20]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v30_lead = v8_lead + (v24_i0 * 16);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 20; ++v25_i1) {
                  float v33_data = glb_m2[(v30_lead + (v25_i1 * 16))];
                  r1[(v24_i0 + v25_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir2[16]{};
              if (v9_g) {
                float v41_data = r0[0];
                float v42_data = r1[0];
                float v45_data = ir2[0];
                ir2[0] = (v45_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 0))));
                float v51_data = ir2[1];
                ir2[1] = (v51_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 1))));
                float v57_data = ir2[2];
                ir2[2] = (v57_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 2))));
                float v63_data = ir2[3];
                ir2[3] = (v63_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 3))));
                float v69_data = ir2[4];
                ir2[4] = (v69_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 4))));
                float v75_data = ir2[5];
                ir2[5] = (v75_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 5))));
                float v81_data = ir2[6];
                ir2[6] = (v81_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 6))));
                float v87_data = ir2[7];
                ir2[7] = (v87_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 7))));
                float v93_data = ir2[8];
                ir2[8] = (v93_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 8))));
                float v99_data = ir2[9];
                ir2[9] = (v99_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 9))));
                float v105_data = ir2[10];
                ir2[10] = (v105_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 10))));
                float v111_data = ir2[11];
                ir2[11] = (v111_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 11))));
                float v117_data = ir2[12];
                ir2[12] = (v117_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 12))));
                float v123_data = ir2[13];
                ir2[13] = (v123_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 13))));
                float v129_data = ir2[14];
                ir2[14] = (v129_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 14))));
                float v135_data = ir2[15];
                ir2[15] = (v135_data + (v41_data * (sycl::group_broadcast(item.get_sub_group(), v42_data, 15))));
              }
              if (v9_g) {
                float v141_data = r0[1];
                float v142_data = r1[1];
                float v145_data = ir2[0];
                ir2[0] = (v145_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 0))));
                float v151_data = ir2[1];
                ir2[1] = (v151_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 1))));
                float v157_data = ir2[2];
                ir2[2] = (v157_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 2))));
                float v163_data = ir2[3];
                ir2[3] = (v163_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 3))));
                float v169_data = ir2[4];
                ir2[4] = (v169_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 4))));
                float v175_data = ir2[5];
                ir2[5] = (v175_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 5))));
                float v181_data = ir2[6];
                ir2[6] = (v181_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 6))));
                float v187_data = ir2[7];
                ir2[7] = (v187_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 7))));
                float v193_data = ir2[8];
                ir2[8] = (v193_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 8))));
                float v199_data = ir2[9];
                ir2[9] = (v199_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 9))));
                float v205_data = ir2[10];
                ir2[10] = (v205_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 10))));
                float v211_data = ir2[11];
                ir2[11] = (v211_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 11))));
                float v217_data = ir2[12];
                ir2[12] = (v217_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 12))));
                float v223_data = ir2[13];
                ir2[13] = (v223_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 13))));
                float v229_data = ir2[14];
                ir2[14] = (v229_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 14))));
                float v235_data = ir2[15];
                ir2[15] = (v235_data + (v141_data * (sycl::group_broadcast(item.get_sub_group(), v142_data, 15))));
              }
              if (v9_g) {
                float v241_data = r0[2];
                float v242_data = r1[2];
                float v245_data = ir2[0];
                ir2[0] = (v245_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 0))));
                float v251_data = ir2[1];
                ir2[1] = (v251_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 1))));
                float v257_data = ir2[2];
                ir2[2] = (v257_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 2))));
                float v263_data = ir2[3];
                ir2[3] = (v263_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 3))));
                float v269_data = ir2[4];
                ir2[4] = (v269_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 4))));
                float v275_data = ir2[5];
                ir2[5] = (v275_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 5))));
                float v281_data = ir2[6];
                ir2[6] = (v281_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 6))));
                float v287_data = ir2[7];
                ir2[7] = (v287_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 7))));
                float v293_data = ir2[8];
                ir2[8] = (v293_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 8))));
                float v299_data = ir2[9];
                ir2[9] = (v299_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 9))));
                float v305_data = ir2[10];
                ir2[10] = (v305_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 10))));
                float v311_data = ir2[11];
                ir2[11] = (v311_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 11))));
                float v317_data = ir2[12];
                ir2[12] = (v317_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 12))));
                float v323_data = ir2[13];
                ir2[13] = (v323_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 13))));
                float v329_data = ir2[14];
                ir2[14] = (v329_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 14))));
                float v335_data = ir2[15];
                ir2[15] = (v335_data + (v241_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 15))));
              }
              if (v9_g) {
                float v341_data = r0[3];
                float v342_data = r1[3];
                float v345_data = ir2[0];
                ir2[0] = (v345_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 0))));
                float v351_data = ir2[1];
                ir2[1] = (v351_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 1))));
                float v357_data = ir2[2];
                ir2[2] = (v357_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 2))));
                float v363_data = ir2[3];
                ir2[3] = (v363_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 3))));
                float v369_data = ir2[4];
                ir2[4] = (v369_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 4))));
                float v375_data = ir2[5];
                ir2[5] = (v375_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 5))));
                float v381_data = ir2[6];
                ir2[6] = (v381_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 6))));
                float v387_data = ir2[7];
                ir2[7] = (v387_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 7))));
                float v393_data = ir2[8];
                ir2[8] = (v393_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 8))));
                float v399_data = ir2[9];
                ir2[9] = (v399_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 9))));
                float v405_data = ir2[10];
                ir2[10] = (v405_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 10))));
                float v411_data = ir2[11];
                ir2[11] = (v411_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 11))));
                float v417_data = ir2[12];
                ir2[12] = (v417_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 12))));
                float v423_data = ir2[13];
                ir2[13] = (v423_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 13))));
                float v429_data = ir2[14];
                ir2[14] = (v429_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 14))));
                float v435_data = ir2[15];
                ir2[15] = (v435_data + (v341_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 15))));
              }
              if (v9_g) {
                float v441_data = r0[4];
                float v442_data = r1[4];
                float v445_data = ir2[0];
                ir2[0] = (v445_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 0))));
                float v451_data = ir2[1];
                ir2[1] = (v451_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 1))));
                float v457_data = ir2[2];
                ir2[2] = (v457_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 2))));
                float v463_data = ir2[3];
                ir2[3] = (v463_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 3))));
                float v469_data = ir2[4];
                ir2[4] = (v469_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 4))));
                float v475_data = ir2[5];
                ir2[5] = (v475_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 5))));
                float v481_data = ir2[6];
                ir2[6] = (v481_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 6))));
                float v487_data = ir2[7];
                ir2[7] = (v487_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 7))));
                float v493_data = ir2[8];
                ir2[8] = (v493_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 8))));
                float v499_data = ir2[9];
                ir2[9] = (v499_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 9))));
                float v505_data = ir2[10];
                ir2[10] = (v505_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 10))));
                float v511_data = ir2[11];
                ir2[11] = (v511_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 11))));
                float v517_data = ir2[12];
                ir2[12] = (v517_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 12))));
                float v523_data = ir2[13];
                ir2[13] = (v523_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 13))));
                float v529_data = ir2[14];
                ir2[14] = (v529_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 14))));
                float v535_data = ir2[15];
                ir2[15] = (v535_data + (v441_data * (sycl::group_broadcast(item.get_sub_group(), v442_data, 15))));
              }
              if (v9_g) {
                float v541_data = r0[5];
                float v542_data = r1[5];
                float v545_data = ir2[0];
                ir2[0] = (v545_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 0))));
                float v551_data = ir2[1];
                ir2[1] = (v551_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 1))));
                float v557_data = ir2[2];
                ir2[2] = (v557_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 2))));
                float v563_data = ir2[3];
                ir2[3] = (v563_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 3))));
                float v569_data = ir2[4];
                ir2[4] = (v569_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 4))));
                float v575_data = ir2[5];
                ir2[5] = (v575_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 5))));
                float v581_data = ir2[6];
                ir2[6] = (v581_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 6))));
                float v587_data = ir2[7];
                ir2[7] = (v587_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 7))));
                float v593_data = ir2[8];
                ir2[8] = (v593_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 8))));
                float v599_data = ir2[9];
                ir2[9] = (v599_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 9))));
                float v605_data = ir2[10];
                ir2[10] = (v605_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 10))));
                float v611_data = ir2[11];
                ir2[11] = (v611_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 11))));
                float v617_data = ir2[12];
                ir2[12] = (v617_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 12))));
                float v623_data = ir2[13];
                ir2[13] = (v623_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 13))));
                float v629_data = ir2[14];
                ir2[14] = (v629_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 14))));
                float v635_data = ir2[15];
                ir2[15] = (v635_data + (v541_data * (sycl::group_broadcast(item.get_sub_group(), v542_data, 15))));
              }
              if (v9_g) {
                float v641_data = r0[6];
                float v642_data = r1[6];
                float v645_data = ir2[0];
                ir2[0] = (v645_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 0))));
                float v651_data = ir2[1];
                ir2[1] = (v651_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 1))));
                float v657_data = ir2[2];
                ir2[2] = (v657_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 2))));
                float v663_data = ir2[3];
                ir2[3] = (v663_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 3))));
                float v669_data = ir2[4];
                ir2[4] = (v669_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 4))));
                float v675_data = ir2[5];
                ir2[5] = (v675_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 5))));
                float v681_data = ir2[6];
                ir2[6] = (v681_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 6))));
                float v687_data = ir2[7];
                ir2[7] = (v687_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 7))));
                float v693_data = ir2[8];
                ir2[8] = (v693_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 8))));
                float v699_data = ir2[9];
                ir2[9] = (v699_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 9))));
                float v705_data = ir2[10];
                ir2[10] = (v705_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 10))));
                float v711_data = ir2[11];
                ir2[11] = (v711_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 11))));
                float v717_data = ir2[12];
                ir2[12] = (v717_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 12))));
                float v723_data = ir2[13];
                ir2[13] = (v723_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 13))));
                float v729_data = ir2[14];
                ir2[14] = (v729_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 14))));
                float v735_data = ir2[15];
                ir2[15] = (v735_data + (v641_data * (sycl::group_broadcast(item.get_sub_group(), v642_data, 15))));
              }
              if (v9_g) {
                float v741_data = r0[7];
                float v742_data = r1[7];
                float v745_data = ir2[0];
                ir2[0] = (v745_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 0))));
                float v751_data = ir2[1];
                ir2[1] = (v751_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 1))));
                float v757_data = ir2[2];
                ir2[2] = (v757_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 2))));
                float v763_data = ir2[3];
                ir2[3] = (v763_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 3))));
                float v769_data = ir2[4];
                ir2[4] = (v769_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 4))));
                float v775_data = ir2[5];
                ir2[5] = (v775_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 5))));
                float v781_data = ir2[6];
                ir2[6] = (v781_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 6))));
                float v787_data = ir2[7];
                ir2[7] = (v787_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 7))));
                float v793_data = ir2[8];
                ir2[8] = (v793_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 8))));
                float v799_data = ir2[9];
                ir2[9] = (v799_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 9))));
                float v805_data = ir2[10];
                ir2[10] = (v805_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 10))));
                float v811_data = ir2[11];
                ir2[11] = (v811_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 11))));
                float v817_data = ir2[12];
                ir2[12] = (v817_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 12))));
                float v823_data = ir2[13];
                ir2[13] = (v823_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 13))));
                float v829_data = ir2[14];
                ir2[14] = (v829_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 14))));
                float v835_data = ir2[15];
                ir2[15] = (v835_data + (v741_data * (sycl::group_broadcast(item.get_sub_group(), v742_data, 15))));
              }
              if (v9_g) {
                float v841_data = r0[8];
                float v842_data = r1[8];
                float v845_data = ir2[0];
                ir2[0] = (v845_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 0))));
                float v851_data = ir2[1];
                ir2[1] = (v851_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 1))));
                float v857_data = ir2[2];
                ir2[2] = (v857_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 2))));
                float v863_data = ir2[3];
                ir2[3] = (v863_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 3))));
                float v869_data = ir2[4];
                ir2[4] = (v869_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 4))));
                float v875_data = ir2[5];
                ir2[5] = (v875_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 5))));
                float v881_data = ir2[6];
                ir2[6] = (v881_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 6))));
                float v887_data = ir2[7];
                ir2[7] = (v887_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 7))));
                float v893_data = ir2[8];
                ir2[8] = (v893_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 8))));
                float v899_data = ir2[9];
                ir2[9] = (v899_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 9))));
                float v905_data = ir2[10];
                ir2[10] = (v905_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 10))));
                float v911_data = ir2[11];
                ir2[11] = (v911_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 11))));
                float v917_data = ir2[12];
                ir2[12] = (v917_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 12))));
                float v923_data = ir2[13];
                ir2[13] = (v923_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 13))));
                float v929_data = ir2[14];
                ir2[14] = (v929_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 14))));
                float v935_data = ir2[15];
                ir2[15] = (v935_data + (v841_data * (sycl::group_broadcast(item.get_sub_group(), v842_data, 15))));
              }
              if (v9_g) {
                float v941_data = r0[9];
                float v942_data = r1[9];
                float v945_data = ir2[0];
                ir2[0] = (v945_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 0))));
                float v951_data = ir2[1];
                ir2[1] = (v951_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 1))));
                float v957_data = ir2[2];
                ir2[2] = (v957_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 2))));
                float v963_data = ir2[3];
                ir2[3] = (v963_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 3))));
                float v969_data = ir2[4];
                ir2[4] = (v969_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 4))));
                float v975_data = ir2[5];
                ir2[5] = (v975_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 5))));
                float v981_data = ir2[6];
                ir2[6] = (v981_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 6))));
                float v987_data = ir2[7];
                ir2[7] = (v987_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 7))));
                float v993_data = ir2[8];
                ir2[8] = (v993_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 8))));
                float v999_data = ir2[9];
                ir2[9] = (v999_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 9))));
                float v1005_data = ir2[10];
                ir2[10] = (v1005_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 10))));
                float v1011_data = ir2[11];
                ir2[11] = (v1011_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 11))));
                float v1017_data = ir2[12];
                ir2[12] = (v1017_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 12))));
                float v1023_data = ir2[13];
                ir2[13] = (v1023_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 13))));
                float v1029_data = ir2[14];
                ir2[14] = (v1029_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 14))));
                float v1035_data = ir2[15];
                ir2[15] = (v1035_data + (v941_data * (sycl::group_broadcast(item.get_sub_group(), v942_data, 15))));
              }
              if (v9_g) {
                float v1041_data = r0[10];
                float v1042_data = r1[10];
                float v1045_data = ir2[0];
                ir2[0] = (v1045_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 0))));
                float v1051_data = ir2[1];
                ir2[1] = (v1051_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 1))));
                float v1057_data = ir2[2];
                ir2[2] = (v1057_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 2))));
                float v1063_data = ir2[3];
                ir2[3] = (v1063_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 3))));
                float v1069_data = ir2[4];
                ir2[4] = (v1069_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 4))));
                float v1075_data = ir2[5];
                ir2[5] = (v1075_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 5))));
                float v1081_data = ir2[6];
                ir2[6] = (v1081_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 6))));
                float v1087_data = ir2[7];
                ir2[7] = (v1087_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 7))));
                float v1093_data = ir2[8];
                ir2[8] = (v1093_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 8))));
                float v1099_data = ir2[9];
                ir2[9] = (v1099_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 9))));
                float v1105_data = ir2[10];
                ir2[10] = (v1105_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 10))));
                float v1111_data = ir2[11];
                ir2[11] = (v1111_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 11))));
                float v1117_data = ir2[12];
                ir2[12] = (v1117_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 12))));
                float v1123_data = ir2[13];
                ir2[13] = (v1123_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 13))));
                float v1129_data = ir2[14];
                ir2[14] = (v1129_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 14))));
                float v1135_data = ir2[15];
                ir2[15] = (v1135_data + (v1041_data * (sycl::group_broadcast(item.get_sub_group(), v1042_data, 15))));
              }
              if (v9_g) {
                float v1141_data = r0[11];
                float v1142_data = r1[11];
                float v1145_data = ir2[0];
                ir2[0] = (v1145_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 0))));
                float v1151_data = ir2[1];
                ir2[1] = (v1151_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 1))));
                float v1157_data = ir2[2];
                ir2[2] = (v1157_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 2))));
                float v1163_data = ir2[3];
                ir2[3] = (v1163_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 3))));
                float v1169_data = ir2[4];
                ir2[4] = (v1169_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 4))));
                float v1175_data = ir2[5];
                ir2[5] = (v1175_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 5))));
                float v1181_data = ir2[6];
                ir2[6] = (v1181_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 6))));
                float v1187_data = ir2[7];
                ir2[7] = (v1187_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 7))));
                float v1193_data = ir2[8];
                ir2[8] = (v1193_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 8))));
                float v1199_data = ir2[9];
                ir2[9] = (v1199_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 9))));
                float v1205_data = ir2[10];
                ir2[10] = (v1205_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 10))));
                float v1211_data = ir2[11];
                ir2[11] = (v1211_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 11))));
                float v1217_data = ir2[12];
                ir2[12] = (v1217_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 12))));
                float v1223_data = ir2[13];
                ir2[13] = (v1223_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 13))));
                float v1229_data = ir2[14];
                ir2[14] = (v1229_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 14))));
                float v1235_data = ir2[15];
                ir2[15] = (v1235_data + (v1141_data * (sycl::group_broadcast(item.get_sub_group(), v1142_data, 15))));
              }
              if (v9_g) {
                float v1241_data = r0[12];
                float v1242_data = r1[12];
                float v1245_data = ir2[0];
                ir2[0] = (v1245_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 0))));
                float v1251_data = ir2[1];
                ir2[1] = (v1251_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 1))));
                float v1257_data = ir2[2];
                ir2[2] = (v1257_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 2))));
                float v1263_data = ir2[3];
                ir2[3] = (v1263_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 3))));
                float v1269_data = ir2[4];
                ir2[4] = (v1269_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 4))));
                float v1275_data = ir2[5];
                ir2[5] = (v1275_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 5))));
                float v1281_data = ir2[6];
                ir2[6] = (v1281_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 6))));
                float v1287_data = ir2[7];
                ir2[7] = (v1287_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 7))));
                float v1293_data = ir2[8];
                ir2[8] = (v1293_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 8))));
                float v1299_data = ir2[9];
                ir2[9] = (v1299_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 9))));
                float v1305_data = ir2[10];
                ir2[10] = (v1305_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 10))));
                float v1311_data = ir2[11];
                ir2[11] = (v1311_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 11))));
                float v1317_data = ir2[12];
                ir2[12] = (v1317_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 12))));
                float v1323_data = ir2[13];
                ir2[13] = (v1323_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 13))));
                float v1329_data = ir2[14];
                ir2[14] = (v1329_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 14))));
                float v1335_data = ir2[15];
                ir2[15] = (v1335_data + (v1241_data * (sycl::group_broadcast(item.get_sub_group(), v1242_data, 15))));
              }
              if (v9_g) {
                float v1341_data = r0[13];
                float v1342_data = r1[13];
                float v1345_data = ir2[0];
                ir2[0] = (v1345_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 0))));
                float v1351_data = ir2[1];
                ir2[1] = (v1351_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 1))));
                float v1357_data = ir2[2];
                ir2[2] = (v1357_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 2))));
                float v1363_data = ir2[3];
                ir2[3] = (v1363_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 3))));
                float v1369_data = ir2[4];
                ir2[4] = (v1369_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 4))));
                float v1375_data = ir2[5];
                ir2[5] = (v1375_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 5))));
                float v1381_data = ir2[6];
                ir2[6] = (v1381_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 6))));
                float v1387_data = ir2[7];
                ir2[7] = (v1387_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 7))));
                float v1393_data = ir2[8];
                ir2[8] = (v1393_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 8))));
                float v1399_data = ir2[9];
                ir2[9] = (v1399_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 9))));
                float v1405_data = ir2[10];
                ir2[10] = (v1405_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 10))));
                float v1411_data = ir2[11];
                ir2[11] = (v1411_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 11))));
                float v1417_data = ir2[12];
                ir2[12] = (v1417_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 12))));
                float v1423_data = ir2[13];
                ir2[13] = (v1423_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 13))));
                float v1429_data = ir2[14];
                ir2[14] = (v1429_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 14))));
                float v1435_data = ir2[15];
                ir2[15] = (v1435_data + (v1341_data * (sycl::group_broadcast(item.get_sub_group(), v1342_data, 15))));
              }
              if (v9_g) {
                float v1441_data = r0[14];
                float v1442_data = r1[14];
                float v1445_data = ir2[0];
                ir2[0] = (v1445_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 0))));
                float v1451_data = ir2[1];
                ir2[1] = (v1451_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 1))));
                float v1457_data = ir2[2];
                ir2[2] = (v1457_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 2))));
                float v1463_data = ir2[3];
                ir2[3] = (v1463_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 3))));
                float v1469_data = ir2[4];
                ir2[4] = (v1469_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 4))));
                float v1475_data = ir2[5];
                ir2[5] = (v1475_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 5))));
                float v1481_data = ir2[6];
                ir2[6] = (v1481_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 6))));
                float v1487_data = ir2[7];
                ir2[7] = (v1487_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 7))));
                float v1493_data = ir2[8];
                ir2[8] = (v1493_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 8))));
                float v1499_data = ir2[9];
                ir2[9] = (v1499_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 9))));
                float v1505_data = ir2[10];
                ir2[10] = (v1505_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 10))));
                float v1511_data = ir2[11];
                ir2[11] = (v1511_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 11))));
                float v1517_data = ir2[12];
                ir2[12] = (v1517_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 12))));
                float v1523_data = ir2[13];
                ir2[13] = (v1523_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 13))));
                float v1529_data = ir2[14];
                ir2[14] = (v1529_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 14))));
                float v1535_data = ir2[15];
                ir2[15] = (v1535_data + (v1441_data * (sycl::group_broadcast(item.get_sub_group(), v1442_data, 15))));
              }
              if (v9_g) {
                float v1541_data = r0[15];
                float v1542_data = r1[15];
                float v1545_data = ir2[0];
                ir2[0] = (v1545_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 0))));
                float v1551_data = ir2[1];
                ir2[1] = (v1551_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 1))));
                float v1557_data = ir2[2];
                ir2[2] = (v1557_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 2))));
                float v1563_data = ir2[3];
                ir2[3] = (v1563_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 3))));
                float v1569_data = ir2[4];
                ir2[4] = (v1569_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 4))));
                float v1575_data = ir2[5];
                ir2[5] = (v1575_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 5))));
                float v1581_data = ir2[6];
                ir2[6] = (v1581_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 6))));
                float v1587_data = ir2[7];
                ir2[7] = (v1587_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 7))));
                float v1593_data = ir2[8];
                ir2[8] = (v1593_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 8))));
                float v1599_data = ir2[9];
                ir2[9] = (v1599_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 9))));
                float v1605_data = ir2[10];
                ir2[10] = (v1605_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 10))));
                float v1611_data = ir2[11];
                ir2[11] = (v1611_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 11))));
                float v1617_data = ir2[12];
                ir2[12] = (v1617_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 12))));
                float v1623_data = ir2[13];
                ir2[13] = (v1623_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 13))));
                float v1629_data = ir2[14];
                ir2[14] = (v1629_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 14))));
                float v1635_data = ir2[15];
                ir2[15] = (v1635_data + (v1541_data * (sycl::group_broadcast(item.get_sub_group(), v1542_data, 15))));
              }
              if (v9_g) {
                float v1641_data = r0[16];
                float v1642_data = r1[16];
                float v1645_data = ir2[0];
                ir2[0] = (v1645_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 0))));
                float v1651_data = ir2[1];
                ir2[1] = (v1651_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 1))));
                float v1657_data = ir2[2];
                ir2[2] = (v1657_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 2))));
                float v1663_data = ir2[3];
                ir2[3] = (v1663_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 3))));
                float v1669_data = ir2[4];
                ir2[4] = (v1669_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 4))));
                float v1675_data = ir2[5];
                ir2[5] = (v1675_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 5))));
                float v1681_data = ir2[6];
                ir2[6] = (v1681_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 6))));
                float v1687_data = ir2[7];
                ir2[7] = (v1687_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 7))));
                float v1693_data = ir2[8];
                ir2[8] = (v1693_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 8))));
                float v1699_data = ir2[9];
                ir2[9] = (v1699_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 9))));
                float v1705_data = ir2[10];
                ir2[10] = (v1705_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 10))));
                float v1711_data = ir2[11];
                ir2[11] = (v1711_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 11))));
                float v1717_data = ir2[12];
                ir2[12] = (v1717_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 12))));
                float v1723_data = ir2[13];
                ir2[13] = (v1723_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 13))));
                float v1729_data = ir2[14];
                ir2[14] = (v1729_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 14))));
                float v1735_data = ir2[15];
                ir2[15] = (v1735_data + (v1641_data * (sycl::group_broadcast(item.get_sub_group(), v1642_data, 15))));
              }
              if (v9_g) {
                float v1741_data = r0[17];
                float v1742_data = r1[17];
                float v1745_data = ir2[0];
                ir2[0] = (v1745_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 0))));
                float v1751_data = ir2[1];
                ir2[1] = (v1751_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 1))));
                float v1757_data = ir2[2];
                ir2[2] = (v1757_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 2))));
                float v1763_data = ir2[3];
                ir2[3] = (v1763_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 3))));
                float v1769_data = ir2[4];
                ir2[4] = (v1769_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 4))));
                float v1775_data = ir2[5];
                ir2[5] = (v1775_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 5))));
                float v1781_data = ir2[6];
                ir2[6] = (v1781_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 6))));
                float v1787_data = ir2[7];
                ir2[7] = (v1787_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 7))));
                float v1793_data = ir2[8];
                ir2[8] = (v1793_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 8))));
                float v1799_data = ir2[9];
                ir2[9] = (v1799_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 9))));
                float v1805_data = ir2[10];
                ir2[10] = (v1805_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 10))));
                float v1811_data = ir2[11];
                ir2[11] = (v1811_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 11))));
                float v1817_data = ir2[12];
                ir2[12] = (v1817_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 12))));
                float v1823_data = ir2[13];
                ir2[13] = (v1823_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 13))));
                float v1829_data = ir2[14];
                ir2[14] = (v1829_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 14))));
                float v1835_data = ir2[15];
                ir2[15] = (v1835_data + (v1741_data * (sycl::group_broadcast(item.get_sub_group(), v1742_data, 15))));
              }
              if (v9_g) {
                float v1841_data = r0[18];
                float v1842_data = r1[18];
                float v1845_data = ir2[0];
                ir2[0] = (v1845_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 0))));
                float v1851_data = ir2[1];
                ir2[1] = (v1851_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 1))));
                float v1857_data = ir2[2];
                ir2[2] = (v1857_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 2))));
                float v1863_data = ir2[3];
                ir2[3] = (v1863_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 3))));
                float v1869_data = ir2[4];
                ir2[4] = (v1869_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 4))));
                float v1875_data = ir2[5];
                ir2[5] = (v1875_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 5))));
                float v1881_data = ir2[6];
                ir2[6] = (v1881_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 6))));
                float v1887_data = ir2[7];
                ir2[7] = (v1887_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 7))));
                float v1893_data = ir2[8];
                ir2[8] = (v1893_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 8))));
                float v1899_data = ir2[9];
                ir2[9] = (v1899_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 9))));
                float v1905_data = ir2[10];
                ir2[10] = (v1905_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 10))));
                float v1911_data = ir2[11];
                ir2[11] = (v1911_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 11))));
                float v1917_data = ir2[12];
                ir2[12] = (v1917_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 12))));
                float v1923_data = ir2[13];
                ir2[13] = (v1923_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 13))));
                float v1929_data = ir2[14];
                ir2[14] = (v1929_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 14))));
                float v1935_data = ir2[15];
                ir2[15] = (v1935_data + (v1841_data * (sycl::group_broadcast(item.get_sub_group(), v1842_data, 15))));
              }
              if (v9_g) {
                float v1941_data = r0[19];
                float v1942_data = r1[19];
                float v1945_data = ir2[0];
                ir2[0] = (v1945_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 0))));
                float v1951_data = ir2[1];
                ir2[1] = (v1951_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 1))));
                float v1957_data = ir2[2];
                ir2[2] = (v1957_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 2))));
                float v1963_data = ir2[3];
                ir2[3] = (v1963_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 3))));
                float v1969_data = ir2[4];
                ir2[4] = (v1969_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 4))));
                float v1975_data = ir2[5];
                ir2[5] = (v1975_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 5))));
                float v1981_data = ir2[6];
                ir2[6] = (v1981_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 6))));
                float v1987_data = ir2[7];
                ir2[7] = (v1987_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 7))));
                float v1993_data = ir2[8];
                ir2[8] = (v1993_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 8))));
                float v1999_data = ir2[9];
                ir2[9] = (v1999_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 9))));
                float v2005_data = ir2[10];
                ir2[10] = (v2005_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 10))));
                float v2011_data = ir2[11];
                ir2[11] = (v2011_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 11))));
                float v2017_data = ir2[12];
                ir2[12] = (v2017_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 12))));
                float v2023_data = ir2[13];
                ir2[13] = (v2023_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 13))));
                float v2029_data = ir2[14];
                ir2[14] = (v2029_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 14))));
                float v2035_data = ir2[15];
                ir2[15] = (v2035_data + (v1941_data * (sycl::group_broadcast(item.get_sub_group(), v1942_data, 15))));
              }
              if (v9_g) {
                #pragma unroll
                for (int32_t v2041_n1 = 0; v2041_n1 < 16; ++v2041_n1) {
                  float v2043_data = ir2[v2041_n1];
                  r2[v2041_n1] = v2043_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v9_g) {
                #pragma unroll
                for (int32_t v2049_i1 = 0; v2049_i1 < 16; ++v2049_i1) {
                  float v2051_data = r2[v2049_i1];
                  glb_m0[(v8_lead + (v2049_i1 * 12))] = v2051_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

