// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f61651fe59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f61651fe59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              sycl::group_barrier(item.get_sub_group());
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead < 12) {
                float v18_data = glb_m1[((v8_lead + 4) - 4)];
                float v19_data = s0[0];
                float v21_data = ir0[0];
                ir0[0] = (v21_data + (v18_data * v19_data));
                float v31_data = glb_m1[((v8_lead + 4) - 4)];
                float v32_data = s0[16];
                float v34_data = ir0[1];
                ir0[1] = (v34_data + (v31_data * v32_data));
                float v44_data = glb_m1[((v8_lead + 4) - 4)];
                float v45_data = s0[33];
                float v47_data = ir0[2];
                ir0[2] = (v47_data + (v44_data * v45_data));
                float v57_data = glb_m1[((v8_lead + 4) - 4)];
                float v58_data = s0[49];
                float v60_data = ir0[3];
                ir0[3] = (v60_data + (v57_data * v58_data));
                float v70_data = glb_m1[((v8_lead + 4) - 4)];
                float v71_data = s0[66];
                float v73_data = ir0[4];
                ir0[4] = (v73_data + (v70_data * v71_data));
                float v83_data = glb_m1[((v8_lead + 4) - 4)];
                float v84_data = s0[82];
                float v86_data = ir0[5];
                ir0[5] = (v86_data + (v83_data * v84_data));
                float v96_data = glb_m1[((v8_lead + 4) - 4)];
                float v97_data = s0[99];
                float v99_data = ir0[6];
                ir0[6] = (v99_data + (v96_data * v97_data));
                float v109_data = glb_m1[((v8_lead + 4) - 4)];
                float v110_data = s0[115];
                float v112_data = ir0[7];
                ir0[7] = (v112_data + (v109_data * v110_data));
              }
              if (v8_lead < 12) {
                float v126_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v127_data = s0[1];
                float v129_data = ir0[0];
                ir0[0] = (v129_data + (v126_data * v127_data));
                float v139_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v140_data = s0[17];
                float v142_data = ir0[1];
                ir0[1] = (v142_data + (v139_data * v140_data));
                float v152_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v153_data = s0[32];
                float v155_data = ir0[2];
                ir0[2] = (v155_data + (v152_data * v153_data));
                float v165_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v166_data = s0[48];
                float v168_data = ir0[3];
                ir0[3] = (v168_data + (v165_data * v166_data));
                float v178_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v179_data = s0[67];
                float v181_data = ir0[4];
                ir0[4] = (v181_data + (v178_data * v179_data));
                float v191_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v192_data = s0[83];
                float v194_data = ir0[5];
                ir0[5] = (v194_data + (v191_data * v192_data));
                float v204_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v205_data = s0[98];
                float v207_data = ir0[6];
                ir0[6] = (v207_data + (v204_data * v205_data));
                float v217_data = glb_m1[(((v8_lead + 4) - 4) + 12)];
                float v218_data = s0[114];
                float v220_data = ir0[7];
                ir0[7] = (v220_data + (v217_data * v218_data));
              }
              if (v8_lead < 12) {
                float v234_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v235_data = s0[2];
                float v237_data = ir0[0];
                ir0[0] = (v237_data + (v234_data * v235_data));
                float v247_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v248_data = s0[18];
                float v250_data = ir0[1];
                ir0[1] = (v250_data + (v247_data * v248_data));
                float v260_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v261_data = s0[35];
                float v263_data = ir0[2];
                ir0[2] = (v263_data + (v260_data * v261_data));
                float v273_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v274_data = s0[51];
                float v276_data = ir0[3];
                ir0[3] = (v276_data + (v273_data * v274_data));
                float v286_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v287_data = s0[64];
                float v289_data = ir0[4];
                ir0[4] = (v289_data + (v286_data * v287_data));
                float v299_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v300_data = s0[80];
                float v302_data = ir0[5];
                ir0[5] = (v302_data + (v299_data * v300_data));
                float v312_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v313_data = s0[97];
                float v315_data = ir0[6];
                ir0[6] = (v315_data + (v312_data * v313_data));
                float v325_data = glb_m1[(((v8_lead + 4) - 4) + 24)];
                float v326_data = s0[113];
                float v328_data = ir0[7];
                ir0[7] = (v328_data + (v325_data * v326_data));
              }
              if (v8_lead < 12) {
                float v342_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v343_data = s0[3];
                float v345_data = ir0[0];
                ir0[0] = (v345_data + (v342_data * v343_data));
                float v355_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v356_data = s0[19];
                float v358_data = ir0[1];
                ir0[1] = (v358_data + (v355_data * v356_data));
                float v368_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v369_data = s0[34];
                float v371_data = ir0[2];
                ir0[2] = (v371_data + (v368_data * v369_data));
                float v381_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v382_data = s0[50];
                float v384_data = ir0[3];
                ir0[3] = (v384_data + (v381_data * v382_data));
                float v394_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v395_data = s0[65];
                float v397_data = ir0[4];
                ir0[4] = (v397_data + (v394_data * v395_data));
                float v407_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v408_data = s0[81];
                float v410_data = ir0[5];
                ir0[5] = (v410_data + (v407_data * v408_data));
                float v420_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v421_data = s0[96];
                float v423_data = ir0[6];
                ir0[6] = (v423_data + (v420_data * v421_data));
                float v433_data = glb_m1[(((v8_lead + 4) - 4) + 36)];
                float v434_data = s0[112];
                float v436_data = ir0[7];
                ir0[7] = (v436_data + (v433_data * v434_data));
              }
              if (v8_lead < 12) {
                float v450_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v451_data = s0[4];
                float v453_data = ir0[0];
                ir0[0] = (v453_data + (v450_data * v451_data));
                float v463_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v464_data = s0[20];
                float v466_data = ir0[1];
                ir0[1] = (v466_data + (v463_data * v464_data));
                float v476_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v477_data = s0[37];
                float v479_data = ir0[2];
                ir0[2] = (v479_data + (v476_data * v477_data));
                float v489_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v490_data = s0[53];
                float v492_data = ir0[3];
                ir0[3] = (v492_data + (v489_data * v490_data));
                float v502_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v503_data = s0[70];
                float v505_data = ir0[4];
                ir0[4] = (v505_data + (v502_data * v503_data));
                float v515_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v516_data = s0[86];
                float v518_data = ir0[5];
                ir0[5] = (v518_data + (v515_data * v516_data));
                float v528_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v529_data = s0[103];
                float v531_data = ir0[6];
                ir0[6] = (v531_data + (v528_data * v529_data));
                float v541_data = glb_m1[(((v8_lead + 4) - 4) + 48)];
                float v542_data = s0[119];
                float v544_data = ir0[7];
                ir0[7] = (v544_data + (v541_data * v542_data));
              }
              if (v8_lead < 12) {
                float v558_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v559_data = s0[5];
                float v561_data = ir0[0];
                ir0[0] = (v561_data + (v558_data * v559_data));
                float v571_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v572_data = s0[21];
                float v574_data = ir0[1];
                ir0[1] = (v574_data + (v571_data * v572_data));
                float v584_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v585_data = s0[36];
                float v587_data = ir0[2];
                ir0[2] = (v587_data + (v584_data * v585_data));
                float v597_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v598_data = s0[52];
                float v600_data = ir0[3];
                ir0[3] = (v600_data + (v597_data * v598_data));
                float v610_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v611_data = s0[71];
                float v613_data = ir0[4];
                ir0[4] = (v613_data + (v610_data * v611_data));
                float v623_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v624_data = s0[87];
                float v626_data = ir0[5];
                ir0[5] = (v626_data + (v623_data * v624_data));
                float v636_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v637_data = s0[102];
                float v639_data = ir0[6];
                ir0[6] = (v639_data + (v636_data * v637_data));
                float v649_data = glb_m1[(((v8_lead + 4) - 4) + 60)];
                float v650_data = s0[118];
                float v652_data = ir0[7];
                ir0[7] = (v652_data + (v649_data * v650_data));
              }
              if (v8_lead < 12) {
                float v666_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v667_data = s0[6];
                float v669_data = ir0[0];
                ir0[0] = (v669_data + (v666_data * v667_data));
                float v679_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v680_data = s0[22];
                float v682_data = ir0[1];
                ir0[1] = (v682_data + (v679_data * v680_data));
                float v692_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v693_data = s0[39];
                float v695_data = ir0[2];
                ir0[2] = (v695_data + (v692_data * v693_data));
                float v705_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v706_data = s0[55];
                float v708_data = ir0[3];
                ir0[3] = (v708_data + (v705_data * v706_data));
                float v718_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v719_data = s0[68];
                float v721_data = ir0[4];
                ir0[4] = (v721_data + (v718_data * v719_data));
                float v731_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v732_data = s0[84];
                float v734_data = ir0[5];
                ir0[5] = (v734_data + (v731_data * v732_data));
                float v744_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v745_data = s0[101];
                float v747_data = ir0[6];
                ir0[6] = (v747_data + (v744_data * v745_data));
                float v757_data = glb_m1[(((v8_lead + 4) - 4) + 72)];
                float v758_data = s0[117];
                float v760_data = ir0[7];
                ir0[7] = (v760_data + (v757_data * v758_data));
              }
              if (v8_lead < 12) {
                float v774_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v775_data = s0[7];
                float v777_data = ir0[0];
                ir0[0] = (v777_data + (v774_data * v775_data));
                float v787_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v788_data = s0[23];
                float v790_data = ir0[1];
                ir0[1] = (v790_data + (v787_data * v788_data));
                float v800_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v801_data = s0[38];
                float v803_data = ir0[2];
                ir0[2] = (v803_data + (v800_data * v801_data));
                float v813_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v814_data = s0[54];
                float v816_data = ir0[3];
                ir0[3] = (v816_data + (v813_data * v814_data));
                float v826_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v827_data = s0[69];
                float v829_data = ir0[4];
                ir0[4] = (v829_data + (v826_data * v827_data));
                float v839_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v840_data = s0[85];
                float v842_data = ir0[5];
                ir0[5] = (v842_data + (v839_data * v840_data));
                float v852_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v853_data = s0[100];
                float v855_data = ir0[6];
                ir0[6] = (v855_data + (v852_data * v853_data));
                float v865_data = glb_m1[(((v8_lead + 4) - 4) + 84)];
                float v866_data = s0[116];
                float v868_data = ir0[7];
                ir0[7] = (v868_data + (v865_data * v866_data));
              }
              if (v8_lead < 12) {
                float v882_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v883_data = s0[8];
                float v885_data = ir0[0];
                ir0[0] = (v885_data + (v882_data * v883_data));
                float v895_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v896_data = s0[24];
                float v898_data = ir0[1];
                ir0[1] = (v898_data + (v895_data * v896_data));
                float v908_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v909_data = s0[41];
                float v911_data = ir0[2];
                ir0[2] = (v911_data + (v908_data * v909_data));
                float v921_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v922_data = s0[57];
                float v924_data = ir0[3];
                ir0[3] = (v924_data + (v921_data * v922_data));
                float v934_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v935_data = s0[74];
                float v937_data = ir0[4];
                ir0[4] = (v937_data + (v934_data * v935_data));
                float v947_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v948_data = s0[90];
                float v950_data = ir0[5];
                ir0[5] = (v950_data + (v947_data * v948_data));
                float v960_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v961_data = s0[107];
                float v963_data = ir0[6];
                ir0[6] = (v963_data + (v960_data * v961_data));
                float v973_data = glb_m1[(((v8_lead + 4) - 4) + 96)];
                float v974_data = s0[123];
                float v976_data = ir0[7];
                ir0[7] = (v976_data + (v973_data * v974_data));
              }
              if (v8_lead < 12) {
                float v990_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v991_data = s0[9];
                float v993_data = ir0[0];
                ir0[0] = (v993_data + (v990_data * v991_data));
                float v1003_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1004_data = s0[25];
                float v1006_data = ir0[1];
                ir0[1] = (v1006_data + (v1003_data * v1004_data));
                float v1016_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1017_data = s0[40];
                float v1019_data = ir0[2];
                ir0[2] = (v1019_data + (v1016_data * v1017_data));
                float v1029_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1030_data = s0[56];
                float v1032_data = ir0[3];
                ir0[3] = (v1032_data + (v1029_data * v1030_data));
                float v1042_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1043_data = s0[75];
                float v1045_data = ir0[4];
                ir0[4] = (v1045_data + (v1042_data * v1043_data));
                float v1055_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1056_data = s0[91];
                float v1058_data = ir0[5];
                ir0[5] = (v1058_data + (v1055_data * v1056_data));
                float v1068_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1069_data = s0[106];
                float v1071_data = ir0[6];
                ir0[6] = (v1071_data + (v1068_data * v1069_data));
                float v1081_data = glb_m1[(((v8_lead + 4) - 4) + 108)];
                float v1082_data = s0[122];
                float v1084_data = ir0[7];
                ir0[7] = (v1084_data + (v1081_data * v1082_data));
              }
              if (v8_lead < 12) {
                float v1098_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1099_data = s0[10];
                float v1101_data = ir0[0];
                ir0[0] = (v1101_data + (v1098_data * v1099_data));
                float v1111_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1112_data = s0[26];
                float v1114_data = ir0[1];
                ir0[1] = (v1114_data + (v1111_data * v1112_data));
                float v1124_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1125_data = s0[43];
                float v1127_data = ir0[2];
                ir0[2] = (v1127_data + (v1124_data * v1125_data));
                float v1137_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1138_data = s0[59];
                float v1140_data = ir0[3];
                ir0[3] = (v1140_data + (v1137_data * v1138_data));
                float v1150_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1151_data = s0[72];
                float v1153_data = ir0[4];
                ir0[4] = (v1153_data + (v1150_data * v1151_data));
                float v1163_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1164_data = s0[88];
                float v1166_data = ir0[5];
                ir0[5] = (v1166_data + (v1163_data * v1164_data));
                float v1176_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1177_data = s0[105];
                float v1179_data = ir0[6];
                ir0[6] = (v1179_data + (v1176_data * v1177_data));
                float v1189_data = glb_m1[(((v8_lead + 4) - 4) + 120)];
                float v1190_data = s0[121];
                float v1192_data = ir0[7];
                ir0[7] = (v1192_data + (v1189_data * v1190_data));
              }
              if (v8_lead < 12) {
                float v1206_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1207_data = s0[11];
                float v1209_data = ir0[0];
                ir0[0] = (v1209_data + (v1206_data * v1207_data));
                float v1219_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1220_data = s0[27];
                float v1222_data = ir0[1];
                ir0[1] = (v1222_data + (v1219_data * v1220_data));
                float v1232_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1233_data = s0[42];
                float v1235_data = ir0[2];
                ir0[2] = (v1235_data + (v1232_data * v1233_data));
                float v1245_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1246_data = s0[58];
                float v1248_data = ir0[3];
                ir0[3] = (v1248_data + (v1245_data * v1246_data));
                float v1258_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1259_data = s0[73];
                float v1261_data = ir0[4];
                ir0[4] = (v1261_data + (v1258_data * v1259_data));
                float v1271_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1272_data = s0[89];
                float v1274_data = ir0[5];
                ir0[5] = (v1274_data + (v1271_data * v1272_data));
                float v1284_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1285_data = s0[104];
                float v1287_data = ir0[6];
                ir0[6] = (v1287_data + (v1284_data * v1285_data));
                float v1297_data = glb_m1[(((v8_lead + 4) - 4) + 132)];
                float v1298_data = s0[120];
                float v1300_data = ir0[7];
                ir0[7] = (v1300_data + (v1297_data * v1298_data));
              }
              if (v8_lead < 12) {
                float v1314_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1315_data = s0[12];
                float v1317_data = ir0[0];
                ir0[0] = (v1317_data + (v1314_data * v1315_data));
                float v1327_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1328_data = s0[28];
                float v1330_data = ir0[1];
                ir0[1] = (v1330_data + (v1327_data * v1328_data));
                float v1340_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1341_data = s0[45];
                float v1343_data = ir0[2];
                ir0[2] = (v1343_data + (v1340_data * v1341_data));
                float v1353_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1354_data = s0[61];
                float v1356_data = ir0[3];
                ir0[3] = (v1356_data + (v1353_data * v1354_data));
                float v1366_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1367_data = s0[78];
                float v1369_data = ir0[4];
                ir0[4] = (v1369_data + (v1366_data * v1367_data));
                float v1379_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1380_data = s0[94];
                float v1382_data = ir0[5];
                ir0[5] = (v1382_data + (v1379_data * v1380_data));
                float v1392_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1393_data = s0[111];
                float v1395_data = ir0[6];
                ir0[6] = (v1395_data + (v1392_data * v1393_data));
                float v1405_data = glb_m1[(((v8_lead + 4) - 4) + 144)];
                float v1406_data = s0[127];
                float v1408_data = ir0[7];
                ir0[7] = (v1408_data + (v1405_data * v1406_data));
              }
              if (v8_lead < 12) {
                float v1422_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1423_data = s0[13];
                float v1425_data = ir0[0];
                ir0[0] = (v1425_data + (v1422_data * v1423_data));
                float v1435_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1436_data = s0[29];
                float v1438_data = ir0[1];
                ir0[1] = (v1438_data + (v1435_data * v1436_data));
                float v1448_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1449_data = s0[44];
                float v1451_data = ir0[2];
                ir0[2] = (v1451_data + (v1448_data * v1449_data));
                float v1461_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1462_data = s0[60];
                float v1464_data = ir0[3];
                ir0[3] = (v1464_data + (v1461_data * v1462_data));
                float v1474_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1475_data = s0[79];
                float v1477_data = ir0[4];
                ir0[4] = (v1477_data + (v1474_data * v1475_data));
                float v1487_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1488_data = s0[95];
                float v1490_data = ir0[5];
                ir0[5] = (v1490_data + (v1487_data * v1488_data));
                float v1500_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1501_data = s0[110];
                float v1503_data = ir0[6];
                ir0[6] = (v1503_data + (v1500_data * v1501_data));
                float v1513_data = glb_m1[(((v8_lead + 4) - 4) + 156)];
                float v1514_data = s0[126];
                float v1516_data = ir0[7];
                ir0[7] = (v1516_data + (v1513_data * v1514_data));
              }
              if (v8_lead < 12) {
                float v1530_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1531_data = s0[14];
                float v1533_data = ir0[0];
                ir0[0] = (v1533_data + (v1530_data * v1531_data));
                float v1543_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1544_data = s0[30];
                float v1546_data = ir0[1];
                ir0[1] = (v1546_data + (v1543_data * v1544_data));
                float v1556_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1557_data = s0[47];
                float v1559_data = ir0[2];
                ir0[2] = (v1559_data + (v1556_data * v1557_data));
                float v1569_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1570_data = s0[63];
                float v1572_data = ir0[3];
                ir0[3] = (v1572_data + (v1569_data * v1570_data));
                float v1582_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1583_data = s0[76];
                float v1585_data = ir0[4];
                ir0[4] = (v1585_data + (v1582_data * v1583_data));
                float v1595_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1596_data = s0[92];
                float v1598_data = ir0[5];
                ir0[5] = (v1598_data + (v1595_data * v1596_data));
                float v1608_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1609_data = s0[109];
                float v1611_data = ir0[6];
                ir0[6] = (v1611_data + (v1608_data * v1609_data));
                float v1621_data = glb_m1[(((v8_lead + 4) - 4) + 168)];
                float v1622_data = s0[125];
                float v1624_data = ir0[7];
                ir0[7] = (v1624_data + (v1621_data * v1622_data));
              }
              if (v8_lead < 12) {
                float v1638_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1639_data = s0[15];
                float v1641_data = ir0[0];
                ir0[0] = (v1641_data + (v1638_data * v1639_data));
                float v1651_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1652_data = s0[31];
                float v1654_data = ir0[1];
                ir0[1] = (v1654_data + (v1651_data * v1652_data));
                float v1664_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1665_data = s0[46];
                float v1667_data = ir0[2];
                ir0[2] = (v1667_data + (v1664_data * v1665_data));
                float v1677_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1678_data = s0[62];
                float v1680_data = ir0[3];
                ir0[3] = (v1680_data + (v1677_data * v1678_data));
                float v1690_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1691_data = s0[77];
                float v1693_data = ir0[4];
                ir0[4] = (v1693_data + (v1690_data * v1691_data));
                float v1703_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1704_data = s0[93];
                float v1706_data = ir0[5];
                ir0[5] = (v1706_data + (v1703_data * v1704_data));
                float v1716_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1717_data = s0[108];
                float v1719_data = ir0[6];
                ir0[6] = (v1719_data + (v1716_data * v1717_data));
                float v1729_data = glb_m1[(((v8_lead + 4) - 4) + 180)];
                float v1730_data = s0[124];
                float v1732_data = ir0[7];
                ir0[7] = (v1732_data + (v1729_data * v1730_data));
              }
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1738_n1 = 0; v1738_n1 < 8; ++v1738_n1) {
                  float v1740_data = ir0[v1738_n1];
                  r0[v1738_n1] = v1740_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead < 12) {
                #pragma unroll
                for (int32_t v1746_i1 = 0; v1746_i1 < 8; ++v1746_i1) {
                  float v1748_data = r0[v1746_i1];
                  glb_m0[(v8_lead + (v1746_i1 * 12))] = v1748_data;
                }
              }
            }
          }
        }
      });
    }
  });
}

