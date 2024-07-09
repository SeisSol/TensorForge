#ifndef YATETO_SUBROUTINE_H_
#define YATETO_SUBROUTINE_H_
void launcher_kernel_386f7503c1(float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags = nullptr, void* streamPtr = nullptr);
void launcher_kernel_9dfefadc54(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags = nullptr, void* streamPtr = nullptr);
void launcher_kernel_44a6a7e323(const float** m0, unsigned m0_extraOffset, float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags = nullptr, void* streamPtr = nullptr);
void launcher_kernel_86dd44de64(const float** m0, unsigned m0_extraOffset, float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, size_t numElements, unsigned* flags = nullptr, void* streamPtr = nullptr);
#endif
