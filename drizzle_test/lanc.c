#include "stdio.h"
#include "math.h"

int main (){
  const int kernel_order = 3;
  const size_t npix = 512;
  const float del = 0.01;
  float lanczos_lut[npix];
  int i;
  const float forder = (float)kernel_order;
  float poff;

  /* Set the first value to avoid arithmetic problems */
  lanczos_lut[0] = 1.0;

  for (i = 1; i < npix; ++i) {
    poff = M_PI * (float)i * del;
    if (poff < M_PI * forder) {
      lanczos_lut[i] = sin(poff) / poff * sin(poff / forder) / (poff / forder);
      printf("%f\n", lanczos_lut[i]);
    } else {
      lanczos_lut[i] = 0.0;
    }
  }

  return 0;

}
