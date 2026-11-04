#ifndef CP_CUTLASS_H
#define CP_CUTLASS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 if CUTLASS fused GEMM can run on the current device. */
int cp_cutlass_device_ok(int dev);
/* 0=INT8, 1=exact FP16/FP32, 2=alternating exact INT8/FP16 panels. */
int cp_tensor_set_mode(int mode);
int cp_tensor_get_mode(void);
void cp_tensor_invalidate_inputs(int dev);
void cp_tensor_invalidate_a(int dev);
void cp_tensor_release(int dev);
void cp_tensor_set_positive_a(int on);
int cp_tensor_positive_a(void);
int cp_tensor_signal_tag(void);
void cp_tensor_set_raw_test(int on);
int cp_tensor_raw_test(void);
void cp_radix_invalidate(int dev,int only_a);
void cp_radix_release(int dev);
void cp_project_invalidate(int dev,int only_a);
void cp_project_release(int dev);
void cp_project_set_noise(int dev,const uint32_t* pairs,const int8_t* ebr);
void cp_narrow_invalidate(int dev);
void cp_narrow_invalidate_a(int dev);
void cp_narrow_release(int dev);

/* Fused GEMM + in-register milestone XOR for one period batch panel.
 * Panel covers row_batch x col_batch CTAs of 128x128 (Case 10 / MMA lane).
 * When jackpot is non-NULL, BLAKE3/target check runs in the GEMM kernel tail
 * and d_tile_xor may be NULL. */
typedef struct {
    uint32_t bound[8];
    const uint32_t* d_a_key8;
    int* d_found;
    int* d_out_t_rows;
    int* d_out_t_cols;
    int row_period0;
    int col_period0;
} CpCutlassJackpotLaunch;
int cp_narrow_period_batch(int dev,int mode,const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp);
int cp_radix_period_batch(int dev,int mode,const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp);
/* Returns 1 for a checked integer fallback, 0 for successful projection, -1 for error. */
int cp_project_period_batch(int dev,int mode,const int8_t* a,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp);
int cp_int8_shape_batch(int mode,const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp);

int cp_cutlass_period_batch(
    int dev,
    const int8_t* d_Ap,
    const int8_t* d_BpT,
    int m,
    int n,
    int row_period0,
    int col_period0,
    int row_batch_count,
    int col_batch_count,
    int step_major,
    uint32_t* d_tile_xor,
    size_t tiles_per_batch,
    const CpCutlassJackpotLaunch* jackpot);

size_t cp_cutlass_tiles_per_batch(int row_batch_count, int col_batch_count);

size_t cp_cutlass_tile_xor_bytes(int row_batch_count, int col_batch_count);

#ifdef __cplusplus
}
#endif

#endif /* CP_CUTLASS_H */
