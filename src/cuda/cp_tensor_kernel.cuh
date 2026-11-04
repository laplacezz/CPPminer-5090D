// Exact PEARL IMMA / HMMA experiment. FP32 partials reset every R=128.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cp_config.h"
#include "cp_cutlass.h"
#include "cp_gpu.cuh"

__device__ __forceinline__ void cp_imma(int* d, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp_hmma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ unsigned cp_ld32(const void* p) { return *reinterpret_cast<const unsigned*>(p); }
__device__ __forceinline__ size_t cp_tensor_index(int row, int k, int rows, bool step) {
    return step ? ((size_t)(k / R_RANK) * rows + row) * R_RANK + k % R_RANK : (size_t)row * K_DIM + k;
}
// MODE 0=IMMA, 1=exact HMMA, 2=alternating IMMA/HMMA K panels.
// DUMP records every cumulative-cell XOR, not just a final checksum.
template<int MODE, bool DUMP>
__global__ __launch_bounds__(256) void cp_tensor_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    const __half* __restrict__ AH, const __half* __restrict__ BH,
    int m, int n, int rp, int cp, bool step, uint32_t* dump,
    size_t tile_count, CpCutlassJackpotLaunch jp, bool jackpot) {
    // Padded row strides avoid the repeated shared-memory bank collisions of
    // tightly packed 32-wide rows for the PTX fragment lane mapping.
    __shared__ __align__(16) int8_t sa[128][48], sb[128][48];
    __shared__ __align__(16) __half ha[128][40], hb[128][40];
    const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    const int wm=warp/4, wn=warp%4, gid=lane/4, t=lane%4;
    const int row0=(rp+blockIdx.y)*128, col0=(cp+blockIdx.x)*128;
    int acc[4][4][4]={};
    uint32_t transcript[16]={};
    const size_t tile=((size_t)blockIdx.y*gridDim.x+blockIdx.x)*256+tid;
    #pragma unroll 1
    for(int milestone=0;milestone<K_DIM/R_RANK;++milestone) {
        float partial[4][4][4]={};
        #pragma unroll
        for(int panel=0;panel<4;++panel) {
            const int k0=milestone*128+panel*32, rr=tid/2, kk=(tid%2)*16;
            const int8_t* ap=A+cp_tensor_index(row0+rr,k0+kk,m,step);
            const int8_t* bp=B+cp_tensor_index(col0+rr,k0+kk,n,step);
            if(MODE==0 || (MODE==2 && panel%2==0)) {
                *reinterpret_cast<int4*>(&sa[rr][kk])=*reinterpret_cast<const int4*>(ap);
                *reinterpret_cast<int4*>(&sb[rr][kk])=*reinterpret_cast<const int4*>(bp);
            } else {
                const __half* ah=AH+cp_tensor_index(row0+rr,k0+kk,m,step);
                const __half* bh=BH+cp_tensor_index(col0+rr,k0+kk,n,step);
                *reinterpret_cast<int4*>(&ha[rr][kk])=*reinterpret_cast<const int4*>(ah);
                *reinterpret_cast<int4*>(&ha[rr][kk+8])=*reinterpret_cast<const int4*>(ah+8);
                *reinterpret_cast<int4*>(&hb[rr][kk])=*reinterpret_cast<const int4*>(bh);
                *reinterpret_cast<int4*>(&hb[rr][kk+8])=*reinterpret_cast<const int4*>(bh+8);
            }
            __syncthreads();
            if(MODE==0 || (MODE==2 && panel%2==0)) {
                unsigned ar[4][4],br[4][2];
                #pragma unroll
                for(int i=0;i<4;++i) {
                    const int r=wm*64+i*16+gid,c=wn*32+i*8+gid;
                    ar[i][0]=cp_ld32(&sa[r][t*4]); ar[i][1]=cp_ld32(&sa[r+8][t*4]);
                    ar[i][2]=cp_ld32(&sa[r][t*4+16]); ar[i][3]=cp_ld32(&sa[r+8][t*4+16]);
                    br[i][0]=cp_ld32(&sb[c][t*4]); br[i][1]=cp_ld32(&sb[c][t*4+16]);
                }
                #pragma unroll
                for(int i=0;i<4;++i)
                    #pragma unroll
                    for(int j=0;j<4;++j) cp_imma(acc[i][j],ar[i],br[j]);
            } else {
                #pragma unroll
                for(int halfk=0;halfk<2;++halfk) {
                    unsigned ar[4][4],br[4][2]; const int k=halfk*16+t*2;
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        const int r=wm*64+i*16+gid,c=wn*32+i*8+gid;
                        ar[i][0]=cp_ld32(&ha[r][k]); ar[i][1]=cp_ld32(&ha[r+8][k]);
                        ar[i][2]=cp_ld32(&ha[r][k+8]); ar[i][3]=cp_ld32(&ha[r+8][k+8]);
                        br[i][0]=cp_ld32(&hb[c][k]); br[i][1]=cp_ld32(&hb[c][k+8]);
                    }
                    #pragma unroll
                    for(int i=0;i<4;++i)
                        #pragma unroll
                        for(int j=0;j<4;++j) cp_hmma(partial[i][j],ar[i],br[j]);
                }
            }
            __syncthreads();
        }
        uint32_t fold=0;
        #pragma unroll
        for(int i=0;i<4;++i)
            #pragma unroll
            for(int j=0;j<4;++j)
                #pragma unroll
                for(int e=0;e<4;++e) {
                    if(MODE!=0) acc[i][j][e]+=__float2int_rn(partial[i][j][e]);
                    fold^=(uint32_t)acc[i][j][e];
                }
        if(DUMP) dump[(size_t)milestone*tile_count+tile]=fold;
        const int w=milestone%16;
        transcript[w]=((transcript[w]<<13)|(transcript[w]>>19))^fold;
    }
    if(jackpot) {
        uint32_t digest[8]; b3_compress64(jp.d_a_key8,transcript,digest);
        bool ok=true;
        for(int w=7;w>=0;--w) {
            if(digest[w]<jp.bound[w]) break;
            if(digest[w]>jp.bound[w]) {ok=false;break;}
        }
        if(ok && atomicCAS(jp.d_found,0,1)==0) {
            *jp.d_out_t_rows=row0+wm*64+gid; *jp.d_out_t_cols=col0+wn*32+2*t;
        }
    }
}
