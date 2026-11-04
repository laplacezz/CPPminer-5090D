// Low precision alternatives with EXACT integer mining semantics.
// Inputs are checked to be integers in [-3,3]. No precision tolerance is used.
// A float accumulator biased by 3*2^22 encodes the exact signed integer in its
// low 22 bits. XOR all 64 encodings, then sign-extend once per milestone.
#pragma once
#include "cp_tensor_kernel.cuh"
#include <type_traits>

__device__ __forceinline__ void cp_bmma(float* d,const unsigned* a,const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void cp_tmma(float* d,const unsigned* a,const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
// FMT: 0 E4M3, 1 E3M2, 2 E2M1, 3 TF32, 4 FP16, 5 BF16, 6 MXFP6.
template<int FMT> __device__ __forceinline__ void cp_lmma(float* d,const unsigned* a,const unsigned* b) {
    if constexpr(FMT==0) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
#if __CUDA_ARCH__ >= 1200
    else if constexpr(FMT==1) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else if constexpr(FMT==2) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else if constexpr(FMT==6) {
        unsigned scale=0x7f7f7f7fu;unsigned short zero=0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e3m2.f32.ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%11,%12},%13,{%14,%15};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
              "r"(scale),"h"(zero),"h"(zero),"r"(scale),"h"(zero),"h"(zero));
    } else if constexpr(FMT==9) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
#endif
}
template<int FMT> __device__ __forceinline__ void cp_low_half_mma(unsigned* d,const unsigned* a,const unsigned* b) {
#if __CUDA_ARCH__ >= 1200
    if constexpr(FMT==2) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e2m1.e2m1.f16 "
            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
            : "+r"(d[0]),"+r"(d[1]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e3m2.e3m2.f16 "
            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
            : "+r"(d[0]),"+r"(d[1]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
#endif
}
__device__ __forceinline__ void cp_narrow_jackpot(uint32_t* transcript,CpCutlassJackpotLaunch jp,bool jackpot,int row,int col) {
    if(!jackpot)return;
    uint32_t digest[8];b3_compress64(jp.d_a_key8,transcript,digest);
    bool ok=true;
    for(int w=7;w>=0;--w) {if(digest[w]<jp.bound[w])break;if(digest[w]>jp.bound[w]){ok=false;break;}}
    if(ok&&atomicCAS(jp.d_found,0,1)==0){*jp.d_out_t_rows=row;*jp.d_out_t_cols=col;}
}
template<int FMT,bool HALFACC,bool DUMP,bool GENERAL=false>
__global__ __launch_bounds__(256) void cp_narrow_kernel(const void* av,const void* bv,
    int m,int n,int rp,int cp,bool step,uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot) {
    using T=typename std::conditional<FMT==3,unsigned,typename std::conditional<(FMT==4||FMT==5),unsigned short,unsigned char>::type>::type;
    constexpr int PITCH=sizeof(T)==1?48:sizeof(T)==2?40:36;
    __shared__ __align__(16) T sa[128][PITCH],sb[128][PITCH];
    const T* A=(const T*)av;const T* B=(const T*)bv;
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,wm=warp/4,wn=warp%4,gid=lane/4,t=lane%4;
    unsigned linear=blockIdx.y*gridDim.x+blockIdx.x,tr=linear%gridDim.y,tc=linear/gridDim.y;
    int row0=(rp+tr)*128,col0=(cp+tc)*128;
    size_t tile=((size_t)tr*gridDim.x+tc)*256+tid;
    float fc[4][4][4];int cumulative[4][4][4]={};uint32_t transcript[16]={};
    #pragma unroll
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fc[i][j][e]=12582912.0f;
    #pragma unroll 1
    for(int s=0;s<32;++s) {
        if constexpr(GENERAL) {
            #pragma unroll
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fc[i][j][e]=0.0f;
        }
        unsigned hc[4][4][2]={};
        #pragma unroll
        for(int panel=0;panel<4;++panel) {
            int k0=s*128+panel*32,rr=tid/2,kk=(tid%2)*16;
            const T* ap=A+cp_tensor_index(row0+rr,k0+kk,m,step);
            const T* bp=B+cp_tensor_index(col0+rr,k0+kk,n,step);
            #pragma unroll
            for(int q=0;q<sizeof(T);++q) {
                ((int4*)&sa[rr][kk])[q]=((const int4*)ap)[q];
                ((int4*)&sb[rr][kk])[q]=((const int4*)bp)[q];
            }
            __syncthreads();
            constexpr int SUB=FMT==3?4:(FMT==4||FMT==5)?2:1;
            #pragma unroll
            for(int sub=0;sub<SUB;++sub) {
                unsigned ar[4][4],br[4][2];
                #pragma unroll
                for(int i=0;i<4;++i) {
                    int r=wm*64+i*16+gid,c=wn*32+i*8+gid;
                    if constexpr(FMT==3) {
                        int k=sub*8+t;
                        ar[i][0]=sa[r][k];ar[i][1]=sa[r+8][k];ar[i][2]=sa[r][k+4];ar[i][3]=sa[r+8][k+4];
                        br[i][0]=sb[c][k];br[i][1]=sb[c][k+4];
                    } else if constexpr(FMT==4||FMT==5) {
                        int k=sub*16+2*t;
                        ar[i][0]=cp_ld32(&sa[r][k]);ar[i][1]=cp_ld32(&sa[r+8][k]);ar[i][2]=cp_ld32(&sa[r][k+8]);ar[i][3]=cp_ld32(&sa[r+8][k+8]);
                        br[i][0]=cp_ld32(&sb[c][k]);br[i][1]=cp_ld32(&sb[c][k+8]);
                    } else {
                        ar[i][0]=cp_ld32(&sa[r][4*t]);ar[i][1]=cp_ld32(&sa[r+8][4*t]);ar[i][2]=cp_ld32(&sa[r][4*t+16]);ar[i][3]=cp_ld32(&sa[r+8][4*t+16]);
                        br[i][0]=cp_ld32(&sb[c][4*t]);br[i][1]=cp_ld32(&sb[c][4*t+16]);
                    }
                }
                #pragma unroll
                for(int i=0;i<4;++i)
                    #pragma unroll
                    for(int j=0;j<4;++j) {
                        if constexpr(HALFACC)cp_low_half_mma<FMT>(hc[i][j],ar[i],br[j]);
                        else if constexpr(FMT==3)cp_tmma(fc[i][j],ar[i],br[j]);
                        else if constexpr(FMT==4)cp_hmma(fc[i][j],ar[i],br[j]);
                        else if constexpr(FMT==5)cp_bmma(fc[i][j],ar[i],br[j]);
                        else cp_lmma<FMT>(fc[i][j],ar[i],br[j]);
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
                    if constexpr(GENERAL) {
                        cumulative[i][j][e]+=__float2int_rn(fc[i][j][e]);
                        fold^=(uint32_t)cumulative[i][j][e];
                    } else if constexpr(HALFACC) {
                        unsigned short bits=(unsigned short)(hc[i][j][e/2]>>(16*(e%2)));
                        cumulative[i][j][e]+=__half2int_rn(__ushort_as_half(bits));
                        fold^=(uint32_t)cumulative[i][j][e];
                    } else fold^=__float_as_uint(fc[i][j][e]);
                }
        if constexpr(!HALFACC&&!GENERAL)fold=(uint32_t)((int32_t)(fold<<9)>>9);
        if(DUMP)dump[(size_t)s*tiles+tile]=fold;
        int w=s%16;transcript[w]=((transcript[w]<<13)|(transcript[w]>>19))^fold;
    }
    cp_narrow_jackpot(transcript,jp,jackpot,row0+wm*64+gid,col0+wn*32+2*t);
}

template<bool DUMP> __global__ __launch_bounds__(256) void cp_int4_kernel(const unsigned char* A,const unsigned char* B,
    int m,int n,int rp,int cp,bool step,uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot) {
    __shared__ __align__(16) unsigned char sa[128][48],sb[128][48];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,wm=warp/4,wn=warp%4,gid=lane/4,t=lane%4;
    unsigned linear=blockIdx.y*gridDim.x+blockIdx.x,tr=linear%gridDim.y,tc=linear/gridDim.y;
    int row0=(rp+tr)*128,col0=(cp+tc)*128;
    size_t tile=((size_t)tr*gridDim.x+tc)*256+tid;
    int acc[8][4][2]={};uint32_t transcript[16]={};
    #pragma unroll 1
    for(int s=0;s<32;++s) {
        #pragma unroll
        for(int panel=0;panel<4;++panel) {
            int k0=s*128+panel*32,rr=tid/2,kk=(tid%2)*8;
            *(uint2*)&sa[rr][kk]=*(const uint2*)(A+cp_tensor_index(row0+rr,k0+2*kk,m,step)/2);
            *(uint2*)&sb[rr][kk]=*(const uint2*)(B+cp_tensor_index(col0+rr,k0+2*kk,n,step)/2);
            __syncthreads();
            unsigned ar[8],br[4];
            #pragma unroll
            for(int i=0;i<8;++i)ar[i]=cp_ld32(&sa[wm*64+i*8+gid][t*4]);
            #pragma unroll
            for(int j=0;j<4;++j)br[j]=cp_ld32(&sb[wn*32+j*8+gid][t*4]);
            #pragma unroll
            for(int i=0;i<8;++i)
                #pragma unroll
                for(int j=0;j<4;++j)
                    asm volatile("mma.sync.aligned.row.col.m8n8k32.s32.s4.s4.s32 {%0,%1},{%2},{%3},{%0,%1};"
                        : "+r"(acc[i][j][0]),"+r"(acc[i][j][1]):"r"(ar[i]),"r"(br[j]));
            __syncthreads();
        }
        uint32_t fold=0;
        #pragma unroll
        for(int i=0;i<8;++i)for(int j=0;j<4;++j)for(int e=0;e<2;++e)fold^=(uint32_t)acc[i][j][e];
        if(DUMP)dump[(size_t)s*tiles+tile]=fold;
        int w=s%16;transcript[w]=((transcript[w]<<13)|(transcript[w]>>19))^fold;
    }
    cp_narrow_jackpot(transcript,jp,jackpot,row0+wm*64+gid,col0+wn*32+2*t);
}

// Also compare the higher absolute throughput of block-scaled FP4. Although
// this class is reduced on the D SKU, its remaining throughput may still beat
// another class. All scales are exactly 1 and all input integers are exact.
template<bool NV> __device__ __forceinline__ void cp_block4_mma(float* d,const unsigned* a,const unsigned* b) {
#if __CUDA_ARCH__ >= 1200
    unsigned short zero=0;
    if constexpr(NV) {
        unsigned scale=0x38383838u;
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%11,%12},%13,{%14,%15};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
              "r"(scale),"h"(zero),"h"(zero),"r"(scale),"h"(zero),"h"(zero));
    } else {
        unsigned scale=0x7f7f7f7fu;
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%11,%12},%13,{%14,%15};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
              "r"(scale),"h"(zero),"h"(zero),"r"(scale),"h"(zero),"h"(zero));
    }
#endif
}
template<bool NV,bool DUMP> __global__ __launch_bounds__(256) void cp_block4_kernel(
    const unsigned char* A,const unsigned char* B,int m,int n,int rp,int cp,bool step,
    uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot) {
    __shared__ __align__(16) unsigned char sa[128][48],sb[128][48];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,wm=warp/4,wn=warp%4,gid=lane/4,t=lane%4;
    unsigned linear=blockIdx.y*gridDim.x+blockIdx.x,tr=linear%gridDim.y,tc=linear/gridDim.y;
    int row0=(rp+tr)*128,col0=(cp+tc)*128;
    size_t tile=((size_t)tr*gridDim.x+tc)*256+tid;
    float fc[4][4][4];uint32_t transcript[16]={};
    #pragma unroll
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fc[i][j][e]=12582912.0f;
    #pragma unroll 1
    for(int s=0;s<32;++s) {
        #pragma unroll
        for(int panel=0;panel<2;++panel) {
            int k0=s*128+panel*64,rr=tid/2,kk=(tid%2)*16;
            *(int4*)&sa[rr][kk]=*(const int4*)(A+cp_tensor_index(row0+rr,k0+2*kk,m,step)/2);
            *(int4*)&sb[rr][kk]=*(const int4*)(B+cp_tensor_index(col0+rr,k0+2*kk,n,step)/2);
            __syncthreads();
            unsigned ar[4][4],br[4][2];
            #pragma unroll
            for(int i=0;i<4;++i) {
                int r=wm*64+i*16+gid,c=wn*32+i*8+gid;
                ar[i][0]=cp_ld32(&sa[r][4*t]);ar[i][1]=cp_ld32(&sa[r+8][4*t]);
                ar[i][2]=cp_ld32(&sa[r][4*t+16]);ar[i][3]=cp_ld32(&sa[r+8][4*t+16]);
                br[i][0]=cp_ld32(&sb[c][4*t]);br[i][1]=cp_ld32(&sb[c][4*t+16]);
            }
            #pragma unroll
            for(int i=0;i<4;++i)
                #pragma unroll
                for(int j=0;j<4;++j)cp_block4_mma<NV>(fc[i][j],ar[i],br[j]);
            __syncthreads();
        }
        uint32_t fold=0;
        #pragma unroll
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fold^=__float_as_uint(fc[i][j][e]);
        fold=(uint32_t)((int32_t)(fold<<10)>>10);
        if(DUMP)dump[(size_t)s*tiles+tile]=fold;
        int w=s%16;transcript[w]=((transcript[w]<<13)|(transcript[w]>>19))^fold;
    }
    cp_narrow_jackpot(transcript,jp,jackpot,row0+wm*64+gid,col0+wn*32+2*t);
}
