#pragma once
#include "cp_narrow_kernel.cuh"

// FMT 0/9: base-8 Karatsuba (3 products), FMT 7: base-16 INT4 (4),
// other formats: LUT-selected base-4 Karatsuba (6 instead of 9).
template<int FMT,bool HALFACC,bool NV,bool DUMP>
__global__ __launch_bounds__(256) void cp_radix_kernel(
    const unsigned char* A,const unsigned char* B,int m,int n,int rp,int cp,bool step,
    uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot) {
    constexpr bool PACKED=FMT==7||FMT==8;
    constexpr int TERMS=(FMT==0||FMT==9)?3:FMT==7?4:6;
    constexpr int KT=FMT==8?64:32,BYTES=PACKED?KT/2:KT;
    constexpr unsigned MAGIC=FMT==9?0x4a400000u:0x4b400000u;
    constexpr int w3[3]={56,-7,8},w4[4]={256,16,16,1},w6[6]={-19,-52,176,4,16,64};
    __shared__ __align__(16) unsigned char sa[128][48],sb[128][48];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,wm=warp/4,wn=warp%4,gid=lane/4,t=lane%4;
    unsigned linear=blockIdx.y*gridDim.x+blockIdx.x,tr=linear%gridDim.y,tc=linear/gridDim.y;
    int row0=(rp+tr)*128,col0=(cp+tc)*128;
    size_t tile=((size_t)tr*gridDim.x+tc)*256+tid;
    size_t astride=(size_t)m*K_DIM/(PACKED?2:1),bstride=(size_t)n*K_DIM/(PACKED?2:1);
    int accum[64]={};uint32_t transcript[16]={};
    #pragma unroll 1
    for(int s=0;s<32;++s) {
        #pragma unroll 1
        for(int term=0;term<TERMS;++term) {
            int pa=term,pb=term;
            if constexpr(FMT==7){pa=term/2;pb=term%2;}
            int weight=TERMS==3?w3[term]:TERMS==4?w4[term]:w6[term];
            float fc[4][4][4];int ip[8][4][2]={};
            #pragma unroll
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fc[i][j][e]=__uint_as_float(MAGIC);
            #pragma unroll
            for(int panel=0;panel<128/KT;++panel) {
                int k0=s*128+panel*KT,rr=tid/2,kk=(tid%2)*(BYTES/2);
                size_t ai=cp_tensor_index(row0+rr,k0+kk*(PACKED?2:1),m,step)/(PACKED?2:1);
                size_t bi=cp_tensor_index(col0+rr,k0+kk*(PACKED?2:1),n,step)/(PACKED?2:1);
                if constexpr(BYTES==32) {
                    *(int4*)&sa[rr][kk]=*(const int4*)(A+(size_t)pa*astride+ai);
                    *(int4*)&sb[rr][kk]=*(const int4*)(B+(size_t)pb*bstride+bi);
                } else {
                    *(uint2*)&sa[rr][kk]=*(const uint2*)(A+(size_t)pa*astride+ai);
                    *(uint2*)&sb[rr][kk]=*(const uint2*)(B+(size_t)pb*bstride+bi);
                }
                __syncthreads();
                if constexpr(FMT==7) {
                    unsigned ar[8],br[4];
                    #pragma unroll
                    for(int i=0;i<8;++i)ar[i]=cp_ld32(&sa[wm*64+i*8+gid][4*t]);
                    #pragma unroll
                    for(int j=0;j<4;++j)br[j]=cp_ld32(&sb[wn*32+j*8+gid][4*t]);
                    #pragma unroll
                    for(int i=0;i<8;++i)for(int j=0;j<4;++j)
                        asm volatile("mma.sync.aligned.row.col.m8n8k32.s32.s4.s4.s32 {%0,%1},{%2},{%3},{%0,%1};"
                            : "+r"(ip[i][j][0]),"+r"(ip[i][j][1]):"r"(ar[i]),"r"(br[j]));
                } else {
                    unsigned ar[4][4],br[4][2],hc[4][4][2]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        int r=wm*64+i*16+gid,c=wn*32+i*8+gid;
                        ar[i][0]=cp_ld32(&sa[r][4*t]);ar[i][1]=cp_ld32(&sa[r+8][4*t]);
                        ar[i][2]=cp_ld32(&sa[r][4*t+16]);ar[i][3]=cp_ld32(&sa[r+8][4*t+16]);
                        br[i][0]=cp_ld32(&sb[c][4*t]);br[i][1]=cp_ld32(&sb[c][4*t+16]);
                    }
                    #pragma unroll
                    for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
                        if constexpr(HALFACC)cp_low_half_mma<FMT>(hc[i][j],ar[i],br[j]);
                        else if constexpr(FMT==8)cp_block4_mma<NV>(fc[i][j],ar[i],br[j]);
                        else cp_lmma<FMT>(fc[i][j],ar[i],br[j]);
                    }
                    if constexpr(HALFACC) {
                        // One K=32 panel: <=32*6*6=1152, exact in FP16.
                        #pragma unroll
                        for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e) {
                            unsigned short bits=(unsigned short)(hc[i][j][e/2]>>(16*(e%2)));
                            accum[(i*4+j)*4+e]+=weight*__half2int_rn(__ushort_as_half(bits));
                        }
                    }
                }
                __syncthreads();
            }
            if constexpr(FMT==7) {
                #pragma unroll
                for(int i=0;i<8;++i)for(int j=0;j<4;++j)for(int e=0;e<2;++e)accum[(i*4+j)*2+e]+=weight*ip[i][j][e];
            } else if constexpr(!HALFACC) {
                #pragma unroll
                for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)
                    accum[(i*4+j)*4+e]+=weight*(__float_as_int(fc[i][j][e])-(int)MAGIC);
            }
        }
        uint32_t fold=0;
        #pragma unroll
        for(int e=0;e<64;++e)fold^=(uint32_t)accum[e];
        if(DUMP)dump[(size_t)s*tiles+tile]=fold;
        int w=s%16;transcript[w]=((transcript[w]<<13)|(transcript[w]>>19))^fold;
    }
    cp_narrow_jackpot(transcript,jp,jackpot,row0+wm*64+gid,col0+wn*32+2*t);
}
