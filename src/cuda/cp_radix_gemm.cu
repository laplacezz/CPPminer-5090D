#include "cp_radix_kernel.cuh"
#include "cp_radix_lut.h"
#include <cstdio>
namespace {
struct Cache {
    unsigned char *a=nullptr,*b=nullptr; int *error=nullptr;
    size_t as=0,bs=0; int mode=-1,step=-1; bool va=false,vb=false,bad=false;
    const int8_t *sa=nullptr,*sb=nullptr;
};
Cache caches[16];
__device__ __constant__ int8_t radix_lut[129][6];
__device__ unsigned encode(int v,int fmt) {
    unsigned neg=v<0; int a=v<0?-v:v;
    if(fmt==7)return v&15;
    if(fmt==2||fmt==8) {
        unsigned code=a==1?2:a==2?4:a==3?5:a==4?6:a==6?7:0;
        code|=neg?8:0;return fmt==2?code<<2:code;
    }
    if(fmt==9) {
        // v denotes twice the represented value, including half-integers.
        const unsigned bits[13]={0,4,8,12,16,18,20,22,24,25,26,27,28};
        return bits[a]|(neg?32:0);
    }
    if(!a)return 0;
    int e=31-__clz(a),mant=fmt==0?3:2,bias=fmt==0?7:3;
    return ((e+bias)<<mant)|((a-(1<<e))<<(mant-e))|(neg?(fmt==0?128:32):0);
}
__device__ int digit(int v,int plane,int fmt) {
    if(fmt==0||fmt==9){int h=(v+68)/8-8,l=v-8*h;return plane==0?h:plane==1?l:h+l;}
    if(fmt==7){int h=(v+72)/16-4;return plane==0?h:v-16*h;}
    return radix_lut[v+64][plane];
}
__global__ void convert(const int8_t* src,unsigned char* dst,size_t count,int fmt,int planes,int* error) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    bool packed=fmt==7||fmt==8;size_t stride=packed?count/2:count;
    if(i>=stride)return;
    int a=src[packed?2*i:i],b=packed?src[2*i+1]:0;
    if(a<-64||a>64||b<-64||b>64){atomicExch(error,1);a=b=0;}
    for(int p=0;p<planes;++p){unsigned x=encode(digit(a,p,fmt),fmt);
        if(packed)x|=encode(digit(b,p,fmt),fmt)<<4;
        dst[(size_t)p*stride+i]=(unsigned char)x;}
}
int format(int mode){switch(mode){case 5:return 0;case 6:case 11:return 1;case 7:case 10:return 2;
    case 8:return 7;case 12:return 6;case 13:case 14:return 8;case 18:return 9;}return -1;}
template<int F,bool H=false,bool NV=false> void run(const Cache& c,int m,int n,int rp,int cp,int rb,int cb,int step,
    uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot) {
    if(dump)cp_radix_kernel<F,H,NV,true><<<dim3(cb,rb),256>>>(c.a,c.b,m,n,rp,cp,step!=0,dump,tiles,jp,jackpot);
    else cp_radix_kernel<F,H,NV,false><<<dim3(cb,rb),256>>>(c.a,c.b,m,n,rp,cp,step!=0,nullptr,tiles,jp,jackpot);
}
}
void cp_radix_invalidate(int dev,int only_a){if(dev>=0&&dev<16){caches[dev].va=false;if(!only_a)caches[dev].vb=false;}}
void cp_radix_release(int dev){if(dev<0||dev>=16)return;cudaSetDevice(dev);auto& c=caches[dev];
    if(c.a)cudaFree(c.a);if(c.b)cudaFree(c.b);if(c.error)cudaFree(c.error);c=Cache{};}
int cp_radix_period_batch(int dev,int mode,const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp) {
    int fmt=format(mode);if(fmt<0)return -1;cudaDeviceProp prop;cudaGetDeviceProperties(&prop,dev);
    if(fmt!=0&&fmt!=7&&prop.major<12)return 1;
    auto& c=caches[dev];int planes=(fmt==0||fmt==9)?3:fmt==7?2:6;
    size_t counta=(size_t)m*K_DIM,countb=(size_t)n*K_DIM;
    size_t as=counta*planes/((fmt==7||fmt==8)?2:1),bs=countb*planes/((fmt==7||fmt==8)?2:1);
    if(c.as!=as||c.bs!=bs||c.mode!=mode)cp_radix_release(dev);
    if(!c.a){if(cudaMalloc(&c.a,as)!=cudaSuccess)return -1;c.as=as;}
    if(!c.b){if(cudaMalloc(&c.b,bs)!=cudaSuccess)return -1;c.bs=bs;}
    if(!c.error&&cudaMalloc(&c.error,4)!=cudaSuccess)return -1;
    bool ca=!c.va||c.sa!=a||c.step!=step,cbv=!c.vb||c.sb!=b||c.step!=step;
    if(c.bad){ca=cbv=true;}
    if(ca||cbv){
        if(cudaMemcpyToSymbol(radix_lut,cp_radix_lut,sizeof(cp_radix_lut))!=cudaSuccess)return -1;
        cudaMemset(c.error,0,4);
        if(ca)convert<<<(unsigned)((counta+255)/256),256>>>(a,c.a,counta,fmt,planes,c.error);
        if(cbv)convert<<<(unsigned)((countb+255)/256),256>>>(b,c.b,countb,fmt,planes,c.error);
        int error=0;if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(&error,c.error,4,cudaMemcpyDeviceToHost)!=cudaSuccess)return -1;
        c.bad=error!=0;c.va=c.vb=true;c.sa=a;c.sb=b;c.step=step;c.mode=mode;
    }
    if(c.bad)return 1; // Exact INT8 fallback; never clamp real mining operands.
    CpCutlassJackpotLaunch j={};if(jp)j=*jp;
    switch(mode){case 5:run<0>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 6:run<1>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 7:run<2>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 8:run<7>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 10:run<2,true>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 11:run<1,true>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 12:run<6>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 13:run<8>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 14:run<8,false,true>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;
    case 18:run<9>(c,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);break;}
    auto err=cudaGetLastError();if(err!=cudaSuccess){fprintf(stderr,"[radix] %s\n",cudaGetErrorString(err));return -1;}return 0;
}
