#include "cp_tensor_kernel.cuh"
#include <stdio.h>
static int tensor_mode=0;
struct HalfCache { __half *a=nullptr,*b=nullptr; size_t as=0,bs=0; const int8_t *srca=nullptr,*srcb=nullptr; bool valid=false; int step=-1; };
static HalfCache half_cache[16];
__global__ void convert_half(const int8_t* src,__half* dst,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)dst[i]=__int2half_rn((int)src[i]);
}
void cp_tensor_invalidate_inputs(int dev) {if(dev>=0&&dev<16)half_cache[dev].valid=false;}
void cp_tensor_release(int dev) {
    if(dev<0||dev>=16)return;
    cudaSetDevice(dev);auto& c=half_cache[dev];
    if(c.a)cudaFree(c.a);if(c.b)cudaFree(c.b);c=HalfCache{};
}
static int prepare_half(int dev,const int8_t* a,const int8_t* b,int m,int n,int step) {
    auto& c=half_cache[dev];size_t as=(size_t)m*K_DIM,bs=(size_t)n*K_DIM;
    if(c.as!=as||c.bs!=bs){cp_tensor_release(dev);}
    if(!c.a){if(cudaMalloc(&c.a,as*2)!=cudaSuccess)return -1;c.as=as;}
    if(!c.b){if(cudaMalloc(&c.b,bs*2)!=cudaSuccess)return -1;c.bs=bs;}
    if(!c.valid||c.srca!=a||c.srcb!=b||c.step!=step) {
        convert_half<<<(unsigned)((as+255)/256),256>>>(a,c.a,as);
        convert_half<<<(unsigned)((bs+255)/256),256>>>(b,c.b,bs);
        if(cudaGetLastError()!=cudaSuccess)return -1;
        c.srca=a;c.srcb=b;c.step=step;c.valid=true;
    }
    return 0;
}
int cp_tensor_set_mode(int mode) {if(mode<0||mode>2)return -1;tensor_mode=mode;return 0;}
int cp_tensor_get_mode(void) {return tensor_mode;}
int cp_cutlass_device_ok(int dev) {cudaDeviceProp p;return cudaGetDeviceProperties(&p,dev)==cudaSuccess&&p.major>=8;}
size_t cp_cutlass_tiles_per_batch(int r,int c) {return (size_t)r*c*256;}
size_t cp_cutlass_tile_xor_bytes(int r,int c) {return cp_cutlass_tiles_per_batch(r,c)*32*sizeof(uint32_t);}
template<int MODE> static void launch(const int8_t* a,const int8_t* b,const __half* ah,const __half* bh,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp) {
    CpCutlassJackpotLaunch p={};if(jp)p=*jp;
    if(dump) cp_tensor_kernel<MODE,true><<<dim3(cb,rb),256>>>(a,b,ah,bh,m,n,rp,cp,step!=0,dump,tiles,p,jp!=nullptr);
    else cp_tensor_kernel<MODE,false><<<dim3(cb,rb),256>>>(a,b,ah,bh,m,n,rp,cp,step!=0,nullptr,tiles,p,jp!=nullptr);
}
int cp_cutlass_period_batch(int dev,const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp) {
    if(dev<0||dev>=16||rb<=0||cb<=0||rp<0||cp<0||(rp+rb)*128>m||(cp+cb)*128>n||tiles!=(size_t)rb*cb*256||cudaSetDevice(dev)!=cudaSuccess)return -1;
    if(tensor_mode!=0 && prepare_half(dev,a,b,m,n,step)!=0)return -1;
    const __half *ah=half_cache[dev].a,*bh=half_cache[dev].b;
    switch(tensor_mode) {
    case 0:launch<0>(a,b,ah,bh,m,n,rp,cp,rb,cb,step,dump,tiles,jp);break;
    case 1:launch<1>(a,b,ah,bh,m,n,rp,cp,rb,cb,step,dump,tiles,jp);break;
    case 2:launch<2>(a,b,ah,bh,m,n,rp,cp,rb,cb,step,dump,tiles,jp);break;
    }
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){fprintf(stderr,"[tensor] %s\n",cudaGetErrorString(err));return -1;}
    return 0;
}
