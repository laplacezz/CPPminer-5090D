#include "cp_tensor_kernel.cuh"
#include <stdio.h>
static int tensor_mode=0;
static int positive_a=0;
static int raw_test=0;
void cp_tensor_set_raw_test(int on){raw_test=on?1:0;}
int cp_tensor_raw_test(void){return raw_test;}
void cp_tensor_set_positive_a(int on){positive_a=on==2?2:on?1:0;}
int cp_tensor_positive_a(void){return positive_a;}
int cp_tensor_signal_tag(void){return tensor_mode>=3||positive_a==2?3:positive_a?2:0;}
static void warn_fallback(){static unsigned warned=0;unsigned flag=1u<<tensor_mode;
    if(!(warned&flag)){fprintf(stderr,"[tensor] mode %d failed exact range/metadata eligibility; using exact INT8 fallback\n",tensor_mode);warned|=flag;}}
struct HalfCache { __half *a=nullptr,*b=nullptr; size_t as=0,bs=0; const int8_t *srca=nullptr,*srcb=nullptr; bool valid=false; int step=-1; };
static HalfCache half_cache[16];
__global__ void convert_half(const int8_t* src,__half* dst,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)dst[i]=__int2half_rn((int)src[i]);
}
void cp_tensor_invalidate_a(int dev){if(dev>=0&&dev<16)half_cache[dev].srca=nullptr;cp_narrow_invalidate_a(dev);cp_radix_invalidate(dev,1);cp_project_invalidate(dev,1);}
void cp_tensor_invalidate_inputs(int dev) {if(dev>=0&&dev<16)half_cache[dev].valid=false;cp_narrow_invalidate(dev);cp_radix_invalidate(dev,0);cp_project_invalidate(dev,0);}
void cp_tensor_release(int dev) {
    if(dev<0||dev>=16)return;
    cp_narrow_release(dev);
    cp_radix_release(dev);
    cp_project_release(dev);
    cudaSetDevice(dev);auto& c=half_cache[dev];
    if(c.a)cudaFree(c.a);if(c.b)cudaFree(c.b);c=HalfCache{};
}
static int prepare_half(int dev,const int8_t* a,const int8_t* b,int m,int n,int step) {
    auto& c=half_cache[dev];size_t as=(size_t)m*K_DIM,bs=(size_t)n*K_DIM;
    if(c.as!=as||c.bs!=bs){cp_tensor_release(dev);}
    if(!c.a){if(cudaMalloc(&c.a,as*2)!=cudaSuccess)return -1;c.as=as;}
    if(!c.b){if(cudaMalloc(&c.b,bs*2)!=cudaSuccess)return -1;c.bs=bs;}
    bool all=!c.valid||c.step!=step;
    if(all||c.srca!=a)convert_half<<<(unsigned)((as+255)/256),256>>>(a,c.a,as);
    if(all||c.srcb!=b)convert_half<<<(unsigned)((bs+255)/256),256>>>(b,c.b,bs);
    if(cudaGetLastError()!=cudaSuccess)return -1;
    c.srca=a;c.srcb=b;c.step=step;c.valid=true;
    return 0;
}
int cp_tensor_set_mode(int mode) {if(mode<0||mode>21)return -1;tensor_mode=mode;return 0;}
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
    if(tensor_mode>=20){int status=cp_int8_shape_batch(tensor_mode,a,b,m,n,rp,cp,rb,cb,step,dump,tiles,jp);
        if(status<=0)return status;
        launch<0>(a,b,nullptr,nullptr,m,n,rp,cp,rb,cb,step,dump,tiles,jp);
        return cudaGetLastError()==cudaSuccess?0:-1;}
    if(tensor_mode>=15&&tensor_mode<=17) {
        int status=cp_project_period_batch(dev,tensor_mode,a,m,n,rp,cp,rb,cb,step,dump,tiles,jp);
        if(status<=0)return status;
        warn_fallback();
        launch<0>(a,b,nullptr,nullptr,m,n,rp,cp,rb,cb,step,dump,tiles,jp);
        return cudaGetLastError()==cudaSuccess?0:-1;
    }
    if(tensor_mode>=3){int status=cp_narrow_period_batch(dev,tensor_mode,a,b,m,n,rp,cp,rb,cb,step,dump,tiles,jp);
        if(status<=0)return status;
        warn_fallback();
        launch<0>(a,b,nullptr,nullptr,m,n,rp,cp,rb,cb,step,dump,tiles,jp);
        return cudaGetLastError()==cudaSuccess?0:-1;}
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
