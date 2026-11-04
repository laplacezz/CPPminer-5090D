#include "cp_tensor_kernel.cuh"
#include <stdio.h>
static int tensor_mode=0;
int cp_tensor_set_mode(int mode) {if(mode<0||mode>2)return -1;tensor_mode=mode;return 0;}
int cp_tensor_get_mode(void) {return tensor_mode;}
int cp_cutlass_device_ok(int dev) {cudaDeviceProp p;return cudaGetDeviceProperties(&p,dev)==cudaSuccess&&p.major>=8;}
size_t cp_cutlass_tiles_per_batch(int r,int c) {return (size_t)r*c*256;}
size_t cp_cutlass_tile_xor_bytes(int r,int c) {return cp_cutlass_tiles_per_batch(r,c)*32*sizeof(uint32_t);}
template<int MODE> static void launch(const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp) {
    CpCutlassJackpotLaunch p={};if(jp)p=*jp;
    if(dump) cp_tensor_kernel<MODE,true><<<dim3(cb,rb),256>>>(a,b,m,n,rp,cp,step!=0,dump,tiles,p,jp!=nullptr);
    else cp_tensor_kernel<MODE,false><<<dim3(cb,rb),256>>>(a,b,m,n,rp,cp,step!=0,nullptr,tiles,p,jp!=nullptr);
}
int cp_cutlass_period_batch(int dev,const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp) {
    if(rb<=0||cb<=0||rp<0||cp<0||(rp+rb)*128>m||(cp+cb)*128>n||tiles!=(size_t)rb*cb*256||cudaSetDevice(dev)!=cudaSuccess)return -1;
    switch(tensor_mode) {
    case 0:launch<0>(a,b,m,n,rp,cp,rb,cb,step,dump,tiles,jp);break;
    case 1:launch<1>(a,b,m,n,rp,cp,rb,cb,step,dump,tiles,jp);break;
    case 2:launch<2>(a,b,m,n,rp,cp,rb,cb,step,dump,tiles,jp);break;
    }
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){fprintf(stderr,"[tensor] %s\n",cudaGetErrorString(err));return -1;}
    return 0;
}
