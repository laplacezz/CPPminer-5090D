#include "cp_narrow_kernel.cuh"
#include <cstdio>
struct NarrowCache {
    void *a=nullptr,*b=nullptr;int* error=nullptr;
    size_t as=0,bs=0;int mode=-1,step=-1;bool valid=false;
    const int8_t *srca=nullptr,*srcb=nullptr;
};
static NarrowCache cache[16];
void cp_narrow_invalidate_a(int dev){if(dev>=0&&dev<16)cache[dev].srca=nullptr;}
void cp_narrow_invalidate(int dev){if(dev>=0&&dev<16)cache[dev].valid=false;}
void cp_narrow_release(int dev) {
    if(dev<0||dev>=16)return;cudaSetDevice(dev);auto& c=cache[dev];
    if(c.a)cudaFree(c.a);if(c.b)cudaFree(c.b);if(c.error)cudaFree(c.error);c=NarrowCache{};
}
static int format_for_mode(int mode) {
    switch(mode){case 3:return 5;case 4:return 3;case 5:return 0;case 6:return 1;
        case 7:return 2;case 8:return 7;case 9:return 4;case 10:return 2;case 11:return 1;case 12:return 6;case 13:case 14:return 8;}
    return -1;
}
__device__ unsigned narrow_bits(int v,int fmt) {
    unsigned neg=v<0;int a=v<0?-v:v;unsigned code=0;
    if(fmt==0){code=a==1?0x38:a==2?0x40:a==3?0x44:0;return code|(neg?0x80:0);}
    if(fmt==1||fmt==6){code=a==1?12:a==2?16:a==3?18:0;return code|(neg?32:0);}
    if(fmt==2){code=a==1?2:a==2?4:a==3?5:0;return (code|(neg?8:0))<<2;}
    if(fmt==4){code=a==1?0x3c00:a==2?0x4000:a==3?0x4200:0;return code|(neg?0x8000:0);}
    if(fmt==5){code=a==1?0x3f80:a==2?0x4000:a==3?0x4040:0;return code|(neg?0x8000:0);}
    return __float_as_uint((float)v);
}
__global__ void narrow_convert(const int8_t* src,void* dst,size_t count,int fmt,int* error) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(fmt==7||fmt==8) {
        if(i>=count/2)return;int a=src[2*i],b=src[2*i+1];
        if(a<-3||a>3||b<-3||b>3)atomicExch(error,1);
        unsigned x=fmt==7?(a&15):(narrow_bits(a,2)>>2),y=fmt==7?(b&15):(narrow_bits(b,2)>>2);
        ((unsigned char*)dst)[i]=(unsigned char)(x|(y<<4));return;
    }
    if(i>=count)return;int v=src[i];
    if(v<-3||v>3){atomicExch(error,1);v=0;}
    unsigned bits=narrow_bits(v,fmt);
    if(fmt==3)((unsigned*)dst)[i]=bits;
    else if(fmt==4||fmt==5)((unsigned short*)dst)[i]=(unsigned short)bits;
    else ((unsigned char*)dst)[i]=(unsigned char)bits;
}
static int prepare(int dev,int mode,const int8_t* a,const int8_t* b,int m,int n,int step) {
    int fmt=format_for_mode(mode);auto& c=cache[dev];
    size_t counta=(size_t)m*K_DIM,countb=(size_t)n*K_DIM;
    size_t bytes=fmt==3?4:(fmt==4||fmt==5)?2:1;
    size_t as=fmt>=7?counta/2:counta*bytes,bs=fmt>=7?countb/2:countb*bytes;
    if(c.as!=as||c.bs!=bs||c.mode!=mode)cp_narrow_release(dev);
    if(!c.a){if(cudaMalloc(&c.a,as)!=cudaSuccess)return -1;c.as=as;}
    if(!c.b){if(cudaMalloc(&c.b,bs)!=cudaSuccess)return -1;c.bs=bs;}
    if(!c.error&&cudaMalloc(&c.error,4)!=cudaSuccess)return -1;
    if(!c.valid||c.srca!=a||c.srcb!=b||c.step!=step||c.mode!=mode) {
        cudaMemset(c.error,0,4);
        bool all=!c.valid||c.step!=step||c.mode!=mode;
        if(all||c.srca!=a)narrow_convert<<<(unsigned)((counta+255)/256),256>>>(a,c.a,counta,fmt,c.error);
        if(all||c.srcb!=b)narrow_convert<<<(unsigned)((countb+255)/256),256>>>(b,c.b,countb,fmt,c.error);
        int error=0;
        if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(&error,c.error,4,cudaMemcpyDeviceToHost)!=cudaSuccess)return -1;
        if(error){fprintf(stderr,"[tensor] narrow mode requires exact input integers in [-3,3]; use the automatic binary A / zero-B generation\n");return -1;}
        c.valid=true;c.srca=a;c.srcb=b;c.step=step;c.mode=mode;
    }
    return 0;
}
template<int FMT,bool HALFACC=false> static void run(const NarrowCache& c,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot) {
    if(dump)cp_narrow_kernel<FMT,HALFACC,true><<<dim3(cb,rb),256>>>(c.a,c.b,m,n,rp,cp,step!=0,dump,tiles,jp,jackpot);
    else cp_narrow_kernel<FMT,HALFACC,false><<<dim3(cb,rb),256>>>(c.a,c.b,m,n,rp,cp,step!=0,nullptr,tiles,jp,jackpot);
}
int cp_narrow_period_batch(int dev,int mode,const int8_t* a,const int8_t* b,int m,int n,
    int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* j) {
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop,dev);
    if((mode==6||mode==7||mode>=10)&&prop.major<12){fprintf(stderr,"[tensor] this mode requires Blackwell SM120a\n");return -1;}
    if(prepare(dev,mode,a,b,m,n,step)!=0)return -1;
    const auto& c=cache[dev];CpCutlassJackpotLaunch jp={};if(j)jp=*j;
    switch(mode) {
    case 3:run<5>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 4:run<3>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 5:run<0>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 6:run<1>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 7:run<2>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 8:
        if(dump)cp_int4_kernel<true><<<dim3(cb,rb),256>>>((const unsigned char*)c.a,(const unsigned char*)c.b,m,n,rp,cp,step!=0,dump,tiles,jp,j!=nullptr);
        else cp_int4_kernel<false><<<dim3(cb,rb),256>>>((const unsigned char*)c.a,(const unsigned char*)c.b,m,n,rp,cp,step!=0,nullptr,tiles,jp,j!=nullptr);
        break;
    case 9:run<4>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 10:run<2,true>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 11:run<1,true>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 12:run<6>(c,m,n,rp,cp,rb,cb,step,dump,tiles,jp,j!=nullptr);break;
    case 13:
        if(dump)cp_block4_kernel<false,true><<<dim3(cb,rb),256>>>((const unsigned char*)c.a,(const unsigned char*)c.b,m,n,rp,cp,step!=0,dump,tiles,jp,j!=nullptr);
        else cp_block4_kernel<false,false><<<dim3(cb,rb),256>>>((const unsigned char*)c.a,(const unsigned char*)c.b,m,n,rp,cp,step!=0,nullptr,tiles,jp,j!=nullptr);
        break;
    case 14:
        if(dump)cp_block4_kernel<true,true><<<dim3(cb,rb),256>>>((const unsigned char*)c.a,(const unsigned char*)c.b,m,n,rp,cp,step!=0,dump,tiles,jp,j!=nullptr);
        else cp_block4_kernel<true,false><<<dim3(cb,rb),256>>>((const unsigned char*)c.a,(const unsigned char*)c.b,m,n,rp,cp,step!=0,nullptr,tiles,jp,j!=nullptr);
        break;
    default:return -1;
    }
    cudaError_t err=cudaGetLastError();if(err!=cudaSuccess){fprintf(stderr,"[tensor] %s\n",cudaGetErrorString(err));return -1;}return 0;
}
