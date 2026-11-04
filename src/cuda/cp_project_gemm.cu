#include "cp_narrow_kernel.cuh"
#include <cstdio>
// Reassociate every required prefix exactly, not just the final GEMM:
// B'[c,k]=E[c,p(k)]-E[c,q(k)], C_s=(A'[:128(s+1)] S_s) E^T.
// B's 4096-wide dense read becomes a reusable 128-wide read.
namespace {
struct Cache {void *a=nullptr,*b=nullptr;int* error=nullptr;size_t as=0,bs=0;
    int mode=-1,step=-1;bool va=false,vb=false,bada=false,badb=false;const int8_t* source=nullptr;};
Cache caches[16];const uint32_t* pair_ptr[16]={};const int8_t* ebr_ptr[16]={};
__device__ unsigned short bf16(float f){unsigned u=__float_as_uint(f);return (unsigned short)((u+0x7fffu+((u>>16)&1u))>>16);}
__global__ void project_a(const int8_t* a,void* out,int m,bool step,const uint32_t* pairs,int mode,int* error) {
    __shared__ int prefix[128],bounds[128];int row=blockIdx.x,t=threadIdx.x;prefix[t]=0;__syncthreads();
    const size_t plane=(size_t)32*m*128;
    for(int s=0;s<32;++s){int k=s*128+t,v=a[cp_tensor_index(row,k,m,step)];
        unsigned p=pairs[2*k],q=pairs[2*k+1];
        if(p>=128||q>=128){atomicExch(error,1);p=q=0;}
        atomicAdd(&prefix[p],v);atomicAdd(&prefix[q],-v);__syncthreads();
        int x=prefix[t];size_t i=((size_t)s*m+row)*128+t;
        if(x<-2048||x>2048)atomicExch(error,1);
        if(mode==16){unsigned short hi=bf16((float)x);float hf=__uint_as_float((unsigned)hi<<16);
            int lo=x-(int)hf;unsigned short lb=bf16((float)lo);
            if(__uint_as_float((unsigned)lb<<16)!=(float)lo)atomicExch(error,1);
            ((unsigned short*)out)[i]=hi;((unsigned short*)out)[plane+i]=lb;
            bounds[t]=(abs((int)hf)+abs(lo))*47;
        }else{bounds[t]=abs(x)*47;
            if(mode==17)((float*)out)[i]=(float)x;else ((__half*)out)[i]=__int2half_rn(x);}
        __syncthreads();
        for(int stride=64;stride;stride/=2){if(t<stride)bounds[t]+=bounds[t+stride];__syncthreads();}
        if(t==0&&bounds[0]>=4194304)atomicExch(error,1);
        __syncthreads();
    }
}
__global__ void project_b(const int8_t* e,void* out,size_t count,int mode,int* error){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;int x=e[i];
    if(x<-16||x>47)atomicExch(error,1);
    if(mode==17)((float*)out)[i]=(float)x;
    else ((unsigned short*)out)[i]=mode==16?bf16((float)x):__half_as_ushort(__int2half_rn(x));
}
template<int MODE,bool DUMP> __global__ __launch_bounds__(256) void kernel(const void* av,const void* bv,int m,int rp,int cp,
    uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot){
    using T=typename std::conditional<MODE==17,unsigned,unsigned short>::type;
    __shared__ __align__(16) T sa[128][MODE==17?36:40],sb[128][MODE==17?36:40];
    const T* A=(const T*)av;const T* B=(const T*)bv;
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,wm=warp/4,wn=warp%4,gid=lane/4,t=lane%4;
    unsigned linear=blockIdx.y*gridDim.x+blockIdx.x,tr=linear%gridDim.y,tc=linear/gridDim.y;
    int row0=(rp+tr)*128,col0=(cp+tc)*128;size_t tile=((size_t)tr*gridDim.x+tc)*256+tid;
    uint32_t transcript[16]={};
    #pragma unroll 1
    for(int s=0;s<32;++s){float fc[4][4][4];
        #pragma unroll
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fc[i][j][e]=12582912.0f;
        #pragma unroll
        for(int plane=0;plane<(MODE==16?2:1);++plane){
            #pragma unroll
            for(int panel=0;panel<4;++panel){int rr=tid/2,kk=(tid%2)*16,k=panel*32+kk;
                const T* ap=A+((size_t)plane*32*m+s*m+row0+rr)*128+k;
                const T* bp=B+(size_t)(col0+rr)*128+k;
                #pragma unroll
                for(int q=0;q<sizeof(T);++q){((int4*)&sa[rr][kk])[q]=((const int4*)ap)[q];((int4*)&sb[rr][kk])[q]=((const int4*)bp)[q];}
                __syncthreads();
                #pragma unroll
                for(int sub=0;sub<(MODE==17?4:2);++sub){unsigned ar[4][4],br[4][2];
                    #pragma unroll
                    for(int i=0;i<4;++i){int r=wm*64+i*16+gid,c=wn*32+i*8+gid;
                        if constexpr(MODE==17){int z=sub*8+t;ar[i][0]=sa[r][z];ar[i][1]=sa[r+8][z];ar[i][2]=sa[r][z+4];ar[i][3]=sa[r+8][z+4];br[i][0]=sb[c][z];br[i][1]=sb[c][z+4];}
                        else{int z=sub*16+2*t;ar[i][0]=cp_ld32(&sa[r][z]);ar[i][1]=cp_ld32(&sa[r+8][z]);ar[i][2]=cp_ld32(&sa[r][z+8]);ar[i][3]=cp_ld32(&sa[r+8][z+8]);br[i][0]=cp_ld32(&sb[c][z]);br[i][1]=cp_ld32(&sb[c][z+8]);}}
                    #pragma unroll
                    for(int i=0;i<4;++i)for(int j=0;j<4;++j){if constexpr(MODE==17)cp_tmma(fc[i][j],ar[i],br[j]);else if constexpr(MODE==16)cp_bmma(fc[i][j],ar[i],br[j]);else cp_hmma(fc[i][j],ar[i],br[j]);}
                }__syncthreads();
            }
        }
        uint32_t fold=0;
        #pragma unroll
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fold^=__float_as_uint(fc[i][j][e]);
        // Bias lives at the midpoint of the [2^23,2^24) binade; ULP=1.
        // XOR of an even 64 encodings, followed by signed 23-bit recovery.
        fold=(uint32_t)((int32_t)(fold<<9)>>9);
        if(DUMP)dump[(size_t)s*tiles+tile]=fold;
        int w=s%16;transcript[w]=((transcript[w]<<13)|(transcript[w]>>19))^fold;
    }
    cp_narrow_jackpot(transcript,jp,jackpot,row0+wm*64+gid,col0+wn*32+2*t);
}
template<int MODE>void run(const Cache& c,int m,int rp,int cp,int rb,int cb,uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot){
    if(dump)kernel<MODE,true><<<dim3(cb,rb),256>>>(c.a,c.b,m,rp,cp,dump,tiles,jp,jackpot);
    else kernel<MODE,false><<<dim3(cb,rb),256>>>(c.a,c.b,m,rp,cp,nullptr,tiles,jp,jackpot);
}
}
void cp_project_set_noise(int dev,const uint32_t* pairs,const int8_t* ebr){if(dev>=0&&dev<16){
    if(pair_ptr[dev]!=pairs||ebr_ptr[dev]!=ebr)cp_project_invalidate(dev,0);pair_ptr[dev]=pairs;ebr_ptr[dev]=ebr;}}
void cp_project_invalidate(int dev,int only_a){if(dev>=0&&dev<16){caches[dev].va=false;if(!only_a)caches[dev].vb=false;}}
void cp_project_release(int dev){if(dev<0||dev>=16)return;cudaSetDevice(dev);auto& c=caches[dev];if(c.a)cudaFree(c.a);if(c.b)cudaFree(c.b);if(c.error)cudaFree(c.error);c=Cache{};}
int cp_project_period_batch(int dev,int mode,const int8_t* a,int m,int n,int rp,int cp,int rb,int cb,int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp){
    if(!pair_ptr[dev]||!ebr_ptr[dev])return 1;
    auto& c=caches[dev];size_t as=(size_t)32*m*128*(mode==16||mode==17?4:2),bs=(size_t)n*128*(mode==17?4:2);
    if(c.as!=as||c.bs!=bs||c.mode!=mode)cp_project_release(dev);
    if(!c.a){if(cudaMalloc(&c.a,as)!=cudaSuccess)return -1;c.as=as;}
    if(!c.b){if(cudaMalloc(&c.b,bs)!=cudaSuccess)return -1;c.bs=bs;}
    if(!c.error&&cudaMalloc(&c.error,4)!=cudaSuccess)return -1;
    if(!c.vb){cudaMemset(c.error,0,4);project_b<<<(unsigned)(((size_t)n*128+255)/256),256>>>(ebr_ptr[dev],c.b,(size_t)n*128,mode,c.error);
        int bad=0;if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(&bad,c.error,4,cudaMemcpyDeviceToHost)!=cudaSuccess)return -1;c.badb=bad!=0;c.vb=true;}
    if(!c.va||c.source!=a||c.step!=step){cudaMemset(c.error,0,4);project_a<<<m,128>>>(a,c.a,m,step!=0,pair_ptr[dev],mode,c.error);
        int bad=0;if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(&bad,c.error,4,cudaMemcpyDeviceToHost)!=cudaSuccess)return -1;c.bada=bad!=0;c.va=true;c.source=a;c.step=step;c.mode=mode;}
    if(c.bada||c.badb)return 1;
    CpCutlassJackpotLaunch j={};if(jp)j=*jp;
    switch(mode){case 15:run<15>(c,m,rp,cp,rb,cb,dump,tiles,j,jp!=nullptr);break;case 16:run<16>(c,m,rp,cp,rb,cb,dump,tiles,j,jp!=nullptr);break;case 17:run<17>(c,m,rp,cp,rb,cb,dump,tiles,j,jp!=nullptr);break;default:return -1;}
    auto err=cudaGetLastError();if(err!=cudaSuccess){fprintf(stderr,"[project] %s\n",cudaGetErrorString(err));return -1;}return 0;
}
