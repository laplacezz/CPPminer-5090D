#include "cp_narrow_kernel.cuh"
// Keep each lane's exact 8x8 hash cells, but reshape the CTA to change reuse.
template<int TM,int TN> __device__ __forceinline__ void load_shape(int8_t* sa,int8_t* sb,
    const int8_t* a,const int8_t* b,int row0,int col0,int k,int m,int n,bool step){
    #pragma unroll
    for(int copy=threadIdx.x;copy<(TM+TN)*2;copy+=256){int r=copy/2,z=(copy%2)*16;
        if(r<TM)cp_async16(sa+r*48+z,a+cp_tensor_index(row0+r,k+z,m,step));
        else{r-=TM;cp_async16(sb+r*48+z,b+cp_tensor_index(col0+r,k+z,n,step));}}
    asm volatile("cp.async.commit_group;");
}
template<int TM,int TN,bool DUMP> __global__ __launch_bounds__(256) void kernel_shape(
    const int8_t* a,const int8_t* b,int m,int n,int rp,int cp,int cb,bool step,
    uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot){
    __shared__ __align__(16) int8_t sa[2][TM][48],sb[2][TN][48];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,gid=lane/4,t=lane%4;
    constexpr int NW=TN/32;int wm=warp/NW,wn=warp%NW;
    unsigned linear=blockIdx.y*gridDim.x+blockIdx.x,tr=linear%gridDim.y,tc=linear/gridDim.y;
    int row0=rp*128+tr*TM,col0=cp*128+tc*TN;
    int row=row0+wm*64+gid,col=col0+wn*32+2*t;
    size_t tile=((size_t)((row-rp*128)/128)*cb+(col-cp*128)/128)*256
        +((row%128/64)*4+(col%128/32))*32+gid*4+t;
    int acc[4][4][4]={};uint32_t transcript[16]={};
    load_shape<TM,TN>(&sa[0][0][0],&sb[0][0][0],a,b,row0,col0,0,m,n,step);
    #pragma unroll 1
    for(int s=0;s<32;++s){
        #pragma unroll
        for(int panel=0;panel<4;++panel){int k=s*128+panel*32,stage=panel%2;
            asm volatile("cp.async.wait_group 0;");__syncthreads();
            if(k+32<K_DIM)load_shape<TM,TN>(&sa[1-stage][0][0],&sb[1-stage][0][0],a,b,row0,col0,k+32,m,n,step);
            unsigned ar[4][4],br[4][2];
            #pragma unroll
            for(int i=0;i<4;++i){int r=wm*64+i*16+gid,c=wn*32+i*8+gid;
                ar[i][0]=cp_ld32(&sa[stage][r][t*4]);ar[i][1]=cp_ld32(&sa[stage][r+8][t*4]);
                ar[i][2]=cp_ld32(&sa[stage][r][t*4+16]);ar[i][3]=cp_ld32(&sa[stage][r+8][t*4+16]);
                br[i][0]=cp_ld32(&sb[stage][c][t*4]);br[i][1]=cp_ld32(&sb[stage][c][t*4+16]);}
            #pragma unroll
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)cp_imma(acc[i][j],ar[i],br[j]);
            __syncthreads();
        }
        uint32_t fold=0;
        #pragma unroll
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int e=0;e<4;++e)fold^=(uint32_t)acc[i][j][e];
        if(DUMP)dump[(size_t)s*tiles+tile]=fold;
        int w=s%16;transcript[w]=((transcript[w]<<13)|(transcript[w]>>19))^fold;
    }
    cp_narrow_jackpot(transcript,jp,jackpot,row,col);
}
template<int TM,int TN>void run_shape(const int8_t* a,const int8_t* b,int m,int n,int rp,int cp,int rb,int cb,
    int step,uint32_t* dump,size_t tiles,CpCutlassJackpotLaunch jp,bool jackpot){
    dim3 grid(cb*128/TN,rb*128/TM);
    if(dump)kernel_shape<TM,TN,true><<<grid,256>>>(a,b,m,n,rp,cp,cb,step!=0,dump,tiles,jp,jackpot);
    else kernel_shape<TM,TN,false><<<grid,256>>>(a,b,m,n,rp,cp,cb,step!=0,nullptr,tiles,jp,jackpot);
}
int cp_int8_shape_batch(int mode,const int8_t* a,const int8_t* b,int m,int n,int rp,int cp,int rb,int cb,
    int step,uint32_t* dump,size_t tiles,const CpCutlassJackpotLaunch* jp){
    if((mode==20&&rb%2)||(mode==21&&cb%2))return 1;
    CpCutlassJackpotLaunch j={};if(jp)j=*jp;
    if(mode==20)run_shape<256,64>(a,b,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);
    else run_shape<64,256>(a,b,m,n,rp,cp,rb,cb,step,dump,tiles,j,jp!=nullptr);
    return cudaGetLastError()==cudaSuccess?0:-1;
}
