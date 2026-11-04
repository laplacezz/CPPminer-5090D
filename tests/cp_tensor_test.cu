#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cp_cutlass.h"
#define CU(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
static uint32_t rng=12345;
static uint32_t next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static std::vector<uint32_t> reference(const std::vector<int8_t>& a,const std::vector<int8_t>& b,int dim,int rp,int cp) {
    std::vector<uint32_t> out(32*256,0);
    // Independent scalar INT64 dot product, including adversarial signed extremes.
    for(int row=0;row<128;++row) for(int col=0;col<128;++col) {
        const int wm=row/64,wn=col/32,gid=row%8,t=(col%8)/2;
        const int tid=(wm*4+wn)*32+gid*4+t;
        int64_t acc=0;
        for(int s=0;s<32;++s) {
            for(int k=s*128;k<(s+1)*128;++k)
                acc+=(int)a[(size_t)(rp*128+row)*4096+k]*(int)b[(size_t)(cp*128+col)*4096+k];
            out[s*256+tid]^=(uint32_t)(int32_t)acc;
        }
    }
    return out;
}
static std::vector<int8_t> pack(const std::vector<int8_t>& x,int dim) {
    std::vector<int8_t> p(x.size());
    for(int row=0;row<dim;++row)for(int k=0;k<4096;++k)
        p[((size_t)(k/128)*dim+row)*128+k%128]=x[(size_t)row*4096+k];
    return p;
}
int main(int argc,char**argv) {
    CU(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));
    printf("GPU %s cc%d.%d\n",prop.name,prop.major,prop.minor);
    if(!cp_cutlass_device_ok(0))return 2;
    bool exact=argc>1 && !strcmp(argv[1],"--bench-exact-large");
    bool narrow=argc>1 && (!strcmp(argv[1],"--bench-narrow") || !strcmp(argv[1],"--bench-narrow-large"));
    bool large=exact || (argc>1 && (!strcmp(argv[1],"--bench-large") || !strcmp(argv[1],"--bench-narrow-large")));
    bool bench=narrow || large || (argc>1 && !strcmp(argv[1],"--bench"));
    const int dim=large?131072:bench?8192:256;
    std::vector<int8_t>a((size_t)dim*4096),b(a.size());
    int8_t *da,*db;uint32_t*dd;
    CU(cudaMalloc(&da,a.size()));CU(cudaMalloc(&db,b.size()));CU(cudaMalloc(&dd,32*1024*sizeof(uint32_t)));
    for(auto&x:a)x=(int8_t)next();for(auto&x:b)x=(int8_t)next();
    uint32_t* pairs=nullptr;int8_t* ebr=nullptr;
    CU(cudaMalloc(&pairs,8192*sizeof(uint32_t)));CU(cudaMalloc(&ebr,(size_t)dim*128));
    if(!bench) {
        for(int pattern=0;pattern<7;++pattern) {
            if(pattern==1) {
                for(size_t i=0;i<a.size();++i){a[i]=(i%11==0)?127:-128;b[i]=(i%17==0)?-127:127;}
            } else if(pattern==2) {
                for(size_t i=0;i<a.size();++i){a[i]=(i%4096<2048)?127:-128;b[i]=(i%13==0)?-128:127;}
            }
            if(pattern==3){for(auto&x:a)x=(int8_t)((int)(next()%7)-3);for(auto&x:b)x=(int8_t)((int)(next()%7)-3);}
            if(pattern==4){for(auto&x:a)x=(int8_t)((int)(next()%128)-63);for(auto&x:b)x=(int8_t)((int)(next()%127)-63);}
            if(pattern==5||pattern==6){
                std::vector<uint32_t> pp(8192);std::vector<int8_t> ee((size_t)dim*128);
                for(int k=0;k<4096;++k){pp[2*k]=next()%128;pp[2*k+1]=(pp[2*k]+1+next()%127)%128;}
                if(pattern==6){for(int k=0;k<4096;++k){pp[2*k]=0;pp[2*k+1]=1;}}
                for(auto&x:ee)x=(int8_t)((int)(next()%64)-16);
                for(auto&x:a)x=pattern==6?64:(int8_t)((int)(next()%128)-63);
                for(int row=0;row<dim;++row)for(int k=0;k<4096;++k)
                    b[(size_t)row*4096+k]=ee[(size_t)row*128+pp[2*k]]-ee[(size_t)row*128+pp[2*k+1]];
                CU(cudaMemcpy(pairs,pp.data(),pp.size()*4,cudaMemcpyHostToDevice));
                CU(cudaMemcpy(ebr,ee.data(),ee.size(),cudaMemcpyHostToDevice));cp_project_set_noise(0,pairs,ebr);
            }else cp_project_set_noise(0,nullptr,nullptr);
            auto ref=reference(a,b,dim,1,1);
            for(int step=0;step<2;++step) {
                auto aa=step?pack(a,dim):a,bb=step?pack(b,dim):b;
                CU(cudaMemcpy(da,aa.data(),aa.size(),cudaMemcpyHostToDevice));
                CU(cudaMemcpy(db,bb.data(),bb.size(),cudaMemcpyHostToDevice));
                for(int mode=0;mode<22;++mode) {
                    if(prop.major<12&&(mode==6||mode==7||(mode>=10&&mode<=14)||mode==18))continue;
                    cp_tensor_set_mode(mode);
                    cp_tensor_invalidate_inputs(0);
                    if((pattern==5||pattern==6)&&mode>=15&&mode<=17){
                        int check=cp_project_period_batch(0,mode,da,dim,dim,1,1,1,1,step,dd,256,nullptr);
                        if(check!=(pattern==6?1:0)){printf("FAIL projection range/fallback pattern=%d mode=%d status=%d\n",pattern,mode,check);return 1;}
                    }
                    if(cp_cutlass_period_batch(0,da,db,dim,dim,1,1,1,1,step,dd,256,nullptr)!=0)return 2;
                    std::vector<uint32_t> got(ref.size());CU(cudaMemcpy(got.data(),dd,got.size()*4,cudaMemcpyDeviceToHost));
                    for(size_t i=0;i<ref.size();++i) if(got[i]!=ref[i]) {
                        printf("FAIL pattern=%d step=%d mode=%d milestone=%zu lane=%zu got=%08x expected=%08x\n",pattern,step,mode,i/256,i%256,got[i],ref[i]);return 1;
                    }
                    printf("PASS pattern=%d step=%d mode=%d 8192 milestone folds\n",pattern,step,mode);
                }
            }
        }
        for(auto&x:a)x=(int8_t)next();for(auto&x:b)x=(int8_t)next();
        for(int step=0;step<2;++step){
            cp_project_set_noise(0,nullptr,nullptr);
            auto aa=step?pack(a,dim):a,bb=step?pack(b,dim):b;
            CU(cudaMemcpy(da,aa.data(),aa.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(db,bb.data(),bb.size(),cudaMemcpyHostToDevice));
            std::vector<uint32_t> refs(32*1024);
            for(int r=0;r<2;++r)for(int c=0;c<2;++c){auto rr=reference(a,b,dim,r,c);
                for(int s=0;s<32;++s)for(int t=0;t<256;++t)refs[(size_t)s*1024+(r*2+c)*256+t]=rr[s*256+t];}
            for(int mode=20;mode<22;++mode){cp_tensor_set_mode(mode);cp_tensor_invalidate_inputs(0);
                if(cp_cutlass_period_batch(0,da,db,dim,dim,0,0,2,2,step,dd,1024,nullptr)!=0)return 2;
                std::vector<uint32_t> got(refs.size());CU(cudaMemcpy(got.data(),dd,got.size()*4,cudaMemcpyDeviceToHost));
                if(got!=refs){printf("FAIL multi-CTA step=%d mode=%d\n",step,mode);return 1;}
                printf("PASS multi-CTA step=%d mode=%d canonical 32768 folds\n",step,mode);}
        }
        printf("ALL EXACTNESS TESTS PASSED\n");
    } else {
        if(exact){
            for(auto&x:a)x=(int8_t)((int)(next()%128)-63);
            std::vector<uint32_t> pp(8192);std::vector<int8_t> ee((size_t)dim*128);
            for(int k=0;k<4096;++k){pp[2*k]=next()%128;pp[2*k+1]=(pp[2*k]+1+next()%127)%128;}
            for(auto&x:ee)x=(int8_t)((int)(next()%64)-16);
            for(int row=0;row<dim;++row)for(int k=0;k<4096;++k)b[(size_t)row*4096+k]=ee[(size_t)row*128+pp[2*k]]-ee[(size_t)row*128+pp[2*k+1]];
            CU(cudaMemcpy(pairs,pp.data(),pp.size()*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(ebr,ee.data(),ee.size(),cudaMemcpyHostToDevice));
            cp_project_set_noise(0,pairs,ebr);
        }
        cp_tensor_set_raw_test(narrow?1:0);
        if(narrow){for(auto&x:a)x=(int8_t)((int)(next()%7)-3);for(auto&x:b)x=(int8_t)((int)(next()%7)-3);}
        CU(cudaMemcpy(da,a.data(),a.size(),cudaMemcpyHostToDevice));CU(cudaMemcpy(db,b.data(),b.size(),cudaMemcpyHostToDevice));
        uint32_t *key;int *found,*rows,*cols;CU(cudaMalloc(&key,32));CU(cudaMemset(key,17,32));
        CU(cudaMalloc(&found,4));CU(cudaMalloc(&rows,4));CU(cudaMalloc(&cols,4));
        CpCutlassJackpotLaunch jp={};jp.d_a_key8=key;jp.d_found=found;jp.d_out_t_rows=rows;jp.d_out_t_cols=cols;
        cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));
        const int rb=large?32:64,cb=large?1024:64; const double mac=(double)rb*cb*128*128*4096;
        int selected=argc>2?atoi(argv[2]):-1;
        for(int mode=0;mode<22;++mode) {
            if(selected>=0&&mode!=selected)continue;
            if((!exact&&!narrow&&mode>=3)||(narrow&&mode>=15))continue;
            if(prop.major<12&&(mode==6||mode==7||(mode>=10&&mode<=14)||mode==18))continue;
            cp_tensor_set_mode(mode);std::vector<float>samples;
            auto unix_ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            printf("BENCH_BEGIN mode=%d unix_ms=%lld\n",mode,(long long)unix_ms);fflush(stdout);
            auto start=std::chrono::steady_clock::now();
            do {
                if(exact)cp_tensor_invalidate_a(0);
                CU(cudaMemset(found,0,4));CU(cudaEventRecord(begin));
                if(cp_cutlass_period_batch(0,da,db,dim,dim,0,0,rb,cb,0,nullptr,(size_t)rb*cb*256,&jp)!=0)return 2;
                CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));float ms;CU(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms);
            }while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<6.0);
            std::sort(samples.begin(),samples.end());float ms=samples[samples.size()/2];
            printf("BENCH mode=%d samples=%zu median_ms=%.4f GEMM+fold+hash_TMAC_s=%.4f %s (not pool hashrate)\n",mode,samples.size(),ms,mac/(ms*1e9),exact?"including repeated A preparation":"cached operands");fflush(stdout);
        }
    }
    CU(cudaFree(da));CU(cudaFree(db));CU(cudaFree(dd));return 0;
}
