// barra-llm — llama.cpp-Treiber, der die Chips des Pixel EXPLIZIT verteilt (barra-Prinzip):
//   Layer-Mathe: GPU (llama.cpp-Vulkan, -ngl) + CPU (Rest)
//   Token-Auswahl (greedy Argmax ueber die Logits): GXP-DSP via barra (Kernel ker_argmax) — oder CPU (--sampler cpu)
//   [TPU: kommt ueber das ggml-barra-Backend, Phase 2]
// Jeder DSP-Token wird gegen den CPU-Argmax geprueft (Mismatch-Zaehler), Zeiten je Stufe gemessen.
// Bau: NDK clang++ (Bionic) gegen libllama.a/libggml*.a (build-android-vulkan) + libbarra (NDK-Build von barra.c).
#include "llama.h"
#include "barra.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int cpu_argmax(const float* l, int n){ int b=0; for(int i=1;i<n;i++) if(l[i]>l[b]) b=i; return b; }

// Argmax auf dem GXP: buf=[N u32][float-bits...] -> Kernel schreibt idx nach buf[0]
static int dsp_argmax(const float* l, int n, std::vector<uint8_t>& io, double* ms){
  io.resize(4+4*(size_t)n);
  uint32_t N=(uint32_t)n; memcpy(io.data(),&N,4); memcpy(io.data()+4,l,4*(size_t)n);
  barra_op op{}; op.device=BARRA_DSP; op.dsp_func="argmax"; op.label="argmax";
  double t0=now_ms();
  int r=barra_run(&op,io.data(),(uint32_t)io.size(),io.data(),(uint32_t)io.size());
  if(ms)*ms=now_ms()-t0;
  if(r<4) return -1;
  int32_t idx; memcpy(&idx,io.data(),4); return idx;
}

int main(int argc,char**argv){
  std::setlocale(LC_NUMERIC,"C");
  std::string model_path, prompt="Erkläre in zwei Sätzen, was ein Pixel 8a ist.", sampler="cpu";   // Default CPU: GXP-Argmax raus aus dem heissen Pfad (21ms/Tok vs CPU 0.1ms, kein Nutzen)
  int ngl=99, n_predict=64; bool chat=true;
  int i=1;
  for(;i<argc;i++){
    if(!strcmp(argv[i],"-m")&&i+1<argc) model_path=argv[++i];
    else if(!strcmp(argv[i],"-n")&&i+1<argc) n_predict=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-ngl")&&i+1<argc) ngl=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--sampler")&&i+1<argc) sampler=argv[++i];
    else if(!strcmp(argv[i],"--raw")) chat=false;
    else break;
  }
  if(model_path.empty()){ fprintf(stderr,"usage: %s -m model.gguf [-n N] [-ngl L] [--sampler dsp|cpu] [--raw] [prompt]\n",argv[0]); return 1; }
  if(i<argc){ prompt=argv[i++]; for(;i<argc;i++){ prompt+=" "; prompt+=argv[i]; } }
  if(chat) prompt="<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n";

  ggml_backend_load_all();
  llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=ngl;
  llama_model* model=llama_model_load_from_file(model_path.c_str(),mp);
  if(!model){ fprintf(stderr,"Modell laedt nicht\n"); return 1; }
  const llama_vocab* vocab=llama_model_get_vocab(model);
  const int n_vocab=llama_vocab_n_tokens(vocab);
  const int n_prompt=-llama_tokenize(vocab,prompt.c_str(),prompt.size(),NULL,0,true,true);
  std::vector<llama_token> ptok(n_prompt);
  if(llama_tokenize(vocab,prompt.c_str(),prompt.size(),ptok.data(),ptok.size(),true,true)<0){ fprintf(stderr,"tokenize fail\n"); return 1; }
  llama_context_params cp=llama_context_default_params(); cp.n_ctx=n_prompt+n_predict; cp.n_batch=n_prompt; cp.no_perf=false;
  llama_context* ctx=llama_init_from_model(model,cp);
  if(!ctx){ fprintf(stderr,"context fail\n"); return 1; }

  fprintf(stderr,"[barra-llm] Modell: %s | vocab=%d | ngl=%d | sampler=%s | prompt-tokens=%d\n",model_path.c_str(),n_vocab,ngl,sampler.c_str(),n_prompt);
  llama_batch batch=llama_batch_get_one(ptok.data(),ptok.size());
  std::vector<uint8_t> io; std::string out;
  int n_gen=0, mism=0; double t_dsp=0, t_dec=0, t_dsp_max=0;
  double t_pf0=now_ms();
  llama_token cur=0;   /* ausserhalb der Schleife: batch haelt &cur ueber Iterationsgrenzen hinweg (kein Dangling) */
  for(int n_pos=0; n_pos+batch.n_tokens<n_prompt+n_predict; ){
    double td0=now_ms();
    if(llama_decode(ctx,batch)){ fprintf(stderr,"decode fail\n"); return 1; }
    const float* logits=llama_get_logits_ith(ctx,-1);   /* Vulkan synchronisiert erst hier -> gehoert zur Decode-Zeit */
    double td=now_ms()-td0; if(n_pos>0) t_dec+=td; else fprintf(stderr,"[barra-llm] Prefill %d Tokens: %.0f ms (%.1f t/s)\n",n_prompt,td,n_prompt*1000.0/td);
    n_pos+=batch.n_tokens;
    int ref=cpu_argmax(logits,n_vocab), tok=ref;
    if(sampler=="dsp"){ double ms=0; int d=dsp_argmax(logits,n_vocab,io,&ms); t_dsp+=ms; if(ms>t_dsp_max)t_dsp_max=ms;
      if(d<0){ fprintf(stderr,"[barra-llm] DSP-Argmax fehlgeschlagen bei Token %d -> CPU\n",n_gen); } else { tok=d; if(d!=ref) mism++; } }
    if(llama_vocab_is_eog(vocab,tok)) break;
    char buf[256]; int n=llama_token_to_piece(vocab,tok,buf,sizeof buf,0,true); if(n>0){ out.append(buf,n); fwrite(buf,1,n,stdout); fflush(stdout); }
    n_gen++;
    cur=tok; batch=llama_batch_get_one(&cur,1);
  }
  double t_all=now_ms()-t_pf0;
  printf("\n");
  fprintf(stderr,"[barra-llm] generiert %d Tokens in %.0f ms gesamt | decode %.1f ms/Token (%.1f t/s) | %s-Argmax %.2f ms/Token (max %.1f) | Mismatch DSP vs CPU: %d/%d\n",
    n_gen,t_all,n_gen?t_dec/n_gen:0,n_gen&&t_dec>0?n_gen*1000.0/t_dec:0,sampler.c_str(),n_gen?t_dsp/n_gen:0,t_dsp_max,mism,n_gen);
  llama_perf_context_print(ctx);
  llama_free(ctx); llama_model_free(model);
  return 0;
}
