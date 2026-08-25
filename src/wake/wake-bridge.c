/* wake-bridge - Container-Seite: liest 48k-mono-S16-PCM vom audiod-record-Socket, jagt es durch
 * den sherpa-onnx-Keyword-Spotter (C-API). Bei "Hey Barra" -> Zeitstempel in Trigger-Datei + Hook.
 * gcc wake-bridge.c -o wake-bridge -I<hdr> -L<lib> -lsherpa-onnx-c-api */
#include "c-api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>

static const char* g_trig; static const char* g_hook;

int main(int argc,char**argv){
  if(argc<7){ fprintf(stderr,"wake-bridge <sock> <enc> <dec> <join> <tokens> <keywords> [trigger] [hook] [threshold]\n"); return 1; }
  const char* sock=argv[1]; g_trig=argc>7?argv[7]:"/tmp/barra-wake/trigger"; g_hook=argc>8?argv[8]:"";
  float thr=argc>9?atof(argv[9]):0.25f;

  SherpaOnnxKeywordSpotterConfig cfg; memset(&cfg,0,sizeof cfg);
  cfg.feat_config.sample_rate=16000; cfg.feat_config.feature_dim=80;
  cfg.model_config.transducer.encoder=argv[2];
  cfg.model_config.transducer.decoder=argv[3];
  cfg.model_config.transducer.joiner=argv[4];
  cfg.model_config.tokens=argv[5];
  cfg.model_config.num_threads=1; cfg.model_config.provider="cpu"; cfg.model_config.debug=0;
  cfg.max_active_paths=4; cfg.num_trailing_blanks=1;
  cfg.keywords_score=1.0f; cfg.keywords_threshold=thr; cfg.keywords_file=argv[6];

  const SherpaOnnxKeywordSpotter* kws=SherpaOnnxCreateKeywordSpotter(&cfg);
  if(!kws){ fprintf(stderr,"[wake] Spotter-Create FAIL\n"); return 1; }
  const SherpaOnnxOnlineStream* st=SherpaOnnxCreateKeywordStream(kws);
  if(!st){ fprintf(stderr,"[wake] Stream-Create FAIL\n"); return 1; }

  int s=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  if(connect(s,(struct sockaddr*)&a,sizeof a)<0){ fprintf(stderr,"[wake] connect %s FAIL\n",sock); return 1; }
  fprintf(stderr,"[wake] bereit, hoert auf \"Hey Barra\" (threshold %.2f)\n",thr);

  short buf[4800]; float fbuf[4800];   // ~0,1s bei 48k
  for(;;){
    ssize_t got=0, want=sizeof buf;
    while(got<want){ ssize_t n=read(s,(char*)buf+got,want-got); if(n<=0){ fprintf(stderr,"[wake] socket zu\n"); return 0; } got+=n; }
    int nf=got/2;
    for(int i=0;i<nf;i++) fbuf[i]=buf[i]/32768.0f;
    SherpaOnnxOnlineStreamAcceptWaveform(st, 48000, fbuf, nf);   // sherpa resampled 48k->16k
    while(SherpaOnnxIsKeywordStreamReady(kws, st)){
      SherpaOnnxDecodeKeywordStream(kws, st);
      const SherpaOnnxKeywordResult* r=SherpaOnnxGetKeywordResult(kws, st);
      if(r && r->keyword && r->keyword[0]){
        time_t now=time(0);
        FILE* f=fopen(g_trig,"w"); if(f){ fprintf(f,"%ld\n",(long)now); fclose(f); }
        fprintf(stderr,"[wake] %ld WAKE: %s\n",(long)now, r->keyword);
        if(g_hook && g_hook[0]){ char cmd[512]; snprintf(cmd,sizeof cmd,"%s >>/tmp/barra-wake/wake.log 2>&1 &",g_hook); system(cmd); }
        SherpaOnnxResetKeywordStream(kws, st);
      }
      if(r) SherpaOnnxDestroyKeywordResult(r);
    }
  }
  return 0;
}
