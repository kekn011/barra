/* audiod - Audio-Ausgabe-Bruecke (Bionic, AAudio) fuer den Ubuntu-Container.
 * Container schickt rohes PCM (48kHz, stereo, S16LE) ueber Unix-Socket -> audiod spielt via AAudio.
 * AAudio geht durch die HAL -> AoC-Block korrekt (raw-ALSA aus dem Container scheitert daran). */
#include <aaudio/AAudio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>

#define RATE 48000
#define CH 2

static AAudioStream* g_stream=0;

static int open_stream(void){
  AAudioStreamBuilder* b=0;
  if(AAudio_createStreamBuilder(&b)!=AAUDIO_OK){ fprintf(stderr,"[audiod] createStreamBuilder FAIL\n"); return -1; }
  AAudioStreamBuilder_setDirection(b,AAUDIO_DIRECTION_OUTPUT);
  AAudioStreamBuilder_setSampleRate(b,RATE);
  AAudioStreamBuilder_setChannelCount(b,CH);
  AAudioStreamBuilder_setFormat(b,AAUDIO_FORMAT_PCM_I16);
  AAudioStreamBuilder_setPerformanceMode(b,AAUDIO_PERFORMANCE_MODE_NONE);
  fprintf(stderr,"[audiod] oeffne AAudio-Stream...\n"); fflush(stderr);
  aaudio_result_t r=AAudioStreamBuilder_openStream(b,&g_stream);
  AAudioStreamBuilder_delete(b);
  if(r!=AAUDIO_OK){ fprintf(stderr,"[audiod] openStream FAIL: %s\n",AAudio_convertResultToText(r)); return -1; }
  AAudioStream_requestStart(g_stream);
  fprintf(stderr,"[audiod] AAudio-Stream offen: rate=%d ch=%d fmt=I16\n",
    AAudioStream_getSampleRate(g_stream),AAudioStream_getChannelCount(g_stream));
  return 0;
}

static void play_conn(int c){
  short buf[2048*CH];   /* 2048 Frames pro Lesen */
  for(;;){
    ssize_t n=read(c,buf,sizeof buf);
    if(n<=0) break;
    long frames=n/(CH*2);
    if(frames<=0) continue;
    aaudio_result_t w=AAudioStream_write(g_stream,buf,frames,200*1000*1000L); /* 200ms Timeout */
    if(w<0){ fprintf(stderr,"[audiod] write FAIL: %s\n",AAudio_convertResultToText(w)); break; }
  }
}

int main(int argc,char**argv){
  const char* sock=argc>1?argv[1]:"/data/local/tmp/audio.sock";
  setvbuf(stderr,NULL,_IONBF,0);
  fprintf(stderr,"[audiod] start (pid %d)\n",getpid());
  signal(SIGPIPE,SIG_IGN);
  if(open_stream()!=0) return 1;

  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ fprintf(stderr,"[audiod] bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,4);
  fprintf(stderr,"[audiod] bereit, lauscht auf %s (48kHz/stereo/S16LE roh)\n",sock);
  for(;;){ int c=accept(srv,0,0); if(c<0) continue; play_conn(c); close(c); }
  return 0;
}
