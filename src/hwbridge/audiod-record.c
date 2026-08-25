/* audiod-record - Audio-EINGABE-Bruecke (Bionic, tinyalsa) fuers Weckwort.
 * Nimmt den Builtin-Mic (card0/device8 = EP1, 48kHz stereo S16) via tinyalsa auf, mischt auf mono
 * und streamt rohes 48k-mono-S16-PCM ueber Unix-Socket an den Container (wake-bridge liest es).
 * Route setzt av-setup (Boot). Bionic-tinyalsa zuverlaessig (glibc haengt im D-State) - wie audiod. */
#include <tinyalsa/pcm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <errno.h>

static void cap_conn(int c){
  struct pcm_config cfg; memset(&cfg,0,sizeof cfg);
  cfg.channels=2; cfg.rate=48000; cfg.period_size=1024; cfg.period_count=4; cfg.format=PCM_FORMAT_S16_LE;
  struct pcm* p=pcm_open(0,8,PCM_IN,&cfg);
  if(!p || !pcm_is_ready(p)){ fprintf(stderr,"[audiod-rec] pcm_open(0,8) FAIL: %s\n", p?pcm_get_error(p):"null"); if(p) pcm_close(p); return; }
  fprintf(stderr,"[audiod-rec] capture offen (card0/device8, 48kHz stereo)\n");
  short stereo[2048]; short mono[1024];   // 1024 frames je read
  for(;;){
    int rc=pcm_read(p, stereo, sizeof stereo);
    if(rc!=0){ fprintf(stderr,"[audiod-rec] pcm_read: %s\n",pcm_get_error(p)); break; }
    for(int i=0;i<1024;i++){ int v=((int)stereo[2*i]+(int)stereo[2*i+1])/2; mono[i]=(short)v; }  // downmix
    ssize_t off=0, tot=1024*2;
    while(off<tot){ ssize_t w=write(c,(char*)mono+off,tot-off); if(w<=0) goto done; off+=w; }
  }
done:
  pcm_close(p);
}

int main(int argc,char**argv){
  const char* sock=argc>1?argv[1]:"/data/local/ubuntu/opt/hwbridge/micrec.sock";
  signal(SIGPIPE,SIG_IGN);
  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ fprintf(stderr,"[audiod-rec] bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,2);
  fprintf(stderr,"[audiod-rec] bereit, lauscht auf %s (48kHz mono S16LE vom Builtin-Mic)\n",sock);
  for(;;){ int c=accept(srv,0,0); if(c<0) continue; cap_conn(c); close(c); }
  return 0;
}
