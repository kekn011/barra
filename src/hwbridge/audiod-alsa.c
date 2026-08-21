/* audiod-alsa - Audio-Ausgabe-Bruecke (Bionic, raw-ALSA via tinyalsa) fuer den Ubuntu-Container.
 * Container schickt rohes PCM (48kHz stereo S16LE) ueber Unix-Socket -> audiod spielt via tinyalsa
 * auf card0/device1 (EP2 -> Cirrus-Speaker). Bionic-tinyalsa ist zuverlaessig (glibc-tinyalsa haengt
 * im Kernel-D-State). Route setzt av-setup. Kein audioserver/Framework noetig. */
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

static void play_conn(int c){
  struct pcm_config cfg; memset(&cfg,0,sizeof cfg);
  cfg.channels=2; cfg.rate=48000; cfg.period_size=1024; cfg.period_count=4; cfg.format=PCM_FORMAT_S16_LE;
  struct pcm* p=pcm_open(0,1,PCM_OUT,&cfg);
  if(!p || !pcm_is_ready(p)){ fprintf(stderr,"[audiod] pcm_open(0,1) FAIL: %s\n", p?pcm_get_error(p):"null"); if(p) pcm_close(p); return; }
  char buf[16384];
  for(;;){ ssize_t n=read(c,buf,sizeof buf); if(n<=0) break;
    if(pcm_write(p,buf,(unsigned)n)!=0){ fprintf(stderr,"[audiod] pcm_write: %s\n",pcm_get_error(p)); break; } }
  pcm_close(p);
}

int main(int argc,char**argv){
  const char* sock=argc>1?argv[1]:"/data/local/tmp/audio.sock";
  signal(SIGPIPE,SIG_IGN);
  int srv=socket(AF_UNIX,SOCK_STREAM,0);
  struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strncpy(a.sun_path,sock,sizeof(a.sun_path)-1);
  unlink(sock);
  if(bind(srv,(struct sockaddr*)&a,sizeof a)<0){ fprintf(stderr,"[audiod] bind %s: %s\n",sock,strerror(errno)); return 1; }
  chmod(sock,0666); listen(srv,4);
  fprintf(stderr,"[audiod] bereit, lauscht auf %s (raw-ALSA card0/device1, 48kHz/stereo/S16LE)\n",sock);
  for(;;){ int c=accept(srv,0,0); if(c<0) continue; play_conn(c); close(c); }
  return 0;
}
