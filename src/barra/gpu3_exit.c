/* gpu3_exit — schickt gpud-zc den v3-EXIT-Befehl (Daemon endet, Supervisor startet die neue Datei). Dev-Loop ohne Reboot. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
int main(void){ int s=socket(AF_UNIX,SOCK_STREAM,0); struct sockaddr_un a; memset(&a,0,sizeof a); a.sun_family=AF_UNIX; strcpy(a.sun_path,"/opt/hwbridge/gpuzc.sock");
  if(connect(s,(struct sockaddr*)&a,sizeof a)){ perror("connect"); return 1; }
  uint32_t hdr[6]={0x47505A33u,7,0,0,0,0}, st=1; if(write(s,hdr,24)!=24) return 1; if(read(s,&st,4)!=4) st=1; printf("EXIT status=%u\n",st); return st; }
