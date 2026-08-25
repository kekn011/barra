/* barra-chat — schlanker Terminal-Chat-Client fuer llama-server (OpenAI-kompatibel).
 * Roh-TCP + HTTP, KEINE externen Libs (Container hat nur gcc). Streaming (SSE) Token fuer Token.
 * Fuehrt die Konversations-History; /reset leert sie, /bye beendet.
 *   Bau:  gcc -O2 barra-chat.c -o barra-chat
 *   Start: ./barra-chat [host] [port]     (Default 127.0.0.1 8080)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

static char* g_hist=0; static size_t g_hlen=0, g_hcap=0;   /* messages-JSON-Fragmente, komma-getrennt */
static void hist_add(const char* role, const char* json_content){
  size_t need=g_hlen+strlen(role)+strlen(json_content)+64;
  if(need>g_hcap){ g_hcap=need*2; g_hist=realloc(g_hist,g_hcap); }
  g_hlen+=sprintf(g_hist+g_hlen, "%s{\"role\":\"%s\",\"content\":%s}", g_hlen?",":"", role, json_content);
}
/* String -> JSON-String-Literal (mit Anfuehrungszeichen), escaped */
static char* json_str(const char* s){
  size_t n=strlen(s); char* o=malloc(n*6+3); char* p=o; *p++='"';
  for(size_t i=0;i<n;i++){ unsigned char c=s[i];
    if(c=='"'||c=='\\'){ *p++='\\'; *p++=c; }
    else if(c=='\n'){ *p++='\\'; *p++='n'; }
    else if(c=='\r'){ }
    else if(c=='\t'){ *p++='\\'; *p++='t'; }
    else if(c<0x20){ p+=sprintf(p,"\\u%04x",c); }
    else *p++=c;
  }
  *p++='"'; *p=0; return o;
}
static int dial(const char* host, const char* port){
  struct addrinfo hints={0}, *res; hints.ai_family=AF_UNSPEC; hints.ai_socktype=SOCK_STREAM;
  if(getaddrinfo(host,port,&hints,&res)) return -1;
  int fd=-1;
  for(struct addrinfo* a=res;a;a=a->ai_next){ fd=socket(a->ai_family,a->ai_socktype,a->ai_protocol); if(fd<0)continue;
    if(connect(fd,a->ai_addr,a->ai_addrlen)==0) break; close(fd); fd=-1; }
  freeaddrinfo(res); return fd;
}
static int writen(int fd,const char*b,size_t n){ while(n){ ssize_t k=write(fd,b,n); if(k<=0)return -1; b+=k; n-=k; } return 0; }

/* Ein "content":"..."-Delta aus einer SSE-data-Zeile extrahieren + unescaped ausgeben. Gibt Laenge des Contents. */
static int emit_content(const char* line){
  const char* p=strstr(line,"\"content\":"); if(!p) return 0; p+=10;
  while(*p==' ')p++; if(*p!='"') return 0; p++;
  int any=0;
  while(*p && *p!='"'){
    if(*p=='\\'){ p++; if(!*p) break;   /* Backslash am Zeilenende: nicht ueber die NUL laufen */
      if(*p=='n'){ putchar('\n'); }
      else if(*p=='t'){ putchar('\t'); }
      else if(*p=='u'){ int v; sscanf(p+1,"%4x",&v); p+=4; if(v<128) putchar(v); else { /* UTF-8 */ if(v<0x800){ putchar(0xC0|(v>>6)); putchar(0x80|(v&0x3F)); } else { putchar(0xE0|(v>>12)); putchar(0x80|((v>>6)&0x3F)); putchar(0x80|(v&0x3F)); } } }
      else putchar(*p);
      p++; any=1;
    } else { putchar(*p++); any=1; }
  }
  return any;
}

int main(int argc,char**argv){
  const char* host=argc>1?argv[1]:"127.0.0.1"; const char* port=argc>2?argv[2]:"8080";
  setvbuf(stdout,0,_IONBF,0);
  printf("\033[1mbarra-chat\033[0m  ->  llama-server @ %s:%s   (/reset leert, /bye beendet)\n\n",host,port);
  char* line=0; size_t lcap=0;
  for(;;){
    printf("\033[1;36mdu>\033[0m "); fflush(stdout);
    ssize_t ln=getline(&line,&lcap,stdin); if(ln<=0){ printf("\n"); break; }
    if(line[ln-1]=='\n') line[--ln]=0;
    if(!*line) continue;
    if(!strcmp(line,"/bye")) break;
    if(!strcmp(line,"/reset")){ g_hlen=0; if(g_hist)g_hist[0]=0; printf("(History geleert)\n\n"); continue; }

    char* uj=json_str(line); hist_add("user",uj); free(uj);
    /* Body bauen */
    size_t blen=g_hlen+256; char* body=malloc(blen);
    int bl=snprintf(body,blen,"{\"messages\":[%s],\"stream\":true,\"cache_prompt\":true,\"temperature\":0.7}",g_hist);
    char req[512];
    int rl=snprintf(req,sizeof req,
      "POST /v1/chat/completions HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",host,bl);
    int fd=dial(host,port);
    if(fd<0){ printf("\033[31m[Server nicht erreichbar]\033[0m\n\n"); free(body); continue; }
    writen(fd,req,rl); writen(fd,body,bl); free(body);

    printf("\033[1;32mki>\033[0m ");
    /* Antwort streamen: SSE-Zeilen "data: {...}" -> content-Deltas. Assistant-Text sammeln fuer History. */
    char buf[8192]; size_t acc_cap=4096, acc_len=0; char* acc=malloc(acc_cap);   /* roher Antworttext */
    char pending[16384]; size_t pl=0; ssize_t k; int in_body=0;
    while((k=read(fd,buf,sizeof buf))>0){
      for(ssize_t i=0;i<k;i++){
        if(pl<sizeof pending-1) pending[pl++]=buf[i];
        if(buf[i]=='\n'){
          pending[pl]=0;
          if(!in_body){ if(pl<=2) in_body=1; }   /* Leerzeile trennt HTTP-Header vom Body */
          else if(!strncmp(pending,"data: ",6)){
            if(strstr(pending,"[DONE]")){ }
            else{
              /* content sammeln (fuer History) + ausgeben */
              const char* cp=strstr(pending,"\"content\":");
              if(cp){ /* in acc anhaengen: roher (noch escaped) content-Teil bis zum schliessenden " */
                cp+=10; while(*cp==' ')cp++; if(*cp=='"'){ cp++; const char* st=cp; while(*cp&&*cp!='"'){ if(*cp=='\\'){ cp++; if(!*cp) break; } cp++; }
                  size_t cl=cp-st; if(acc_len+cl+1>acc_cap){ acc_cap=(acc_len+cl)*2; acc=realloc(acc,acc_cap);} memcpy(acc+acc_len,st,cl); acc_len+=cl; acc[acc_len]=0; }
                emit_content(pending);
              }
            }
          }
          pl=0;
        }
      }
    }
    close(fd);
    printf("\n\n");
    /* Assistant-Antwort in die History (acc ist bereits JSON-escaped, in Quotes packen) */
    char* aj=malloc(acc_len+3); aj[0]='"'; memcpy(aj+1,acc,acc_len); aj[acc_len+1]='"'; aj[acc_len+2]=0;
    hist_add("assistant",aj); free(aj); free(acc);
  }
  printf("Tschuess.\n");
  return 0;
}
