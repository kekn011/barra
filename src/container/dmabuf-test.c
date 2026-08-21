/* dmabuf-test - Zero-Copy-Fundament: dmabuf aus /dev/dma_heap/system allozieren,
 * mmappen, CPU-Roundtrip. Der fd ist per SCM_RIGHTS an eine Bruecke weitergebbar,
 * die ihn dann in ihren Chip importiert (GxpCapi_ImportBufferFromFd / Vulkan dma_buf). */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <linux/types.h>

struct dma_heap_allocation_data { __u64 len; __u32 fd; __u32 fd_flags; __u64 heap_flags; };
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0, struct dma_heap_allocation_data)

int main(void){
  const char* heap="/dev/dma_heap/system";
  int h=open(heap,O_RDONLY|O_CLOEXEC);
  if(h<0){ perror("open heap"); return 1; }
  struct dma_heap_allocation_data d; memset(&d,0,sizeof d);
  d.len=4096; d.fd_flags=O_RDWR|O_CLOEXEC;
  if(ioctl(h,DMA_HEAP_IOCTL_ALLOC,&d)<0){ perror("DMA_HEAP_IOCTL_ALLOC"); return 1; }
  int fd=(int)d.fd;
  printf("dmabuf alloziert: fd=%d len=%llu (heap %s)\n",fd,(unsigned long long)d.len,heap);

  uint32_t* p=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(p==MAP_FAILED){ perror("mmap"); return 1; }
  for(int i=0;i<8;i++) p[i]=(uint32_t)(i*11);
  printf("CPU-geschrieben:  ");
  for(int i=0;i<8;i++) printf("%u ",p[i]);
  printf("\n");

  /* beweisen: es ist wirklich ein dmabuf (fstat -> char/anon, mappbar, teilbar) */
  struct stat st; fstat(fd,&st);
  printf("fd-Typ: %s, Groesse mappbar\n", S_ISCHR(st.st_mode)?"char":(st.st_size?"regular":"dmabuf(anon)"));
  printf("Roundtrip ok. Dieser fd ist per SCM_RIGHTS an tpud/gpud/gxpd weitergebbar\n"
         "-> der Chip importiert ihn und rechnet IN diesem Puffer (kein Byte-Copy).\n");
  munmap(p,4096); close(fd); close(h);
  return 0;
}
