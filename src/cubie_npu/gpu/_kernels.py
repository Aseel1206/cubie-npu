"""OpenCL C kernels. Images are interleaved (HxWxC) unless noted. No local memory is used (the GPU has only 4 KiB of it)."""

SRC = r"""
inline int refl101(int p, int n){ if(n==1) return 0; if(p<0) p=-p; if(p>=n) p=2*n-2-p; return clamp(p,0,n-1); }

// ---- resize / letterbox: dst canvas dw x dh; the source is scaled into the region (ox,oy,rw,rh), the rest is 'pad' ----
__kernel void resize_bilinear(__global const uchar* src, __global uchar* dst, int sw, int sh, int c,
                              int dw, int dh, int ox, int oy, int rw, int rh, uchar pad){
  int x=get_global_id(0), y=get_global_id(1); if(x>=dw||y>=dh) return; int o=(y*dw+x)*c;
  int rx=x-ox, ry=y-oy;
  if(rx<0||ry<0||rx>=rw||ry>=rh){ for(int k=0;k<c;k++) dst[o+k]=pad; return; }
  float fx=(rx+0.5f)*((float)sw/rw)-0.5f, fy=(ry+0.5f)*((float)sh/rh)-0.5f;
  int x0=(int)floor(fx), y0=(int)floor(fy); float ax=fx-x0, ay=fy-y0;
  int xa=clamp(x0,0,sw-1), xb=clamp(x0+1,0,sw-1), ya=clamp(y0,0,sh-1), yb=clamp(y0+1,0,sh-1);
  for(int k=0;k<c;k++){
    float v=(1-ay)*((1-ax)*src[(ya*sw+xa)*c+k]+ax*src[(ya*sw+xb)*c+k])+ay*((1-ax)*src[(yb*sw+xa)*c+k]+ax*src[(yb*sw+xb)*c+k]);
    dst[o+k]=convert_uchar_sat_rte(v); }
}
__kernel void resize_area(__global const uchar* src, __global uchar* dst, int sw, int sh, int c,
                          int dw, int dh, int ox, int oy, int rw, int rh, uchar pad){
  int x=get_global_id(0), y=get_global_id(1); if(x>=dw||y>=dh) return; int o=(y*dw+x)*c;
  int rx=x-ox, ry=y-oy;
  if(rx<0||ry<0||rx>=rw||ry>=rh){ for(int k=0;k<c;k++) dst[o+k]=pad; return; }
  float scx=(float)sw/rw, scy=(float)sh/rh; float x0=rx*scx, x1=x0+scx, y0=ry*scy, y1=y0+scy;
  int ix0=(int)floor(x0), ix1=min((int)ceil(x1),sw), iy0=(int)floor(y0), iy1=min((int)ceil(y1),sh);
  float a0=0,a1=0,a2=0,wsum=0;
  for(int yy=iy0; yy<iy1; yy++){ float wy=min((float)(yy+1),y1)-max((float)yy,y0); __global const uchar* row=src+(yy*sw)*c;
    for(int xx=ix0; xx<ix1; xx++){ float w=wy*(min((float)(xx+1),x1)-max((float)xx,x0)); wsum+=w;
      if(c==3){ uchar3 v=vload3(xx,row); a0+=w*v.x; a1+=w*v.y; a2+=w*v.z; } else { a0+=w*row[xx*c]; } } }
  float inv=1.0f/wsum;
  if(c==3){ dst[o]=convert_uchar_sat_rte(a0*inv); dst[o+1]=convert_uchar_sat_rte(a1*inv); dst[o+2]=convert_uchar_sat_rte(a2*inv); }
  else { dst[o]=convert_uchar_sat_rte(a0*inv); }
}

// ---- colour ----
__kernel void swap_rb(__global const uchar* src, __global uchar* dst, int n){ int i=get_global_id(0); if(i>=n) return; dst[i*3]=src[i*3+2]; dst[i*3+1]=src[i*3+1]; dst[i*3+2]=src[i*3]; }
// same fixed-point weights as OpenCV: Y = (R*4899 + G*9617 + B*1868 + 8192) >> 14 ; rgb!=0 means channel order is R,G,B, else B,G,R
__kernel void to_gray(__global const uchar* src, __global uchar* dst, int n, int rgb){
  int i=get_global_id(0); if(i>=n) return; int a=src[i*3], b=src[i*3+1], c=src[i*3+2]; int r=rgb?a:c, bl=rgb?c:a;
  dst[i]=(uchar)((r*4899 + b*9617 + bl*1868 + 8192) >> 14); }

// ---- model input preparation: (x-mean)/std, optional channel swap, layout NHWC or NCHW, float32 or float16 out ----
__kernel void normalize(__global const uchar* src, __global void* dst, int w, int h, int swap, int nchw, int half_out,
                        float m0, float m1, float m2, float s0, float s1, float s2){
  int x=get_global_id(0), y=get_global_id(1); if(x>=w||y>=h) return; int i=(y*w+x)*3;
  float v[3]; v[0]=(src[i+(swap?2:0)]-m0)*s0; v[1]=(src[i+1]-m1)*s1; v[2]=(src[i+(swap?0:2)]-m2)*s2;
  for(int k=0;k<3;k++){ int o = nchw ? k*w*h + y*w + x : i+k;
    if(half_out) vstore_half(v[k], o, (__global half*)dst); else ((__global float*)dst)[o]=v[k]; } }

// ---- warp: dst(x,y) = src(M*[x,y,1]) , bilinear, constant border 0 (M maps dst->src like cv2.WARP_INVERSE_MAP) ----
__kernel void warp_affine(__global const uchar* src, __global uchar* dst, int sw, int sh, int c, int dw, int dh,
                          float m00, float m01, float m02, float m10, float m11, float m12){
  int x=get_global_id(0), y=get_global_id(1); if(x>=dw||y>=dh) return; int o=(y*dw+x)*c;
  float fx=m00*x+m01*y+m02, fy=m10*x+m11*y+m12; int x0=(int)floor(fx), y0=(int)floor(fy); float ax=fx-x0, ay=fy-y0;
  for(int k=0;k<c;k++){ float v=0;
    for(int j=0;j<2;j++) for(int i=0;i<2;i++){ int xx=x0+i, yy=y0+j; float w=(i?ax:1-ax)*(j?ay:1-ay);
      if(xx>=0&&yy>=0&&xx<sw&&yy<sh) v+=w*src[(yy*sw+xx)*c+k]; }
    dst[o+k]=convert_uchar_sat_rte(v); } }

// ---- separable gaussian blur (BORDER_REFLECT_101), float intermediate; interior pixels take a branch-free path ----
__kernel void blur_h(__global const uchar* src, __global float* tmp, __global const float* k, int r, int w, int h, int c){
  int x=get_global_id(0), y=get_global_id(1); if(x>=w||y>=h) return; bool in=(x>=r&&x<w-r);
  for(int ch=0;ch<c;ch++){ float s=0;
    if(in){ __global const uchar* p=src+(y*w+x-r)*c+ch; for(int j=0;j<=2*r;j++) s+=k[j]*p[j*c]; }
    else { for(int j=-r;j<=r;j++) s+=k[j+r]*src[(y*w+refl101(x+j,w))*c+ch]; }
    tmp[(y*w+x)*c+ch]=s; } }
__kernel void blur_v(__global const float* tmp, __global uchar* dst, __global const float* k, int r, int w, int h, int c){
  int x=get_global_id(0), y=get_global_id(1); if(x>=w||y>=h) return; bool in=(y>=r&&y<h-r);
  for(int ch=0;ch<c;ch++){ float s=0;
    if(in){ __global const float* p=tmp+((y-r)*w+x)*c+ch; for(int j=0;j<=2*r;j++) s+=k[j]*p[j*w*c]; }
    else { for(int j=-r;j<=r;j++) s+=k[j+r]*tmp[(refl101(y+j,h)*w+x)*c+ch]; }
    dst[(y*w+x)*c+ch]=convert_uchar_sat_rte(s); } }

// ---- sobel 3x3 on a gray image -> int16 gx, gy ----
__kernel void sobel(__global const uchar* src, __global short* gx, __global short* gy, int w, int h){
  int x=get_global_id(0), y=get_global_id(1); if(x>=w||y>=h) return;
  int xm=refl101(x-1,w), xp=refl101(x+1,w), ym=refl101(y-1,h), yp=refl101(y+1,h);
  int a=src[ym*w+xm], b=src[ym*w+x], cc=src[ym*w+xp], d=src[y*w+xm], f=src[y*w+xp], g=src[yp*w+xm], hh=src[yp*w+x], i=src[yp*w+xp];
  gx[y*w+x]=(short)((cc+2*f+i)-(a+2*d+g)); gy[y*w+x]=(short)((g+2*hh+i)-(a+2*b+cc)); }

// ---- FAST-9 corner score (OpenCV definition: largest t for which the pixel is still a corner), 0 if not a corner ----
// 16 circle pixels -> two 16-bit masks (brighter / darker than p by more than thr); a corner has 9 consecutive set bits. Only corners get the exact score.
#define S16(i,dx,dy) int d##i=(int)img[(y+(dy))*w+x+(dx)]-p;
#define ARC9(m) ({ uint m2=(m)|((m)<<16); uint a_=m2&(m2>>1); uint b_=a_&(a_>>2); uint c_=b_&(b_>>4); (c_&(m2>>8)&0xFFFFu)!=0; })
__kernel void fast_score(__global const uchar* img, __global ushort* score, int w, int h, int thr){
  int x=get_global_id(0), y=get_global_id(1); if(x>=w||y>=h) return; score[y*w+x]=0;
  if(x<3||y<3||x>=w-3||y>=h-3) return; int p=img[y*w+x];
  S16(0,0,-3) S16(4,3,0) S16(8,0,3) S16(12,-3,0)
  int nb=(d0>thr)+(d4>thr)+(d8>thr)+(d12>thr), nd=(-d0>thr)+(-d4>thr)+(-d8>thr)+(-d12>thr);
  if(nb<2 && nd<2) return;                                     // any 9-long arc contains at least 2 of the 4 compass points
  S16(1,1,-3) S16(2,2,-2) S16(3,3,-1) S16(5,3,1) S16(6,2,2) S16(7,1,3) S16(9,-1,3) S16(10,-2,2) S16(11,-3,1) S16(13,-3,-1) S16(14,-2,-2) S16(15,-1,-3)
  uint mb=0, md=0;
  #define BIT(i) mb|=(uint)(d##i>thr)<<i; md|=(uint)(-d##i>thr)<<i;
  BIT(0) BIT(1) BIT(2) BIT(3) BIT(4) BIT(5) BIT(6) BIT(7) BIT(8) BIT(9) BIT(10) BIT(11) BIT(12) BIT(13) BIT(14) BIT(15)
  if(!ARC9(mb) && !ARC9(md)) return;
  int dd[16]={d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15}; int best=0;
  for(int s=0;s<16;s++){ int a=1000, b=1000;
    for(int j=0;j<9;j++){ int v=dd[(s+j)&15]; a=min(a,v); b=min(b,-v); }
    best=max(best,max(a,b)); }
  score[y*w+x]=(ushort)(best-1); }
// non-max suppression over 3x3 (ties: earlier raster position wins) + compaction of survivors into out[] as (x,y,score,0)
__kernel void fast_nms(__global const ushort* score, __global int4* out, __global volatile int* count, int w, int h, int maxn, int do_nms){
  int x=get_global_id(0), y=get_global_id(1); if(x>=w||y>=h) return; int s=score[y*w+x]; if(s==0) return;
  if(do_nms){ for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){ if(i==0&&j==0) continue; int xx=x+i, yy=y+j; if(xx<0||yy<0||xx>=w||yy>=h) continue;
      int t=score[yy*w+xx]; bool before=(j<0)||(j==0&&i<0); if(before ? (t>=s) : (t>s)) return; } }
  int idx=atomic_inc(count); if(idx<maxn) out[idx]=(int4)(x,y,s,0); }
"""
