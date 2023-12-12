/* -*- mode: c++ -*- */

#include "cuda_lattice.h"

#include <cassert>
#include <cuda_runtime.h>

/// from global.h without pulling in the whole header
#define _GSI(_ix) (24*(_ix))
#ifdef __restrict__
#define _RESTR __restrict__
#else
#define _RESTR
#endif

__device__ __constant__ int gamma_permutation[16][24] = {
  {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
  {19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16, 7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4},
  {18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5},
  {13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10},
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
  {19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16, 7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4},
  {18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5},
  {13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10},
  {7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4, 19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16},
  {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17},
  {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22},
  {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22},
  {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17},
  {7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4, 19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16}
};
__device__ __constant__ int gamma_sign[16][24] = {
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {+1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {-1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {-1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1},
  {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1}
};
__device__ __constant__ IdxComb idx_comb {
  .comb = {
    {0,1},
    {0,2},
    {0,3},
    {1,2},
    {1,3},
    {2,3}
  }};


/**
 * See geometry comment in cuda_lattice.h
 */
const int BS = CUDA_BLOCK_SIZE;
__device__ inline Coord get_thread_origin(Geom local_geom) {
  int x = BS*(blockIdx.x * blockDim.x + threadIdx.x);
  int t = BS*(x / local_geom.LX);
  x %= local_geom.LX;
  int y = BS*(blockIdx.y * blockDim.y + threadIdx.y);
  int z = BS*(blockIdx.z * blockDim.z + threadIdx.z);
  return Coord { .t = t, .x = x, .y = y, .z = z };
}
__device__ inline size_t coord2lexic(Coord coord, Geom local_geom) {
  return (((coord.t*local_geom.LX) + coord.x)*local_geom.LY + coord.y)*local_geom.LZ + coord.z;
}
__device__ inline Coord lexic2coord(size_t ind, Geom local_geom) {
  return Coord {
    .t = (int)(ind / (local_geom.LX * local_geom.LY * local_geom.LZ)),
    .x = (int)((ind / (local_geom.LY * local_geom.LZ)) % local_geom.LX),
    .y = (int)((ind / local_geom.LZ) % local_geom.LY),
    .z = (int)(ind % local_geom.LZ)
  };
}

/**
 * Given length-24 spin vector in, multiply by appropriate gamma matrix, writing
 * to out (non-aliasing assumed).
 */
__device__ inline void _fv_eq_gamma_ti_fv(double* _RESTR out, int gamma_index, const double* _RESTR in) {
  for (int i = 0; i < 24; ++i) {
    out[i] = in[gamma_permutation[gamma_index][i]] * gamma_sign[gamma_index][i];
  }
}
__device__ inline void _fv_ti_eq_g5(double* in_out) {
  for (int i = 12; i < 24; ++i) {
    in_out[i] *= -1;
  }
}

/**
 * 1D kernels: operate over CUDA_BLOCK_SIZE spinor elements each.
 *  - `len`: num *doubles* in the input/output array (must be divisible by 24)
 */
__global__ void ker_spinor_field_eq_gamma_ti_spinor_field(
    double* _RESTR out, int gamma_index, const double* _RESTR in, size_t len) {
  int start_ind = BS*_GSI(blockIdx.x * blockDim.x + threadIdx.x);
  for (int ind = start_ind; ind < len && ind < (start_ind + BS*24); ind += 24) {
    double* rr = out + ind;
    const double* ss = in + ind;
    _fv_eq_gamma_ti_fv(rr, gamma_index, ss);
  }
}

__global__ void ker_g5_phi(double* spinor, size_t len) {
  /* invert sign of spin components 2 and 3 */
  int start_ind = BS*_GSI(blockIdx.x * blockDim.x + threadIdx.x);
  for (int ind = start_ind; ind < len && ind < (start_ind + BS*24); ind += 24) {
    _fv_ti_eq_g5(&spinor[ind]);
  }
}

/**
 * Wrap coords to [-L/2, L/2]. Assumes inputs are in [0, ..., L-1]
 */
__device__ int coord_map(int xi, int Li) {
  return (xi >= Li / 2) ? (xi - Li) : xi;
}
__device__ int coord_map_zerohalf(int xi, int Li) {
  return (xi > Li / 2) ? xi - Li : ( (xi < Li / 2) ? xi : 0 );
}


/**
 * 4D kernels: operate over CUDA_BLOCK_SIZE^4 spinor elements each.
 */
__global__ void ker_dzu_dzsu(
    double* _RESTR dzu, double* _RESTR dzsu, const double* _RESTR fwd_src, const double* _RESTR fwd_y,
    int iflavor, Coord g_proc_coords, Coord gsx,
    Geom global_geom, Geom local_geom) {

  // Coord origin = get_thread_origin(local_geom);
  int gsx_arr[4] = {gsx.t, gsx.x, gsx.y, gsx.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};

  // double dzu_work[6 * 12 * 12 * 2] = { 0 };
  // double dzsu_work[4 * 12 * 12 * 2] = { 0 };
  double spinor_work_0[24] = { 0 };
  double spinor_work_1[24] = { 0 };

  for (int ia = 0; ia < 12; ++ia) {
    const double* fwd_base = &fwd_src[_GSI(VOLUME) * (iflavor * 12 + ia)];
    for (int k = 0; k < 6; ++k) {
      const int sigma = idx_comb.comb[k][1];
      const int rho = idx_comb.comb[k][0];
      double dzu_work[12 * 2] = { 0 };
      for (int iz = blockIdx.x * blockDim.x + threadIdx.x;
           iz < VOLUME; iz += blockDim.x * gridDim.x) {
        const Coord coord = lexic2coord(iz, local_geom);
        const int tt = coord.t;
        const int xx = coord.x;
        const int yy = coord.y;
        const int zz = coord.z;
        const double* _u = &fwd_base[_GSI(iz)];
        double* _t_sigma = spinor_work_0;
        double* _t_rho = spinor_work_1;
        _fv_eq_gamma_ti_fv(_t_sigma, sigma, _u);
        _fv_ti_eq_g5(_t_sigma);
        _fv_eq_gamma_ti_fv(_t_rho, rho, _u);
        _fv_ti_eq_g5(_t_rho);
        int coord_arr[4] = {tt, xx, yy, zz};
        int zrho = coord_arr[rho] + proc_coord_arr[rho] * local_geom_arr[rho] - gsx_arr[rho];
        zrho = (zrho + global_geom_arr[rho]) % global_geom_arr[rho];
        int zsigma = coord_arr[sigma] + proc_coord_arr[sigma] * local_geom_arr[sigma] - gsx_arr[sigma];
        zsigma = (zsigma + global_geom_arr[sigma]) % global_geom_arr[sigma];
        int factor_rho = coord_map_zerohalf(zrho, global_geom_arr[rho]);
        int factor_sigma = coord_map_zerohalf(zsigma, global_geom_arr[sigma]);
        for (int ib = 0; ib < 12; ++ib) {
          for (int i = 0; i < 12; ++i) {
            double fwd_y_re = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i];
            double fwd_y_im = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i+1];
            double s_re = (_t_sigma[2*i] * factor_rho - _t_rho[2*i] * factor_sigma);
            double s_im = (_t_sigma[2*i+1] * factor_rho - _t_rho[2*i+1] * factor_sigma);
            // dzu_work[((k * 12 + ia) * 12 + ib) * 2 + 0] += fwd_y_re * s_re + fwd_y_im * s_im;
            // dzu_work[((k * 12 + ia) * 12 + ib) * 2 + 1] += fwd_y_re * s_im - fwd_y_im * s_re;
            dzu_work[2*ib] += fwd_y_re * s_re + fwd_y_im * s_im;
            dzu_work[2*ib+1] += fwd_y_re * s_im - fwd_y_im * s_re;
          }
        }
      } // end vol loop

      // reduce (TODO faster reduce algo?)
      for (int ib = 0; ib < 12; ++ib) {
        int ind = ((k * 12 + ia) * 12 + ib) * 2;
        atomicAdd_system(&dzu[ind], dzu_work[2*ib]);
        atomicAdd_system(&dzu[ind+1], dzu_work[2*ib+1]);
      }
    }

    for (int sigma = 0; sigma < 4; ++sigma) {
      // const double* fwd_base = &fwd_src[_GSI(VOLUME) * (iflavor * 12 + ia)];
      for (int ib = 0; ib < 12; ++ib) {
        double dzsu_work_re = 0.0;
        double dzsu_work_im = 0.0;
        for (int iz = blockIdx.x * blockDim.x + threadIdx.x;
             iz < VOLUME; iz += blockDim.x * gridDim.x) {
          // const Coord coord = lexic2coord(iz, local_geom);
          // const int tt = coord.t;
          // const int xx = coord.x;
          // const int yy = coord.y;
          // const int zz = coord.z;
          const double* _u = &fwd_base[_GSI(iz)];
          double* _t = spinor_work_0;
          _fv_eq_gamma_ti_fv(_t, sigma, _u);
          _fv_ti_eq_g5(_t);

          for (int i = 0; i < 12; ++i) {
            double fwd_y_re = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i];
            double fwd_y_im = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i+1];
            double s_re = _t[2*i];
            double s_im = _t[2*i+1];
            // dzsu_work[((sigma * 12 + ia) * 12 + ib) * 2 + 0] += fwd_y_re * s_re + fwd_y_im * s_im;
            // dzsu_work[((sigma * 12 + ia) * 12 + ib) * 2 + 1] += fwd_y_re * s_im - fwd_y_im * s_re;
            dzsu_work_re += fwd_y_re * s_re + fwd_y_im * s_im;
            dzsu_work_im += fwd_y_re * s_im - fwd_y_im * s_re;
          }
        } // end vol loop

        // reduce (TODO faster reduce algo?)
        int ind = ((sigma * 12 + ia) * 12 + ib) * 2;
        atomicAdd_system(&dzsu[ind], dzsu_work_re);
        atomicAdd_system(&dzsu[ind+1], dzsu_work_im);
      }
    }
  }
}



// scalar product of two four-vectors
__device__
static inline double
SCALPROD( const double xv[4] ,
	  const double yv[4] )
{
  return xv[0]*yv[0] + xv[1]*yv[1] + xv[2]*yv[2] + xv[3]*yv[3] ;
}

// little helper function
__device__
static inline double
lerp( const double a ,
      const double T1 ,
      const double T2 )
{
  return a*T1 + (1.0-a)*T2 ;
}

__device__
double
chebUsum( const int nk ,
	  const double x ,
	  const double *co )
{
  double ya=0.0;
  double yb=0.0;
  const double twox = 2.0*x;
  double ytmp ;
  int j;
  for(j=nk-1;j>0;j--) {
    ytmp = yb;
    yb = twox*yb - ya + co[j];
    ya = ytmp;
  }
  return(-ya + twox*yb + co[0]);
}

// Clenshaw for Sum[co[k]*Derivative[0,1][ChebyshevU][k,x],{k,0,nk-1}]
// alf(n,z) = 2z(n+1)/n       beta(n,z) = -(n+2)/n
// y_k = alf(k,x)*y_{k+1} + beta(k+1,x)*y_{k+2} + co[k]
__device__
double
dchebUsum( const int nk ,
	   const double x,
	   const double *co)
{
  double ya=0.0;
  double yb=0.0;
  const double twox = 2.0*x;
  double ytmp ;
  int j ;
  for(j=nk-1;j>0;j--) {
    ytmp = yb;
    yb = twox*(j+1)*yb/j - (j+3)*ya/(j+1) + co[j];
    ya = ytmp;
  }
  return(2.0*yb);
}

// Clenshaw for Sum[co[k]*Derivative[0,2][ChebyshevU][k,x],{k,0,nk-1}]
// alf(n,z) = 2z(n+1)/(n-1)       beta(n,z) = -(n+3)/(n-1)
__device__
double
ddchebUsum( const int nk,
	    const double x,
	    const double *co)
{
  double ya=0.0;
  double yb=0.0;
  const double twox = 2.0*x;
  double ytmp ;
  int j ;
  for(j=nk-1;j>1;j--) {
    ytmp = yb;
    yb = twox*(j+1)*yb/(j-1) ;
    yb -= (j+4)*ya/j ;
    yb += co[j];
    ya = ytmp;
  }
  return 8*yb ;
}

// Clenshaw for Sum[co[k]*Derivative[0,3][ChebyshevU][k,x],{k,0,nk-1}]
// alf(n,z) = 2z(n+1)/(n-2)       beta(n,z) = -(n+4)/(n-2)
__device__
double
dddchebUsum( const int nk ,
	     const double x ,
	     const double *co )
{
  double ya=0.0;
  double yb=0.0;
  const double twox = 2.0*x;
  double ytmp ;
  int j ;
  for(j=nk-1;j>2;j--) {
    ytmp = yb;
    yb = twox*(j+1)*yb/(j-2) - (j+5)*ya/(j-1) + co[j];
    ya = ytmp;
  }
  return 48*yb ;
}

// interpolation function
// dy should be set to y1-y2
__device__
static inline double
interpol3( const double y ,
	   const double y1 ,
	   const double y2 ,
	   const double dy ,
	   const double f1 ,
	   const double f2 ,
	   const double g1 ,
	   const double g2 )
{
  return ((-f1* (y - y2)*(y - y2)*(2* y - 3* y1 + y2) +
	   (y - y1)* (f2* (y - y1)*(2* y + y1 - 3* y2)
		      + (y - y2)*dy* (g1* y + g2* y - g2* y1 - g1* y2)))/(dy*dy*dy)) ;
}

// precompute all this business for x or y
__device__
void
precompute_INV( struct intprecomp *INVy ,
		const double y ,
		const double y1 ,
		const double y2 ,
		const size_t idx )
{
  INVy -> idx = idx ;
  const double dy = y1-y2 ;
  const double ymy1 = y-y1 ;
  const double ymy2 = y-y2 ;
  const double ym2sq = (ymy2*ymy2) ;
  const double ym1sq = (ymy1*ymy1) ;
  INVy -> A = -(ym2sq)*(2*y - 3*y1 + y2) ;
  INVy -> B = (ym1sq)*(2*y + y1 - 3*y2) ;
  INVy -> C1 = (ymy1)*(ym2sq)*dy ;
  INVy -> C2 = (ym1sq)*(ymy2)*dy ;
  INVy -> D = 1./(dy*dy*dy) ;
  INVy -> lA = (y2-y)/(y2-y1) ;
}

__device__
static inline double
interpol4( const struct intprecomp& INVx ,
	   const double f1 ,
	   const double f2 ,
	   const double g1 ,
	   const double g2 )
{
  return (f1*INVx.A + f2*INVx.B + g1*INVx.C1 + g2*INVx.C2)*INVx.D ; 
}

// function pointer for cheby stuff
// static double (*Func_usm[4])( const int , const double , const double *) =
// { chebUsum , dchebUsum , ddchebUsum , dddchebUsum } ;
__device__
static double Func_usm( const int i,
    const int n, const double x, const double *f ) {
  switch(i) {
    case 0: return chebUsum(n, x, f);
    case 1: return dchebUsum(n, x, f);
    case 2: return ddchebUsum(n, x, f);
    case 3: return dddchebUsum(n, x, f);
    default: return 0.0;
  }
}

// returns the form factor, given the coefficients fm[0..(nf-1)] and fp[0..(nf-1)]
// e.g. fm = alpha^{(3)}_{m-}  and fp = alpha^{(3)}_{m+}
// nf = length of vectors fm[nm][ix/iy] and similarly for fp
// nm = index of the sum sigma that appeared in the integrand == outer index of ff
// ndy and ndcb = # derivatives with respect to y and cos(beta) respectively
// x is not used in this function
__device__
static void
getff2( double res[2] ,
	const int nf, const FFidx nm ,
	const bool ndy , const NDCB ndcb, 
	const double y, const double x ,
	const float *fm, const float *fp )
{
  const double y1 = 1.0/y;

  // enum guarantees these are set but avoid stupid
  // gcc maybe unitialized warning
  double yp = 1 ;
  int mshm = 2 ;
  switch(nm) {
  case QG0    : case dxQG0 : yp = 1.0   ; mshm=2 ; break;
  case QG1    : case dxQG1 : yp = 1.0   ; mshm=2 ; break;
  case QG2    : case dxQG2 : case d2xQG2: yp =  y1 ; mshm=3 ; break;
  case QG3    : case dxQG3 : case d2xQG3: yp = 1.0 ; mshm=2 ; break;
  case QL4    : case dxQL4 : yp =   y   ; mshm=1 ; break;
  case QL2    : case dxQL2 : yp = y1*y1 ; mshm=4 ; break;
  }

  // these two parameters are simply related to mshm
  int mshp = 2 - mshm ;
  // mm used to be set by the global map Idm, but that was unnecessary
  const int mm = abs( mshp ) ;

  // set fval to zero, actually not really needed
  double fval[128], fvalD[128];
  // TODO: do something more reasonable if too long
  if (nf+mm >= 128) return;
  memset( fval , 0 , (nf+mm)*sizeof( double ) ) ;
  memset( fvalD , 0 , (nf+mm)*sizeof( double ) ) ;
  double facm = y1*y1*yp ;
  double facp = yp ;

  for(int j = 0 ; j < mm ; j++ ) {
    facm *= y1 ;
    facp *= y ;
  }

  // double *Pfval = (double*)fval + mm ;
  // double *PfvalD = (double*)fvalD + mm ;

  mshm += mm ; mshp += mm ;
  for(int j = 0; j < nf; j++) {
    fvalD[mm + j] = y1*mshp*facp*__ldg(fp + j) - y1*mshm*facm*__ldg(fm + j) ;
    fval[mm + j] = facm*__ldg(fm) + facp*__ldg(fp) ;
    facm *= y1; facp *= y;    
    mshm++ ; mshp++ ;
  }

  res[1] = Func_usm(mm + ndcb, nf+mm, x , fvalD ) ;
  res[0] = ndy? res[1] : Func_usm(mm + ndcb, nf+mm, x , fval ) ;
}

// case where you have read in the weight functions upon initialization
// interpolates the form factor[nm] to the target point y using the grid
__device__
double
accessv( const bool flag_hy, const bool use_y_derivs,
	 const int ix, const int iy,
	 const FFidx nm, const bool ndy, const NDCB ndcb,
	 const double cb, const double y, 
	 const struct Grid_coeffs& Grid )
{
  const int nx = __ldg(Grid.nfx+ix) ;
  const double y1 = __ldg(Grid.YY+iy);
  double res1[2] = {0.,0.} ;
    
  // occasionally ndy gets set to 1, if use_y_derivs is set we always do the deriv
  getff2( res1 , nx, nm, ndy, ndcb, y1, cb,
	  getFfm(&Grid, nm, ix, iy) , getFfp(&Grid, nm, ix, iy) ) ;
  
  // if we are not at the upper limit of Y we can use info from the next point
  if(!flag_hy) {
    const int iy2 = iy+1;
    const double y2 = __ldg(Grid.YY+iy2);
    double res2[2] = {0.,0.} ;

    getff2( res2 , nx, nm, ndy, ndcb, y2, cb,
	    getFfm(&Grid, nm, ix, iy2) ,
	    getFfp(&Grid, nm, ix, iy2) ) ;
    
    if( use_y_derivs ) {
      return interpol3( y, y1, y2, y1-y2,
			res1[0], res2[0], res1[1], res2[1] ) ;      
    } else {
      // value interpolated to target y, at x=x1[0];
      return lerp( (y2-y)/(y2-y1) , res1[0] , res2[0] ) ;
    }
  }
  return res1[0] ;
}

// linear search variant targeted at avoiding warp divergence and keeping
// regular memory access on CUDA.
__device__
static int
lsrch( const double *arr, const double target,
    const int lo, const int hi ) {
  int index = lo;
  for (int i = lo; i < hi; ++i) {
    index = (__ldg(arr+i) <= target && target <= __ldg(arr+i+1)) ? i : index;
  }
  return index;
}

// returns the lower index that bounds "target"
// e.g arr[lo] < target < arr[lo+1]
// (assumes a monotonically increasing arr)
__device__
inline int
find_ind(const double *arr, const double target,
    const int lo, const int hi) {
  return lsrch(arr, target, lo, hi);
}

// extract the form factor
__device__
double
extractff( const FFidx nm, const bool ndy, const NDCB ndcb,
	   const struct invariants& Inv , const struct Grid_coeffs& Grid )
{  
  const bool flag_hx = ( Inv.x >= __ldg(Grid.XX + Grid.nstpx-1) ) ;
  const bool flag_hy = ( Inv.y >= __ldg(Grid.YY + Grid.nstpy-1) ) ;
 
  const bool use_x_derivs = (nm<dxQG0 || nm==dxQG2 || nm==dxQG3) ;

  const int ix = Inv.INVx.idx , iy = Inv.INVy.idx ; 
  const bool use_y_derivs = (ndy==false) ;
  const double f1iy = accessv( flag_hy, use_y_derivs, ix, iy, nm,
			       ndy, ndcb, Inv.cb, Inv.y, Grid ) ;

  if(!flag_hx) {
    const double f2iy = accessv( flag_hy, use_y_derivs, ix+1, iy, nm,
				 ndy, ndcb, Inv.cb, Inv.y , Grid ) ;
    if(use_x_derivs) {
      const int offset = nm < dxQG0 ? dxQG0 : QL4 ;
      // These enum additions are pretty sketchy...
      const double g1iy = accessv( flag_hy, use_y_derivs, ix, iy, (FFidx)(nm+offset),
				   ndy, ndcb, Inv.cb, Inv.y, Grid ) ;
      const double g2iy = accessv( flag_hy, use_y_derivs, ix+1, iy, (FFidx)(nm+offset),
				   ndy, ndcb, Inv.cb, Inv.y, Grid ) ;

      return interpol4( Inv.INVx, f1iy, f2iy, g1iy, g2iy ) ;
    } else {
      // lerpity lerp
      return lerp( Inv.INVx.lA , f1iy , f2iy ) ;
    }
  }
  return f1iy ;
}

// extract the form factor
__device__
void
extractff2( const FFidx nm,
	    const NDCB ndcb,
	    const struct invariants& Inv ,
	    const struct Grid_coeffs& Grid ,
	    double F[4] )
{  
  // derivative map
  const NDCB ndcb2 = (NDCB)(ndcb+1 < 5 ? ndcb+1 : ndcb) ;
  // map for the x-derivative, incomplete as some derivatives aren't used
  const FFidx dxmap[14] = { dxQG0  , dxQG1 , dxQG2  , dxQG3  , dxQL4 , dxQL2 ,
			  dxQG0  , dxQG1 , d2xQG2 , d2xQG3 , dxQL4 , dxQL2 ,
			  d2xQG2 , d2xQG3 } ;
  // derivative wrt dcb
  F[0] = extractff( nm, false , ndcb2 , Inv , Grid ) ;
  // derivative wrt x
  F[1] = extractff( dxmap[nm], false , ndcb  , Inv , Grid ) ;
  // derivative wrt y
  F[2] = extractff( nm, true  , ndcb  , Inv , Grid ) ;
  // no derivative
  F[3] = extractff( nm, false , ndcb  , Inv , Grid ) ;
  return ;
}

// Initialises the invariants used in chnr_*
__device__ __noinline__
static void // struct invariants
set_invariants( const double xv[4] ,
		const double yv[4] ,
		const struct Grid_coeffs& Grid,
                struct invariants& Inv )
{
  // struct invariants Inv ;
  const double EPSIN = 1E-7 ;
  
  Inv.xsq = SCALPROD( xv , xv ) ;
  Inv.xsq = Inv.xsq < EPSIN ? EPSIN : Inv.xsq ;
  Inv.x = sqrt( fabs( Inv.xsq ) ) ;

  // y needs a little fudge factor as it is badly behaved for very small y
  Inv.ysq = SCALPROD( yv , yv ) ;
  Inv.ysq = Inv.ysq < EPSIN ? EPSIN : Inv.ysq ;
  Inv.y = sqrt( fabs( Inv.ysq ) ) ;

  Inv.xdoty = SCALPROD( xv , yv ) ;
  Inv.cb = Inv.xdoty/( Inv.x * Inv.y ) ;

  Inv.cborig = Inv.cb ;
  Inv.yorig = Inv.y ;
  Inv.xmysq = ( Inv.xsq + Inv.ysq )*(1.00000000000001)-2.0*Inv.xdoty ;
  Inv.xmy = sqrt( fabs( Inv.xmysq ) ) ;

  Inv.flag2 = false ;
  const double rx = ( Inv.x > Inv.xmy ? Inv.xmy/Inv.x : Inv.x/Inv.xmy);
  const double rxy = ( Inv.x > Inv.y ? Inv.y/Inv.x : Inv.x/Inv.y);

  if( rx < rxy
      && Inv.xmy < __ldg(Grid.YY + Grid.nstpy-1)
      && fabs(Inv.xmysq) > 1E-28 ) {
    Inv.cb = (Inv.x-Inv.y*Inv.cb)/Inv.xmy ;
    Inv.y = Inv.xmy ;    
    Inv.flag2 = true ;
  }

  // setup INVx
  const size_t ix1 = (size_t)find_ind( Grid.XX , Inv.x , 0 , Grid.nstpx ) ;
  // const size_t ix1 = (size_t)find_bin( Grid.xstp , Inv.x , Grid.nstpx ) ;
  size_t ix2 = ix1+1 ;
  if( ix2 >= (size_t)Grid.nstpx ) {
    ix2 = ix1 ;
  }
  precompute_INV( &Inv.INVx, Inv.x, __ldg(Grid.XX+ix1), __ldg(Grid.XX+ix2) , ix1 ) ;
  
  // setup InvY
  const size_t iy1 = (size_t)find_ind( Grid.YY , Inv.y , 0 , Grid.nstpy ) ;  
  // const size_t iy1 = (size_t)find_bin( Grid.ystp , Inv.y , Grid.nstpy ) ;
  // y edge case
  size_t iy2 = iy1 + 1 ;
  if( iy2 >= (size_t)Grid.nstpy ) {
    iy2 = iy1 ;
  }
  precompute_INV( &Inv.INVy , Inv.y , __ldg(Grid.YY+iy1) , __ldg(Grid.YY+iy2) , iy1 ) ;
  
  // return Inv ;
}

// __device__
// static int
// compute_dg_TAYLORX( double *dg0dy ,
// 		    double *dg0dx ,
// 		    double *dg0dcb ,
// 		    const struct invariants& Inv ,
// 		    const struct Grid_coeffs& Grid )
// {  
//   const int iy_tay = find_ind( getTX(&Grid,YY) , Inv.y , 0 , Grid.NY_tay ) ;
//   const int iy = find_ind( Grid.YY , Inv.y , 0 , Grid.nstpy ) ;
  
//   if( iy_tay == Grid.NY_tay-1 || iy == Grid.nstpy-1 ) {
//     return 1 ;
//   }

//   const int iy2_tay = iy_tay+1;
//   const int ix  = 0;
  
//   const double ay = (__ldg(getTX(&Grid,YY)+iy2_tay)-Inv.y)
//     /(getTX(&Grid,YY)[iy2_tay]-getTX(&Grid,YY)[iy_tay]);
//   const double ax = ( __ldg(Grid.XX)-Inv.x)/( __ldg(Grid.XX));
  
//   double f1 = Inv.cb*lerp( ay , getTX(&Grid,G0dx)[iy_tay] , getTX(&Grid,G0dx)[iy2_tay] ) ;  
//   double f2 = accessv( false, true, ix, iy, dxQG0, false, d0cb, Inv.cb, Inv.y, Grid );
//   *dg0dx = lerp( ax , f1 , f2 ) ;

//   f1 = lerp( ay , getTX(&Grid,G0dy)[iy_tay] , getTX(&Grid,G0dy)[iy2_tay ] ) ;
//   f2 = accessv( false, false, ix, iy, QG0, true , d0cb, Inv.cb, Inv.y, Grid );
//   *dg0dy = lerp( ax , f1 , f2 ) ;
    
//   f2 = accessv( false, true , ix, iy, QG0, false, d1cb, Inv.cb, Inv.y, Grid );
//   *dg0dcb = lerp( ax , 0.0 , f2 ) ;

//   return 0 ;
// }

// sets dyv[bet]= \partial^(y)_\beta < I >_epsilon
// sets dxv[bet]= \partial^(x)_\beta < I >_epsilon
__device__
int
chnr_dS( const double xv[4] ,
	 const double yv[4] ,
	 const struct invariants& Inv ,
	 const struct Grid_coeffs& Grid ,
	 double dxv[4] ,
	 double dyv[4] )
{
  double dg0dy, dg0dx, dg0dcb;
  
  if( Inv.x < __ldg(Grid.XX) ) {
    // FORNOW
    // if( compute_dg_TAYLORX( &dg0dy , &dg0dx , &dg0dcb , Inv, Grid ) == 1 ) {
    //   return 1 ;
    // }
  } else if( Inv.x > __ldg(Grid.XX + Grid.nstpx -1) ) {
    return 1 ;
  } else { 
    double f[4];
    extractff2( QG0 , d0cb, Inv, Grid , f );
    dg0dcb = f[0] ;
    dg0dx  = f[1] ;
    dg0dy  = f[2] ;
  }
  
  if( Inv.flag2 ) {
    const double dgs0dx  = dg0dx;
    const double dgs0dcb = dg0dcb;
    const double dgs0dy  = dg0dy;
    const double ca = Inv.cb; 
    const double cd = (Inv.yorig-Inv.x*Inv.cborig)/Inv.xmy;
    dg0dx  = dgs0dx + (1.0-ca*ca)/Inv.xmy*dgs0dcb + ca*dgs0dy;
    dg0dcb = -(Inv.ysq*cd*dgs0dcb + Inv.x*Inv.yorig*Inv.xmy*dgs0dy)/Inv.xmysq; 
    dg0dy  = -(Inv.cborig+ca*cd)/Inv.xmy*dgs0dcb+cd*dgs0dy;
  } 

  int bet ;
  for(bet=0;bet<4;bet++) {
    const double yhat = yv[bet]/Inv.yorig;
    const double xhat = xv[bet]/Inv.x;
    const double c2 = (xhat-Inv.cborig*yhat)/Inv.yorig;
    const double c4 = (yhat-Inv.cborig*xhat)/Inv.x;
    dyv[bet] = yhat*dg0dy+c2*dg0dcb;
    dxv[bet] = xhat*dg0dx+c4*dg0dcb;
  }
  return 0 ;
}

// little storage for the temporary variables
struct ttmps {
  double ell1, ell2, ell3, v1, ell4, dv1dx, dv1dy, dv1dcb, dell2dx, dell2dy, dell2dcb;
  double dell1dx, dell1dy, dell1dcb, dell3dx, dell3dy, dell3dcb, dell4dy, dell4dx, dell4dcb;
} ;

// __device__
// static int
// TAYLORX_contribution( struct ttmps *T ,
// 		      const struct invariants& Inv ,
// 		      const struct Grid_coeffs& Grid )
// {
//   // Just like non taylor-expanded version need to set this due to
//   // it being set differently compared to the V and S
//   const double ysq = Inv.flag2 ? Inv.xmysq : Inv.ysq ;
  
//   // set the indices
//   const int iy_tay  = find_ind( getTX(&Grid,YY) , Inv.y , 0 , Grid.NY_tay ) ;      
//   const int iy = find_ind( Grid.YY , Inv.y , 0 , Grid.nstpy ) ;

//   // check they are reasonable
//   if( iy_tay == Grid.NY_tay-1 || iy == Grid.nstpy-1 ) {
//     // fprintf( stderr , "chnr_dT TAYLORX y is at or above edge of grid\n" ) ;
//     // fprintf( stderr , "%f >= %f\n" , Inv.y , getTX(&Grid,YY)[iy_tay] ) ;
//     // fprintf( stderr , "%f >= %f\n" , Inv.y , Grid.YY[Grid.nstpy-1] ) ;
//     return 1 ;
//   }

//   const int iy2_tay = iy_tay+1;
//   const int ix = 0 ;
  
//   const double ay = (__ldg(getTX(&Grid,YY)+iy2_tay)-Inv.y)/
//       (__ldg(getTX(&Grid,YY)+iy2_tay)- __ldg(getTX(&Grid,YY)+iy_tay));
//   const double ax = (__ldg(Grid.XX)-Inv.x)/(__ldg(Grid.XX));
  
//   const double xb=__ldg(Grid.XX);
//   const double xbsq=xb*xb;
  
//   const double ell2a = lerp( ay , __ldg(getTX(&Grid,Gl2)+iy_tay) , __ldg(getTX(&Grid,Gl2)+iy2_tay) ) ;
//   const double ell2b = accessv(false, true, ix, iy, QL2, false, d0cb, Inv.cb, Inv.y, Grid );
//   T -> ell2 = lerp(ax, ell2a, ell2b) ;
  
//   const double dell2adx = lerp( ay , __ldg(getTX(&Grid,Gl21) + iy_tay) , __ldg(getTX(&Grid,Gl21) + iy2_tay) )*Inv.cb ;
//   const double dell2bdx = accessv(false, true, ix, iy, dxQL2, false, d0cb, Inv.cb, Inv.y , Grid );
//   T -> dell2dx = lerp(ax, dell2adx, dell2bdx) ;

//   const double dell2ady = lerp( ay , __ldg(getTX(&Grid,Gl2dy) + iy_tay) , __ldg(getTX(&Grid,Gl2dy) + iy2_tay) ) ;
//   const double dell2bdy = accessv(false, false, ix, iy, QL2 , true, d0cb, Inv.cb, Inv.y , Grid ) ;
//   T -> dell2dy = lerp(ax, dell2ady, dell2bdy) ;

//   const double dell2adcb = 0.0;
//   const double dell2bdcb = accessv(false, true, ix, iy, QL2 , false, d1cb, Inv.cb, Inv.y , Grid );
//   T -> dell2dcb = lerp(ax ,dell2adcb ,dell2bdcb ) ;
  
//   const double v1b       = accessv(false, true , ix, iy, QG1  , false, d0cb , Inv.cb, Inv.y , Grid );
//   const double dv1bdx    = accessv(false, true , ix, iy, dxQG1, false, d0cb , Inv.cb, Inv.y , Grid );
//   const double dv1bdy    = accessv(false, false, ix, iy, QG1  , true , d0cb , Inv.cb, Inv.y , Grid );
//   const double dv1bdcb   = accessv(false, true , ix, iy, QG1  , false, d1cb , Inv.cb, Inv.y , Grid );
//   const double ell4b     = accessv(false, true , ix, iy, QL4  , false, d0cb , Inv.cb, Inv.y , Grid );
//   const double dell4bdx  = accessv(false, true , ix, iy, dxQL4, false, d0cb , Inv.cb, Inv.y , Grid );
//   const double dell4bdy  = accessv(false, false, ix, iy, QL4  , true , d0cb , Inv.cb, Inv.y , Grid );
//   const double dell4bdcb = accessv(false, true , ix, iy, QL4  , false, d1cb , Inv.cb, Inv.y , Grid );
  
//   const double ell3a = lerp( ay , __ldg(getTX(&Grid,Gl3) + iy_tay) , __ldg(getTX(&Grid,Gl3) + iy2_tay) ) ;
//   const double ell3b = ell4b/(2.0*xbsq*ysq) - Inv.y*Inv.cb*ell2b/xb;
//   T -> ell3 = lerp( ax , ell3a , ell3b ) ;

//   const double ell1b = 4.0/(3.0*xbsq*xbsq)*(v1b - xbsq*ysq*(Inv.cb*Inv.cb-0.25)*ell2b - 1.5*xbsq*xb*Inv.y*Inv.cb*ell3b);
//   T -> ell1 = ell1b;

//   const double dell3ady = lerp( ay , __ldg(getTX(&Grid,Gl3dy) + iy_tay) , __ldg(getTX(&Grid,Gl3dy) + iy2_tay) ) ;
//   const double dell3bdy = (-ell4b/(ysq*Inv.y)
// 			   + dell4bdy/(2.0*ysq)
// 			   - Inv.cb*xb*ell2b
// 			   - Inv.y*xb*Inv.cb*dell2bdy)/(xbsq);
//   T -> dell3dy = lerp( ax, dell3ady , dell3bdy ) ; 

//   const double dell3adcb = 0.0;
//   const double dell3bdcb = (dell4bdcb/(2.0*xb*ysq)
// 			    -Inv.y*ell2b - Inv.y*Inv.cb*dell2bdcb)/xb;
//   T -> dell3dcb = lerp( ax , dell3adcb , dell3bdcb ) ; 

//   const double dell3bdx = (-ell4b/(xb*ysq)
// 			   + dell4bdx/(2.0*ysq)
// 			   + Inv.y*Inv.cb*ell2b
// 			   - Inv.y*xb*Inv.cb*dell2bdx)/xbsq;
//   T -> dell3dx = dell3bdx;

//   T -> dell1dy = 4.0/(3.0*xbsq*xbsq)*(dv1bdy-2.0*xbsq*Inv.y*(Inv.cb*Inv.cb-0.25)*ell2b
// 				      -xbsq*ysq*(Inv.cb*Inv.cb-0.25)*dell2bdy
// 				      -1.5*xbsq*xb*Inv.cb*ell3b
// 				      -1.5*xbsq*xb*Inv.y*Inv.cb*dell3bdy) ;
  
//   T -> dell1dcb = 4.0/(3.0*xbsq*xbsq)
//     *(dv1bdcb - 2.0*xbsq*ysq*Inv.cb*ell2b-xbsq*ysq*(Inv.cb*Inv.cb-0.25)*dell2bdcb
//       -1.5*xbsq*xb*Inv.y*ell3b-1.5*xbsq*xb*Inv.y*Inv.cb*dell3bdcb);

//   T -> dell1dx = -4.0*ell1b/xb + 4.0/(3.0*xbsq*xbsq)
//     *(dv1bdx - 2.0*xb*ysq*(Inv.cb*Inv.cb-0.25)*ell2b-xbsq*ysq*(Inv.cb*Inv.cb-0.25)*dell2bdx
//       -4.5*xbsq*Inv.y*Inv.cb*ell3b - 1.5*xbsq*xb*Inv.y*Inv.cb*dell3bdx);

//   return 0 ;
// }

// __device__
// static void
// XMYSWAP_T_contrib( struct ttmps *T ,
// 		   const struct invariants& Inv )
// {
//   const double ell2s = T -> ell2;
//   const double dells2dx  = T -> dell2dx;
//   const double dells2dcb = T -> dell2dcb;
//   const double dells2dy  = T -> dell2dy;
//   const double cd = ( Inv.yorig-Inv.x*Inv.cborig)/Inv.xmy;
//   T->ell2     = ell2s;
//   T->dell2dx  = dells2dx + (1.0-Inv.cb*Inv.cb)/Inv.xmy*dells2dcb + Inv.cb*dells2dy;
//   T->dell2dcb = -(Inv.ysq*cd*dells2dcb + Inv.x*Inv.yorig*Inv.xmy*dells2dy)/Inv.xmysq;
//   T->dell2dy  = -(Inv.cborig+Inv.cb*cd)/Inv.xmy*dells2dcb+cd*dells2dy;
  
//   const double ell3s     = -T->ell3;
//   const double dells3dx  = -T->dell3dx;
//   const double dells3dcb = -T->dell3dcb;
//   const double dells3dy  = -T->dell3dy;

//   T->ell3    =  ell3s - T->ell2; 
//   T->dell3dx =  dells3dx + (1.0-Inv.cb*Inv.cb)/Inv.xmy*dells3dcb + Inv.cb*dells3dy - T->dell2dx;
//   T->dell3dcb =  -(Inv.ysq*cd*dells3dcb + Inv.x*Inv.yorig*Inv.xmy*dells3dy)/Inv.xmysq - T->dell2dcb;
//   T->dell3dy  =  -(Inv.cborig+Inv.cb*cd)/Inv.xmy*dells3dcb+cd*dells3dy - T->dell2dy;
  
//   const double ell1s = T->ell1;
//   const double dells1dx  = T->dell1dx;
//   const double dells1dcb = T->dell1dcb;
//   const double dells1dy  = T->dell1dy;
//   T->ell1     = ell1s - T->ell2 -2.0*T->ell3;
//   T->dell1dx  = dells1dx + (1.0-Inv.cb*Inv.cb)/Inv.xmy*dells1dcb
//     + Inv.cb*dells1dy - T->dell2dx - 2.0*T->dell3dx;
//   T->dell1dcb = -(Inv.ysq*cd*dells1dcb + Inv.x*Inv.yorig*Inv.xmy*dells1dy)/Inv.xmysq
//     - T->dell2dcb -2.0*T->dell3dcb;
//   T->dell1dy  = -(Inv.cborig+Inv.cb*cd)/Inv.xmy*dells1dcb+cd*dells1dy
//     - T->dell2dy -2.0*T->dell3dy;
// }

// set dyv[alf][bet][dta]= \partial^(y)_\beta <(epsilon_alf epsilon_dta - 1/4 delta_{alf dta}) I>_epsilon
// and dxv[alf][bet][dta]= \partial^(x)_\beta <(epsilon_alf epsilon_dta - 1/4 delta_{alf dta}) I>_epsilon
__device__
int
chnr_dT( const double xv[4] ,
	 const double yv[4] ,
	 const struct invariants& Inv ,
	 const struct Grid_coeffs& Grid ,
	 double dxv[4][4][4] ,
	 double dyv[4][4][4] )
{
  struct ttmps T ;
  const double ysq = Inv.flag2 ? Inv.xmysq : Inv.ysq ;
  
  if( Inv.x < __ldg(Grid.XX)) {
    // FORNOW
    // if( TAYLORX_contribution( &T , Inv , Grid ) == 1 ) {
    //   return 1 ;
    // }
  } else if( Inv.x > __ldg(Grid.XX + Grid.nstpx -1) ) {
    return 1 ;
  } else {
    double f[4] KQED_ALIGN ;
    extractff2( QL2 , d0cb , Inv , Grid , f ) ;
    T.dell2dx  = f[1] ;
    T.dell2dy  = f[2] ;
    T.ell2     = f[3] ;
    T.dell2dcb = f[0] ;

    extractff2( QL4 , d0cb , Inv , Grid , f ) ;
    T.dell4dx  = f[1] ;
    T.dell4dy  = f[2] ;
    T.ell4     = f[3] ;
    T.dell4dcb = f[0] ; 
    
    extractff2( QG1 , d0cb , Inv , Grid , f ) ;
    T.dv1dx  = f[1] ;
    T.dv1dy  = f[2] ;
    T.v1     = f[3] ;
    T.dv1dcb = f[0] ;
    
    T.ell3 = T.ell4/(2.0*Inv.xsq*ysq) - Inv.y*Inv.cb*T.ell2/Inv.x;
    T.ell1 = 4.0/(3.0*Inv.xsq*Inv.xsq)*
      (T.v1 - Inv.xsq*ysq*(Inv.cb*Inv.cb-0.25)*T.ell2
       - 1.5*Inv.xsq*Inv.x*Inv.y*Inv.cb*T.ell3);
        
    T.dell3dy  = (-T.ell4/(ysq*Inv.y) + T.dell4dy/(2.0*ysq)
		  - Inv.cb*Inv.x*T.ell2 -
		  Inv.y*Inv.x*Inv.cb*T.dell2dy)/(Inv.xsq);
    
    T.dell3dcb = (T.dell4dcb/(2.0*Inv.x*ysq) -
		  Inv.y*T.ell2 - Inv.y*Inv.cb*T.dell2dcb)/Inv.x;
    
    T.dell3dx  = (-T.ell4/(Inv.x*ysq)
		  + T.dell4dx/(2.0*ysq)
		  + Inv.y*Inv.cb*T.ell2
		  - Inv.y*Inv.x*Inv.cb*T.dell2dx)/Inv.xsq;
    
    T.dell1dy  = 4.0/(3.0*Inv.xsq*Inv.xsq)*(T.dv1dy-2.0*Inv.xsq*Inv.y*(Inv.cb*Inv.cb-0.25)*T.ell2
				    -Inv.xsq*ysq*(Inv.cb*Inv.cb-0.25)*T.dell2dy
				    -1.5*Inv.xsq*Inv.x*Inv.cb*T.ell3-1.5*Inv.xsq*Inv.x*Inv.y*Inv.cb*T.dell3dy);
    
    T.dell1dcb = 4.0/(3.0*Inv.xsq*Inv.xsq)*(T.dv1dcb
				    - 2.0*Inv.xsq*ysq*Inv.cb*T.ell2-Inv.xsq*ysq*(Inv.cb*Inv.cb-0.25)*T.dell2dcb
				    -1.5*Inv.xsq*Inv.x*Inv.y*T.ell3-1.5*Inv.xsq*Inv.x*Inv.y*Inv.cb*T.dell3dcb);
    
    T.dell1dx  = -4.0*T.ell1/Inv.x
      + 4.0/(3.0*Inv.xsq*Inv.xsq)*(T.dv1dx- 2.0*Inv.x*ysq*(Inv.cb*Inv.cb-0.25)*T.ell2
			   -Inv.xsq*ysq*(Inv.cb*Inv.cb-0.25)*T.dell2dx
			   -4.5*Inv.xsq*Inv.y*Inv.cb*T.ell3
				   - 1.5*Inv.xsq*Inv.x*Inv.y*Inv.cb*T.dell3dx);
  }

  if( Inv.flag2 ) {
    // FORNOW
    // XMYSWAP_T_contrib( &T , Inv ) ;
  } 
   
  int alf, bet, dta ;
  for(bet=0;bet<4;bet++)  {  
    const double c1 = yv[bet]/Inv.yorig;
    const double c3 = xv[bet]/Inv.x;
    const double c2 = (c3-Inv.cborig*c1)/Inv.yorig;
    const double c4 = (c1-Inv.cborig*c3)/Inv.x;
    for(alf=0;alf<4;alf++)  {
      const int d_ab = ( alf==bet ) ;
      for(dta=0;dta<4;dta++) {
	const int d_bd = ( bet==dta ) ;
	const int d_ad = ( alf==dta ) ;
	const double t1 = d_ab*yv[dta]+d_bd*yv[alf]-0.5*d_ad*yv[bet];
	const double t2 = d_ab*xv[dta]+d_bd*xv[alf]-0.5*d_ad*xv[bet];
	const double t3 = xv[alf]*xv[dta]-d_ad*Inv.xsq/4;
	const double t4 = yv[alf]*yv[dta]-d_ad*Inv.ysq/4;
	const double t5 = xv[alf]*yv[dta]+yv[alf]*xv[dta]-0.5*Inv.xdoty*d_ad;
	dyv[alf][bet][dta] =
	  + t1*T.ell2
	  + t2*T.ell3
	  + t3*(c1*T.dell1dy + c2*T.dell1dcb)
	  + t4*(c1*T.dell2dy + c2*T.dell2dcb)
	  + t5*(c1*T.dell3dy + c2*T.dell3dcb)
	  ;
	dxv[alf][bet][dta] =
	  + t2*T.ell1
	  + t1*T.ell3
	  + t3*(c3*T.dell1dx + c4*T.dell1dcb)
	  + t4*(c3*T.dell2dx + c4*T.dell2dcb)
	  + t5*(c3*T.dell3dx + c4*T.dell3dcb)
	  ;
      }
    }
  }
  
  return 0 ;
}

// little struct definition to store all these variables
struct vtmp {
  double g2, dg1dx, dg1dy, dg1dcb, dg2dx, dg2dy, dg2dcb , dg3dx, dg3dy, dg3dcb;
  double ddg2dxdx, ddg2dxdy, ddg2dxdcb, ddg2dydcb, ddg2dcbdcb;
  double ddg3dxdx, ddg3dxdy, ddg3dxdcb, ddg3dydcb, ddg3dcbdcb;
  double ddg1dxdx, ddg1dxdy, ddg1dxdcb, ddg1dydcb, ddg1dcbdcb;
  double ddg1dydy , ddg2dydy ;
} ;

// __device__
// static int
// taylorx_contrib( const struct Grid_coeffs Grid ,
// 		 struct vtmp *D ,
// 		 const double x ,
// 		 const double y ,
// 		 const double xsq ,
// 		 const bool flag2 ,
// 		 double cb )
// { 
//   const int iy_tay  = find_ind( getTX(&Grid, YY) , y , 0 , Grid.NY_tay ) ;
//   const int iy = find_ind( Grid.YY , y , 0 , Grid.nstpy ) ;
  
//   if( iy_tay == Grid.NY_tay-1 || iy == Grid.nstpy-1 ) {
//     return 1 ;
//   }

//   const int iy2_tay = iy_tay+1;
//   const int ix=0;
  
//   const double ay = (__ldg(getTX(&Grid,YY)+iy2_tay)-y)/(__ldg(getTX(&Grid,YY)+iy2_tay)-__ldg(getTX(&Grid,YY)+iy_tay));
//   const double ax = (__ldg(Grid.XX)-x)/(__ldg(Grid.XX));

//   const double xb=__ldg(Grid.XX) ;
//   const double xbsq=xb*xb;

//   const double g2a = 0.0;
//   const double g2b = accessv( false, true, ix, iy, QG2 , false, d0cb, cb, y, Grid );
//   D -> g2 = lerp( ax , g2a , g2b ) ;

//   const double ddg2adxdcb = lerp( ay , getTX(&Grid,G21)[iy_tay] , getTX(&Grid,G21)[iy2_tay] ) ;
//   const double ddg2bdxdcb = accessv( false, true, ix, iy, dxQG2, false, d1cb, cb, y, Grid );
//   D -> ddg2dxdcb = lerp( ax , ddg2adxdcb , ddg2bdxdcb ) ;

//   const double dg2bdx = accessv( false, true, ix, iy, dxQG2, false, d0cb, cb, y , Grid );
//   D -> dg2dx = D -> ddg2dxdcb*cb; // ax*dg2adx + (1.0-ax)*dg2bdx;

//   const double dg2bdcb = accessv( false, true , ix, iy, QG2, false, d1cb, cb, y , Grid );
//   D -> dg2dcb = D -> ddg2dxdcb*x; // dg2dx*x/cb; // ax*dg2adcb + (1.0-ax)*dg2bdcb;

//   const double dg2bdy = accessv( false, false, ix, iy, QG2, true, d0cb, cb, y , Grid );
//   D -> dg2dy = lerp( ax , 0.0 , dg2bdy ) ;

//   const double ddg2adxdy = cb*lerp( ay , getTX(&Grid,G21dy)[iy_tay] , getTX(&Grid,G21dy)[iy2_tay] ) ;
//   const double ddg2bdxdy = accessv( false, false, ix, iy, dxQG2, true, d0cb, cb, y , Grid );
//   D -> ddg2dxdy = lerp( ax , ddg2adxdy , ddg2bdxdy ) ;

//   const double ddg2adxdx = 2.0*( lerp( ay , getTX(&Grid,G22A)[iy_tay] , getTX(&Grid,G22A)[iy2_tay] ) 
// 				 + lerp( ay , getTX(&Grid,G22B)[iy_tay] , getTX(&Grid,G22B)[iy2_tay] )
// 				 *(0.4+0.6*(2.0*cb*cb-1)));
//   const double ddg2bdxdx = accessv( false, true , ix, iy, d2xQG2, false, d0cb, cb, y , Grid );
//   D -> ddg2dxdx = lerp( ax , ddg2adxdx , ddg2bdxdx ) ;


//   const double ddg2bdcbdcb = accessv(false, true, ix, iy, QG2, false, d2cb, cb, y , Grid );
//   D -> ddg2dcbdcb = ddg2bdcbdcb*xsq/xbsq; // the function starts quadratically

//   const double ddg2bdydcb = accessv( false, false, ix, iy, QG2, true, d1cb, cb, y, Grid );
//   D -> ddg2dydcb = lerp( ax , 0.0 , ddg2bdydcb ) ;
  
//   const double dg3bdy      = accessv( false, false, ix, iy, QG3   , true , d0cb, cb, y , Grid );
//   const double ddg3bdxdy   = accessv( false, false, ix, iy, dxQG3 , true , d0cb, cb, y , Grid );
//   const double ddg3bdxdcb  = accessv( false, true , ix, iy, dxQG3 , false, d1cb, cb, y , Grid );
//   const double ddg3bdxdx   = accessv( false, true , ix, iy, d2xQG3, false, d0cb, cb, y , Grid );
//   const double ddg3bdcbdcb = accessv( false, true , ix, iy, QG3   , false, d2cb, cb, y , Grid );
//   const double ddg3bdydcb  = accessv( false, false, ix, iy, QG3   , true , d1cb, cb, y , Grid );

//   const double dg1bdy = dg3bdy - cb/xb*g2b -(y/xb)*cb*dg2bdy;
//   const double ddg1bdxdx =  ddg3bdxdx -2.0*y/(xbsq*xb)*cb*g2b+y/xbsq*cb*dg2bdx + y/xbsq*cb*dg2bdx - y/xb*cb*ddg2bdxdx;
//   const double ddg1bdxdy = ddg3bdxdy +cb/xbsq*g2b + y*cb/xbsq*dg2bdy -cb/xb*dg2bdx - y/xb*cb*ddg2bdxdy;
//   const double ddg1bdxdcb = ddg3bdxdcb +y/xbsq*g2b +y/xbsq*cb*dg2bdcb -y/xb*dg2bdx -y/xb*cb*ddg2bdxdcb;
//   const double ddg1bdydcb = ddg3bdydcb - g2b/xb -cb/xb*dg2bdcb -y/xb*dg2bdy - y/xb*cb*ddg2bdydcb;
//   const double ddg1bdcbdcb = ddg3bdcbdcb - 2.0*y/xb*dg2bdcb - y/xb*cb*ddg2bdcbdcb;
    
//   const double dg1ady = lerp( ay , getTX(&Grid,G3Ady)[iy_tay] , getTX(&Grid,G3Ady)[iy2_tay] )
//     - lerp( ay , getTX(&Grid,G3Bdy)[iy_tay] , getTX(&Grid,G3Bdy)[iy2_tay] ) ;
  
//   const double ddg1adxdy = cb*( lerp( ay , getTX(&Grid,G31Ady)[iy_tay] , getTX(&Grid,G31Ady)[iy2_tay] )
// 				-(2/3.)*lerp( ay , getTX(&Grid,G31Bdy)[iy_tay] , getTX(&Grid,G31Bdy)[iy2_tay] )
// 				-lerp( ay , getTX(&Grid,G22A)[iy_tay] , getTX(&Grid,G22A)[iy2_tay] )
// 				-y*lerp( ay , getTX(&Grid,G22Ady)[iy_tay] , getTX(&Grid,G22Ady)[iy2_tay] )) ;
//   const double ddg1adxdcb =
//     lerp( ay , getTX(&Grid,G31A)[iy_tay] , getTX(&Grid,G31A)[iy2_tay] )
//     -(2/3.)*lerp( ay , getTX(&Grid,G31B)[iy_tay] , getTX(&Grid,G31B)[iy2_tay] )
//     -y*lerp( ay , getTX(&Grid,G22A)[iy_tay] , getTX(&Grid,G22A)[iy2_tay] ) ;

//   const double ddg1adxdx = ddg1bdxdx; 
//   D -> ddg1dxdcb  = lerp( ax , ddg1adxdcb , ddg1bdxdcb ) ;
//   D -> dg1dx      = D -> ddg1dxdcb*cb; // ax*dg1adx + (1.0-ax)*dg1bdx;     
//   D -> dg1dcb     = D -> ddg1dxdcb*x; // ax*dg1adcb + (1.0-ax)*dg1bdcb;     
//   D -> dg1dy      = lerp( ax , dg1ady , dg1bdy ) ;
//   D -> ddg1dxdy   = lerp( ax , ddg1adxdy , ddg1bdxdy ) ;
//   D -> ddg1dxdx   = lerp( ax , ddg1adxdx , ddg1bdxdx ) ;
//   D -> ddg1dcbdcb = ddg1bdcbdcb*xsq/xbsq; // the function starts quadratically 
//   D -> ddg1dydcb  = lerp( ax , 0.0 , ddg1bdydcb ) ;

//   if(flag2) {
//     // in the case of a swap, you need the second derivative wrt y
//     const double ya = __ldg(Grid.YY+iy);
//     const double yb = __ldg(Grid.YY+iy+1) ; //ya+Grid.ystp;
//     const double ddg2adydy= 0.0;

//     const double dg2bdy_ya = accessv( false, false, ix, iy, QG2, true, d0cb, cb, ya, Grid );
//     const double dg2bdy_yb = accessv( false, false, ix, iy, QG2, true, d0cb, cb, yb, Grid );
//     const double ddg2bdydy = (dg2bdy_yb-dg2bdy_ya)/Grid.ystp;
//     D -> ddg2dydy = ax*ddg2adydy + (1.0-ax)*ddg2bdydy;
       
//     const double ddg1adydy =
//         (__ldg(getTX(&Grid,G3Ady)+iy2_tay)- __ldg(getTX(&Grid,G3Bdy)+iy2_tay)-
//          ( __ldg(getTX(&Grid,G3Ady)+iy_tay)- __ldg(getTX(&Grid,G3Bdy)+iy_tay)))
//         /(__ldg(getTX(&Grid,YY)+iy2_tay) - __ldg(getTX(&Grid,YY)+iy_tay) ) ;

//     const double dg3bdy_ya = accessv( false, false, ix, iy, QG3, true, d0cb, cb, ya, Grid );
//     const double dg3bdy_yb = accessv( false, false, ix, iy, QG3, true, d0cb, cb, yb, Grid );

//     const double ddg3bdydy = (dg3bdy_yb-dg3bdy_ya)/Grid.ystp;
//     const double ddg1bdydy = ddg3bdydy -(y*ddg2bdydy+2.0*dg2bdy)*cb/x;
//     D -> ddg1dydy = ax*ddg1adydy + (1.0-ax)*ddg1bdydy;
//   }

//   return 0 ; 
// }

// // flips the sign of all the dg2 values
// __device__
// static struct vtmp
// v_flip2( const struct vtmp D )
// {
//   struct vtmp F = D ; // memcpy
//   F.g2         = -D.g2 ;
//   F.dg2dx      = -D.dg2dx ;
//   F.dg2dy      = -D.dg2dy ;
//   F.dg2dcb     = -D.dg2dcb ;
//   F.ddg2dxdx   = -D.ddg2dxdx ;
//   F.ddg2dxdy   = -D.ddg2dxdy ;
//   F.ddg2dydy   = -D.ddg2dydy ;
//   F.ddg2dxdcb  = -D.ddg2dxdcb ;
//   F.ddg2dydcb  = -D.ddg2dydcb ;
//   F.ddg2dcbdcb = -D.ddg2dcbdcb ;  
//   return F ;
// }

// __device__
// static void
// XMYSWAP_V_contrib2( struct vtmp *D ,
// 		   const double x ,
// 		   const double xmy ,
// 		   const double y ,
// 		   const double xsq ,
// 		   const double xmysq ,
// 		   const double ysq ,
// 		   const double ca ,
// 		   const double cborig )
// { 
//   const struct vtmp F = v_flip2( *D ) ; 
//   const double cd = (y-x*cborig)/xmy;
  
//   D -> g2     = F.g2;
//   D -> dg2dx  = F.dg2dx + (1.0-ca*ca)/xmy*F.dg2dcb + ca * F.dg2dy;
//   D -> dg2dcb = -(ysq*cd*F.dg2dcb + x*y*xmy*F.dg2dy)/xmysq;
//   D -> dg2dy  = -(cborig+ca*cd)/xmy*F.dg2dcb+cd*F.dg2dy;

//   D -> dg1dx  = F.dg1dx + (1.0-ca*ca)/xmy*F.dg1dcb + ca * F.dg1dy - D -> dg2dx;
//   D -> dg1dcb = -(ysq*cd*F.dg1dcb + x*y*xmy*F.dg1dy)/xmysq - D -> dg2dcb;
//   D -> dg1dy  = -(cborig+ca*cd)/xmy*F.dg1dcb+cd*F.dg1dy - D -> dg2dy;

//   // ddg2*
//   D -> ddg2dxdx = (F.ddg2dxdx + (1.0-ca*ca)/xmy*F.ddg2dxdcb + ca * F.ddg2dxdy)
//     - 3.0*ca*(1.0-ca*ca)/xmysq*F.dg2dcb
//     + (1.0-ca*ca)/xmy*(F.ddg2dxdcb + (1.0-ca*ca)/xmy*F.ddg2dcbdcb + ca * F.ddg2dydcb)
//     + (1.0-ca*ca)/xmy*F.dg2dy + ca*(F.ddg2dxdy + (1.0-ca*ca)/xmy*F.ddg2dydcb + ca * F.ddg2dydy);
     
//   D -> ddg2dxdcb = ysq/(xmysq*xmy)*(cborig+3.0*ca*cd)*F.dg2dcb
//     -ysq/xmysq*cd*(F.ddg2dxdcb + (1.0-ca*ca)/xmy*F.ddg2dcbdcb + ca * F.ddg2dydcb)
//     + y/xmysq*(x*ca-xmy)*F.dg2dy-x*y/xmy*(F.ddg2dxdy + (1.0-ca*ca)/xmy*F.ddg2dydcb
// 					  + ca * F.ddg2dydy);
  
//   D -> ddg2dxdy = (-(cborig+ca*cd)/xmy*F.ddg2dxdcb+cd*F.ddg2dxdy)
//     +(2.0*ca*cborig+cd*(3.0*ca*ca-1.0))/xmysq*F.dg2dcb
//     +(1.0 - ca*ca)/xmy*(-(cborig + ca*cd)/xmy*F.ddg2dcbdcb + cd*F.ddg2dydcb)
//     -(cborig+ca*cd)/xmy*F.dg2dy + ca*(-(cborig+ca*cd)/xmy*F.ddg2dydcb+cd*F.ddg2dydy);

//   D -> ddg2dydcb = y/(xmysq*xmy)*((3.0*cd*cd-1.0)*y-2*xmy*cd)*F.dg2dcb
//     -ysq/(xmysq)*cd*(-(cborig+ca*cd)/xmy*F.ddg2dcbdcb+cd*F.ddg2dydcb)
//     + x/xmysq*(y*cd-xmy)*F.dg2dy
//     -x*y/xmy*(-(cborig+ca*cd)/xmy*F.ddg2dydcb+cd*F.ddg2dydy);

//   D -> ddg2dcbdcb = x*ysq/(xmysq*xmysq)*(xmy-3.0*y*cd)*F.dg2dcb
//     - ysq/(xmysq)*cd*(-(ysq*cd*F.ddg2dcbdcb + x*y*xmy*F.ddg2dydcb)/xmysq)
//     - xsq*ysq/(xmysq*xmy)*F.dg2dy
//     - x*y/xmy*(-(ysq*cd*F.ddg2dydcb + x*y*xmy*F.ddg2dydy)/xmysq);

//   // these have the 1s in
//   D -> ddg1dxdx = ( F.ddg1dxdx + (1.0-ca*ca)/xmy*F.ddg1dxdcb + ca * F.ddg1dxdy)
//     - 3.0*ca*(1.0-ca*ca)/xmysq*F.dg1dcb
//     + (1.0-ca*ca)/xmy*(F.ddg1dxdcb + (1.0-ca*ca)/xmy*F.ddg1dcbdcb + ca * F.ddg1dydcb)
//     + (1.0-ca*ca)/xmy*F.dg1dy + ca*(F.ddg1dxdy + (1.0-ca*ca)/xmy*F.ddg1dydcb + ca * F.ddg1dydy)
//     - D -> ddg2dxdx;
  
//   D -> ddg1dxdcb = ysq/(xmysq*xmy)*(cborig+3.0*ca*cd)*F.dg1dcb
//     - ysq/xmysq*cd*(F.ddg1dxdcb + (1.0-ca*ca)/xmy*F.ddg1dcbdcb + ca * F.ddg1dydcb)
//     + y/xmysq*(x*ca-xmy)*F.dg1dy
//     - x*y/xmy*(F.ddg1dxdy+ (1.0-ca*ca)/xmy*F.ddg1dydcb + ca * F.ddg1dydy)
//     - D -> ddg2dxdcb;

//   D -> ddg1dxdy = (-(cborig+ca*cd)/xmy*F.ddg1dxdcb+cd*F.ddg1dxdy)
//     +(2.0*ca*cborig+cd*(3.0*ca*ca-1.0))/xmysq*F.dg1dcb
//     +(1.0 - ca*ca)/xmy*(-(cborig + ca*cd)/xmy*F.ddg1dcbdcb + cd*F.ddg1dydcb)
//     -(cborig+ca*cd)/xmy*F.dg1dy + ca*(-(cborig+ca*cd)/xmy*F.ddg1dydcb+cd*F.ddg1dydy)
//     - D -> ddg2dxdy;

//   D -> ddg1dydcb = y/(xmysq*xmy)*((3.0*cd*cd-1.0)*y-2*xmy*cd)*F.dg1dcb
//     -ysq/(xmysq)*cd*(-(cborig+ca*cd)/xmy*F.ddg1dcbdcb+cd*F.ddg1dydcb)
//     + x/xmysq*(y*cd-xmy)*F.dg1dy-x*y/xmy*(-(cborig+ca*cd)/xmy*F.ddg1dydcb+cd*F.ddg1dydy)
//     - D -> ddg2dydcb;

//   D -> ddg1dcbdcb = x*ysq/(xmysq*xmysq)*(xmy-3.0*y*cd)*F.dg1dcb
//     - ysq/(xmysq)*cd*(-(ysq*cd*F.ddg1dcbdcb + x*y*xmy*F.ddg1dydcb)/xmysq)
//     - xsq*ysq/(xmysq*xmy)*F.dg1dy
//     - x*y/xmy*(-(ysq*cd*F.ddg1dydcb + x*y*xmy*F.ddg1dydy)/xmysq)
//     - D -> ddg2dcbdcb;
  
//   return ;
// }

// using the computed values in D sets the array dv
__device__
static void
get_dv( const double x ,
	const double xv[4] ,
	const double y ,
	const double yv[4] ,
	const double cb ,
	struct vtmp D ,
	double dv[4][4][4] )
{
  // look up tables and general precomputations
  const double xhat[4] = { xv[0]/x , xv[1]/x , xv[2]/x , xv[3]/x } ;
  const double yhat[4] = { yv[0]/y , yv[1]/y , yv[2]/y , yv[3]/y } ;
  const double c2v[4] = { (xhat[0]-cb*yhat[0])/y , (xhat[1]-cb*yhat[1])/y ,
			  (xhat[2]-cb*yhat[2])/y , (xhat[3]-cb*yhat[3])/y } ;
  const double c4v[4] = { (yhat[0]-cb*xhat[0])/x , (yhat[1]-cb*xhat[1])/x ,
			  (yhat[2]-cb*xhat[2])/x , (yhat[3]-cb*xhat[3])/x } ;
  const double D1[4] = { xv[0]*D.ddg1dxdx+yv[0]*D.ddg2dxdx ,
			 xv[1]*D.ddg1dxdx+yv[1]*D.ddg2dxdx ,
			 xv[2]*D.ddg1dxdx+yv[2]*D.ddg2dxdx ,
			 xv[3]*D.ddg1dxdx+yv[3]*D.ddg2dxdx } ;
  const double D2[4] = { xv[0]*D.ddg1dxdy+yv[0]*D.ddg2dxdy ,
			 xv[1]*D.ddg1dxdy+yv[1]*D.ddg2dxdy ,
			 xv[2]*D.ddg1dxdy+yv[2]*D.ddg2dxdy ,
			 xv[3]*D.ddg1dxdy+yv[3]*D.ddg2dxdy } ;
  const double D3[4] = { xv[0]*D.ddg1dxdcb+yv[0]*D.ddg2dxdcb ,
			 xv[1]*D.ddg1dxdcb+yv[1]*D.ddg2dxdcb ,
			 xv[2]*D.ddg1dxdcb+yv[2]*D.ddg2dxdcb ,
			 xv[3]*D.ddg1dxdcb+yv[3]*D.ddg2dxdcb } ;
  const double D4[4] = { xv[0]*D.ddg1dydcb+yv[0]*D.ddg2dydcb ,
			 xv[1]*D.ddg1dydcb+yv[1]*D.ddg2dydcb ,
			 xv[2]*D.ddg1dydcb+yv[2]*D.ddg2dydcb ,
			 xv[3]*D.ddg1dydcb+yv[3]*D.ddg2dydcb } ;
  const double D5[4] = { xv[0]*D.ddg1dxdcb+yv[0]*D.ddg2dxdcb ,
			 xv[1]*D.ddg1dxdcb+yv[1]*D.ddg2dxdcb ,
			 xv[2]*D.ddg1dxdcb+yv[2]*D.ddg2dxdcb ,
			 xv[3]*D.ddg1dxdcb+yv[3]*D.ddg2dxdcb } ;
  const double D6[4] = { xv[0]*D.ddg1dcbdcb+yv[0]*D.ddg2dcbdcb ,
			 xv[1]*D.ddg1dcbdcb+yv[1]*D.ddg2dcbdcb ,
			 xv[2]*D.ddg1dcbdcb+yv[2]*D.ddg2dcbdcb ,
			 xv[3]*D.ddg1dcbdcb+yv[3]*D.ddg2dcbdcb } ;
  const double D7[4] = { (xv[0]*D.dg1dx+yv[0]*D.dg2dx)/x ,
			 (xv[1]*D.dg1dx+yv[1]*D.dg2dx)/x ,
			 (xv[2]*D.dg1dx+yv[2]*D.dg2dx)/x ,
			 (xv[3]*D.dg1dx+yv[3]*D.dg2dx)/x } ;
  const double D8[4] = { (xv[0]*D.dg1dcb+yv[0]*D.dg2dcb)/x ,
			 (xv[1]*D.dg1dcb+yv[1]*D.dg2dcb)/x ,
			 (xv[2]*D.dg1dcb+yv[2]*D.dg2dcb)/x ,
			 (xv[3]*D.dg1dcb+yv[3]*D.dg2dcb)/x } ;

  int bet , alf , dta ;
  for(alf=0;alf<4;alf++)  {
    const double astuff = (D.dg1dx+D.dg2dx)*xhat[alf]+(D.dg1dcb+D.dg2dcb)*c4v[alf] ;
    for(bet=0;bet<4;bet++)  {
      // this only depends on beta
      const double bstuff = (D.dg1dx*xhat[bet]+D.dg1dy*yhat[bet]
			     +D.dg1dcb*(xhat[bet]/y+yhat[bet]/x
					-cb*(xhat[bet]/x+yhat[bet]/y))) ;
      const double c2c4 = c2v[bet]+c4v[bet] ;
      const int d_ab = ( alf==bet ) ;
      const double xaxb = xhat[alf]*xhat[bet] ;
      const double yayb = yhat[alf]*yhat[bet] ;
      const double xayb = xhat[alf]*yhat[bet] ;
      const double yaxb = yhat[alf]*xhat[bet] ;
      const double c1 = ((cb*(3.0*xaxb-d_ab)-xayb-yaxb)/x+(d_ab+xayb*cb-xaxb-yayb)/y) ;
      
      for(dta=0;dta<4;dta++) {
	const int d_bd = (bet==dta) ;
	const int d_ad = (alf==dta) ;
	
	double f = d_bd*astuff; 
	 
	f += d_ad*bstuff;
	f += D1[dta]*xaxb+D2[dta]*xayb;
	f += (D3[dta]*xhat[bet]+D4[dta]*yhat[bet])*c4v[alf];
	f += (D5[dta]*xhat[alf]+D6[dta]*c4v[alf])*c2c4 ;
	f += D7[dta]*(d_ab-xaxb);

	dv[alf][bet][dta] = f + D8[dta]*c1 ;
      }
    }
  }
  return ;
}

// returns dv[alf][bet][dta] = \partial^(x)_\alpha (\partial^(x)_\beta + \partial^(y)_\beta) < \epsilon_\delta I>_\epsilon
__device__
int
chnr_dV( const double xv[4] ,
	 const double yv[4] ,
	 const struct invariants& Inv ,
	 const struct Grid_coeffs& Grid ,
	 double dv[4][4][4] )
{
  struct vtmp D = {} ; // zero this guy

  if( Inv.x < __ldg(Grid.XX) ) {
    // FORNOW:
    // if( taylorx_contrib( Grid , &D , Inv.x , Inv.y , Inv.xsq ,
    //     		 Inv.flag2 , Inv.cb ) == 1 ) {
    //   return 1 ;
    // }
  } else if( Inv.x > __ldg(Grid.XX + Grid.nstpx -1 ) ) {
    return 1 ;
  } else {

    // this one is all on its own and that is sad
    D.dg3dy = extractff( QG3   , true , d0cb, Inv, Grid );
    
    // use the new extract code
    double f[4];
    extractff2( QG2 , d0cb , Inv , Grid , f ) ;
    D.g2    = f[3] ;
    D.dg2dy = f[2] ;

    extractff2( dxQG2 , d0cb , Inv , Grid , f ) ;
    D.ddg2dxdx  = f[1] ;
    D.dg2dx     = f[3] ;
    D.ddg2dxdy  = f[2] ;
    D.ddg2dxdcb = f[0] ;

    extractff2( dxQG3 , d0cb , Inv , Grid , f ) ;
    D.ddg3dxdx  = f[1] ;
    D.dg3dx     = f[3] ;
    D.ddg3dxdy  = f[2] ;
    D.ddg3dxdcb = f[0] ;
    
    extractff2( QG2 , d1cb , Inv , Grid , f ) ;
    D.dg2dcb     = f[3] ;
    D.ddg2dydcb  = f[2] ;
    D.ddg2dcbdcb = f[0] ;

    extractff2( QG3 , d1cb , Inv , Grid , f ) ;
    D.dg3dcb     = f[3] ;
    D.ddg3dydcb  = f[2] ;
    D.ddg3dcbdcb = f[0] ;

    D.dg1dx  = D.dg3dx + Inv.y/Inv.xsq*Inv.cb*D.g2 - Inv.y/Inv.x*Inv.cb*D.dg2dx;
    D.dg1dy  = D.dg3dy - Inv.cb/Inv.x*D.g2 -(Inv.y/Inv.x)*Inv.cb*D.dg2dy;
    D.dg1dcb = D.dg3dcb - Inv.y/Inv.x*D.g2 - Inv.y/Inv.x*Inv.cb*D.dg2dcb;
    
    D.ddg1dxdx   =  D.ddg3dxdx
      - 2.0*Inv.y/(Inv.xsq*Inv.x)*Inv.cb*D.g2
      + Inv.y/Inv.xsq*Inv.cb*D.dg2dx
      + Inv.y/Inv.xsq*Inv.cb*D.dg2dx
      - Inv.y/Inv.x*Inv.cb*D.ddg2dxdx;

    D.ddg1dxdy   = D.ddg3dxdy +Inv.cb/Inv.xsq*D.g2
      + Inv.y*Inv.cb/Inv.xsq*D.dg2dy
      - Inv.cb/Inv.x*D.dg2dx
      - Inv.y/Inv.x*Inv.cb*D.ddg2dxdy;

    D.ddg1dxdcb  = D.ddg3dxdcb
      + Inv.y/Inv.xsq*D.g2
      + Inv.y/Inv.xsq*Inv.cb*D.dg2dcb
      - Inv.y/Inv.x*D.dg2dx
      - Inv.y/Inv.x*Inv.cb*D.ddg2dxdcb;
    
    D.ddg1dydcb  = D.ddg3dydcb
      - D.g2/Inv.x
      - Inv.cb/Inv.x*D.dg2dcb
      - Inv.y/Inv.x*D.dg2dy
      - Inv.y/Inv.x*Inv.cb*D.ddg2dydcb;
    
    D.ddg1dcbdcb = D.ddg3dcbdcb
      - 2.0*Inv.y/Inv.x*D.dg2dcb
      - Inv.y/Inv.x*Inv.cb*D.ddg2dcbdcb;
      
    if( Inv.flag2 ) {
      // in case of swap, you need the second derivative wrt y
      const int iy1 = find_ind( Grid.YY , Inv.y , 0 , Grid.nstpy ) ;

      if( iy1 == Grid.nstpy-1 ) {
	      return 1 ;
      }
      
      struct invariants Inv1 = Inv , Inv2 = Inv ;
      Inv1.y = __ldg(Grid.YY+iy1) ;
      Inv2.y = __ldg(Grid.YY+iy1+1) ;
      
      double fq1 = extractff( QG2, true, d0cb, Inv1, Grid );
      double fq2 = extractff( QG2, true, d0cb, Inv2, Grid );
      D.ddg2dydy = (fq2-fq1)/Grid.ystp;
      
      fq1 = extractff( QG3, true, d0cb, Inv1, Grid );
      fq2 = extractff( QG3, true, d0cb, Inv2, Grid );
      const double ddg3dydy = (fq2-fq1)/Grid.ystp; 
      D.ddg1dydy = ddg3dydy -(Inv.y*D.ddg2dydy+2.0*D.dg2dy)*Inv.cb/Inv.x;
    }
  }
  
  if( Inv.flag2) {
    // here, convert the derivatives of g2(x,ca,xmy) into the derivatives of g2(x,cb,y)
    // FORNOW
    // XMYSWAP_V_contrib2( &D , Inv.x , Inv.xmy , Inv.yorig ,
    //     	       Inv.xsq , Inv.xmysq , Inv.ysq ,
    //     	       Inv.cb , Inv.cborig ) ;
  } 
  
  get_dv( Inv.x , xv , Inv.yorig , yv , Inv.cborig , D , dv ) ;

  return 0 ;
}

// check that a 4-vector is zero
__device__
static inline bool
is_zero( const double x[4] )
{
  return (x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]) < 1E-28 ;
}

// check if x vector is equal to y vector
__device__
static inline bool
x_is_y( const double x[4] ,
	const double y[4] )
{
  const double xmy[4] = {x[0]-y[0],x[1]-y[1],x[2]-y[2],x[3]-y[3]} ;
  return is_zero( xmy ) ;
}


// L0 is the standard version of the kernel
__device__
void
my_QED_kernel_L0(
    const double xv[4] , const double yv[4] , const struct QED_kernel_temps& t ,
    double S, double kerv[6][4][4][4] )
{
  const bool x_is_zero  = is_zero( xv ) ;
  const bool y_is_zero  = is_zero( yv ) ;

  // do the logic
  if( x_is_zero && y_is_zero ) {
    return ;
  }
  if( x_is_zero ) {
    kernelQED_xoryeq0_axpy( yv , t , kerv , S, false, KQED_TABD_XEQ0 ) ;
    return ;
  }
  if( y_is_zero ) {
    kernelQED_xoryeq0_axpy( xv , t , kerv , S, false, KQED_TABD_YEQ0 ) ;
    return ;
  }
  if( x_is_y( xv , yv ) ) {
    kernelQED_xoryeq0_axpy( xv , t , kerv , S, true /* mulam_minus */, KQED_TABD_YEQ0 ) ;
    return ;
  }
  
  // if neither are zero or small we do the standard one
  // TODO:
  // kernelQED_axpy( xv , yv , t , S , kerv ) ;
  
  return ;
}

__device__
int
init_STV( const double xv[4] ,
	  const double yv[4] ,
	  const struct QED_kernel_temps& t ,
	  struct STV *k )
{
  struct invariants Inv;
  set_invariants( xv , yv , t.Grid, Inv ) ;
  // const struct invariants Inv = set_invariants( xv , yv , t.Grid ) ;

  // Not needed
  // memset( k -> Sxv , 0 , 4*sizeof( double ) ) ;
  // memset( k -> Syv , 0 , 4*sizeof( double ) ) ;
  // memset( k -> Txv , 0 , 64*sizeof( double ) ) ;
  // memset( k -> Tyv , 0 , 64*sizeof( double ) ) ;
  // memset( k -> Vv  , 0 , 64*sizeof( double ) ) ;

  // precompuations for the interpolations
  const size_t ix1 = Inv.INVx.idx ;
  size_t ix2 = ix1+1 ;
  if( ix2 >= (size_t)t.Grid.nstpx ) {
    ix2 = ix1 ;
  }

  if( chnr_dS( xv, yv, Inv, t.Grid, k->Sxv, k->Syv ) ||
      chnr_dT( xv, yv, Inv, t.Grid, k->Txv, k->Tyv ) ||
      chnr_dV( xv, yv, Inv, t.Grid, k->Vv ) ) {
    return 1 ;
  }

  return 0 ;
}


__device__
void KQED_LX(
    int ikernel, const double xm[4], const double ym[4],
    const struct QED_kernel_temps& kqed_t, double kerv[6][4][4][4]) {
#if CUDA_N_QED_KERNEL != 2
  #error "Number of QED kernels does not match implementation"
#endif
  // FFs of QED kernel and their derivatives wrt x,cb,y
  struct STV x_y;
  assert(init_STV( xm, ym, kqed_t, &x_y ) == 0);
  if (ikernel == 0) {
    // QED_kernel_L3( xm, ym, kqed_t, kerv );
    // FORNOW
    /*
    const double zero[4] = { 0 , 0 , 0 , 0 } ;
    my_QED_kernel_L0( xm , ym , kqed_t , 1.0, kerv ) ;
    my_QED_kernel_L0( xm , xm , kqed_t , -1.0, kerv) ;
    my_QED_kernel_L0( zero , xm , kqed_t , 1.0, kerv ) ;
    my_QED_kernel_L0( zero , ym , kqed_t , -1.0, kerv ) ;
    */
    kernelQED_axpy( xm , ym , kqed_t , x_y, 1.0, kerv ) ;
  }
  else {
    // QED_Mkernel_L2( 0.4, xm, ym, kqed_t, kerv );
    kernelQED_axpy( xm , ym , kqed_t , x_y, 1.0, kerv ) ;
    // TODO:
    // const double M = 0.4;
    // my_QED_kernel_L0( xm, ym, kqed_t, 1.0, kerv ) ;
    // const double gaussX = exp( -M*(xm[0]*xm[0]+xm[1]*xm[1]+
    //                                xm[2]*xm[2]+xm[3]*xm[3] )/2. ) ;
    // const double gaussY = exp( -M*(ym[0]*ym[0]+ym[1]*ym[1]+
    //                                ym[2]*ym[2]+ym[3]*ym[3] )/2. ) ;
    // const double kt[6][4][4][4] = { 0 } ;
    // const double zero[4] = { 0 , 0 , 0 , 0 } ;
    // my_QED_kernel_L0( zero, ym, kqed_t, 1.0, kt );
    // for (int k = 0; k < 6; ++k) {
    //   for (int mu = 0; mu < 4; ++mu) {
    //     for (int nu = 0; nu < 4; ++nu) {
    //       for (int lambda = 0; lambda < 4; ++lambda) {
    //         kerv[k][mu][nu][lambda] -=
    //             gaussX * (
    //                 kt[k][mu][nu][lambda] -
    //                 M * xm[mu] * ( xm[0]*kt[k][0][nu][lambda] +
    //                                xm[1]*kt[k][1][nu][lambda] +
    //                                xm[2]*kt[k][2][nu][lambda] +
    //                                xm[3]*kt[k][3][nu][lambda] ) ) ;
    //       }
    //     }
    //   }
    // }
    // for (int k = 0; k < 6; ++k) {
    //   for (int mu = 0; mu < 4; ++mu) {
    //     for (int nu = 0; nu < 4; ++nu) {
    //       for (int lambda = 0; lambda < 4; ++lambda) {
    //         kt[k][mu][nu][lambda] = 0;
    //       }
    //     }
    //   }
    // }
    // my_QED_kernel_L0( xm, zero, kqed_t, 1.0, kt );
    // for (int k = 0; k < 6; ++k) {
    //   for (int mu = 0; mu < 4; ++mu) {
    //     for (int nu = 0; nu < 4; ++nu) {
    //       for (int lambda = 0; lambda < 4; ++lambda) {
    //         kerv[k][mu][nu][lambda] -=
    //             gaussY * (
    //                 kt[k][mu][nu][lambda] -
    //                 M * ym[nu] * ( ym[0]*kt[k][mu][0][lambda] +
    //                                ym[1]*kt[k][mu][1][lambda] +
    //                                ym[2]*kt[k][mu][2][lambda] +
    //                                ym[3]*kt[k][mu][3][lambda] ) ) ;
    //       }
    //     }
    //   }
    // }
  }
}

__global__
void ker_4pt_contraction(
    double* _RESTR kernel_sum, const double* _RESTR g_dzu, const double* _RESTR g_dzsu,
    const double* _RESTR fwd_src, const double* _RESTR fwd_y, int iflavor, Coord g_proc_coords,
    Coord gsx, Pair xunit, Coord yv, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom) {

  // Coord origin = get_thread_origin(local_geom);
  int gsx_arr[4] = {gsx.t, gsx.x, gsx.y, gsx.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};

  // double corr_I[6 * 4 * 4 * 4 * 2];
  // double corr_II[6 * 4 * 4 * 4 * 2];
  // double dxu[4 * 12 * 12 * 2];
  // __shared__ double corr_I_shared[CUDA_THREAD_DIM_1D * 6 * 4 * 4 * 4];
  // __shared__ double corr_II_shared[CUDA_THREAD_DIM_1D * 6 * 4 * 4 * 4];
  // double* corr_I_re = &corr_I_shared[threadIdx.x * 6 * 4 * 4 * 4];
  // double* corr_II_re = &corr_II_shared[threadIdx.x * 6 * 4 * 4 * 4];
  double corr_I_re[6 * 4 * 4 * 4];
  double corr_II_re[6 * 4 * 4 * 4];

  double kernel_sum_work[CUDA_N_QED_KERNEL] = { 0 };
  double spinor_work_0[24], spinor_work_1[24];
  double kerv[6][4][4][4] = { 0 };

  for (int ix = blockIdx.x * blockDim.x + threadIdx.x;
       ix < VOLUME; ix += blockDim.x * gridDim.x) {
    Coord coord = lexic2coord(ix, local_geom);
    const int tt = coord.t;
    const int xx = coord.x;
    const int yy = coord.y;
    const int zz = coord.z;
    int coord_arr[4] = {tt, xx, yy, zz};
    int xv[4], xvzh[4];
    #pragma unroll
    for (int rho = 0; rho < 4; ++rho) {
      int xrho = coord_arr[rho] + proc_coord_arr[rho] * local_geom_arr[rho] - gsx_arr[rho];
      xrho = (xrho + global_geom_arr[rho]) % global_geom_arr[rho];
      xv[rho] = coord_map(xrho, global_geom_arr[rho]);
      xvzh[rho] = coord_map_zerohalf(xrho, global_geom_arr[rho]);
    }

    #pragma unroll
    for (int mu = 0; mu < 4; ++mu) {
      /// COMPUTE DXU v1
      double dxu[12 * 12 * 2] = { 0 };
      for (int ia = 0; ia < 12; ++ia) {
        const double* _d = &fwd_src[((1-iflavor) * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
        double* _t = spinor_work_1;
        _fv_eq_gamma_ti_fv(_t, mu, _d);
        _fv_ti_eq_g5(_t);
        for (int ib = 0; ib < 12; ++ib) {
          const double* _u = &fwd_y[(iflavor * 12 + ib) * _GSI(VOLUME) + _GSI(ix)];
          for (int i = 0; i < 12; ++i) {
            double _t_re = _t[2*i];
            double _t_im = _t[2*i+1];
            double _u_re = _u[2*i];
            double _u_im = _u[2*i+1];
            /* -1 factor due to (g5 gmu)^+ = -g5 gmu */
            dxu[(ib * 12 + ia) * 2 + 0] += -(_t_re * _u_re + _t_im * _u_im);
            dxu[(ib * 12 + ia) * 2 + 1] += -(_t_re * _u_im - _t_im * _u_re);
          }
        }
      }

      #pragma unroll
      for (int k = 0; k < 6; ++k) {
        const int sigma = idx_comb.comb[k][1];
        const int rho = idx_comb.comb[k][0];
        #pragma unroll
        for (int nu = 0; nu < 4; ++nu) {
          #pragma unroll
          for (int lambda = 0; lambda < 4; ++lambda) {
            double *_corr_I_re = &corr_I_re[((k * 4 + mu) * 4 + nu) * 4 + lambda];
            double *_corr_II_re = &corr_II_re[((k * 4 + mu) * 4 + nu) * 4 + lambda];
            _corr_I_re[0] = 0.0;
            _corr_II_re[0] = 0.0;
            for (int ia = 0; ia < 12; ++ia) {
              /// COMPUTE DXU v2
              // double dxu[12 * 2];
              // const double* _u = &fwd_y[(iflavor * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
              // for (int ic = 0; ic < 12; ++ic) {
              //   const double* _d = &fwd_src[((1-iflavor) * 12 + ic) * _GSI(VOLUME) + _GSI(ix)];
              //   double* _t = spinor_work_1;
              //   _fv_eq_gamma_ti_fv(_t, mu, _d);
              //   _fv_ti_eq_g5(_t);
              //   for (int i = 0; i < 12; ++i) {
              //     double _t_re = _t[2*i];
              //     double _t_im = _t[2*i+1];
              //     double _u_re = _u[2*i];
              //     double _u_im = _u[2*i+1];
              //     /* -1 factor due to (g5 gmu)^+ = -g5 gmu */
              //     dxu[2*ic] += -(_t_re * _u_re + _t_im * _u_im);
              //     dxu[2*ic+1] += -(_t_re * _u_im - _t_im * _u_re);
              //   }
              // }
              // double *_dxu = &dxu[0];
              double *_dxu = &dxu[ia * 12 * 2];
              double *_t = spinor_work_0;
              _fv_eq_gamma_ti_fv(_t, 5, _dxu);
              double *_g_dxu = spinor_work_1;
              _fv_eq_gamma_ti_fv(_g_dxu, lambda, _t);
              for (int ib = 0; ib < 12; ++ib) {
                double u_re = _g_dxu[2*ib];
                double u_im = _g_dxu[2*ib+1];
                double v_re = g_dzu[(((k * 4 + nu) * 12 + ib) * 12 + ia) * 2];
                double v_im = g_dzu[(((k * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1];
                _corr_I_re[0] -= u_re * v_re - u_im * v_im;
                v_re = (
                    xvzh[rho] * g_dzsu[(((sigma * 4 + nu) * 12 + ib) * 12 + ia) * 2] -
                    xvzh[sigma] * g_dzsu[(((rho * 4 + nu) * 12 + ib) * 12 + ia) * 2] );
                v_im = (
                    xvzh[rho] * g_dzsu[(((sigma * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1] -
                    xvzh[sigma] * g_dzsu[(((rho * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1] );
                _corr_II_re[0] -= u_re * v_re - u_im * v_im;
              }
            }
          }
        }
      }
    }

    double const xm[4] = {
      xv[0] * xunit.a,
      xv[1] * xunit.a,
      xv[2] * xunit.a,
      xv[3] * xunit.a };

    double const ym[4] = {
      yv.t * xunit.a,
      yv.x * xunit.a,
      yv.y * xunit.a,
      yv.z * xunit.a };

    double * const _corr_I_re  = corr_I_re;
    double * const _corr_II_re = corr_II_re;

    double const xm_mi_ym[4] = {
      xm[0] - ym[0],
      xm[1] - ym[1],
      xm[2] - ym[2],
      xm[3] - ym[3] };

    for (int ikernel = 0; ikernel < CUDA_N_QED_KERNEL; ++ikernel) {
      // dtmp += (
      //     kerv1[k][mu][nu][lambda] + kerv2[k][nu][mu][lambda]
      //     - kerv3[k][lambda][nu][mu] ) * _corr_I[2*i]
      //     + kerv3[k][lambda][nu][mu] * _corr_II[2*i];
      double dtmp = 0.;
      int i;
      KQED_LX( ikernel, xm, ym, kqed_t, kerv );
      i = 0;
      for( int k = 0; k < 6; k++ ) {
        for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int lambda = 0; lambda < 4; lambda++ ) {
              dtmp += kerv[k][mu][nu][lambda] * _corr_I_re[i];
              // dtmp += _corr_I_re[i];
              i++;
            }
          }
        }
      }
      KQED_LX( ikernel, ym, xm,       kqed_t, kerv );
      i = 0;
      for( int k = 0; k < 6; k++ ) {
        for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int lambda = 0; lambda < 4; lambda++ ) {
              dtmp += kerv[k][nu][mu][lambda] * _corr_I_re[i];
              // dtmp += _corr_I_re[i];
              i++;
            }
          }
        }
      }
      KQED_LX( ikernel, xm, xm_mi_ym, kqed_t, kerv );
      i = 0;
      for( int k = 0; k < 6; k++ ) {
        for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int lambda = 0; lambda < 4; lambda++ ) {
              dtmp -= kerv[k][lambda][nu][mu] * _corr_I_re[i];
              dtmp += kerv[k][lambda][nu][mu] * _corr_II_re[i];
              // dtmp += _corr_II_re[i] - _corr_I_re[i];
              i++;
            }
          }
        }
      }
      kernel_sum_work[ikernel] += dtmp;
    }

  } // end coord loop

  // reduce (TODO faster reduce algo?)
  for (int ikernel = 0; ikernel < CUDA_N_QED_KERNEL; ++ikernel) {
    atomicAdd_system(&kernel_sum[ikernel], kernel_sum_work[ikernel]);
  }
}

__global__
void ker_2p2_pieces(
    double* _RESTR P1, double* _RESTR P2, double* _RESTR P3,
    const double* _RESTR fwd_y, int iflavor, Coord g_proc_coords,
    Coord gsw, int n_y, Coord* gycoords, Pair xunit, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom, int Lmax) {
  int gsw_arr[4] = {gsw.t, gsw.x, gsw.y, gsw.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};

  double pimn[4][4];
  double spinor_work_0[24], spinor_work_1[24];
  double kerv[6][4][4][4] = { 0 };

  for (int ix = blockIdx.x * blockDim.x + threadIdx.x;
       ix < VOLUME; ix += blockDim.x * gridDim.x) {
    Coord coord = lexic2coord(ix, local_geom);
    int coord_arr[4] = {coord.t, coord.x, coord.y, coord.z};

    for (int nu = 0; nu < 4; ++nu) {
      for (int mu = 0; mu < 4; ++mu) {
        pimn[mu][nu] = 0.0;
        // V2: Sparse summation using properties of gamma matrices
        for (int ib = 0; ib < 12; ++ib) {
          int ia = (gamma_permutation[nu][2*ib]) / 2;
          int sign_re = ((ib >= 6) ? -1 : 1) * gamma_sign[nu][2*ib];
          int sign_im = ((ib >= 6) ? -1 : 1) * gamma_sign[nu][2*ib+1];
          bool re_im_swap = gamma_permutation[nu][2*ib] % 2 == 1;
          
          const double* _u = &fwd_y[(iflavor * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
          double* _t = spinor_work_0;
          _fv_eq_gamma_ti_fv(_t, mu, _u);
          _fv_ti_eq_g5(_t);

          const double* _d = &fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(ix)];
          for (int i = 0; i < 12; ++i) {
            const double _t_re = _t[2*i];
            const double _t_im = _t[2*i+1];
            const double _d_re = _d[2*i];
            const double _d_im = _d[2*i+1];
            if (!re_im_swap) {
              pimn[mu][nu] += sign_re * (_d_re * _t_re + _d_im * _t_im);
            }
            else {
              pimn[mu][nu] += sign_im * (_d_re * _t_im - _d_im * _t_re);
            }
          }
        }
        
        // V1: Direct loop and trace
        /*
        for (int ia = 0; ia < 12; ++ia) {
          const double* _u = &fwd_y[(iflavor * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
          double* _t = spinor_work_0;
          _fv_eq_gamma_ti_fv(_t, mu, _u);
          _fv_ti_eq_g5(_t);
          double* _s = spinor_work_1;
          for (int ib = 0; ib < 12; ++ib) {
            const double* _d = &fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(ix)];
            double w_re = 0.0;
            double w_im = 0.0;
            for (int i = 0; i < 12; ++i) {
              const double _t_re = _t[2*i];
              const double _t_im = _t[2*i+1];
              const double _d_re = _d[2*i];
              const double _d_im = _d[2*i+1];
              w_re += _d_re * _t_re + _d_im * _t_im;
              w_im += _d_re * _t_im - _d_im * _t_re;
            }
            _s[2*ib] = w_re;
            _s[2*ib+1] = w_im;
          }
          _fv_ti_eq_g5(_s);
          _fv_eq_gamma_ti_fv(_t, nu, _s);
          // real part
          pimn[mu][nu] += _t[2*ia];
        }
        */
      }
    }

    int z[4];
    #pragma unroll
    for (int rho = 0; rho < 4; ++rho) {
      int zrho = coord_arr[rho] + proc_coord_arr[rho] * local_geom_arr[rho] - gsw_arr[rho];
      z[rho] = (zrho + global_geom_arr[rho]) % global_geom_arr[rho];
    }
    // reduce (TODO faster reduce algo?)
    for (int sigma = 0; sigma < 4; ++sigma) {
      for (int nu = 0; nu < 4; ++nu) {
        for (int rho = 0; rho < 4; ++rho) {
          atomicAdd_system(
              &P1[(((rho * 4) + sigma) * 4 + nu) * Lmax + z[rho]],
              pimn[sigma][nu]);
        }
      }
    }

    for (int yi = 0; yi < n_y; yi++) {
      // For P2: y = (gsy - gsw)
      // For P3: y = (gsw - gsy)
      // We define y = (gsy - gsw) and use -y as input for P3.
      int const gsy_arr[4] = { gycoords[yi].t, gycoords[yi].x, gycoords[yi].y, gycoords[yi].z };
      int const y[4] = {
        (gsy_arr[0] - gsw_arr[0] + global_geom_arr[0]) % global_geom_arr[0],
        (gsy_arr[1] - gsw_arr[1] + global_geom_arr[1]) % global_geom_arr[1],
        (gsy_arr[2] - gsw_arr[2] + global_geom_arr[2]) % global_geom_arr[2],
        (gsy_arr[3] - gsw_arr[3] + global_geom_arr[3]) % global_geom_arr[3]
      };
      int const yv[4] = {
        coord_map_zerohalf(y[0], global_geom.T),
        coord_map_zerohalf(y[1], global_geom.LX),
        coord_map_zerohalf(y[2], global_geom.LY),
        coord_map_zerohalf(y[3], global_geom.LZ)
      };
      double const ym[4] = {
        yv[0] * xunit.a,
        yv[1] * xunit.a,
        yv[2] * xunit.a,
        yv[3] * xunit.a };
      double const ym_minus[4] = {
        -yv[0] * xunit.a,
        -yv[1] * xunit.a,
        -yv[2] * xunit.a,
        -yv[3] * xunit.a };
      
      int const xv[4] = {
        coord_map_zerohalf(z[0], global_geom_arr[0]),
        coord_map_zerohalf(z[1], global_geom_arr[1]),
        coord_map_zerohalf(z[2], global_geom_arr[2]),
        coord_map_zerohalf(z[3], global_geom_arr[3])
      };
      double const xm[4] = {
        xv[0] * xunit.a,
        xv[1] * xunit.a,
        xv[2] * xunit.a,
        xv[3] * xunit.a };
        

      int const x_mi_y[4] = {
        (z[0] - y[0] + global_geom_arr[0]) % global_geom_arr[0],
        (z[1] - y[1] + global_geom_arr[1]) % global_geom_arr[1],
        (z[2] - y[2] + global_geom_arr[2]) % global_geom_arr[2],
        (z[3] - y[3] + global_geom_arr[2]) % global_geom_arr[3]
      };
      int xv_mi_yv[4] = {
        coord_map_zerohalf(x_mi_y[0], global_geom_arr[0]),
        coord_map_zerohalf(x_mi_y[1], global_geom_arr[1]),
        coord_map_zerohalf(x_mi_y[2], global_geom_arr[2]),
        coord_map_zerohalf(x_mi_y[3], global_geom_arr[3])
      };
      double const xm_mi_ym[4] = {
        xv_mi_yv[0] * xunit.a,
        xv_mi_yv[1] * xunit.a,
        xv_mi_yv[2] * xunit.a,
        xv_mi_yv[3] * xunit.a
      };

      #pragma unroll
      for (int ikernel = 0; ikernel < CUDA_N_QED_KERNEL; ++ikernel) {
        double local_P2[4][4][4] = { 0 };
        double local_P3[4][4][4] = { 0 };
        KQED_LX( ikernel, xm, ym, kqed_t, kerv );
        for( int k = 0; k < 6; k++ ) {
          int const rho = idx_comb.comb[k][0];
          int const sigma = idx_comb.comb[k][1];
          for ( int nu = 0; nu < 4; nu++ ) {
            local_P2[rho][sigma][nu] = 0.0;
            for ( int mu = 0; mu < 4; mu++ ) {
              for ( int lambda = 0; lambda < 4; lambda++ ) {
                // FORNOW:
                // kerv[k][mu][nu][lambda] = (xm[mu] - ym[mu]) * (xm[nu] - ym[nu]) + lambda;
                local_P2[rho][sigma][nu] += kerv[k][mu][nu][lambda] * pimn[mu][lambda];
                kerv[k][mu][nu][lambda] = 0.0;
              }
            }
          }
        }
        KQED_LX( ikernel, ym, xm, kqed_t, kerv );
        for( int k = 0; k < 6; k++ ) {
          int const rho = idx_comb.comb[k][0];
          int const sigma = idx_comb.comb[k][1];
          for ( int mu = 0; mu < 4; mu++ ) {
            for ( int nu = 0; nu < 4; nu++ ) {
              for ( int lambda = 0; lambda < 4; lambda++ ) {
                // FORNOW:
                // kerv[k][mu][nu][lambda] += k * (xm[nu] - ym[nu]) * (xm[nu] - ym[nu]);
                local_P2[rho][sigma][nu] += kerv[k][nu][mu][lambda] * pimn[mu][lambda];
                kerv[k][mu][nu][lambda] = 0.0;
              }
            }
          }
        }
        KQED_LX( ikernel, xm_mi_ym, ym_minus, kqed_t, kerv );
        for( int k = 0; k < 6; k++ ) {
          int const rho = idx_comb.comb[k][0];
          int const sigma = idx_comb.comb[k][1];
          for ( int nu = 0; nu < 4; nu++ ) {
            local_P3[rho][sigma][nu] = 0.0;
            for ( int mu = 0; mu < 4; mu++ ) {
              for ( int lambda = 0; lambda < 4; lambda++ ) {
                // FORNOW:
                // kerv[k][mu][nu][lambda] += (xm[mu] - ym[mu]) * (xm[nu] - ym[mu]) - k;
                local_P3[rho][sigma][nu] += kerv[k][mu][lambda][nu] * pimn[mu][lambda];
                kerv[k][mu][nu][lambda] = 0.0;
              }
            }
          }
        }

        // reduce (TODO faster reduce algo?)
        for (int rho = 0; rho < 4; ++rho) {
          for (int sigma = 0; sigma < 4; ++sigma) {
            for (int nu = 0; nu < 4; ++nu) {
              atomicAdd_system(
                  &P2[(((yi * CUDA_N_QED_KERNEL + ikernel) * 4 + rho) * 4 + sigma) * 4 + nu],
                  local_P2[rho][sigma][nu]);
              atomicAdd_system(
                  &P3[(((yi * CUDA_N_QED_KERNEL + ikernel) * 4 + rho) * 4 + sigma) * 4 + nu],
                  local_P3[rho][sigma][nu]);
            }
          }
        }

      }
    }
  }
}

/**
 * Top-level operations.
 */
void cu_spinor_field_eq_gamma_ti_spinor_field(double* out, int mu, const double* in, size_t len) {
  const size_t BS_spinor = 12 * CUDA_BLOCK_SIZE * CUDA_THREAD_DIM_1D;
  size_t nx = (len + BS_spinor - 1) / BS_spinor;
  dim3 kernel_nblocks(nx);
  dim3 kernel_nthreads(CUDA_THREAD_DIM_1D);
  ker_spinor_field_eq_gamma_ti_spinor_field<<<kernel_nblocks, kernel_nthreads>>>(
      out, mu, in, len);
}

void cu_g5_phi(double* spinor, size_t len) {
  const size_t BS_spinor = 12 * CUDA_BLOCK_SIZE * CUDA_THREAD_DIM_1D;
  size_t nx = (len + BS_spinor - 1) / BS_spinor;
  dim3 kernel_nblocks(nx);
  dim3 kernel_nthreads(CUDA_THREAD_DIM_1D);
  ker_g5_phi<<<kernel_nblocks, kernel_nthreads>>>(spinor, len);
}

void cu_dzu_dzsu(
    double* d_dzu, double* d_dzsu, const double* fwd_src, const double* fwd_y,
    int iflavor, Coord proc_coords, Coord gsx,
    Geom global_geom, Geom local_geom) {
  size_t T = local_geom.T;
  size_t LX = local_geom.LX;
  size_t LY = local_geom.LY;
  size_t LZ = local_geom.LZ;
  size_t VOLUME = T * LX * LY * LZ;
  size_t kernel_nthreads = CUDA_THREAD_DIM_1D;
  size_t kernel_nblocks = (VOLUME + kernel_nthreads - 1) / kernel_nthreads;
  // const size_t BS_TX = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE;
  // const size_t BS_Y = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // const size_t BS_Z = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // size_t nx = (T*LX + BS_TX - 1) / BS_TX;
  // size_t ny = (LY + BS_Y - 1) / BS_Y;
  // size_t nz = (LZ + BS_Z - 1) / BS_Z;
  // dim3 kernel_nblocks(nx, ny, nz);
  // dim3 kernel_nthreads(CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D);
  ker_dzu_dzsu<<<kernel_nblocks, kernel_nthreads>>>(
      d_dzu, d_dzsu, fwd_src, fwd_y, iflavor, proc_coords, gsx,
      global_geom, local_geom);
}

void cu_4pt_contraction(
    double* d_kernel_sum, const double* d_g_dzu, const double* d_g_dzsu,
    const double* fwd_src, const double* fwd_y, int iflavor, Coord proc_coords,
    Coord gsx, Pair xunit, Coord yv, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom) {
  size_t T = local_geom.T;
  size_t LX = local_geom.LX;
  size_t LY = local_geom.LY;
  size_t LZ = local_geom.LZ;
  size_t VOLUME = T * LX * LY * LZ;
  size_t kernel_nthreads = CUDA_THREAD_DIM_1D;
  size_t kernel_nblocks = (VOLUME + kernel_nthreads - 1) / kernel_nthreads;
  // const size_t BS_TX = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE;
  // const size_t BS_Y = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // const size_t BS_Z = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // size_t nx = (T*LX + BS_TX - 1) / BS_TX;
  // size_t ny = (LY + BS_Y - 1) / BS_Y;
  // size_t nz = (LZ + BS_Z - 1) / BS_Z;
  // dim3 kernel_nblocks(nx, ny, nz);
  // dim3 kernel_nthreads(CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D);
  ker_4pt_contraction<<<kernel_nblocks, kernel_nthreads>>>(
      d_kernel_sum, d_g_dzu, d_g_dzsu, fwd_src, fwd_y, iflavor, proc_coords,
      gsx, xunit, yv, kqed_t, global_geom, local_geom);
}

void cu_2p2_pieces(
    double* d_P1, double* d_P2, double* d_P3, const double* fwd_y, int iflavor,
    Coord proc_coords, Coord gsw, int n_y, Coord* d_ycoords, Pair xunit,
    QED_kernel_temps kqed_t, Geom global_geom, Geom local_geom) {
  size_t T = local_geom.T;
  size_t LX = local_geom.LX;
  size_t LY = local_geom.LY;
  size_t LZ = local_geom.LZ;
  size_t VOLUME = T * LX * LY * LZ;
  size_t kernel_nthreads = CUDA_THREAD_DIM_1D;
  size_t kernel_nblocks = (VOLUME + kernel_nthreads - 1) / kernel_nthreads;
  int Lmax = 0;
  if (global_geom.T >= Lmax) Lmax = global_geom.T;
  if (global_geom.LX >= Lmax) Lmax = global_geom.LX;
  if (global_geom.LY >= Lmax) Lmax = global_geom.LY;
  if (global_geom.LZ >= Lmax) Lmax = global_geom.LZ;
  ker_2p2_pieces<<<kernel_nblocks, kernel_nthreads>>>(
      d_P1, d_P2, d_P3, fwd_y, iflavor, proc_coords, gsw, n_y, d_ycoords, xunit,
      kqed_t, global_geom, local_geom, Lmax);
}


/**
 * Simple interface to KQED.
 * NOTE: One should not really use these single-point evaluations in production,
 * they are only intended to test that the CUDA QED kernels work.
 */
#if 0
#include "KQED.h"

struct __attribute__((packed, aligned(8))) Vec4 {
  double x[4];
};
inline Vec4 vec4(const double xv[4]) {
  Vec4 pt;
  for (int i = 0; i < 4; ++i) {
    pt.x[i] = xv[i];
  }
  return pt;
}
struct __attribute__((packed, aligned(8))) OneKernel {
  double k[6][4][4][4] ;
};

__global__
void
ker_QED_kernel_L0(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L0(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}
__global__
void
ker_QED_kernel_L1(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L1(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}
__global__
void
ker_QED_kernel_L2(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L2(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}
__global__
void
ker_QED_kernel_L3(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L3(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}

void
cu_pt_QED_kernel_L0(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L0<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
void
cu_pt_QED_kernel_L1(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L1<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
void
cu_pt_QED_kernel_L2(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L2<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
void
cu_pt_QED_kernel_L3(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L3<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
#endif
