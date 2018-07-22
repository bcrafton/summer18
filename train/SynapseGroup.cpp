
#include "SynapseGroup.h"

using namespace std;
using namespace boost::numeric::ublas;

SynapseGroup::SynapseGroup(uint32_t N, 
                           uint32_t M, 
                           matrix<float> w, 
                           bool stdp, 
                           float tc_pre_ee, 
                           float tc_post_1_ee, 
                           float tc_post_2_ee, 
                           float nu_ee_pre, 
                           float nu_ee_post, 
                           float wmax_ee) {
  this->N = N;
  this->M = M;
  this->w = w;
  this->stdp = stdp;
  this->tc_pre_ee = tc_pre_ee;
  this->tc_post_1_ee = tc_post_1_ee;
  this->tc_post_2_ee = tc_post_2_ee;
  this->nu_ee_pre = nu_ee_pre;
  this->nu_ee_post = nu_ee_post;
  this->wmax_ee = wmax_ee;
  
  this->last_pre.resize(this->N, 1);
  this->last_pre.clear();

  this->last_post.resize(this->N, 1);
  this->last_post.clear();
}
               
matrix<float> SynapseGroup::step(float t, float dt, matrix<bool> pre_spk, matrix<bool> post_spk)
{
  // element_prod
  // we can use this!

  matrix<float> I = prod(trans(pre_spk), this->w);
  return I;
  matrix<float> dw(this->N, this->M);
  
  uint32_t ii, jj;
  
  if (this->stdp) {
    
    // evaluate pres no matter what.
    ///////////////////////////
    //////////PRE//////////////
    ///////////////////////////
    for(ii=0; ii<this->N; ii++) {
      if (pre_spk(ii, 0)) {
        this->last_pre(ii, 0) = t;
      }
      
      for(jj=0; jj<this->M; jj++) {
        float post = exp(-(t - this->last_post(jj, 0)) / this->tc_post_1_ee);
        dw(ii, jj) = -1 * this->nu_ee_pre * pre_spk(ii, 0) * post;
      }
    }
   
    // this is benefit from only evaluating when we get a post since far less posts than dts
    ///////////////////////////
    //////////POST/////////////
    ///////////////////////////
    bool got_post = false;
    for(ii=0; ii<M; ii++) {
      got_post |= post_spk(ii, 0);
    }
    if(got_post) {
    
      for(jj=0; jj<this->M; jj++) {
        if(post_spk(jj, 0)) {
          this->last_post(jj, 0) = t; 
        }
      
        float post2 = exp(-(t - this->last_post(jj, 0)) / this->tc_post_2_ee);
        for(ii=0; ii<this->N; ii++) {
          float pre = exp(-(t - this->last_pre(ii, 0)) / this->tc_pre_ee);
          dw(ii, jj) += this->nu_ee_post * pre * post2 * post_spk(jj, 0);
        }
      } 
    }
    
  } // if (this->stdp) {
  return I;
} // matrix<float> SynapseGroup::step ...

void SynapseGroup::reset()
{
  this->last_pre.resize(this->N, 1);
  this->last_pre.clear();

  this->last_post.resize(this->N, 1);
  this->last_post.clear();
  
  if(this->stdp) {
    // do the normalization
    // use a row vector of ones.
    /*
    col_sum = np.sum(np.copy(self.w), axis=0)
    col_factor = 78.0 / col_sum
    for i in range(self.M):
      self.w[:, i] *= col_factor[i]
    */
  }
}





