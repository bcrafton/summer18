
#include "LIFGroup.h"

using namespace std;
using namespace boost::numeric::ublas;

LIFGroup::LIFGroup(uint32_t N, 
                   float adapt, 
                   float tau, 
                   matrix<float> theta, 
                   float vthr, 
                   float vrest, 
                   float vreset, 
                   float refrac_per, 
                   float i_offset, 
                   float tc_theta, 
                   float theta_plus_e) {
  this->N = N;
  this->adapt = adapt;
  this->tau = tau;
  this->theta = theta;
  this->vthr = vthr;
  this->vrest = vrest;
  this->vreset = vreset;
  this->refrac_per = refrac_per;
  this->i_offset = i_offset;
  this->tc_theta = tc_theta;
  this->theta_plus_e = theta_plus_e;
  
  this->ge_tau = 0.001;
  this->gi_tau = 0.002;
  
  this->ge.resize(this->N, 1);
  this->ge.clear();

  this->gi.resize(this->N, 1);
  this->gi.clear();
  
  this->v.resize(this->N, 1);
  this->gi.clear();
  
  this->last_spk.resize(this->N, 1);
  uint32_t ii;
  for (ii=0; ii<this->N; ii++) { 
    this->last_spk(ii, 0) = -1.0;
  }
}

matrix<uint32_t> LIFGroup::step(float t, float dt, matrix<float> Iine, matrix<float> Iini)
{
  matrix<uint32_t> spkd(this->N, 1);
  
  uint32_t ii;
  for(ii=0; ii<this->N; ii++) {
    uint32_t nrefrac = ((t - this->last_spk(ii, 0) - this->refrac_per) > 0);
    
    float IsynE = -1 * this->ge(ii, 0) * this->v(ii, 0);
    float IsynI = this->gi(ii, 0) * (this->i_offset - this->v(ii, 0));
    
    float dv = ((this->vrest - this->v(ii, 0)) + (IsynE + IsynI)) / this->tau * dt;
    float dge = -1 * (this->ge(ii, 0) / this->ge_tau * dt);
    float dgi = -1 * (this->gi(ii, 0) / this->gi_tau * dt);
    
    this->v(ii, 0) += dv * nrefrac;
    this->ge(ii, 0) += (dge + Iine(ii, 0)) * nrefrac;
    this->gi(ii, 0) += (dgi + Iini(ii, 0)) * nrefrac;
    
    spkd(ii, 0) = this->v(ii, 0) > (this->theta(ii, 0) + this->vthr);
    uint32_t nspkd = !spkd(ii, 0);
    
    if (spkd(ii, 0)) {
      this->last_spk(ii, 0) = t;
      this->v(ii, 0) = this->vreset;
      this->ge(ii, 0) = 0.0;
      this->gi(ii, 0) = 0.0;
    }
    
    if (this->adapt) {
      float dtheta = -1 * this->theta(ii, 0) / this->tc_theta * dt;
      this->theta(ii, 0) += dtheta + spkd(ii, 0) * this->theta_plus_e;
    }
  }
  
  return spkd;
}

matrix<uint32_t> LIFGroup::step(float t, float dt, matrix<float> Iine)
{
  matrix<uint32_t> spkd(this->N, 1);
  
  uint32_t ii;
  for(ii=0; ii<this->N; ii++) {
    uint32_t nrefrac = ((t - this->last_spk(ii, 0) - this->refrac_per) > 0);
    
    float IsynE = -1 * this->ge(ii, 0) * this->v(ii, 0);
    float IsynI = this->gi(ii, 0) * (this->i_offset - this->v(ii, 0));
    
    float dv = ((this->vrest - this->v(ii, 0)) + (IsynE + IsynI)) / this->tau * dt;
    float dge = -1 * (this->ge(ii, 0) / this->ge_tau * dt);
    float dgi = -1 * (this->gi(ii, 0) / this->gi_tau * dt);
    
    this->v(ii, 0) += dv * nrefrac;
    this->ge(ii, 0) += (dge + Iine(ii, 0)) * nrefrac;
    this->gi(ii, 0) += dgi * nrefrac;
    
    spkd(ii, 0) = this->v(ii, 0) > (this->theta(ii, 0) + this->vthr);
    uint32_t nspkd = !spkd(ii, 0);
    
    if (spkd(ii, 0)) {
      this->last_spk(ii, 0) = t;
      this->v(ii, 0) = this->vreset;
      this->ge(ii, 0) = 0.0;
      this->gi(ii, 0) = 0.0;
    }
    
    if (this->adapt) {
      float dtheta = -1 * this->theta(ii, 0) / this->tc_theta * dt;
      this->theta(ii, 0) += dtheta + spkd(ii, 0) * this->theta_plus_e;
    }
  }
  
  return spkd;
}

void LIFGroup::reset()
{
  this->ge.clear();
  this->gi.clear();
  this->gi.clear();
  uint32_t ii;
  for (ii=0; ii<this->N; ii++) {
    this->last_spk(ii, 0) = -1.0;
  }
}







