
#include "LIFGroup.h"

using namespace std;
using namespace boost::numeric::ublas;

LIFGroup::LIFGroup(uint32_t N, 
                   float adapt, 
                   float tau, 
                   float theta, 
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
  
  this->ge.clear();
  this->gi.clear();
  this->v.clear();
  this->last_spk += 1.0;
  // this->last_spk.clear();
  // this->last_spk -= 1.0;
  
  
}

matrix<float> LIFGroup::step(float t, float dt, matrix<float> Iine, matrix<float> Iini)
{
  matrix<bool> nrefrac = ((t - this->last_spk - this->refrac_per) > 0);
  
  IsynE = this->ge * -1 * this->v;
  IsynI = this->gi * (this->i_offset - this->v);
}

void LIFGroup::reset()
{
}
