
#include "defines.h"
#include "LIFGroup.h"
#include "SynapseGroup.h"

using namespace std;
using namespace boost::numeric::ublas;

matrix<double> load_csv(string filename, unsigned int rows, unsigned int cols)
{
  FILE* fp = fopen(filename.c_str(), "r");
  matrix<double> m = matrix<double>(rows, cols);
  
  float next;
  
  int ii, jj;
  for(ii=0; ii<rows; ii++) {
    for(jj=0; jj<cols; jj++) {
      assert(fscanf(fp, "%f", &next));
      m(ii, jj) = next;
    }
  }
  
  return m;
}

int main() {
  matrix<float> XeAe = load_csv("XeAe.csv", 784, 400);
  matrix<float> AeAi = load_csv("AeAi.csv", 400, 400);
  matrix<float> AiAe = load_csv("AiAe.csv", 400, 400);
  matrix<float> theta_e = load_csv("theta_A.csv", 400, 1);
  matrix<float> theta_i(400, 1); theta_i.clear();
  
  float dt = 0.0005;
  
  float active_T = 0.35;
  int active_steps = active_T / dt;
  // active_Ts = np.linspace(0, active_T, active_steps)
  
  float rest_T = 0.15;
  int rest_steps = rest_T / dt;
  // rest_Ts = np.linspace(active_T, active_T + rest_T, rest_steps)

  uint32_t NUM_EX = 1000;
  
  SynapseGroup Syn(784, 400, XeAe, true, 0.02, 0.02, 0.04, 0.0001, 0.01, 1.0);
  LIFGroup lif_exc(400, true, 0.1, theta_e, -0.072, -0.065, -0.065, 0.005, -0.1, 0.0001, 0.00005);
  LIFGroup lif_inh(400, false, 0.01, theta_i, -0.04, -0.06, -0.045, 0.002, -0.085, 0.0001, 0.00005);

  cout << "starting sim" << endl;

}
