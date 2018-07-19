
#include <cstdio>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost::numeric::ublas;

matrix<double>* load_csv(string filename, unsigned int rows, unsigned int cols)
{
  FILE* fp = fopen(filename.c_str(), "r");
  matrix<double>* m = new matrix<double>(rows, cols);
  
  float next;
  
  int ii, jj;
  for(ii=0; ii<rows; ii++) {
    for(jj=0; jj<cols; jj++) {
      assert(fscanf(fp, "%f", &next));
      (*m)(ii, jj) = next;
    }
  }
  
  return m;
}

int main() {
  matrix<double>* XeAe = load_csv("XeAe.csv", 784, 400);
  matrix<double>* AeAi = load_csv("AeAi.csv", 400, 400);
  matrix<double>* AiAe = load_csv("AiAe.csv", 400, 400);
  
  
}
