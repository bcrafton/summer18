
// lapack 
// blas
// eigan
// boost

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost::numeric::ublas;

int main () {
    
    matrix<double> m (3, 3);
    matrix<double> n (3, 3);
    for (unsigned i = 0; i < m.size1 (); ++ i) {
        for (unsigned j = 0; j < m.size2 (); ++ j) {
            m (i, j) = 3 * i + j;
            n (i, j) = 3 * i + j;
        }
    }
            
    cout << m << endl;
    cout << prod(m, n) << endl;
}
