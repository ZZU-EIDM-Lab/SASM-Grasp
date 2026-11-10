 /** 
 * Licensed under the MIT License. To view a copy of this license, visit:
 *     https://opensource.org/licenses/MIT
 */


#include "knn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("knn", &knn, "k-nearest neighbors");
}
