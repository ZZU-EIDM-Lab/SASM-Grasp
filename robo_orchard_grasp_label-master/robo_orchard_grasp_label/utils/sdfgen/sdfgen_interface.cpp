 /** 
 * Licensed under the MIT License. To view a copy of this license, visit:
 *     https://opensource.org/licenses/MIT
 */

#include <torch/extension.h>
#include "makelevelset3.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

void run_sdf(const std::string &filename, int dim, int padding)
{
  if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj"))
  {
    std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
    exit(-1);
  }

  if (padding < 1)
    padding = 1;
  if (2 * padding > dim)
  {
    std::cout << "Padding greater than cube dim" << std::endl;
    exit(1);
  }

  // start with a massive inside out bound box.
  Vec3f min_box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()),
      max_box(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

  std::cout << "Reading data.\n";

  std::ifstream infile(filename);
  if (!infile)
  {
    std::cerr << "Failed to open. Terminating.\n";
    exit(-1);
  }

  int ignored_lines = 0;
  int count = 0;
  std::string slash = "/";
  std::string line;
  std::vector<Vec3f> vertList;
  std::vector<Vec3ui> faceList;
  while (!infile.eof())
  {
    std::getline(infile, line);
    if (line.substr(0, 2) == std::string("v "))
    {
      std::stringstream data(line);
      char c;
      Vec3f point;
      data >> c >> point[0] >> point[1] >> point[2];
      vertList.push_back(point);
      update_minmax(point, min_box, max_box);
    }
    else if (line.substr(0, 2) == std::string("f "))
    {
      std::stringstream data(line);
      char c;
      std::string v0str, v1str, v2str;
      int v0, v1, v2;
      data >> c >> v0str >> v1str >> v2str;

      // get vertex index from slash parsing
      size_t pos = 0;
      std::string token = v0str;
      if ((pos = v0str.find(slash)) != std::string::npos)
      {
        token = v0str.substr(0, pos);
      }
      v0 = atoi(token.c_str());

      pos = 0;
      token = v1str;
      if ((pos = v1str.find(slash)) != std::string::npos)
      {
        token = v1str.substr(0, pos);
      }
      v1 = atoi(token.c_str());

      pos = 0;
      token = v2str;
      if ((pos = v2str.find(slash)) != std::string::npos)
      {
        token = v2str.substr(0, pos);
      }
      v2 = atoi(token.c_str());

      // if (v0 - 1 == 2649) {
      //   std::cout << "line " << line << std::endl;
      //   std::cout << "data " << c << " " << v0 << " " << v1 << " " << v2 << std::endl;
      // }
      count++;
      faceList.push_back(Vec3ui(v0 - 1, v1 - 1, v2 - 1));
    }
    else
    {
      ++ignored_lines;
    }
  }
  infile.close();

  if (ignored_lines > 0)
    std::cout << "Warning: " << ignored_lines << " lines were ignored since they did not contain faces or vertices.\n";

  std::cout << "Read in " << vertList.size() << " vertices and " << faceList.size() << " faces." << std::endl;

  // Add padding around the box.
  Vec3f unit(1, 1, 1);
  // min_box -= padding*dx*unit;
  // max_box += padding*dx*unit;
  float max_diff = std::max(max_box.v[0] - min_box.v[0], std::max(max_box.v[1] - min_box.v[1], max_box.v[2] - min_box.v[2]));
  //  float max_bound = std::max(max_box.v[0], std::max(max_box.v[1], max_box.v[2]));
  float dx = max_diff / (dim - 2 * padding);
  std::cout << "Resolution: " << dx << " with real dimension " << max_diff << std::endl;
  Vec3ui sizes = Vec3ui(dim);

  // remap the min bounds so that the object is centered
  Vec3f center = min_box + (max_box - min_box) / 2;
  std::cout << "Center of grid " << center << std::endl;
  std::cout << "Original bounding box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;
  min_box = center - Vec3f((dim * dx) / 2);

  //  min_box -= padding*dx*unit;

  std::cout << "Rescaled bounding box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;

  std::cout << "Computing signed distance field.\n";
  Array3f phi_grid;
  make_level_set3(faceList, vertList, min_box, dx, sizes[0], sizes[1], sizes[2], phi_grid);

  // Very hackily strip off file suffix.
  std::string outname = filename.substr(0, filename.size() - 4) + std::string(".sdf");
  std::cout << "Writing results to: " << outname << "\n";
  std::ofstream outfile(outname.c_str());
  outfile << phi_grid.ni << " " << phi_grid.nj << " " << phi_grid.nk << std::endl;
  outfile << min_box[0] << " " << min_box[1] << " " << min_box[2] << std::endl;
  outfile << dx << std::endl;
  for (unsigned int i = 0; i < phi_grid.a.size(); ++i)
  {
    outfile << phi_grid.a[i] << std::endl;
  }
  outfile.close();
  std::cout << "Processing complete.\n";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("run_sdf", &run_sdf, "Generate SDF from OBJ (filename, dim, padding)");
}
