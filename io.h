
/**
 *  Example code for IO, read binary data vectors and write knng to path.
 *
 */

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "assert.h"
#include <Eigen/Dense>
using std::cout;
using std::endl;
using std::string;
using std::vector;

const int DD = 200;

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knng a (N * 200) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNNG(const std::vector<std::vector<uint32_t>> &knng,
              const std::string &path = "output.bin") {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  const int K = 100;
  //const uint32_t N = knng.size();
  std::cout << "Saving KNN Graph (" << knng.size() << " X 100) to " << path
            << std::endl;
//  cout<<"knng.front().size()" << knng.front().size()<<"\n";
  assert(knng.front().size() == K);
  for (unsigned i = 0; i < knng.size(); ++i) {
    auto const &knn = knng[i];
    ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
  }
  ofs.close();
}

void SaveKNNGRaw(const uint32_t*knng, int size,
              const std::string &path = "output.bin") {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  std::cout << "Saving KNN Graph (" << size/100 << " X 100) to " << path
            << std::endl;
//printf("data %p\n", knng); for(int i=0; i < 100;i++) { printf("%d ", knng[i]); } printf("\n");
  ofs.write(reinterpret_cast<char const *>(knng), size * sizeof(uint32_t));
  ofs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x 200)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             std::vector<std::vector<float>> &data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data.resize(N);
  std::cout << "# of points: " << N << std::endl;

  const int num_dimensions = DD;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    // new: pad 0
//    std::vector<float> row(num_dimensions + 4);

    std::vector<float> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<float>(buff[d]);
    }

    // new: pad 0
//    for (int i =0;i<4;i++)
//      row[100 + i] = 0;
    data[counter++] = std::move(row);
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

/// Read Bin Data for Eigen
/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBinEigen(const std::string &file_path,
             Eigen::MatrixXf&  data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data = Eigen::MatrixXf((int)N, DD);
  std::cout << "# of points: " << N << std::endl;
  const int num_dimensions = DD;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    for (int d = 0; d < num_dimensions; d++) {
      data(counter, d) = static_cast<float>(buff[d]);
    }
    ++counter;
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}


void ReadBinEigenColMajor(const std::string &file_path,
                  Eigen::MatrixXf&  data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data = Eigen::MatrixXf(DD, (int)N);
  std::cout << "# of points: " << N << std::endl;
  const int num_dimensions = DD;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    for (int d = 0; d < num_dimensions; d++) {
      //data(d, counter) = int(static_cast<float>(buff[d])*500)/1.0;
      data(d, counter) = (static_cast<float>(buff[d]));
    }
    ++counter;
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

void ReadBinInt(const std::string &file_path,
                  Eigen::MatrixXd&  data) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points

  ifs.read((char *)&N, sizeof(uint32_t));
  data = Eigen::MatrixXd(DD, (int)N);
  std::cout << "# of points: " << N << std::endl;
  const int num_dimensions = DD;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    for (int d = 0; d < num_dimensions; d++) {
      data(d, counter) = double(static_cast<float>(buff[d]));
    }
    ++counter;
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}
    // generate size distinct random numbers < N
    template<typename RNG>
    static void GenRandom(RNG &rng, uint32_t *addr, uint32_t size, uint32_t N) {
      if (N == size) {
        for (uint32_t i = 0; i < size; ++i) {
          addr[i] = i;
        }
        return;
      }
      for (uint32_t i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
      }
      std::sort(addr, addr + size);
      for (uint32_t i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
          addr[i] = addr[i - 1] + 1;
        }
      }
      uint32_t off = rng() % N;
      for (uint32_t i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
      }
    }
