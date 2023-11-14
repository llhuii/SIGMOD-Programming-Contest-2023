#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_AVX512
#define EIGEN_DONT_PARALLELIZE

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_AVX512
//#define EIGEN_USE_THREADS 1
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_RUNTIME_NO_MALLOC


#include <sys/time.h>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include "assert.h"
#include "io.h"
#include "nn-descent/kgraph.h"
#include <omp.h>
// for avx
#include <x86intrin.h>

// for timer
#include <boost/timer/timer.hpp>
//#include "include/efanna2e/index_kdtree.h"
//#define timer timer_for_boost_progress_t



// for Eigen
#include <Eigen/Dense>
using namespace boost;

#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#endif
using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace kgraph;

#define _INT_MAX 2147483640



// Modify for Eigen
typedef Eigen::MatrixXf MyType;

const int K = 100;


typedef float ResultType;


// 【Eigen dist】
float compare(const MyType& a, const MyType& b) {
	//  return ((a - b)*(a - b).transpose())(0,0);
	return ((a - b).transpose() *(a-b))(0,0);
}

// 【Eigen dist id】
float compare_with_id(const MyType& a, const MyType& b, uint32_t id_a, uint32_t id_b) {
	//  return (urn ((a - b)*(a - b).transpose())(0,0);
	//  return ((a - b).transpose() *(a-b))(0,0);
	//  Eigen::MatrixXf tmp = (- 2 * (a.transpose()) * b);
	//  cout<<KGraph::square_sums.size() <<" "<<KGraph::square_sums.rows() <<" "<<KGraph::square_sums.cols()<<"\n";
	//  float ret = KGraph::square_sums(id_a, 0);
	//  float ret = KGraph::square_sums(id_a, 0) + KGraph::square_sums(id_b, 0) -2 * (a.transpose()) * b)(0,0);
	//  return ret;
	return (KGraph::square_sums(id_a, Eigen::all) + KGraph::square_sums(id_b, Eigen::all) - (((a.transpose()) * b)))(0,0);
}



// 【Eigen version】
typedef kgraph::VectorOracle<MyType, MyType> MyOracle;



int main1(int argc, char **argv) {
	boost::timer::cpu_timer timer;
	string source_path = "dummy-data.bin";

	// Also accept other path for source data
	if (argc > 1) {
		source_path = string(argv[1]);
	}
	omp_set_num_threads(32);

	// Read data points
	//  ReadBinEigen(source_path, KGraph::nodes);   // Eigen version
	ReadBinEigenColMajor(source_path, KGraph::nodes);   // Eigen version


	cout<<KGraph::nodes.cols()<<"\n";
	int n =  KGraph::nodes.cols();  // note: this should be rows rather than size!


	// K-graph related
	MyOracle oracle(KGraph::nodes, compare, compare_with_id);


	KGraph *index = KGraph::create();



	KGraph::IndexParams params;

	params.S = 100;
	params.K = 100;
	params.L=  165;
	params.R = 251;
	params.iterations= 7;


	params.recall = 1.0;
	params.delta = 0.0002;

	// 【For submit】
	  params.controls = 0;

	// 【For local evaluation】
// params.controls= 100;

           uint32_t *data= (uint32_t*)KGraph::nodes.data();
	printf("Build starting with S: %d, K: %d, L: %d, R:%d !\n", params.S, params.K, params.L, params.R);

	index->build(oracle, params);

	auto times_build = timer.elapsed();
	printf("Build finished S: %d, K :%d, L: %d, R:%d !\n", params.S, params.K, params.L, params.R);
	std::cerr << "Build time: " << times_build.wall / 1e9 <<"\n";

	// Save to ouput.bin
	// SaveKNNG(index->knng);
	SaveKNNGRaw(data, n * K);
	auto times_save = timer.elapsed();
	std::cerr << "Save time: " << (times_save.wall - times_build.wall) / 1e9 << "\n";

	return 0;
}


#define PINT0

#ifdef PINT
typedef Eigen::MatrixXd PerfType;
	Eigen::VectorXd square_sums;
#else
typedef Eigen::MatrixXf PerfType;
	Eigen::VectorXf square_sums;

#endif
	PerfType nodes;


static void squared_dist(const vector<uint32_t>& idA, const vector<uint32_t>& idB, PerfType& D){  // Compute squared Euclidean dist between 2 matrixes
Eigen::internal::set_is_malloc_allowed(false);
	//          // (100, na)   (100, nb)  ->
	PerfType A = nodes(Eigen::all, idA); // (100, na)
	PerfType B = nodes(Eigen::all, idB).transpose(); // (nb, 100)
	// get square sum
	PerfType A2 = square_sums(idA, Eigen::all).transpose();  // (1, na)
	PerfType B2 = square_sums(idB, Eigen::all);  // (nb, 1)

	D.noalias() =  B * -2 * A;
#ifdef PINT
	D.noalias() += B2 * Eigen::MatrixXd::Ones (1, A2.cols());
	D.noalias() += Eigen::MatrixXd::Ones (B2.rows(),1) * (A2);
#else
	D.noalias() += B2 * Eigen::MatrixXf::Ones (1, A2.cols());
	D.noalias() += Eigen::MatrixXf::Ones (B2.rows(),1) * (A2);
#endif

	Eigen::internal::set_is_malloc_allowed(true);
	return;
}

int main2(int argc, char **argv) {
	string source_path = "dummy-data.bin";

	// Also accept other path for source data
	if (argc > 1) {
		source_path = string(argv[1]);
	}
	omp_set_num_threads(32);


	// Read data points
	//  ReadBinEigen(source_path, KGraph::nodes);   // Eigen version
#ifdef PINT
	ReadBinInt(source_path, nodes);   // Eigen version
#else
ReadBinEigenColMajor(source_path, nodes);   // Eigen version

#endif



square_sums =  nodes.colwise().squaredNorm();
	cout<<nodes.cols()<<"\n";
	cout<<nodes.col(nodes.cols()-2).size() << endl;
	cout << "square: " << square_sums[0]<<"\n";
int N = nodes.cols();

                vector<uint32_t> nn_new, nn_old;
int size = 100;
		nn_new.resize(size);
		nn_old.resize(size);
		
		std::mt19937 rng(99);
		GenRandom(rng, &nn_new[0], nn_new.size(), N);
		GenRandom(rng, &nn_old[0], nn_old.size(), N);
                PerfType D(nn_old.size(), nn_new.size());
	boost::timer::cpu_timer timer;
            for (uint32_t n = 0; n < 10000; ++n) {
                //PerfType D(Eigen::seq(0, 100), Eigen::seq(0, 100));
		squared_dist(nn_new, nn_old, D);
	    }
	auto times = timer.elapsed();
	std::cerr << "cal time: " << times.wall / 1e9 <<"\n";
return 0;
}

using namespace std;
using namespace Eigen;
int perfIntTest()
{
 int dimension = 50;

 Matrix <int, Dynamic, Dynamic> intMatrixA(dimension, dimension);
 Matrix <int, Dynamic, Dynamic> intMatrixB(dimension, dimension);
 intMatrixA.setRandom();
 intMatrixB.setRandom();

 MatrixXf floatMatrixA(dimension, dimension);
 MatrixXf floatMatrixB(dimension, dimension);
 floatMatrixA.setRandom();
 floatMatrixB.setRandom();

 MatrixXd doubleMatrixA(dimension, dimension);
 MatrixXd doubleMatrixB(dimension, dimension);
 doubleMatrixA.setRandom();
 doubleMatrixB.setRandom();

{
	boost::timer::cpu_timer timer;
 MatrixXi resultI = intMatrixA * intMatrixB;
	auto times = timer.elapsed();
	std::cerr << "cal int time: " << times.wall / 1e9 <<"\n";
}

{
	boost::timer::cpu_timer timer;
 MatrixXf resultF = floatMatrixA * floatMatrixB;
	auto times = timer.elapsed();
	std::cerr << "cal float time: " << times.wall / 1e9 <<"\n";
}

{
	boost::timer::cpu_timer timer;
 MatrixXd resultD = doubleMatrixA * doubleMatrixB;
	auto times = timer.elapsed();
	std::cerr << "cal double time: " << times.wall / 1e9 <<"\n";

}

return 0;
}
    struct Neighbor {
        uint32_t id;
	uint16_t dist;
    };

int main(int argc, char **argv) {
	//return main2(argc, argv);
//	printf("got %d\n", sizeof(Neighbor));
return main1(argc, argv);
//	perfIntTest();
}
