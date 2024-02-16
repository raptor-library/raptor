#include <iostream>
#include <array>
#include <fstream>
#include <sstream>
#include <memory>

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "ruge_stuben/ruge_stuben_solver.hpp"

namespace {
using size = std::size_t;
using scalar = double;
struct header {
	size nrows;
	size ncols;
	size nnz;
	bool symmetric;
};

header read_header(std::ifstream & fh) {
	header hdr;
	std::array<size, 3> sizes;
	std::string line;

	hdr.symmetric = false;
	while (std::getline(fh, line)) {
		if (line.find("symmetric") != std::string::npos) {
			hdr.symmetric = true;
		}
		if (line[0] != '%') {
			std::istringstream iss(line);
			for (int i = 0; i < 3; i++) {
				std::string tok;
				std::getline(iss, tok, ' ');
				sizes[i] = std::atoi(tok.c_str());
			}
			break;
		}
	}

	hdr.nrows = sizes[0];
	hdr.ncols = sizes[1];
	hdr.nnz = sizes[2];

	return hdr;
}


auto readmat(const char * fname) {
	std::ifstream fh(fname);
	auto hdr = read_header(fh);

	std::vector<int> I;
	std::vector<int> J;
	std::vector<double> V;
	[=](auto&...v) { ((v.reserve(hdr.nnz)),...); }(I,J,V);

	int i, j;
	double v;
	while (fh >> i >> j >> v) {
		--i; --j;
		[=](auto &...vec){
			[&](auto & ... ent){
				((vec.push_back(ent)), ...);
			}(i,j,v);
		}(I,J,V);
	}

	return std::make_unique<raptor::COOMatrix>(hdr.nrows, hdr.ncols, I, J, V);
}

auto readvec(const char *fname) {
	std::ifstream fh(fname);
	std::vector<double> v;
	double val;
	while (fh >> val) v.push_back(val);

	raptor::Vector ret(v.size());
	ret.values = v;
	return ret;
}

}
int main(int argc, char *argv[]) {
	std::string basepath{"MFEM_EXAMPLE_DIR"};
	auto A_coo = readmat((basepath + "/A.mtx").c_str());

	auto A = A_coo->to_CSR();
	auto x = readvec((basepath + "/x.txt").c_str());
	auto b = readvec((basepath + "/b.txt").c_str());

    raptor::RugeStubenSolver ml;
    ml.setup(A);
	auto iters = ml.solve(x, b);
	for (int i = 0; i < iters; ++i)
		std::cout << ml.get_residuals()[i] << std::endl;

	delete A;
}
