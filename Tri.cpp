// Concept: Andrew Polar and Mike Poluektov
// Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

// This code is friendly to integrated circuits: it uses only integers and no divisions.
// It avoids nested if/else blocks with bodies of different sizes, preventing irregular latency.
//
// Problem: predict the area of random triangles given their vertex coordinates.
// Triangle vertices are generated in a square with coordinates from 100 to 2000.
// Training set: 8192 records, Validation set: 2048 records.
// Accuracy metric: average absolute residual error, typically ~1% of max target range after 32 epochs.
// Feb. 8, 2026.

#include <iostream>
#include <random>
#include <stdint.h>

using RNG = std::mt19937;

//generate features
std::vector<std::vector<int>> MakeRandomMatrix(RNG& rng, int rows, int cols, int min, int max) {
	std::uniform_int_distribution<int> dist(min, max);
	std::vector<std::vector<int>> matrix(rows);
	for (int i = 0; i < rows; ++i) {
		matrix[i] = std::vector<int>(cols);
		for (int j = 0; j < cols; ++j) {
			matrix[i][j] = (int)dist(rng);
		}
	}
	return matrix;
}
//area of triange
int AreaOfTriangle(int x1, int y1, int x2, int y2, int x3, int y3) {
	int buffer = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
	if (buffer < 0) buffer = -buffer;
	return buffer / 2;
}
//generate targets
std::vector<int> ComputeAreas(std::vector<std::vector<int>>& matrix) {
	size_t N = (int)matrix.size();
	std::vector<int> u(N);
	for (size_t i = 0; i < N; ++i) {
		u[i] = AreaOfTriangle(matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3], matrix[i][4], matrix[i][5]);
	}
	return u;
}

//core functions 
struct Function {
	std::vector<int> f;  //nodes
	int xmin;            //field of definition
	int xmax;			 //field of definition
	int delta_shift;     //the size of single linear segment as 1 << delta_shift 
	int offset;			 //saved distance within linear segment from the left
	int index;           //saved left index of linear segement
};

//we start stuctures at random
void InitializeFunction(Function& F, int nPoints,
	int xmin, int xmax, int fmin, int fmax, int delta_shift, RNG& rng) {

	//we know function range from fmin to fmax, but instantiate within narrower 
	//limit on the purpose of further computational stability
	int range = (fmax - fmin) >> 2;
	F.f.resize(nPoints);
	std::uniform_int_distribution<int> dist(fmin + range, fmax - range);
	for (int j = 0; j < nPoints; ++j) {
		F.f[j] = dist(rng);
	}
	F.xmin = xmin;
	F.xmax = xmax;
	F.delta_shift = delta_shift;
}

//only a linear interpolation and clipping when x out of the range
int Compute(int x, Function& F) {
	if (x <= F.xmin) {
		F.index = 0;
		F.offset = 512;
		return F.f[0];
	}
	else if (x >= F.xmax) {
		F.index = (int)F.f.size() - 2;
		F.offset = (1 << F.delta_shift) - 512;
		return F.f[F.f.size() - 1];
	}
	else {
		int D = x - F.xmin;
		F.index = D >> F.delta_shift;
		F.offset = D & ((1 << F.delta_shift) - 1);
		int64_t Q = (int64_t)(F.f[F.index + 1] - F.f[F.index]);
		Q *= F.offset;
		Q >>= F.delta_shift;
		Q += F.f[F.index];
		return (int)Q;
	}
}

//compute operation saves F.index, now we can get difference without argument
//it is numerator in partial derivative for gradient of the layer
int GetDifference(Function& F) {
	return F.f[F.index + 1] - F.f[F.index];
}

//update of two points of particular linear segment identified at Compute
//residual shows a direction to more accurate model for one specific record
void Update(int64_t residual, Function& F) {
	int tmp = (int)((residual * F.offset) >> F.delta_shift);
	F.f[F.index + 1] += tmp;
	F.f[F.index] += (int)residual - tmp;
}
//end core functions

int main() {   
	RNG rng(std::random_device{}());
	const int nFeatureMin = 100;
	const int nFeatureMax = 2000;
	const int nFeatures = 6;
	const int nTrainingRecords = 1 << 13;
	const int nValidationRecords = 1 << 11;

	auto features_training = MakeRandomMatrix(rng, nTrainingRecords, nFeatures, nFeatureMin, nFeatureMax);
	auto features_validation = MakeRandomMatrix(rng, nValidationRecords, nFeatures, nFeatureMin, nFeatureMax);
	auto targets_training = ComputeAreas(features_training);
	auto targets_validation = ComputeAreas(features_validation);

	clock_t start_application = clock();
	clock_t current_time = clock();

	int nTargetMin = targets_training[0];
	int nTargetMax = nTargetMin;
	for (size_t i = 0; i < targets_training.size(); ++i) {
		if (nTargetMin > targets_training[i]) nTargetMin = targets_training[i];
		if (nTargetMax < targets_training[i]) nTargetMax = targets_training[i];
	}

	//configuration of network
	//all parameters are logically related
	//they can be changed only in coordinated way
	//the core element of the model is one block y = 1/n sum of functions F_i(x_i)). it maps vector to scalar
	//layer has several blocks and maps vector to vector.
	const int nU0 = 53;  //blocks in inner layer
	const int nU1 = 11;  //blocks in next layer
	const int nU2 = 4;   //blocks in next layer
	const int nU3 = 1;   //one block in outer layer
	const int base_shift = 13;  //we can't have division but we need compute average of nU blocks, so we 
	//use multiplication and shift instead.  (x * 1365) >> base_shift must be approx to x / 6, because 6
	//is number of features.  
	//
	const int nPoints0 = 2;  //points functions. when changed all other parameters must be updated
	const int xMin0 = 0;     //definition function arguments, defined only by computational statility
	const int xMax0 = 1 << 11;
	const int delta_shift0 = 11;   //1 << delta_shift0 is size of linear segment
	const int alpha0_shift = 13;   //learning rate applied as shift to the right 
	const int mult0 = 1365;  //for 6
	//
	const int nPoints1 = 14;
	const int xMin1 = -10'000;
	const int xMax1 = 1'693'936;
	const int delta_shift1 = 17;
	const int alpha1_shift = 8;
	const int mult1 = 154;  //for 53
	//
	const int nPoints2 = 14;
	const int xMin2 = -10'000;
	const int xMax2 = 1'693'936;
	const int delta_shift2 = 17;
	const int alpha2_shift = 7;
	const int mult2 = 744;  //for 11
	//
	const int nPoints3 = 14;
	const int xMin3 = -10'000;
	const int xMax3 = 1'693'936;
	const int delta_shift3 = 17;
	const int alpha3_shift = 6;
	const int mult3 = 2048;  //for 4

	const int nEpochs = 32;

	std::vector<std::unique_ptr<Function>> layer0;
	for (int i = 0; i < nU0 * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints0, xMin0, xMax0, nTargetMin, nTargetMax, delta_shift0, rng);
		layer0.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer1;
	for (int i = 0; i < nU1 * nU0; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints1, xMin1, xMax1, nTargetMin, nTargetMax, delta_shift1, rng);
		layer1.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer2;
	for (int i = 0; i < nU2 * nU1; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints2, xMin2, xMax2, nTargetMin, nTargetMax, delta_shift2, rng);
		layer2.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer3;
	for (int i = 0; i < nU3 * nU2; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints3, xMin3, xMax3, nTargetMin, nTargetMax, delta_shift3, rng);
		layer3.push_back(std::move(function));
	}

	//auxiliary buffers
	std::vector<int> models0(nU0);
	std::vector<int> models1(nU1);
	std::vector<int> models2(nU2);
	std::vector<int> models3(nU3);

	std::vector<std::vector<int>> differences2(nU3, std::vector<int>(nU2));
	std::vector<std::vector<int>> differences1(nU2, std::vector<int>(nU1));
	std::vector<std::vector<int>> differences0(nU1, std::vector<int>(nU0));

	std::vector<int64_t> deltas3(nU3);
	std::vector<int64_t> deltas2(nU2);
	std::vector<int64_t> deltas1(nU1);
	std::vector<int64_t> deltas0(nU0);

	printf("Targets are areas of random triangles, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < nEpochs; ++epoch) {
		//training
		for (int record = 0; record < nTrainingRecords; ++record) {
			//forward pass
			//FPGA single-cycle block 
			for (int k = 0; k < nU0; ++k) {
				int64_t m0 = 0;
				for (int j = 0; j < nFeatures; ++j) {
					m0 += Compute(features_training[record][j], *layer0[k * nFeatures + j]);
				}
				m0 *= mult0;
				m0 >>= base_shift;
				models0[k] = (int)m0;
			}
			//next FPGA single-cycle block
			for (int k = 0; k < nU1; ++k) {
				int64_t m1 = 0;
				for (int j = 0; j < nU0; ++j) {
					m1 += Compute(models0[j], *layer1[k * nU0 + j]);
				}
				m1 *= mult1;
				m1 >>= base_shift;
				models1[k] = (int)m1; 
			}
			//next FPGA single-cycle block
			for (int k = 0; k < nU2; ++k) {
				int64_t m2 = 0;
				for (int j = 0; j < nU1; ++j) {
					m2 += Compute(models1[j], *layer2[k * nU1 + j]);
				}
				m2 *= mult2;
				m2 >>= base_shift;
				models2[k] = (int)m2;
			}
			//next FPGA single-cycle block
			for (int k = 0; k < nU3; ++k) {
				int64_t m3 = 0;
				for (int j = 0; j < nU2; ++j) {
					m3 += Compute(models2[j], *layer3[k * nU2 + j]);
				}
				m3 *= mult3;
				m3 >>= base_shift;
				models3[k] = (int)m3;
			}
			//end of forward pass

			//get all differences
			//next FPGA single-cycle block
			for (int k = 0; k < nU3; ++k) {
				for (int j = 0; j < nU2; ++j) {
					differences2[k][j] = GetDifference(*layer3[k * nU2 + j]);
				}
			}
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					differences1[k][j] = GetDifference(*layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					differences0[k][j] = GetDifference(*layer1[k * nU0 + j]);
				}
			}

			//compute all delta vectors for each layer
			//next FPGA single-cycle block
			deltas3[0] = targets_training[record] - models3[0];  //first one is scalar
			for (int j = 0; j < nU2; ++j) {
				deltas2[j] = 0;
				for (int i = 0; i < nU3; ++i) {
					deltas2[j] += (differences2[i][j] * deltas3[i]) >> delta_shift3;
				}
			}
			//next FPGA single-cycle block
			for (int j = 0; j < nU1; ++j) {
				deltas1[j] = 0;
				for (int i = 0; i < nU2; ++i) {
					deltas1[j] += differences1[i][j] * deltas2[i] >> delta_shift2;
				}
			}
			//next FPGA single-cycle block
			for (int j = 0; j < nU0; ++j) {
				deltas0[j] = 0;
				for (int i = 0; i < nU1; ++i) {
					deltas0[j] += differences0[i][j] * deltas1[i] >> delta_shift1;
				}
			}

			//step: update all layers by deltas
			//last FPGA single-cycle block
			for (int k = 0; k < nU3; ++k) {
				for (int j = 0; j < nU2; ++j) {
					Update((deltas3[k] >> alpha3_shift), *layer3[k * nU2 + j]);
				}
			}
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					Update((deltas2[k] >> alpha2_shift), *layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					Update((deltas1[k] >> alpha1_shift), *layer1[k * nU0 + j]);
				}
			}
			for (int k = 0; k < nU0; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update((deltas0[k] >> alpha0_shift), *layer0[k * nFeatures + j]);
				}
			}
		}
		//9 FPGA cycles per record

		//this validation runs every epoch, we don't need it that often
		int error = 0;
		for (int record = 0; record < nValidationRecords; ++record) {

			for (int k = 0; k < nU0; ++k) {
				int64_t m0 = 0;
				for (int j = 0; j < nFeatures; ++j) {
					m0 += Compute(features_validation[record][j], *layer0[k * nFeatures + j]);
				}
				m0 *= mult0;
				m0 >>= base_shift;
				models0[k] = (int)m0;
			}
			for (int k = 0; k < nU1; ++k) {
				int64_t m1 = 0;
				for (int j = 0; j < nU0; ++j) {
					m1 += Compute(models0[j], *layer1[k * nU0 + j]);
				}
				m1 *= mult1;
				m1 >>= base_shift;
				models1[k] = (int)m1;
			}
			for (int k = 0; k < nU2; ++k) {
				int64_t m2 = 0;
				for (int j = 0; j < nU1; ++j) {
					m2 += Compute(models1[j], *layer2[k * nU1 + j]);
				}
				m2 *= mult2;
				m2 >>= base_shift;
				models2[k] = (int)m2;
			}
			for (int k = 0; k < nU3; ++k) {
				int64_t m3 = 0;
				for (int j = 0; j < nU2; ++j) {
					m3 += Compute(models2[j], *layer3[k * nU2 + j]);
				}
				m3 *= mult3;
				m3 >>= base_shift;
				models3[k] = (int)m3;
			}

			int e = targets_validation[record] - models3[0];
			if (e < 0) e = -e;
			error += e;
		}
		current_time = clock();
		printf("Epoch %d, time %2.3f, average error %d, target limit %d\n", epoch, 
			(double)(current_time - start_application) / CLOCKS_PER_SEC, error / nValidationRecords, 
			(nFeatureMax - nFeatureMin)* (nFeatureMax - nFeatureMin) / 2);
	}
	printf("\n");

	// Training: 8192 records, 32 epochs, 9 cycles per record = 2,359,296 cycles
	// On a 100 MHz FPGA board, processing time ≈ 23 ms (latency)
	// The same code runs on a PC in 1.335 s, so this elementary example
	// is ~58× faster on the board, even though the board’s clock is 40× slower
	// Adding more functions does not increase latency; only additional layers do
}
