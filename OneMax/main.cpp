#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

#define INDIVIDUAL_SIZE 100
#define POPULATION_SIZE 200
#define MUTATE_RATE 0.1
#define MAX_GENERATION 1000

using namespace std;

random_device seed_gen;
mt19937 engine(seed_gen());
uniform_real_distribution<> dist(0.0, 1.0);
uniform_int_distribution<> dist_p(0, INDIVIDUAL_SIZE-1);

vector<int> createIndividual(int);
vector<vector<int>> tournament(vector<vector<int>>, vector<pair<int, int>> &, int tour_size);
vector<vector<int>> crossover(vector<vector<int>> &);
void mutate(vector<vector<int>> &);
int evaluate(vector<int>);
void printIndividual(vector<int>);
int findMax(vector<vector<int>>);
float findMean(vector<vector<int>>);

ofstream ofs("result.csv");

int main(int argc, char const* argv[]) {
	auto start = std::chrono::system_clock::now();
	size_t i, j;
	int generation = 0, max_fit = 0;
	vector<vector<int>> pop;
	vector<pair<int, int>> fit;
	vector<thread> ths;
	for (i = 0; i < POPULATION_SIZE; ++i) {
		pop.emplace_back(createIndividual(INDIVIDUAL_SIZE));
	}

	while (max_fit < INDIVIDUAL_SIZE) {
		//cout << "Generation " << generation << endl;
		//for (i = 0; i < pop.size(); ++i) {
		//	pair<int, int> temp(i, evaluate(pop[i]));
		//	fit.emplace_back(temp);
		//}

		for (i = 0; i < pop.size(); ++i) {
			ths.emplace_back(
				[i, &pop, &fit]() {
					pair<int, int> temp(i, evaluate(pop[i]));
					fit.emplace_back(temp);
				}
			);
		}

		for (thread& th : ths) th.join();
		ths.clear();

		pop = tournament(pop, fit, 10);
		pop = crossover(pop);
		mutate(pop);
		findMean(pop);
		max_fit = findMax(pop);
		generation++;
		//cout << endl;
	}
	auto end = std::chrono::system_clock::now();
	auto dur = end - start;
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	std::cout << msec << " milli sec \n";
	cout << "Press Enter to exit...";
	cin.ignore();

	return 0;
}

int findMax(vector<vector<int>> v) {
	size_t i;
	int res = 0;
	vector<int> fit(v.size());
	for (i = 0; i < v.size(); ++i) {
		fit[i] = evaluate(v[i]);
	}

	vector<int>::iterator iter = max_element(fit.begin(), fit.end());
	size_t idx = distance(fit.begin(), iter);

	res = fit[idx];
	cout << "Max fitness individual: " << fit[idx] << endl;
	ofs << res << endl;
	cout << "Best Individual: " << endl;
	printIndividual(v[idx]);

	fit.clear();

	return res;
}

float findMean(vector<vector<int>> v) {
	size_t i;
	float res = 0.0;
	for (i = 0; i < v.size(); ++i) {
		res += evaluate(v[i]);
	}
	res /= v.size();
	cout << "Mean fitness individual: " << res << endl;
	ofs << res << ",";
	return res;
}

void mutate(vector<vector<int>> &v) {
	size_t i;
	int point = 0;
	cout << "Mutate point:  ";
	for (i = 0; i < v.size(); ++i) {
		if (dist(engine) < MUTATE_RATE) {
			point = dist_p(engine);
			cout << point << " ";
			if (v[i][point] == 0) v[i][point] = 1;
			else if (v[i][point] == 1) v[i][point] = 0;
		}
	}
	cout << endl;
}

vector<vector<int>> crossover(vector<vector<int>> &v) {
	int point_1 = 0;
	int point_2 = 0;
	size_t i, j;
	vector<vector<int>> res;
	vector<vector<int>> offspring(2);
	vector<vector<int>> offspring_copy(2);
	cout << "Cross over method: Two-Point Crossover" << endl;
	while (v.size() != 0) {
		vector<vector<int>> offspring(2);
		vector<vector<int>> offspring_copy(2);
		offspring[0] = v.back();
		v.pop_back();
		offspring[1] = v.back();
		v.pop_back();
		
		copy(offspring.begin(), offspring.end(), offspring_copy.begin());

		point_1 = dist_p(engine);
		point_2 = dist_p(engine);

		if (point_2 > point_1) {
			for (i = point_1; i < point_2; ++i) {
				offspring[0][i] = offspring_copy[1][i];
				offspring[1][i] = offspring_copy[0][i];
			}
		}
		else {
			for (i = point_2; i < point_1; ++i) {
				offspring[0][i] = offspring_copy[1][i];
				offspring[1][i] = offspring_copy[0][i];
			}
		}

		res.emplace_back(offspring[0]);
		res.emplace_back(offspring[1]);

		offspring.clear();
		offspring_copy.clear();
	}
	return res;
}

vector<vector<int>> tournament(vector<vector<int>> v, vector<pair<int, int>> &fit, int tour_size) {
	size_t i;
	vector<vector<int>> res;
	cout << "Select Method: Tournament, size " << tour_size << endl;
	do {
		vector<pair<int, int>> temp(tour_size);
		int max_fit, max_idx = 0;
		shuffle(fit.begin(), fit.end(), engine);
		
		for (i = 0; i < tour_size; ++i) {
			temp[i] = fit.back();
			fit.pop_back();
		}
		max_fit = temp[0].second;
		if (temp[1].second > max_fit) max_idx = 1;
		if (temp[2].second > max_fit) max_idx = 2;
		for (i = 0; i < tour_size; ++i) {
			res.emplace_back(v[temp[max_idx].first]);
		}
	} while ((fit.size() > tour_size));
	for (i = 0; i < fit.size(); ++i) {
		res.emplace_back(v[fit[i].first]);
	}
	fit.clear();
	shuffle(res.begin(), res.end(), engine);
	return res;
}

vector<int> createIndividual(int size) {
	vector<int> v;
	for (size_t i = 0; i < size; ++i) {
		float p1 = dist(engine);
		if (p1 > 0.85) v.emplace_back(1);
		else v.emplace_back(0);	
	}
	return v;
}

int evaluate(vector<int> v) {
	int res = 0;
	for (size_t i = 0; i < v.size(); ++i) {
		res += v[i];
	}
	return res;
}

void printIndividual(vector<int> v) {
	cout << "[";
	for (size_t i = 0; i < v.size(); ++i) {
		if (i < v.size() - 1) cout << v[i] << " ";
		else cout << v[i];
	}
	cout << "]" << endl;
}