#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

#define DIMENSIONS 2
#define UP_LIMIT 512
#define LOW_LIMIT -512
#define POPULATION_SIZE 300
#define TOURNAMENT_SIZE 5
#define CROSSOVER_RATE 0.9
#define MUTATION_RATE 0.01
#define MAX_GENERATIONS 500
#define HALL_OF_FAME_SIZE 30
#define CROWDING_FACTOR 20.0

random_device seed_gen;
mt19937 engine(seed_gen());
uniform_real_distribution<> dist_prob(0.0, 1.0);
uniform_real_distribution<> dist_variable(LOW_LIMIT, UP_LIMIT);

struct tuple_chromo {
	float x;
	float y;
	float val;
};

float eggholder(float, float);
void tournament(vector<tuple_chromo> &, int);
void crossover(vector<tuple_chromo>&, float, float);
void mutate(vector<tuple_chromo> &, float);
void generateIndividuals(vector<tuple_chromo> &);
void printAnswer();
void printBest(vector<tuple_chromo>);

ofstream ofs("result.csv");

int main(int argc, char* argv[]) {
	int g=0;
	int max_gen = stoi(argv[1]);
	vector<tuple_chromo> pop(POPULATION_SIZE);
	generateIndividuals(pop);
	printBest(pop);
	while (g < max_gen) {
		cout << "\nGeneration " << g << endl;
		tournament(pop, TOURNAMENT_SIZE);
		crossover(pop, CROWDING_FACTOR, CROSSOVER_RATE);
		mutate(pop, MUTATION_RATE);
		printBest(pop);
		g++;
	}
	cout << "\nFinal result" << endl;
	printAnswer();
	printBest(pop);
	return 0;
}

void mutate(vector<tuple_chromo> &v, float rate) {
	size_t v_size = v.size();
	size_t i;
	for (i = 0; i < v_size; i++) {
		if (dist_prob(engine) < rate) v[i].x = dist_variable(engine);
		if (dist_prob(engine) < rate) v[i].y = dist_variable(engine);
		v[i].val = eggholder(v[i].x, v[i].y);
	}
}

float beta(float eta) {
	float u = dist_prob(engine);
	if (u <= 0.5) return pow(2 * u, 1 / (eta + 1));
	else return pow(1/(2 - 2 * u), 1 / (eta + 1));
}

void crossover(vector<tuple_chromo> &v, float eta, float rate) {
	size_t v_size = int(v.size() / 2);
	size_t i;
	float b = beta(eta);
	shuffle(v.begin(), v.end(), engine);
	for (i = 0; i < v_size; ++i) {
		if (dist_prob(engine) < rate) continue;
		float x1, x2, y1, y2;
		float offx1, offx2, offy1, offy2;
		
		if (v[2 * i].x > v[2 * i + 1].x) { x1 = v[2 * i].x; x2 = v[2 * i + 1].x; }
		else { x2 = v[2 * i].x; x1 = v[2 * i + 1].x; }

		if (v[2 * i].y > v[2 * i + 1].y) { y1 = v[2 * i].y; y2 = v[2 * i + 1].y; }
		else { y2 = v[2 * i].y; y1 = v[2 * i + 1].y; }

		offx1 = 0.5 * ((1.0 + b) * x1 + (1.0 - b) * x2);
		if (offx1 <= LOW_LIMIT) offx1 = LOW_LIMIT;
		else if (offx1 >= UP_LIMIT) offx1 = UP_LIMIT;
		offx2 = 0.5 * ((1.0 + b) * x2 + (1.0 - b) * x1);
		if (offx2 <= LOW_LIMIT) offx2 = LOW_LIMIT;
		else if (offx2 >= UP_LIMIT) offx2 = UP_LIMIT;

		offy1 = 0.5 * ((1.0 + b) * y1 + (1.0 - b) * y2);
		if (offy1 <= LOW_LIMIT) offy1 = LOW_LIMIT;
		else if (offy1 >= UP_LIMIT) offy1 = UP_LIMIT;
		offy2 = 0.5 * ((1.0 + b) * y2 + (1.0 - b) * y1);
		if (offy2 <= LOW_LIMIT) offy2 = LOW_LIMIT;
		else if (offy2 >= UP_LIMIT) offy2 = UP_LIMIT;

		v[2 * i].x = offx1; v[2 * i].y = offy1; v[2 * i].val = eggholder(offx1, offy1);
		v[2 * i+1].x = offx2; v[2 * i + 1].y = offy2; v[2 * i + 1].val = eggholder(offx2, offy2);
	}
	return;
}

void tournament(vector<tuple_chromo> &v, int tour_size) {
	size_t v_size = int(v.size()/tour_size);
	size_t i, j;
	shuffle(v.begin(), v.end(), engine);
	for (i = 0; i < v_size; ++i) {
		vector<tuple_chromo> temp;
		for (j = 0; j < tour_size; ++j) temp.emplace_back(v[tour_size*i+j]);
		sort(temp.begin(), temp.end(), 
			[](tuple_chromo const& iLhs, tuple_chromo const& iRhs){return iLhs.val < iRhs.val;}
		);
		for (j = 0; j < tour_size; ++j) v[tour_size * i + j] = temp[0];
	}
	return;
}

float eggholder(float x, float y) {
	return (-(y + 47.0) * sin(sqrt(abs(x / 2.0 + (y + 47.0)))) - x * sin(sqrt(abs(x - (y + 47.0)))));
}

void generateIndividuals(vector<tuple_chromo> &v) {
	size_t v_size = v.size();
	size_t i;
	for (i = 0; i < v_size; ++i) {
		v[i].x = dist_variable(engine);
		v[i].y = dist_variable(engine);
		v[i].val = eggholder(v[i].x, v[i].y);
	}
	return;
}

void printAnswer() {
	float x, y, f;
	x = 512;
	y = 404.23129;
	f = eggholder(x, y);
	cout << "Known best value x: " << x << " y: " << y << endl;
	cout << "f(x, y) value: " << f << endl;
	cout << endl;
	return;
}

float findMean(vector<tuple_chromo> v) {
	size_t v_size = v.size();
	size_t i;
	float res = 0;
	for (i = 0; i < v_size; ++i) {
		res += v[i].val;
	}
	return res / v.size();
}

void printBest(vector<tuple_chromo> v) {
	vector<tuple_chromo> temp(v.size());
	copy(v.begin(), v.end(), temp.begin());
	sort(temp.begin(), temp.end(),
		[](tuple_chromo const& iLhs, tuple_chromo const& iRhs) {return iLhs.val < iRhs.val; }
	);
	float x, y, f, m;
	x = temp[0].x;
	y = temp[0].y;
	f = temp[0].val;
	m = findMean(temp);
	cout << "Found best value x: " << x << " y: " << y << endl;
	cout << "f(x, y) value: " << f << endl;
	cout << "Mean value: " << m << endl;
	ofs << x << "," << y << "," << f << "," << m << endl;
	temp.clear();
	return;
}
