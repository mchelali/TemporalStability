#include <stdio.h>
#include <vector>
using namespace std;
/* ==== Prototypes =================================== */

vector<int> rle(double *series, int size_s);                 // default RLE
vector<int> decode_rle(vector<int> code);

int count_tShift(double val, double *serie, int size_s);
vector<int> rle_temp(double *series, int size_s, int w = 3); //rle with temporal relaxations

int count_sShift(double val, double **serie, int size_s, int w = 3);
vector<int> rle_spatio(double **series, int size_s, int w = 3); // rle with spatial relaxations

int count_stShift(double val, double **serie, int size_s, int ws = 3, int wt = 3);
vector<int> rle_spatiotemp(double **series, int size_s, int ws = 3, int wt = 3); // rle with both temporal and spatial relaxations

/* ==== utils =================================== */
double *getData(double *mat, int row, int col); // get data from `row` to `col`
void stack(double *a, int size_a, double *b, int size_b, double *result);
double **extractElements(double **srcArray, int start, int end, int lines);
void destroyTwoDimenArrayOnHeapUsingFree(double **ptr, int row, int col);