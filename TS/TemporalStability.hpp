#include <Python.h>
#include <numpy/arrayobject.h>
#include "RunLengthEncoding.hpp"

/* Header to test of C modules for arrays for Python: C_test.c */
#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

/* ==== Prototypes =================================== */

/* .... C vector utility functions ..................*/
//PyArrayObject *pyvector(PyObject *objin);
//double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
//int  not_doublevector(PyArrayObject *vec);

/* .... C matrix utility functions ..................*/
int not_doublematrix(PyArrayObject *mat);
PyArrayObject *pymatrix(PyObject *objin);

/* .... C 2D array utility functions ..................*/
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);

void C_showMat(double **a, int lin, int col);

/* .... C 3D array utility functions ..................*/
double ***ptrvector3d(int n);
double ***malloc3D(int l, int c, int d);
void free_Carrayptrs3D(double ***v);
double ***py3dArray_to_Carrayptrs(PyArrayObject *arrayin);

void C_show3DArray(double ***a, int lin, int col, int dim);

/* .... C data extraction from array utility functions ..................*/

double *getTemporalProfil(double ***ndarray, int dim, int pos_x, int pos_y);
double **getTempProfilsNeighbors(double ***ndarray, int lines, int cols, int dim, int pos_x, int pos_y, int w, int &nb_pixels);

/* .... TS features extraction from Image Sequence ..................*/

void getTS(double ***sits, int lin, int col, int dim, int *dates, double ***features);
void getTS_temp(double ***sits, int lin, int col, int dim, int w, int *dates, double ***features);
void getTS_spatio(double ***sits, int lin, int col, int dim, int w, int *dates, double ***features);
void getTS_spatiotemp(double ***sits, int lin, int col, int dim, int ws, int wt, int *dates, double ***features);

void getReconstruncted_TS(double ***sits, int lin, int col, int dim, int *dates, double ***reconstruncted_data);
void getReconstruncted_TS_temp(double ***sits, int lin, int col, int dim, int *dates, double ***reconstruncted_data);
void getReconstruncted_TS_spatio(double ***sits, int lin, int col, int dim, int w, int *dates, double ***reconstruncted_data);
void getReconstruncted_TS_spatiotemp(double ***sits, int lin, int col, int dim, int ws, int wt, int *dates, double ***reconstruncted_data);

/* .... Python callable Matrix functions ..................*/
static PyObject *showMat(PyObject *self, PyObject *args);
static PyObject *showMat2(PyObject *self, PyObject *args);                        // To show content of 2D/2D array
static PyObject *version(PyObject *self);                                         // Version of the package

static PyObject *getTemporalStability(PyObject *self, PyObject *args);            // function that extract the Temporal Stablity features
static PyObject *getTemporalStability_temp(PyObject *self, PyObject *args);       // function that extract the Temporal Stablity features with temporal relaxations
static PyObject *getTemporalStability_spatio(PyObject *self, PyObject *args);     // function that extract the Temporal Stablity features with spatial relaxations
static PyObject *getTemporalStability_spatiotemp(PyObject *self, PyObject *args); // function that extract the Temporal Stablity features with spatio-temporal relaxations

static PyObject *getReconstruncted_TemporalStability(PyObject *self, PyObject *args); // function that extract the Temporal Stability features and reconstrunct the Image Time Series
static PyObject *getReconstruncted_TemporalStability_temp(PyObject *self, PyObject *args); // function that extract the Temporal Stability features and reconstrunct the Image Time Series
static PyObject *getReconstruncted_TemporalStability_spatio(PyObject *self, PyObject *args); // function that extract the Temporal Stability features and reconstrunct the Image Time Series
static PyObject *getReconstruncted_TemporalStability_spatiotemp(PyObject *self, PyObject *args); // function that extract the Temporal Stability features and reconstrunct the Image Time Series