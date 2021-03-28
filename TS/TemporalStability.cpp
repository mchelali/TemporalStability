#include "TemporalStability.hpp"
#include <stdio.h>
#include <vector>
#include <algorithm>

using namespace std;

/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef C_TemporalStabilityMethods[] = {
    {"showMat", showMat, METH_VARARGS},
    {"showMat2", showMat2, METH_VARARGS},
    {"version", (PyCFunction)version, METH_NOARGS},
    {"getTemporalStability", (PyCFunction)getTemporalStability, METH_VARARGS},
    {"getTemporalStability_temp", (PyCFunction)getTemporalStability_temp, METH_VARARGS},
    {"getTemporalStability_spatio", (PyCFunction)getTemporalStability_spatio, METH_VARARGS},
    {"getTemporalStability_spatiotemp", (PyCFunction)getTemporalStability_spatiotemp, METH_VARARGS},
    {"getReconstruncted_TemporalStability", (PyCFunction)getReconstruncted_TemporalStability, METH_VARARGS},
    {"getReconstruncted_TemporalStability_temp", (PyCFunction)getReconstruncted_TemporalStability_temp, METH_VARARGS},
    {"getReconstruncted_TemporalStability_spatio", (PyCFunction)getReconstruncted_TemporalStability_spatio, METH_VARARGS},
    {"getReconstruncted_TemporalStability_spatiotemp", (PyCFunction)getReconstruncted_TemporalStability_spatiotemp, METH_VARARGS},
    {NULL, NULL, 0} /* Sentinel - marks the end of this structure */
};

/* ==== Set up the Module structure ====================== */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef C_Arraytest = {
    PyModuleDef_HEAD_INIT,
    "TemporalStability",
    "show matrix module",
    -1,
    C_TemporalStabilityMethods};
#endif

/* ==== Initialize the C_test functions ====================== */
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_TemporalStability(void)
{
    return PyModule_Create(&C_Arraytest);
}
#else
PyMODINIT_FUNC PyInit_TemporalStability(void)
{
    Py_InitModule("TemporalStability", C_TemporalStabilityMethods, "Temporal Stability computation methods", NULL, 1.0);
}
#endif
//}

/* =========== Implementation of declared functions ===================== */

PyArrayObject *pymatrix(PyObject *objin)
{
    return (PyArrayObject *)PyArray_ContiguousFromObject(objin,
                                                         NPY_DOUBLE, 2, 2);
}

int not_doublematrix(PyArrayObject *mat)
{
    if (mat->descr->type_num != NPY_DOUBLE || mat->nd < 2)
    {
        PyErr_SetString(PyExc_ValueError,
                        "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
        return 1;
    }
    return 0;
}

/* .... C 2D array utility functions ..................*/

double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)
{
    double **c, *a;
    int i, n, m;

    n = (int) arrayin->dimensions[0];
    m = (int) arrayin->dimensions[1];
    c = ptrvector(n);
    a = (double *)arrayin->data; /* pointer to arrayin data as double */
    for (i = 0; i < n; i++)
    {
        c[i] = a + i * m;
    }
    return c;
}

double **ptrvector(long n)
{
    double **v;
    v = (double **)malloc((size_t)(n * sizeof(double)));
    if (!v)
    {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);
    }
    return v;
}

void free_Carrayptrs(double **v)
{
    free((char *)v);
}

void C_showMat(double **a, int lin, int col)
{
    printf("Array content :\n[");
    for (int i = 0; i < lin; i++)
    {
        printf(" [ ");
        for (int j = 0; j < col; j++)
        {
            printf("%f\t", a[i][j]);
        }
        printf(" ];\n");
    }
    printf("];\n");
}

/* .... C 3D array utility functions ..................*/

double ***py3dArray_to_Carrayptrs(PyArrayObject *arrayin)
{
    double ***c;
    int i, j, n, m, o;

    n = (int) arrayin->dimensions[0];
    m = (int) arrayin->dimensions[1];
    o = (int) arrayin->dimensions[1];

    c = (double ***)malloc((size_t)(n * sizeof(double **)));

    for (i = 0; i < n; i++)
    {
        c[i] = (double **)malloc((size_t)(m * sizeof(double *)));
        for (j = 0; j < m; j++)
        {
            c[i][j] = (double *)malloc((size_t)(o * sizeof(double)));
            c[i][j] = (double *)PyArray_GETPTR2(arrayin, i, j);
        }
    }

    return c;
}

double ***malloc3D(int l, int c, int d)
{
    double ***mat;
    mat = (double ***)malloc((size_t)(l * sizeof(double **)));

    for (int i = 0; i < l; i++)
    {
        mat[i] = (double **)malloc((size_t)(c * sizeof(double *)));
        for (int j = 0; j < c; j++)
        {
            mat[i][j] = (double *)malloc((size_t)(d * sizeof(double)));
        }
    }
    return mat;
}

double ***ptrvector3d(int n)
{
    double ***v;
    v = (double ***)malloc((size_t)(n * sizeof(double)));
    if (!v)
    {
        printf("In **ptrvector. Allocation of memory for 3D array failed.");
        exit(0);
    }
    return v;
}

void free_Carrayptrs3D(double ***v)
{
    free((char *)v);
}

void C_show3DArray(double ***a, int lin, int col, int dim)
{
    printf("Array content :\n[");
    for (int k = 0; k < dim; k++)
    {
        printf("Dim %d\n", k);
        for (int i = 0; i < lin; i++)
        {
            printf(" [ ");
            for (int j = 0; j < col; j++)
            {
                printf("%f\t", a[i][j][k]);
            }
            printf(" ];\n");
        }
        printf("\n");
    }
    printf("];\n");
}

/* .... C data extraction from array utility functions ..................*/
double *getTemporalProfil(double ***ndarray, int dim, int pos_x, int pos_y)
{
    double *res;
    res = (double *)malloc((size_t)(dim * sizeof(double)));

    for (int i = 0; i < dim; i++)
    {
        res[i] = ndarray[pos_x][pos_y][i];
    }

    return res;
}

double **getTempProfilsNeighbors(double ***ndarray, int lines, int cols, int dim, int pos_x, int pos_y, int w, int &nb_pixels)
{
    //printf("=========================================\n");
    int step = (w / 2);
    //printf("Position %d; %d\n", pos_x, pos_y);

    // Count number of valid pixels (those who are in the range of rows and cols)
    nb_pixels = 0;
    vector<int> x_coord;
    vector<int> y_coord;
    // Ajouter d'abord les cood pos_x et po_y pour qu'ils soient dans la 1ere line
    x_coord.push_back(pos_x);
    y_coord.push_back(pos_y);

    for (int i = -step; i <= step; i++)
    {
        for (int j = -step; j <= step; j++)
        {
            bool x_cond = ((pos_x + i) >= 0) && ((pos_x + i) < lines);
            bool y_cond = ((pos_y + j) >= 0) && ((pos_y + j) < cols);
            if (x_cond && y_cond)
            {
                //printf("(%d, %d) coord : %d; %d\n", i, j, pos_x + i, pos_y + j);
                nb_pixels++;
                if (((pos_x + i) != pos_x) || ((pos_y + j) != pos_y))
                {
                    //printf("\t(%d, %d) coord : %d; %d\n", i, j, pos_x + i, pos_y + j);
                    x_coord.push_back(pos_x + i);
                    y_coord.push_back(pos_y + j);
                }
            }
        }
    }
    //printf("Nombre de pixels du voisinnage est %d; %d\n", nb_pixels, x_coord.size());
    //exit(-1);
    // Allocation de la memoire pour recuperer les pixels
    //printf("Allocation de la memoire\n");
    double **profiles;
    profiles = (double **)malloc((nb_pixels) * sizeof(double *));
    for (int i = 0; i < nb_pixels; i++)
    {
        profiles[i] = (double *)malloc(dim * sizeof(double));
    }

    //printf("recuperation des valeurs\n");
    // copier les pixels
    int idx_lin = 0;
    for (int i = 0; i < x_coord.size(); i++)
    {
        for (int k = 0; k < dim; k++)
        {
            profiles[idx_lin][k] = ndarray[x_coord[i]][y_coord[i]][k];
        }
        idx_lin++;
    }

    return profiles;
}
/* .... TS features extraction from SITS ..................*/
/*
getTS : cette fonction calcul le RLE sur chaque pixel temporel et recupere les features MS, NB, MSS
Les parametres : 
    sits : la serie d'image temporelle
    lin : nombre de line de la serie
    col : nombre de colone de la serie
    dim : le nombred'image dans la serie
    date : les dates associers à chaque image ( si != NULL alors on tiens compte de ce vecteur)  
*/
void getTS(double ***sits, const int lin, const int col, const int dim, int *dates, double ***features)
{
    /*for (int i = 0; i < dim; i++)
    {
        printf("%d\t", dates[i]);
    }
    printf("\n");
    exit(-1);*/
    for (int i = 0; i < lin; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int ms, nb, mss;
            double *profil = getTemporalProfil(sits, dim, i, j);
            vector<int> ts = rle(profil, dim);
            nb = (int) ts.size();

            vector<int>::iterator result;
            result = max_element(ts.begin(), ts.end());

            int arg_ms = 0;
            mss = 0;
            for_each(ts.begin(), result, [&](int n) {
                mss += dates[n];
                arg_ms++;
            });
            ms = dates[arg_ms + *result - 1] - dates[arg_ms] + 1;
            //printf("MS = %d; ArgMax=%d; max=%d\n", ms, arg_ms, *result);
            //printf("\tDate1= %d; Date2 = %d; range=%d\n", dates[arg_ms], dates[arg_ms + *result - 1], dates[arg_ms + *result - 1] - dates[arg_ms] + 1);
            //printf("max=%d; argmax=%d ;date0=%d, date1=%d; ==> %d\n", *result, arg_ms, dates[(arg_ms + *result) - 1], dates[arg_ms], ms);
            features[i][j][0] = ms;
            features[i][j][1] = nb;
            features[i][j][2] = mss;
        }
    }
}

void getTS_temp(double ***sits, int lin, int col, int dim, int w, int *dates, double ***features)
{

    for (int i = 0; i < lin; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int ms, nb, mss;
            double *profil = getTemporalProfil(sits, dim, i, j);
            vector<int> ts = rle_temp(profil, dim, w);
            nb = (int) ts.size();

            vector<int>::iterator result;
            result = max_element(ts.begin(), ts.end());

            int arg_ms = 0;
            mss = 0;
            for_each(ts.begin(), result, [&](int n) {
                mss += dates[n];
                arg_ms++;
            });
            ms = dates[(arg_ms + *result) - 1] - dates[arg_ms] + 1;

            features[i][j][0] = ms;
            features[i][j][1] = nb;
            features[i][j][2] = mss;
        }
    }
}

void getTS_spatio(double ***sits, int lin, int col, int dim, int w, int *dates, double ***features){
    for (int i = 0; i < lin; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int ms, nb, mss, nb_profils = 0;
            double **profil = getTempProfilsNeighbors(sits, lin, col, dim, i, j, w, nb_profils);
            //printf("nb de profils est %d\n", nb_profils);
            vector<int> ts = rle_spatio(profil, dim, nb_profils);

            destroyTwoDimenArrayOnHeapUsingFree(profil, nb_profils, dim);
            nb = (int) ts.size();

            vector<int>::iterator result;
            result = max_element(ts.begin(), ts.end());

            int arg_ms = 0;
            mss = 0;
            for_each(ts.begin(), result, [&](int n) {
                mss += dates[n];
                arg_ms++;
            });
            ms = dates[(arg_ms + *result) - 1] - dates[arg_ms] + 1;
            //printf("MS = %d; ArgMax=%d; max=%d", ms, arg_ms, *result);
            //printf("(%d,%d);nb_profil=%d\tnb de changement est %d; ms=%d; mss=%d\n", i, j, nb_profils, nb, ms, mss);
            //exit(-1);
            features[i][j][0] = ms;
            features[i][j][1] = nb;
            features[i][j][2] = mss;
        }
    }
}

void getTS_spatiotemp(double ***sits, int lin, int col, int dim, int ws, int wt, int *dates, double ***features){
    for (int i = 0; i < lin; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int ms, nb, mss, nb_profils;
            double **profil = getTempProfilsNeighbors(sits, lin, col, dim, i, j, ws, nb_profils);
            vector<int> ts = rle_spatiotemp(profil, dim, nb_profils, wt);
            nb = (int) ts.size();

            vector<int>::iterator result;
            result = max_element(ts.begin(), ts.end());

            int arg_ms = 0;
            mss = 0;
            for_each(ts.begin(), result, [&](int n) {
                mss += dates[n];
                arg_ms++;
            });
            ms = dates[(arg_ms + *result) - 1] - dates[arg_ms] + 1;

            features[i][j][0] = ms;
            features[i][j][1] = nb;
            features[i][j][2] = mss;
        }
    }
}


/* .... SITS to TS to Reversed SITS based on  TS ..................*/

void getReconstruncted_TS(double ***sits, int lin, int col, int dim, int *dates, double ***reconstruncted_data){
    for (int i = 0; i < lin; i++){
        for (int j = 0; j < col; j++){
            //int ms, nb, mss;
            double *profil = getTemporalProfil(sits, dim, i, j);
            vector<int> ts = rle(profil, dim);

            vector<int> reco = decode_rle(ts);
            for (int k = 0; k < dim; k++){
                reconstruncted_data[i][j][k] = reco[k];
            }
        }
    }
}

void getReconstruncted_TS_temp(double ***sits, int lin, int col, int dim, int *dates, double ***reconstruncted_data){
    for (int i = 0; i < lin; i++){
        for (int j = 0; j < col; j++){
            //int ms, nb, mss;
            double *profil = getTemporalProfil(sits, dim, i, j);
            vector<int> ts = rle_temp(profil, dim);

            vector<int> reco = decode_rle(ts);
            for (int k = 0; k < dim; k++){
                reconstruncted_data[i][j][k] = reco[k];
            }
        }
    }
}

void getReconstruncted_TS_spatio(double ***sits, int lin, int col, int dim, int w, int *dates, double ***reconstruncted_data){
    for (int i = 0; i < lin; i++){
        for (int j = 0; j < col; j++){
            int nb_profils;
            double **profil = getTempProfilsNeighbors(sits, lin, col, dim, i, j, w, nb_profils);
            //printf("nb de profils est %d\n", nb_profils);
            vector<int> ts = rle_spatio(profil, dim, nb_profils);

            vector<int> reco = decode_rle(ts);
            for (int k = 0; k < dim; k++){
                reconstruncted_data[i][j][k] = reco[k];
            }
        }
    }
}

void getReconstruncted_TS_spatiotemp(double ***sits, int lin, int col, int dim, int ws, int wt, int *dates, double ***reconstruncted_data){
    for (int i = 0; i < lin; i++){
        for (int j = 0; j < col; j++){
            int nb_profils;
            double **profil = getTempProfilsNeighbors(sits, lin, col, dim, i, j, ws, nb_profils);
            //printf("nb de profils est %d\n", nb_profils);
            vector<int> ts = rle_spatiotemp(profil, dim, nb_profils);

            vector<int> reco = decode_rle(ts);
            for (int k = 0; k < dim; k++){
                reconstruncted_data[i][j][k] = reco[k];
            }
        }
    }
}


/* .... Python callable Matrix functions ..................*/

static PyObject *showMat(PyObject *self, PyObject *args){
    PyArrayObject *matIn; // The Python Array Object that is eaxtracted from the args

    /* Parse tuples separately since args will differ between C fcns */
    if (!PyArg_ParseTuple(args, "O", &matIn))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == matIn)
    {
        printf("Null Array");
        return NULL;
    }

    /* Check that objects are 'double' type and vectors
         Not needed if python wrapper function checks before call to this routine */
    if (not_doublematrix(matIn))
    {
        printf("Not double Array");
        return NULL;
    }

    int dim = PyArray_NDIM(matIn);

    /* Get vector dimension. */
    if (dim == 2)
    {
        int l, c;
        l = (int) matIn->dimensions[0];
        c = (int) matIn->dimensions[1];

        double **c_matIn;
        c_matIn = pymatrix_to_Carrayptrs(matIn);

        C_showMat(c_matIn, l, c);

        free_Carrayptrs(c_matIn);
    }
    if (dim == 3)
    {
        int l, c, d;
        l = (int) matIn->dimensions[0];
        c = (int) matIn->dimensions[1];
        d = (int) matIn->dimensions[2];

        double ***c_matIn;
        c_matIn = py3dArray_to_Carrayptrs(matIn);

        C_show3DArray(c_matIn, l, c, d);

        free_Carrayptrs3D(c_matIn);
    }

    return Py_BuildValue("i", 0);
}

static PyObject *showMat2(PyObject *self, PyObject *args){
    PyArrayObject *matIn; // The Python Array Object that is eaxtracted from the args

    /* Parse tuples separately since args will differ between C fcns */
    if (!PyArg_ParseTuple(args, "O", &matIn))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == matIn)
    {
        printf("Null Array");
        return NULL;
    }

    /* Check that objects are 'double' type and vectors
         Not needed if python wrapper function checks before call to this routine */
    if (not_doublematrix(matIn))
    {
        printf("Not double Array");
        return NULL;
    }

    npy_intp *strides = PyArray_STRIDES(matIn);
    char *data = (char *)PyArray_DATA(matIn);
    int dim = PyArray_NDIM(matIn);
    int l, c, d;
    l = (int) matIn->dimensions[0];
    c = (int) matIn->dimensions[1];
    if (dim > 2)
    {
        d = (int) matIn->dimensions[2];
    }
    else
    {
        d = 1;
    }

    printf("Array content :\n[");
    for (int k = 0; k < d; k++)
    {
        printf("Dim %d\n", k);
        for (int i = 0; i < l; i++)
        {
            printf(" [ ");
            for (int j = 0; j < c; j++)
            {
                printf("%f\t", *(double *)&data[i * strides[0] + j * strides[1] + k * strides[2]]);
            }
            printf(" ];\n");
        }
        printf("\n");
    }
    printf("];\n");

    return Py_BuildValue("i", 0);
}

static PyObject *version(PyObject *self){
    return Py_BuildValue("s", "Version 1.0");
}

static PyObject *getTemporalStability(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((dims * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");
    //exit(-5);
    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, 3);
    // double ***sits, int lin, int col, int dim, int *dates, int ***features
    getTS(sits, lines, cols, dims, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
    //return Py_BuildValue("s", "yes");
}

static PyObject *getTemporalStability_temp(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((pyDates->dimensions[0] * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//pyDates->data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");

    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, 3);
    // double ***sits, int lin, int col, int dim, int w, int *dates, double ***features
    getTS_temp(sits, lines, cols, dims, 3, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
    //return Py_BuildValue("s", "yes");
}

static PyObject *getTemporalStability_spatio(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((pyDates->dimensions[0] * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//pyDates->data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");

    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, 3);
    // double ***sits, int lin, int col, int dim, int w, int *dates, double ***features
    getTS_spatio(sits, lines, cols, dims, 3, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    //printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
}

static PyObject *getTemporalStability_spatiotemp(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((pyDates->dimensions[0] * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//pyDates->data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");

    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, 3);
    // double ***sits, int lin, int col, int dim, int w, int *dates, double ***features
    getTS_spatiotemp(sits, lines, cols, dims, 3, 3, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    //printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
}


static PyObject *getReconstruncted_TemporalStability(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((dims * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");
    //exit(-5);
    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, dims);
    // double ***sits, int lin, int col, int dim, int *dates, int ***features
    getReconstruncted_TS(sits, lines, cols, dims, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < dims; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
    //return Py_BuildValue("s", "yes");
}

static PyObject *getReconstruncted_TemporalStability_temp(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((dims * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");
    //exit(-5);
    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, dims);
    // double ***sits, int lin, int col, int dim, int *dates, int ***features
    getReconstruncted_TS_temp(sits, lines, cols, dims, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < dims; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
    //return Py_BuildValue("s", "yes");
}

static PyObject *getReconstruncted_TemporalStability_spatio(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((dims * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");
    //exit(-5);
    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, dims);
    // double ***sits, int lin, int col, int dim, int *dates, int ***features
    getReconstruncted_TS_spatio(sits, lines, cols, dims, 3, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < dims; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
    //return Py_BuildValue("s", "yes");
}

static PyObject *getReconstruncted_TemporalStability_spatiotemp(PyObject *self, PyObject *args){
    PyArrayObject *array3d; // The Python Array Object that is eaxtracted from the args
    PyArrayObject *pyDates;
    PyArrayObject *output;
    // Parse tuples separately since args will differ between C fcns
    if (!PyArg_ParseTuple(args, "OOO", &array3d, &pyDates, &output))
    {
        printf("jai pas pu get le array\n");
        return NULL;
    }

    if (NULL == array3d || NULL == pyDates)
    {
        printf("Null Array");
        return NULL;
    }

    if (not_doublematrix(array3d))
    {
        printf("Not double Array");
        return NULL;
    }

    int lines, cols, dims;

    lines = (int) array3d->dimensions[0];
    cols = (int) array3d->dimensions[1];
    dims = (int) array3d->dimensions[2];
    //printf("Dimention de l'entrer est %d, %d, %d\n", lines, cols, dims);

    double ***sits;
    double ***features;
    sits = py3dArray_to_Carrayptrs(array3d); // convert numpy array to C array
    //printf("Convertion de numpy a c reussi; \n");

    int *dates;
    dates = (int *)malloc((dims * sizeof(int)));
    //printf("dates shape %d\n", pyDates->dimensions[0]);
    //dates = (int *)PyArray_GetPtr(pyDates, (npy_intp[]){0});
    npy_intp *strides = PyArray_STRIDES(pyDates);

    for (int i = 0; i < dims; i++)
    {
        dates[i] = *((int*) PyArray_GETPTR1(pyDates, i));//data[i * strides[0]];
        //printf("%d\t", dates[i]);
    }
    //printf("\n");
    //exit(-5);
    //printf("featuresmat allocation \n");
    features = malloc3D(lines, cols, dims);
    // double ***sits, int lin, int col, int dim, int *dates, int ***features
    getReconstruncted_TS_spatiotemp(sits, lines, cols, dims, 3, 3, dates, features);

    //printf("Conversion to PyObject \n");
    //PyObject *pyFeatures;
    npy_intp dimFeatures[] = {lines, cols, dims};
    int nd = 3;
    //pyFeatures = PyArray_SimpleNewFromData(3, dimFeatures, NPY_LONG, features);
    double ***c_intout;
    c_intout = py3dArray_to_Carrayptrs(output);
    //printf("Copiying data\n");
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < dims; k++)
            {
                c_intout[i][j][k] = features[i][j][k];
            }
        }
    }

    printf("Liberation de la memoire \n");
    free_Carrayptrs3D(sits);
    free_Carrayptrs3D(features);
    return Py_BuildValue("O", output);
    //return Py_BuildValue("s", "yes");
}
// EOF
