#include "RunLengthEncoding.hpp"
#include <stdlib.h>
#include <algorithm>

double **extractElements(double **srcArray, int start, int end, int lines)
{
    //if (subArray == NULL)
    //{
    double **subArray;
    subArray = (double **)malloc((lines) * sizeof(double *));
    for (int i = 0; i < lines; i++)
    {
        subArray[i] = (double *)malloc((end - start + 1) * sizeof(double));
    }
    //}

    //printf("copying\n lines=%d\n", lines);
    for (int i = 0; i < lines; i++)
    {
        for (int j = start; j < end + 1; j++)
        {
            //printf("i=%d; j=%d; (j-start)=%d; val=%f\n", i, j, j - start, srcArray[i][j]);
            subArray[i][j - start] = srcArray[i][j];
        }
    }
    //printf("end copying\n");
    return subArray;
}

void destroyTwoDimenArrayOnHeapUsingFree(double **ptr, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        free(ptr[i]);
    }
    free(ptr);
}

/*
stack : cette fonction permet de fusionnner 2 tableaux
     a: tableau num 1
     size_a : nb element de a
     b: tableau num 2
     size_b : nb element de b
     *res : addresse du tableau pour recuperer les 2 tableau en un seul 
*/
void stack(double *a, int size_a, double *b, int size_b, double *result)
{

    int idx = 0;
    for (int i = 0; i < size_a; i++)
    {
        result[idx] = a[i];
        idx++;
    }

    for (int i = 0; i < size_b; i++)
    {
        result[idx] = b[i];
        idx++;
    }

    //return result;
}

vector<int> rle(double *series, int size_s)
{
    vector<int> a;
    int rep = 0;
    for (int i = 1; i < size_s; i++)
    {
        if (series[i - 1] == series[i])
        {
            rep++;
        }
        else
        {
            a.push_back(rep + 1);
            rep = 0;
        }
    }
    a.push_back(rep + 1);
    return a;
}
vector<int> decode_rle(vector<int> code){
    vector<int> uncoded;
    for(int i = 0; i < code.size(); i++){
        for(int j = 0; j<code[i]; j++){
        uncoded.push_back(code[i]);
        }
    }
    return uncoded;
}
/*-------- RLE with temporal relaxationn ------*/

int count_tShift(double val, double *serie, int size_s)
{
    int i, rep;
    i = 0;
    rep = 0;

    while (i < size_s)
    {
        if (val == serie[i])
        {
            rep++;
            i++;
        }
        else
        {
            if ((i < size_s - 1) && (val == serie[i + 1]))
            {
                rep = rep + 2;
                i = i + 2;
            }
            else
            {
                return rep;
            }
        }
    }
    return rep;
}

vector<int> rle_temp(double *series, int size_s, int w)
{
    //printf("Input size = %d\n", size_s);
    vector<int> a;
    if (size_s == 0)
    {
        return a;
    }
    else
    {
        vector<int> code;
        for (int i = 0; i < size_s; i++)
        {
            int rep;
            rep = count_tShift(series[i], series + i, size_s - i);
            //printf("series[%d]=%f; rep=%d\n", i, series[i], rep);
            code.push_back(rep);
        }

        int max, stratMax, endMax;
        vector<int>::iterator max_it;
        max_it = max_element(code.begin(), code.end());
        max = *max_it;

        stratMax = (int) distance(code.begin(), max_it);
        endMax = stratMax + max;
        //printf("max=%d; startMx=%d; endMax=%d\n", max, stratMax, endMax);

        /*printf("Left Part of the serie\n");
        for (int i = 0; i < stratMax; i++)
        {
            printf("%f\t", series[i]);
        }
        printf("\n");
        printf("Right Part of the serie\n");
        for (int i = endMax; i < (size_s - endMax); i++)
        {
            printf("%f\t", series[i]);
        }
        printf("\n");*/

        vector<int> kept_code;
        a = rle_temp(series, stratMax, w);                         // compute the max for the left part of the series
        a.push_back(max);                                          // append the max value to the vector `a`
        kept_code = rle_temp(series + endMax, size_s - endMax, w); //compute the max for the roght part of the series
        a.insert(a.end(), kept_code.begin(), kept_code.end());
        return a;
    }
}

/*-------- RLE with spatiale relaxationn ------*/
int count_sShift(double val, double **serie, int size_s, int w)
{
    int i, rep;
    i = 0;
    rep = 0;

    while (i < size_s)
    {
        bool decision = false;
        int j = 0;
        while (j < w)
        {
            //printf("val=%f; pos(%d,%d)=%f\n", val, j, i, serie[j][i]);
            decision = decision || (val == serie[j][i]);
            j++;
        }

        if (decision)
        {
            rep++;
            i++;
        }
        else
        {
            //i++;
            return rep;
        }
    }
    return rep;
}

vector<int> rle_spatio(double **series, int size_s, int w)
{
    //printf("rle debug\n");
    vector<int> a;
    if (size_s == 0)
    {
        return a;
    }
    else
    {
        vector<int> code;
        //printf("============\n");
        for (int i = 0; i < size_s; i++)
        {
            int rep;

            double **subArray;
            int start = i, end = size_s;
            subArray = extractElements(series, start, end, w);
            //printf("avant\n");
            rep = count_sShift(series[0][i], subArray, size_s - i, w);
            //printf("series[%d]=%f; rep=%d\n", i, series[0][i], rep);
            destroyTwoDimenArrayOnHeapUsingFree(subArray, w, (end - start + 1));
            code.push_back(rep);
        }

        int max, stratMax, endMax;
        vector<int>::iterator max_it;
        max_it = max_element(code.begin(), code.end());
        max = *max_it;

        stratMax = (int) distance(code.begin(), max_it);
        endMax = stratMax + max;

        vector<int> kept_code;
        a = rle_spatio(series, stratMax, w); // compute the max for the left part of the series
        a.push_back(max);
        // append the max value to the vector `a`
        double **subArray;
        subArray = extractElements(series, endMax, size_s, w); // Exxtract element from columns `endMax` to the end
        kept_code = rle_spatio(subArray, size_s - endMax, w);  //compute the max for the right part of the series
        destroyTwoDimenArrayOnHeapUsingFree(subArray, w, (size_s - endMax + 1));

        a.insert(a.end(), kept_code.begin(), kept_code.end());
        return a;
    }
}

/*-------- RLE with spatio-temporal relaxationn ------*/

int count_stShift(double val, double **serie, int size_s, int ws, int wt)
{
    int i, rep;
    i = 0;
    rep = 0;

    while (i < size_s)
    {

        bool decision = false;
        int j = 0;
        while (j < ws)
        {
            /*
                on regarde a la position i+1
            */
            //printf("val=%f; pos(%d,%d)=%f\n", val, j, i, serie[j][i]);
            decision = decision || (val == serie[j][i]);
            j++;
        }

        if (decision)
        {
            rep++;
            i++;
        }
        else
        {
            if (i < size_s - 1)
            {
                decision = false;
                j = 0;
                while (j < ws)
                {
                    /*
                on regarde a la position i+2
                */
                    //printf("val=%f; pos(%d,%d)=%f\n", val, j, i, serie[j][i]);
                    decision = decision || (val == serie[j][i + 1]);
                    j++;
                }
            }

            if (decision)
            {
                rep = rep + 2;
                i = i + 2;
            }
            else
            {
                return rep;
            }
        }
    }
    return rep;
}

vector<int> rle_spatiotemp(double **series, int size_s, int ws, int wt)
{
    vector<int> a;
    if (size_s == 0)
    {
        return a;
    }
    else
    {
        vector<int> code;
        //printf("============\n");
        for (int i = 0; i < size_s; i++)
        {
            int rep;

            double **subArray;
            int start = i, end = size_s;
            subArray = extractElements(series, start, end, ws);
            //printf("avant\n");
            rep = count_stShift(series[0][i], subArray, size_s - i, ws, wt);
            //printf("series[%d]=%f; rep=%d\n", i, series[0][i], rep);
            destroyTwoDimenArrayOnHeapUsingFree(subArray, ws, (end - start + 1));
            code.push_back(rep);
        }

        int max, stratMax, endMax;
        vector<int>::iterator max_it;
        max_it = max_element(code.begin(), code.end());
        max = *max_it;

        stratMax = (int) distance(code.begin(), max_it);
        endMax = stratMax + max;

        vector<int> kept_code;
        a = rle_spatiotemp(series, stratMax, ws, wt); // compute the max for the left part of the series
        a.push_back(max);
        // append the max value to the vector `a`
        double **subArray;
        subArray = extractElements(series, endMax, size_s, ws);        // Exxtract element from columns `endMax` to the end
        kept_code = rle_spatiotemp(subArray, size_s - endMax, ws, wt); //compute the max for the right part of the series
        destroyTwoDimenArrayOnHeapUsingFree(subArray, ws, (size_s - endMax + 1));

        a.insert(a.end(), kept_code.begin(), kept_code.end());
        return a;
    }
}