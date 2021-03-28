#include <stdio.h>
#include <stdlib.h>
#include "RunLengthEncoding.hpp"
#include <vector>
#include <algorithm>

using namespace std;

int main()
{
    double tab[] = {2, 5, 2, 5, 2, 5, 5, 1};
    double tab2[] = {2, 2, 8, 8, 8};
    int s_tab = 8, s_tab2 = 5, s_res;

    s_res = s_tab + s_tab2;
    double *res;

    res = (double *)malloc(s_res * sizeof(double));

    stack(tab, s_tab, tab2, s_tab2, res);

    for (size_t i = 0; i < s_res; i++)
    {
        printf("%f\t", res[i]);
    }
    printf("\n");
    vector<int> a;
    a = rle(res, s_res);
    printf("size of rle code is %ld\tcode=\t\n", a.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        printf("%d\t", a[i]);
    }
    printf("\n");

    printf("\n\nRLE Temp\n\n");
    vector<int> rle_t, rle_s, rle_st;
    rle_t = rle_temp(res, s_res, 3);

    printf("size of rle code is %ld\tcode=\t\n", rle_t.size());

    for (size_t i = 0; i < rle_t.size(); i++)
    {
        printf("%d\t", rle_t[i]);
    }
    printf("\n");

    double dd[3][6] = {{0, 2, 2, 1, 6, 5},
                       {3, 3, 1, 1, 4, 4},
                       {1, 1, 1, 1, 2, 6}};
    int s = 6;
    int w = 3;
    double **tab3;
    tab3 = (double **)malloc((size_t)(w * sizeof(double *)));
    for (int j = 0; j < w; j++)
    {
        tab3[j] = (double *)malloc((size_t)(s * sizeof(double)));
        tab3[j] = dd[j];
    }

    double **subArray;
    int start = 1, end = s;
    subArray = extractElements(tab3, start, end, w);

    printf("show content\n");
    if (subArray != NULL)
    {
        /* code */

        for (int i = 0; i < (end - start + 1); i++)
        {
            printf("val=%f\n", subArray[0][i]);
        }
    }

    int count = count_sShift(subArray[0][0], subArray, s, w);
    printf("count=%d\n", count);

    rle_s = rle_spatio(tab3, s, w);

    for (size_t i = 0; i < rle_s.size(); i++)
    {
        printf("%d\t", rle_s[i]);
    }
    printf("\n");

    printf("-------- rle Spatio Temp ---------\n\n");
    int count_st = count_stShift(tab3[0][0], tab3, s, w, w);
    printf("count_st=%d\n", count_st);
    rle_st = rle_spatiotemp(tab3, s, w, w);

    for (size_t i = 0; i < rle_st.size(); i++)
    {
        printf("%d\t", rle_st[i]);
    }
    printf("\n");

    rle_st = decode_rle(rle_st);
    for (size_t i = 0; i < rle_st.size(); i++)
    {
        printf("%d\t", rle_st[i]);
    }
    printf("\n");
    return 0;
}