---
layout: post
title: The Maximum Subarray Problem
subtitle: When ugly and specialized algorithms beat elegant solutions
gh-repo: thiagolcmelo/thiagolcmelo.github.io
gh-badge: [star, follow]
tags: [c, algorithms]
thumbnail-img: /assets/img/subarray.png
comments: true
---

Given a sequence of numbers $S=\\{a_1, \dots, a_N\\}$, the maximum subarray is the subsequence $s=\\{a_{i}, \dots, a_{j}\\}$, with $s \subset S$, such that $\displaystyle\sum_{l=i}^{j} a_l$ is maximum among all possible values of $i$ and $j$ obeying $1 \leq i \leq j \leq N$.

An example using numbers can be:

$S = \\{ 3, 11, -2, -2, 2, -9, -4, -3, -8, 1, 5, 13, -1, 4, -7, -16 \\}$

For the sequence above, the maximum subarray is the subsequence:

$s = \\{ 1, 5, 13, -1, 4 \\}$

whose sum is $22$.

## Brute force algorithm

One way of of solving this problem is by using brute force: find the subsequence with the maximum sum among all possible subsequences.

We can define a `struct` to keep the result:

{% highlight c linenos %}
#include <stdio.h>

struct subarray
{
    int low;
    int high;
    int sum;
};
{% endhighlight %}

It is a very simple `struct`: it stores the index of the beginning of the maximum subarray, the index of its end, and the sum of its elements.

The following function is capable of searching for the maximum subarray among all possible subsequences. The reason why we define its parameters as an array `A`, and integers `low` and `high`, is because we will reuse this function later for optimizing another function, and this prototype will make things easier. But for now, `low=0` and `high=N - 1` will be fine.

{% highlight c linenos %}
struct subarray find_max_subarray_brute_force(int* A, int low, int high) {
    struct subarray max_subarray;
    max_subarray.low = low; // [1.1]
    max_subarray.high = low; // [1.2]
    max_subarray.sum = A[low]; // [1.3]

    for (int i = low; i < high; i++) { // [2]
        int sum = A[i];

        for (int j = i + 1; j <= high; j++) { // [3]
            sum += A[j];

            if (sum > max_subarray.sum) { // [4]
                max_subarray.sum = sum;
                max_subarray.low = i;
                max_subarray.high = j;
            } else if (A[j] > max_subarray.sum) { // [5]
                max_subarray.sum = A[j];
                max_subarray.low = j;
                max_subarray.high = j;
            }
        }
    }
    return max_subarray;
}
{% endhighlight %}

- In `[1.1]`, `[1.2]`, and `[1.3]` we start by assuming that the first element in the array `A` is by itself the maximum subarray.
- In `[2]` and `[3]` we loop over all possible subsequences.
- In `[4]` we check whether a current subsequence sum is greater than the maximum subsequence sum found so far, in positive case we update the result subarray.
- In `[5]` we cover a situation where the current subsequence sum might not be greater than the maximum subsequence sum found so far. However, the current element is by itself greater than that maximum. A case when its purpose becomes clear is when the maximum sum is negative, say $-50$, and the new element is bigger than that, say $-49$.

We can test this function against our sequence using:

{% highlight c linenos %}
int main() {
    int sequence[16] = { 3, 11, -2, -2, 2, -9, -4, -3,
                         -8, 1, 5, 13, -1, 4, -7, -16 };
    
    struct subarray max_subarray = find_max_subarray_brute_force(
        sequence, 0, 15);
    
    printf("Maximum subarray is:\n\n\t");
    
    for (int i = max_subarray.low; i <= max_subarray.high; i++) {
        printf("%3d ", sequence[i]);
    }
 
    printf("\n\nsum: %d\n", max_subarray.sum);
 
    return 0;   
}
{% endhighlight %}

If we store the three snippets above in a file called `maximum_subarray_brute_force.c`, then we can see the results by running:

```
$ gcc maximum_subarray_brute_force.c -o maximum_subarray_brute_force && ./maximum_subarray_brute_force
Maximum subarray is:

          1   5  13  -1   4 

sum: 22
```

This algorithm is not so ugly, it is quite legible and easy to understand actually. The problem is its asymptotic running time, basically $\Theta(n^2)$.

## Divide and conquer

An elegant and faster alternative to the brute force algorithm arises from the observation that the maximum subarray in a sequence $S=\\{a_1, \dots, a_N\\}$ is in either one of the following subsequences:

1. $s_{L}=\\{a_1, \dots, a_{N/2}\\}$
2. $s_{R}=\\{a_{N/2+1}, \dots, a_{N}\\}$
3. $s_{C}=\\{ a_i, \dots, a_j\\}$, where $i \leq N/2 < j$

Although cases **1.** and **2.** are just sub-problems of the original one, the third case is different.

Finding the maximum subarray satisfying the constrain $i \leq N/2 < j$ can be done using the following function:

{% highlight c linenos %}
#include <limits.h>

struct subarray find_max_crossing_subarray(int* A, int low, int mid, int high)
{
    int sum_left = INT_MIN;
    int sum_right = INT_MIN;
    int left_max_idx, right_max_idx;
    int sum = 0;

    for (int i = mid; i >= low; i--) // [1]
    {
        sum += A[i];
        if (sum > sum_left)
        {
            left_max_idx = i;
            sum_left = sum;
        }
    }

    sum = 0;
    for (int i = mid + 1; i <= high; i++) // [2]
    {
        sum += A[i];
        if (sum > sum_right)
        {
            right_max_idx = i;
            sum_right = sum;
        }
    }

    struct subarray subarray;
    subarray.low = left_max_idx; // [3.1]
    subarray.high = right_max_idx; // [3.2]
    subarray.sum = sum_left + sum_right; // [3.3]
    return subarray;
}
{% endhighlight %}

- In `[1]` we start at the middle and move to the left keeping track of the index whose sum of all elements from this index until the middle is maximum.
- In `[2]` we start at the middle and move to the right keeping track of the index whose sum of all elements from the middle until this index is maximum.
- In `[3.1]`, `[3.2]`, and `[3.3]` we compose the maximum subarray crossing the middle using the information obtained previously.

Now we need to use this solution as a complement for a function that solves the problem by recursively splitting the original sequence into subsequences a selecting the one with highest summation.

{% highlight c linenos %}
struct subarray find_max_subarray(int* A, int low, int high)
{
    if (low == high)
    {
        struct subarray subarray;
        subarray.low = low;
        subarray.high = high;
        subarray.sum = A[low];
        return subarray;
    }

    int mid = (high + low) / 2;
    struct subarray left = find_max_subarray(A, low, mid);
    struct subarray right = find_max_subarray(A, mid + 1, high);
    struct subarray crossing = find_max_crossing_subarray(A, low, mid, high);

    if (left.sum > right.sum && left.sum > crossing.sum)
    {
        return left;
    }
    else if(right.sum > left.sum && right.sum > crossing.sum)
    {
        return right;
    }
    return crossing;
}
{% endhighlight %}

When I first wrote this algorithm I was confused how it could work. It was until I realize the role of `find_max_crossing_subarray`, which is the real work horse of this algorithm.

We can test this function against our sequence using:

{% highlight c linenos %}
int main() {
    int sequence[16] = { 3, 11, -2, -2, 2, -9, -4, -3,
                         -8, 1, 5, 13, -1, 4, -7, -16 };
    
    struct subarray max_subarray = find_max_subarray(
        sequence, 0, 15);
    
    printf("Maximum subarray is:\n\n\t");
    
    for (int i = max_subarray.low; i <= max_subarray.high; i++) {
        printf("%3d ", sequence[i]);
    }
 
    printf("\n\nsum: %d\n", max_subarray.sum);
 
    return 0;   
}
{% endhighlight %}

If we store the snippets snippets above in a file called `maximum_subarray_divide_and_conquer.c`, also including the `struct subarray` and the headers `<stdio.h>` and `<limits.h>`, then we can see the results by running:

```
$ gcc maximum_subarray_divide_and_conquer.c -o maximum_subarray_divide_and_conquer && ./maximum_subarray_divide_and_conquer
Maximum subarray is:

          1   5  13  -1   4 

sum: 22
```

This algorithm has a much better asymptotic running time: $\Theta(n \, \log(n))$.

However, for $n$ sufficiently small, the brute force algorithm still beats the above algorithm, as we can observe using the following code:

{% highlight c linenos %}
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <time.h>

/*
 * Include the struct subarray and other functions here
* /

int main()
{
    for (int i = 2; i <= pow(2, 16); i = i * 2) // [1]
    {
        srand(0);
        int sequence[i];
        for (int j = 0; j < i; j++)
        {
            sequence[j] = rand() % 30 -15; // [2]
        }

        clock_t begin_divide_conquer = clock();
        struct subarray x = find_max_subarray(sequence, 0, i - 1);
        clock_t end_divide_conquer = clock();

        clock_t begin_brute_force = clock();
        struct subarray y = find_max_subarray_brute_force(sequence, 0, i - 1);
        clock_t end_brute_force = clock();

        double divide_conquer_t = (double)(end_divide_conquer - begin_divide_conquer) / CLOCKS_PER_SEC; // [3.1]
        double brute_force_t = (double)(end_brute_force - begin_brute_force) / CLOCKS_PER_SEC; // [3.2]
        
        printf("i = %7d, divide and conquer: %e, brute force: %e\n", i, divide_conquer_t, brute_force_t);
    }

    return 0;
}
{% endhighlight %}

- In `[1]` we generate sequences of sizes $2, 4, \dots, 2^{16}$.
- In `[2]` we define that these sequences will have random numbers ranging from $-15$ until $15$.
- In `[3.1]` we measure the execution time of the divide and conquer algorithm for each sequence.
- In `[3.2]` we measure the execution time of the brute force algorithm for each sequence.

The output of the code above is the following:

```
i =       2, divide and conquer: 2.00e-06, brute force: 1.00e-06
i =       4, divide and conquer: 1.00e-06, brute force: 1.00e-06
i =       8, divide and conquer: 2.00e-06, brute force: 2.00e-06
i =      16, divide and conquer: 3.00e-06, brute force: 2.00e-06
i =      32, divide and conquer: 5.00e-06, brute force: 4.00e-06
i =      64, divide and conquer: 8.00e-06, brute force: 1.20e-05
i =     128, divide and conquer: 1.40e-05, brute force: 4.60e-05
i =     256, divide and conquer: 3.40e-05, brute force: 1.79e-04
i =     512, divide and conquer: 6.00e-05, brute force: 8.05e-04
i =    1024, divide and conquer: 1.29e-04, brute force: 3.25e-03
i =    2048, divide and conquer: 2.79e-04, brute force: 1.34e-02
i =    4096, divide and conquer: 5.44e-04, brute force: 4.54e-02
i =    8192, divide and conquer: 9.33e-04, brute force: 1.46e-01
i =   16384, divide and conquer: 2.03e-03, brute force: 5.86e-01
i =   32768, divide and conquer: 4.63e-03, brute force: 2.15e+00
i =   65536, divide and conquer: 8.05e-03, brute force: 7.91e+00
```

Looking at the results above we can see that the divide and conquer algorithm only starts to bead the brute force algorithm for sequences of length $i \gt 32$.

It gives us the opportunity to come up with an algorithm which is a combination of both:
- for sequences shorter than $64$ elements: use brute force
- for sequences with size $64$ or greater: use divide and conquer

This improved algorithm is shown bellow.

{% highlight c linenos %}
struct subarray find_max_subarray_improved(int* A, int low, int high)
{
    if (high - low < 64)
    {
        return find_max_subarray_brute_force(A, low, high);
    }

    int mid = (high + low) / 2;
    struct subarray left = find_max_subarray_improved(A, low, mid);
    struct subarray right = find_max_subarray_improved(A, mid + 1, high);
    struct subarray crossing = find_max_crossing_subarray(A, low, mid, high);

    if (left.sum > right.sum && left.sum > crossing.sum)
    {
        return left;
    }
    else if(right.sum > left.sum && right.sum > crossing.sum)
    {
        return right;
    }
    return crossing;
}
{% endhighlight %}

And the comparison goes as follows:

```
i =       2, divConq: 2.00e-06, brForce: 1.00e-06, comb: 1.00e-06
i =       4, divConq: 1.00e-06, brForce: 1.00e-06, comb: 1.00e-06
i =       8, divConq: 2.00e-06, brForce: 1.00e-06, comb: 1.00e-06
i =      16, divConq: 3.00e-06, brForce: 2.00e-06, comb: 2.00e-06
i =      32, divConq: 5.00e-06, brForce: 4.00e-06, comb: 5.00e-06
i =      64, divConq: 9.00e-06, brForce: 1.70e-05, comb: 1.00e-05
i =     128, divConq: 1.60e-05, brForce: 5.30e-05, comb: 1.80e-05
i =     256, divConq: 3.20e-05, brForce: 2.14e-04, comb: 3.70e-05
i =     512, divConq: 6.40e-05, brForce: 7.98e-04, comb: 8.20e-05
i =    1024, divConq: 1.35e-04, brForce: 3.57e-03, comb: 1.63e-04
i =    2048, divConq: 3.11e-04, brForce: 1.38e-02, comb: 3.70e-04
i =    4096, divConq: 6.33e-04, brForce: 5.33e-02, comb: 5.84e-04
i =    8192, divConq: 1.17e-03, brForce: 2.03e-01, comb: 1.24e-03
i =   16384, divConq: 2.56e-03, brForce: 6.70e-01, comb: 2.09e-03
i =   32768, divConq: 4.09e-03, brForce: 2.13e+00, comb: 3.77e-03
i =   65536, divConq: 7.91e-03, brForce: 7.89e+00, comb: 8.73e-03
```

Above, `divComq` stands for divide and conquer, `brForce` stands for brute force and `comb` stands for combined.

What we see now is that the combined algorithm sometimes beats the divide and conquer one for bigger sequence lengths, for instance in $i=4096$, $i=16,384$, and $i=32,768$. However, it can very well be due to CPU oscillations (not sure what else was running on my background exactly).

## Linear running time

The brute force algorithm has an asymptotic running time of $\Theta(n^2)$, whereas the divide and conquer has a running time of $\Theta(n \, \log(n))$.

There is a way to tackle this problem with running time of $\Theta(n)$. It arises from the observation that by knowing the maximum subarray of a sequence $S'=\\{a_1, \dots, a_j\\}$, the maximum subarray of another subsequence $S''=\\{a_1, \dots, a_j, a_{j+1}\\}$ is either the maximum subarray of $S'$ or another subarray in the interval $S''=\\{a_i, \dots, a_j, a_{j+1}\\}$, where $i \geq 1$.

This observation by itself sounds clear, but not sufficient. It can be complemented by the fact that to judge whether a new candidate is actually better than the maximum subarray already found, we need to somehow keep track of a **second best guess**.

The following function implements this idea.

{% highlight c linenos %}
struct subarray find_max_subarray_linear(int *A, int length)
{
    struct subarray result;
    result.low = 0; // [1.1]
    result.high = 0; // [1.2]
    result.sum = A[0]; // [1.3]

    struct subarray second_guess;
    second_guess.low = 0; // [2.1]
    second_guess.high = 0; // [2.2]
    second_guess.sum = A[0]; // [2.3]

    for (int i = 1; i < length; i++)
    {
        if (A[i] > 0 && result.high == i - 1) // [3]
        {
            result.high = i;
            result.sum += A[i];
        }
        else if (A[i] > result.sum) // [4]
        {
            result.low = i;
            result.high = i;
            result.sum = A[i];
        }

        second_guess.sum += A[i]; // [5.1]
        second_guess.high = i; // [5.2]

        if (second_guess.sum > result.sum) // [6]
        {
            result.low = second_guess.low;
            result.high = second_guess.high;
            result.sum = second_guess.sum;
        }
        else if(second_guess.sum < A[i]) // [7]
        {
            second_guess.low = i;
            second_guess.high = i;
            second_guess.sum = A[i];
        }
    }
    return result;
}
{% endhighlight %}

- In `[1.1]`, `[1.2]`, and `[1.3]` we initialize the result assuming that the first element by itself is the maximum subarray.
- In `[2.1]`, `[2.2]`, and `[2.3]` we also initialize the *second best guess* assuming that the first element by itself is the maximum subarray.
- In `[3]` we update the upper bound of our current maximum subarray if the new element is greater than `0` and if the current maximum subarray is aligned to the right of the subsequence analysed so fat, which is to say there is no gap between this new element and the current maximum subarray.
- In `[4]` we replace the whole current maximum subarray if the current element by itself is greater than the sum.
- In `[5.1]` and `[5.2]` we update our *second best guess* naïvely: no judgement of the current element, because even if it is very negative, maybe the next one will be a *game changer*.
- In `[6]` we check whether our naïve *second best guess* has a sum bigger than our current maximum subarray by any chance, if it has, replace it.
- In `[7]`, very similar to `[4]`, we replace the current *second best guess* with the current element if its value is greater that the *second best guess* sum.

The comparison between the function above and the divide and conquer algorithm is done using the following code:

{% highlight c linenos %}
int main()
{
    for (int i = 2; i <= pow(2, 20); i = i * 2)
    {
        srand(0);
        int sequence[i];
        for (int j = 0; j < i; j++)
        {
            sequence[j] = rand() % 30 -15;
        }

        clock_t begin_divide_conquer = clock();
        struct subarray x = find_max_subarray(sequence, 0, i - 1);
        clock_t end_divide_conquer = clock();

        clock_t begin_linear = clock();
        struct subarray y = find_max_subarray_linear(sequence, i);
        clock_t end_linear = clock();

        double divide_conquer_t = (double)(end_divide_conquer - begin_divide_conquer) / CLOCKS_PER_SEC;
        double linear_t = (double)(end_linear - begin_linear) / CLOCKS_PER_SEC;

        printf("i = %7d, divConq: %.2e, linear: %.2e\n", i, divide_conquer_t, linear_t);
    }

    return 0;
}
{% endhighlight %}

And the results are shown bellow.

```
i =       2, divConq: 2.00e-06, linear: 1.00e-06
i =       4, divConq: 2.00e-06, linear: 1.00e-06
i =       8, divConq: 2.00e-06, linear: 1.00e-06
i =      16, divConq: 3.00e-06, linear: 1.00e-06
i =      32, divConq: 3.00e-06, linear: 2.00e-06
i =      64, divConq: 7.00e-06, linear: 2.00e-06
i =     128, divConq: 1.40e-05, linear: 3.00e-06
i =     256, divConq: 2.80e-05, linear: 5.00e-06
i =     512, divConq: 5.50e-05, linear: 9.00e-06
i =    1024, divConq: 1.18e-04, linear: 2.00e-05
i =    2048, divConq: 2.30e-04, linear: 2.90e-05
i =    4096, divConq: 5.09e-04, linear: 6.60e-05
i =    8192, divConq: 9.31e-04, linear: 1.16e-04
i =   16384, divConq: 1.93e-03, linear: 2.28e-04
i =   32768, divConq: 4.15e-03, linear: 4.70e-04
i =   65536, divConq: 8.12e-03, linear: 9.23e-04
i =  131072, divConq: 1.87e-02, linear: 2.14e-03
i =  262144, divConq: 3.63e-02, linear: 3.69e-03
i =  524288, divConq: 7.96e-02, linear: 9.27e-03
i = 1048576, divConq: 1.78e-01, linear: 1.49e-02
```

The results show that the linear algorithm beats the divide and conquer mercilessly.

## Conclusion

I am fond of recursive algorithms for solving problems using divide and conquer strategies. Although it will always require some specialization, it is a terrific framework for approaching problems without much thinking and most of times leading to good results.

 However, sometimes, detailed observations of a problem can give birth to better strategies. The issue with these strategies is that they cannot be applied to other problems easily, in other words they are hardly a lift and shift solution. Probably they provide opportunity for exercising our brain, sharpening our problem solving skills though.

 The lesson I take from here is to first try divide and conquer, it is very likely that the solution will be elegant and deliver fair performance. When there is time to deep dive, then search for particular solutions that better explore the nature of the problem at hand.

## References

1. Introduction to Algorithms. Cormen, Thomas H. and Leiserson, Charles E. and Rivest, Ronald L. and Stein, Clifford. Third edition. The MIT Press, 2009. ISBN 0262033844.