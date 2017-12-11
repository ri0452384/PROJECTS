import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
"""
MP in Sorting - a course requirement for CMSC 142 - Analysis of Algorithms
by: Rayven N. Ingles, BSCS 4
Description:
    Each graph is headed by a Figure #, and each graph in the figure has a part #.
    Simply navigate to each graph to make changes to its settings, labels, etc. 
    There is also a max_count variable that will determine the n for every figure. 
    Simply adjust this value to increase or decrease n.
    
Disclaimer:
    This MP solution was created via Python 3.6
    Some formulae for the # of comparisons taken from:
        http://watson.latech.edu/book/algorithms
"""

"""
helper function definitions here
"""


def insertion(array):
    """insertion sort subroutine for bucket sort"""
    ans = 0
    for index in range(1,len(array)):
        curr = array[index]
        pos = index
        ans += 1

        while pos > 0 and array[pos-1] > curr:
            array[pos] = array[pos-1]
            pos = pos-1
            ans += 1
        array[pos] = curr
    return ans


def shell_insertion_helper(array,s,gap, ans):
    """insertion sort subroutine for shell sort"""
    for i in range(s+gap,len(array),gap):
        ans += 1
        curr = array[i]
        pos = i

        while pos>=gap and array[pos-gap]>curr:
            ans += 1
            array[pos]=array[pos-gap]
            pos = pos-gap
        array[pos]=curr
    return ans


def shell_sort(array):
    """ implements insertion sort as a subroutine"""
    ans = 0
    div = len(array)//2

    while div > 0:
        for s in range(div):
            ans = shell_insertion_helper(array,s,div, ans)
        div = div // 2
    return ans


def ciura_gaps_shell_sort(array):
    """ implements ciura's gaps in shell sort"""
    ans = 0
    gaps = [i for i in [701, 301, 132, 57, 23, 10, 4, 1] if i < len(array)] #ciura's gaps
    for gap in gaps:
        for i in range(gap, len(array)):
            temp = array[i]
            j = i
            ans += 1

            while j >= gap and array[j-gap] > temp:
                ans += 1
                array[j] = array[j - gap]
                j -= gap
            array[j] = temp
    return ans


def graph(formula, x_range, c,label,c_patch,line_style):
    """ helps graph figures given a formula"""
    elements = np.array(x_range)
    y=[]
    for x in elements:
        y.append(eval(formula))

    plt.plot(elements, y, color= c, label=label,linestyle=line_style)
    plt.legend(handles=[c_patch])
    plt.legend()
    return plt


def bucket_sort(array):
    """bucket sort w/ insertion as subroutine"""
    b_size = 10
    ans = 0
    min_val = min(array)
    max_val = max(array)
    b_count = int(math.floor((max_val - min_val) / b_size) + 1)
    buckets = []

    for i in range(0, b_count):
        buckets.append([])

    """fill each bucket"""
    for i in range(0, len(array)):
        ans += 1
        buckets[int(math.floor((array[i] - min_val) / b_size))].append(array[i])

    """sort buckets"""
    array = []
    for i in range(0, len(buckets)):
        ans += insertion(buckets[i])
        for j in range(0, len(buckets[i])):
            array.append(buckets[i][j])
    return ans


def radix_with_counting(array, radix=10):
    """implements counting sort as a subroutine"""
    ans = 0
    if len(array) == 0:
        return ans

    # Determine minimum and maximum values
    min_val = array[0]
    max_val = array[0]

    for i in range(1, len(array)):
        if array[i] < min_val:
            min_val = array[i]
        elif array[i] > max_val:
            max_val = array[i]

    # LSD(not the drug) counting sort
    exp = 1
    while (max_val - min_val) / exp >= 1:
        array, tmp = counting_sort(array, radix, exp, min_val)
        ans += tmp
        exp *= radix

    return ans


def counting_sort(array, radix, exp, min_val):
    """counting sort subroutine for radix sort"""
    ans = 0
    rad_index = -1
    group = [0] * radix
    ans_array = [None] * len(array)

    for i in range(0, len(array)):
        ans += 1
        rad_index = int(math.floor(((array[i] - min_val) / exp) % radix))
        group[rad_index] += 1
    for i in range(1, radix):
        ans += 1
        group[i] += group[i - 1]
    for i in range(len(array) - 1, -1, -1):
        ans += 1
        rad_index = int(math.floor(((array[i] - min_val) / exp) % radix))
        group[rad_index] -= 1
        ans_array[group[rad_index]] = array[i]
    return ans_array, ans


def radix_with_insert(array):
    """implements insertion sort as a subroutine, implementation already within the function"""
    radix = 10
    max_length = False
    temp, pos = -1, 1
    ans = 0

    while not max_length:
        temp = 0
        max_length = True
        group = [list() for _ in range(radix)]

        """grouping"""
        for i in array:
            temp = i / pos
            ans += 1
            group[int(math.floor(temp % radix))].append(i)
            if max_length and temp > 0:
                max_length = False

        x = 0
        for y in range(radix):
            g = group[y]
            for i in g:
                array[x] = i
                x += 1
        pos *= radix
    return ans


def merge_sort(array):
    ans = 0
    if len(array)>1:
        mid = len(array)//2
        left = array[:mid]
        right = array[mid:]
        ans += merge_sort(left)
        ans += merge_sort(right)
        i=0
        j=0
        k=0

        while i < len(left) and j < len(right):
            ans += 1
            if left[i] < right[j]:
                array[k]=left[i]
                i += 1
            else:
                array[k]=right[j]
                j += 1
            k += 1

        while i < len(left):
            ans += 1
            array[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            ans += 1
            array[k]=right[j]
            j += 1
            k += 1
    return ans


def quick_sort(array):
    ans = 0
    ans += quick_sort_helper(array, 0, len(array)-1)
    return ans


def quick_sort_helper(array, first, last):
    ans = 0
    if first < last:
        pivot, ans = split(array, first, last)
        ans += quick_sort_helper(array, first, pivot-1)
        ans += quick_sort_helper(array, pivot+1, last)
    return ans


def split(array,first,last):
    """helper function to partition the array according to the pivot"""
    ans = 0
    pivot = array[first]
    left = first+1
    right = last

    done = False
    while not done:
        while left <= right and array[left] <= pivot:
            ans += 1
            left = left + 1

        while array[right] >= pivot and right >= left:
            ans += 1
            right = right -1

        if right < left:
            done = True

        else:
            temp = array[left]
            array[left] = array[right]
            array[right] = temp

    temp = array[first]
    array[first] = array[right]
    array[right] = temp

    return right, ans


""" end of helper function definitions"""

"""
Figure 1. O(n^2) algorithms 
Comparison for Insertion, Selection, and Bubble Sort

Insertion sort: 
Best:  n - 1
Worst:  n/2(n - 1)
Average:  n/4(n - 1)

Selection sort:
Best: n(n - 1)/2
Worst: n(n - 1)/2
Average: n(n - 1)/2

Bubble sort:
Best: n - 1
Worst: n(n - 1)/2
Average: n(n - 1)/2

"""
max_range = 300
plt.close()
plt.figure(figsize=(10,5))
plt.suptitle("O(n^2) algorithms")
plt.rcParams.update({'font.size': 8})
"""
    Part I - Best Case O(n^2) algorithms
"""
plt.subplot(131)
plt.title('Best Case')
plt.ylabel('Comparison count(f(n))')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
plot = graph('x-1',range(1,max_range),'r','Insertion Sort',red_patch,'-')
blue_patch = mpatches.Patch(color='blue' )
graph('(x*(x - 1))/2',range(1,max_range),'b','Selection Sort',blue_patch,'-')
green_patch = mpatches.Patch(color='green')
graph('x - 1',range(1,max_range),'g','Bubble Sort',green_patch,'--')

"""
    Part II - Worst Case O(n^2) algorithms
"""
plt.subplot(132)
plt.title('Worst Case')
plt.ylabel('Comparison count(f(n))')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
plot = graph('(x*(x - 1))/2',range(1,max_range),'r','Insertion Sort',red_patch,'--')
blue_patch = mpatches.Patch(color='blue' )
graph('(x*(x - 1))/2',range(1,max_range),'b','Selection Sort',blue_patch,'-.')
green_patch = mpatches.Patch(color='green')
graph('(x*(x - 1))/2',range(1,max_range),'g','Bubble Sort',green_patch,':')

"""
    Part III - Average Case O(n^2) algorithms
"""
plt.subplot(133)
plt.title('Average Case')
plt.ylabel('Comparison count(f(n))')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
plot = graph('(x*(x - 1))/4',range(1,max_range),'r','Insertion Sort',red_patch,'--')
blue_patch = mpatches.Patch(color='blue' )
graph('(x*(x - 1))/2',range(1,max_range),'b','Selection Sort',blue_patch,'-.')
green_patch = mpatches.Patch(color='green')
graph('(x*(x - 1))/2',range(1,max_range),'g','Bubble Sort',green_patch,':')
plt.show()

"""
Figure 2. Insertion Sort vs. Shell sort with insertion sort as subroutine  
"""
max_range = 300
plt.close()
plt.figure(figsize=(10,5))
plt.suptitle("Insertion Sort vs. Shell sort with insertion sort as subroutine")
plt.rcParams.update({'font.size': 8})
"""
    Part I - Best Case 
"""
plt.subplot(131)
plt.title('Pre-sorted sequences')
plt.ylabel('Comparisons')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
graph('x-1',range(1,max_range),'r','Insertion Sort',red_patch,'--')
blue_patch = mpatches.Patch(color='blue' )
graph('(x * math.log2(x) )-x',range(1,max_range),'b','Shell Sort w/ insertion',blue_patch,'-')
"""
    Part II - Worst Case
"""
plt.subplot(132)
plt.title('Reversed sequences')
plt.ylabel('Comparisons')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
graph('(x*(x - 1))/2',range(1,max_range),'r','Insertion Sort',red_patch,'--')
blue_patch = mpatches.Patch(color='blue' )
graph('(x * math.log2(x) )-x',range(1,max_range),'b','Shell Sort w/ insertion',blue_patch,'-')
"""
    Part III - Average Case
"""
plt.subplot(133)
plt.title('Random sequences')
plt.ylabel('Comparisons')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
graph('(x*(x - 1))/4',range(1,max_range),'r','Insertion Sort',red_patch,'--')
blue_patch = mpatches.Patch(color='blue' )
graph('(x * math.log2(x) )-x',range(1,max_range),'b','Shell Sort w/ insertion',blue_patch,'-')
plt.show()

"""
Figure 3. Shell's Gaps vs. Cuira's Gaps  
"""
max_range = 300
plt.close()
plt.figure(figsize=(10,5))
plt.title("Shell's Gaps vs. Cuira's Gaps  ")
plt.rcParams.update({'font.size': 8})
"""
    Part I - Random Numbers
"""
plt.ylabel('Comparison count(f(n))')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
elements = np.array(range(1,max_range))
y=[]
count = 0
for x in elements:
    x = random.sample(range(1000), k=count)
    y.append(shell_sort(x))
    count +=1

plt.plot(elements, y, color='r', label='Shell Sort using Shell\'s gaps',linestyle='-')
plt.legend(handles=[red_patch])

blue_patch = mpatches.Patch(color='blue')
elements = np.array(range(1,max_range))
y=[]
count = 0
for x in elements:
    x = random.sample(range(1000), k=count)
    y.append(ciura_gaps_shell_sort(x))
    count +=1

plt.plot(elements, y, color='b', label='Shell Sort w/ Cuira\'s gaps',linestyle='-')
plt.legend(handles=[blue_patch])
plt.legend()

plt.show()

"""
Figure 4.  Bucket Sort for different distributions
"""
max_range = 300
plt.close()
plt.figure(figsize=(10,5))
plt.suptitle("Bucket Sort for different distributions")
plt.rcParams.update({'font.size': 8})
"""
    Part I - Gaussian distribution
"""
plt.subplot(321)
plt.title('Frequency Distribution')
#generate the elements of the array to be sorted
elements = []
y=[]
for i in range(0,max_range):
    elements.append(int(1000*random.gauss(500,800)))
plt.hist(elements,normed=True)

"""
Part I-2 - runtime graph for Gaussian distribution
"""
plt.subplot(322)
plt.title('Runtime Computation')
plt.ylabel('Comparisons')
plt.xlabel('n')
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = bucket_sort(temp)
    x.append(i)
    y.append((compare_count/4)*(compare_count - 1))
plt.plot(x,y)
"""
    Part II - More data near mean
"""
plt.subplot(323)
#generate the elements of the array to be sorted
elements = []
y=[]
for i in range(0,int(max_range/2)):
    elements.append(int(1000*random.normalvariate(500,430)))
for i in range(int(max_range/2) +1,max_range):
    elements.append(int(1000*random.normalvariate(500,1)))
plt.hist(elements,normed=True)

"""
Part II-2 - runtime graph for 2nd distribution
"""
plt.subplot(324)
plt.ylabel('Comparisons')
plt.xlabel('n')
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = bucket_sort(temp)
    x.append(i)
    y.append((compare_count/4)*(compare_count - 1))
plt.plot(x,y)

"""
    Part III - Even More data near mean
"""
plt.subplot(325)
#generate the elements of the array to be sorted
elements = []
y=[]

for i in range(0,int(max_range/4)):
    elements.append(int(1000*random.normalvariate(500,430)))
for i in range(int(max_range/4) +1,max_range):
    elements.append(int(1000*random.normalvariate(500,1)))
plt.hist(elements,normed=True)

"""
Part III-2 - runtime graph for 3rd distribution
"""
plt.subplot(326)
plt.ylabel('Comparisons')
plt.xlabel('n')
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = bucket_sort(temp)
    x.append(i)
    y.append((compare_count/4)*(compare_count - 1))
plt.plot(x,y)
plt.show()

"""
Figure 5. Radix Sort Optimization  
"""
max_range = 100
plt.close()
plt.figure(figsize=(10,5))
plt.suptitle("Radix Sort Optimization ")
plt.rcParams.update({'font.size': 8})
"""
    Part I - Pre sorted Numbers
"""
plt.subplot(131)
plt.title('Pre sorted Numbers')
plt.ylabel('Comparison count(f(n))')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
elements = np.array(range(1,max_range))
count = 0
for x in elements:
    y.append(count)
    count +=1
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = radix_with_insert(temp)
    y.append(compare_count)
    x.append(i)

plt.plot(x, y, color='r', label='Radix Sort w/ Insertion as subroutine',linestyle='-')
plt.legend(handles=[red_patch])

blue_patch = mpatches.Patch(color='blue')
elements = np.array(range(1,max_range))
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = radix_with_counting(temp)
    y.append(compare_count)
    x.append(i)

plt.plot(x, y, color='b', label='Radix Sort w/ Counting Sort as subroutine',linestyle='-')
plt.legend(handles=[blue_patch])
plt.legend()

"""
    Part II - Reverse sorted Numbers
"""
plt.subplot(132)
plt.title('Reverse sorted Numbers')
red_patch = mpatches.Patch(color='red')
elements = np.array(range(1,max_range))
elements = sorted(elements,reverse=True)
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = radix_with_insert(temp)
    y.append(compare_count)
    x.append(i)
plt.plot(x, y, color='r', label='Radix Sort w/ Insertion as subroutine',linestyle='-')
blue_patch = mpatches.Patch(color='blue')
elements = np.array(range(1,max_range))
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = radix_with_counting(temp)
    y.append(compare_count)
    x.append(i)
plt.plot(x, y, color='b', label='Radix Sort w/ Counting Sort as subroutine',linestyle='-')
"""
    Part III - Random Numbers
"""
plt.subplot(133)
plt.title('Random Numbers')
red_patch = mpatches.Patch(color='red')
elements = np.array(range(1,max_range))

for x in elements:
    x = int(300*random.random())
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = radix_with_insert(temp)
    y.append(compare_count)
    x.append(i)
plt.plot(x, y, color='r', label='Radix Sort w/ Insertion as subroutine',linestyle='-')
blue_patch = mpatches.Patch(color='blue')
elements = np.array(range(1,max_range))
y=[]
x = []
temp = []
for i in range(0,max_range):
    temp.clear()
    temp.append(elements[0])
    for j in range(1,i):
        temp.append(elements[j])
    compare_count = radix_with_counting(temp)
    y.append(compare_count)
    x.append(i)
plt.plot(x, y, color='b', label='Radix Sort w/ Counting Sort as subroutine',linestyle='-')
plt.show()
"""
Figure 6. Merge Sort vs Quick Sort on Random numbers
Merge Sort:
Best: n log2 n
Worst: n log2 n
Average: n log2 n
"""
max_range = 300
plt.close()
plt.figure(figsize=(10,5))
plt.title("Quick Sort vs Merge Sort on Random Numbers ")
plt.rcParams.update({'font.size': 8})
"""
    Part I - Random Numbers
"""
plt.ylabel('Comparison count(f(n))')
plt.xlabel('n')
red_patch = mpatches.Patch(color='red')
elements = np.array(range(1,max_range))
y=[]
count = 0
for x in elements:
    x = random.sample(range(1000), k=count)
    y.append(merge_sort(x))
    count +=1

plt.plot(elements, y, color='r', label='Quick Sort',linestyle='-')
plt.legend(handles=[red_patch])
blue_patch = mpatches.Patch(color='blue')
elements = np.array(range(1,max_range))
y=[]
count = 0
for x in elements:
    x = random.sample(range(1000), k=count)
    y.append(quick_sort(x))
    count +=1
plt.plot(elements, y, color='b', label='Merge Sort',linestyle='-')
plt.legend(handles=[blue_patch])
plt.legend()
plt.show()
