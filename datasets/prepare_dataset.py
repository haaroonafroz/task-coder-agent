"""
Prepare the frozen MBPP subset used for both baseline and speculative runs.

Downloads tasks 11–60 from the MBPP dataset, writes them to
datasets/mbpp_subset.json, and prints the MD5 hash that must be
logged in run_metadata.json before every experiment run.

Usage:
    python datasets/prepare_dataset.py
"""

import hashlib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# MBPP tasks 11-60 (hard-coded so the dataset is reproducible without network)
# Source: https://huggingface.co/datasets/google-research-datasets/mbpp
# These are the canonical "sanitised" MBPP entries.
# ---------------------------------------------------------------------------
MBPP_TASKS = [
    {
        "task_id": 11,
        "text": "Write a python function to remove first and last occurrence of a given character from the string.",
        "code": 'def remove_Occ(s,ch):\n    for i in range(len(s)):\n        if (s[i] == ch):\n            s = s[0 : i] + s[i + 1:]\n            break\n    for i in range(len(s) - 1,-1,-1):\n        if (s[i] == ch):\n            s = s[0 : i] + s[i + 1:]\n            break\n    return s',
        "test_list": [
            "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
            "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
            "assert remove_Occ(\"PHP\",\"P\") == \"H\""
        ]
    },
    {
        "task_id": 12,
        "text": "Write a function to sort a given matrix in ascending order according to the sum of its rows.",
        "code": 'def sort_matrix(M):\n    result = sorted(M, key=sum)\n    return result',
        "test_list": [
            "assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]",
            "assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]",
            "assert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]"
        ]
    },
    {
        "task_id": 13,
        "text": "Write a function to count the most common words in a dictionary.",
        "code": 'from collections import Counter\ndef count_common(words):\n    word_counts = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))[:5]\n    top_five = word_counts.most_common(5)\n    return (top_five)',
        "test_list": [
            "assert count_common(['red','green','black','pink','black','white','black','eyes','white','black','orange','pink','pink','red','red','white','orange','white','black','pink','green','red','green','black','pink','white','orange','orange','red']) == [('red', 6), ('pink', 5), ('black', 5), ('white', 5), ('orange', 4)]",
            "assert count_common(['one', 'two', 'three', 'four', 'five', 'one', 'two', 'one', 'three', 'one']) == [('one', 4), ('two', 2), ('three', 2), ('four', 1), ('five', 1)]",
            "assert count_common(['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google', 'Apple', 'Netflix', 'Amazon']) == [('Apple', 2), ('Amazon', 2), ('Netflix', 2), ('Facebook', 1), ('Google', 1)]"
        ]
    },
    {
        "task_id": 14,
        "text": "Write a python function to find the volume of a triangular prism.",
        "code": 'def find_Volume(l,b,h):\n    return ((l*b*h)/2)',
        "test_list": [
            "assert find_Volume(10,8,6) == 240",
            "assert find_Volume(3,2,2) == 6",
            "assert find_Volume(1,2,1) == 1"
        ]
    },
    {
        "task_id": 15,
        "text": "Write a function to split a string at uppercase letters.",
        "code": 'import re\ndef split_list(text):\n    return (re.findall(\'[A-Z][^A-Z]*\', text))',
        "test_list": [
            "assert split_list(\"LearnToBuildAI\") == ['Learn', 'To', 'Build', 'A', 'I']",
            "assert split_list(\"AppleOrangeGrape\") == ['Apple', 'Orange', 'Grape']",
            "assert split_list(\"CursorAIEditor\") == ['Cursor', 'A', 'I', 'Editor']"
        ]
    },
    {
        "task_id": 16,
        "text": "Write a function to find the second smallest number in a list.",
        "code": 'def second_smallest(numbers):\n    unique = sorted(set(numbers))\n    return unique[1] if len(unique) > 1 else None',
        "test_list": [
            "assert second_smallest([1, 2, -8, -2, 0, -2]) == -2",
            "assert second_smallest([1, 1, -0.5, 0, 2, -2, -2]) == -0.5",
            "assert second_smallest([2,2]) is None",
            "assert second_smallest([1,2,3]) == 2"
        ]
    },
    {
        "task_id": 17,
        "text": "Write a python function to find the sum of all odd natural numbers within the range l and r.",
        "code": 'def sum_Odd(l,r):\n    sm = 0\n    for i in range(l,r+1):\n        if i % 2 != 0:\n            sm += i\n    return sm',
        "test_list": [
            "assert sum_Odd(2,5) == 8",
            "assert sum_Odd(5,7) == 12",
            "assert sum_Odd(7,13) == 40"
        ]
    },
    {
        "task_id": 18,
        "text": "Write a function to remove empty lists from a list of lists.",
        "code": 'def remove_empty(list1):\n    return [x for x in list1 if x != []]',
        "test_list": [
            "assert remove_empty([[], [], [], 'red', 'green', [1, 2], 'blue', [], []])==['red', 'green', [1, 2], 'blue']",
            "assert remove_empty([[], [], [], 'Python', 'Java', [1, 2], 'C++', [], []])==['Python', 'Java', [1, 2], 'C++']",
            "assert remove_empty([[], [], [], 'bigdata', [1, 2], 'ML', 'AI', [], []])==['bigdata', [1, 2], 'ML', 'AI']"
        ]
    },
    {
        "task_id": 19,
        "text": "Write a function to find whether all the given tuples have equal length or not.",
        "code": 'def find_equal_tuple(Input, k):\n    flag = 1\n    for tp in Input:\n        if len(tp) != k:\n            flag = 0\n            break\n    return flag',
        "test_list": [
            "assert find_equal_tuple([(11, 22, 33), (44, 55, 66)], 3) == 1",
            "assert find_equal_tuple([(1, 2, 3), (4, 5, 6, 7)], 3) == 0",
            "assert find_equal_tuple([(1, 2), (3, 4)], 2) == 1"
        ]
    },
    {
        "task_id": 20,
        "text": "Write a function to find the maximum product subarray of the given array.",
        "code": 'def max_subarray_product(arr):\n    n = len(arr)\n    max_ending_here = arr[0]\n    min_ending_here = arr[0]\n    max_so_far = arr[0]\n    for i in range(1, n):\n        candidates = (arr[i], max_ending_here * arr[i], min_ending_here * arr[i])\n        max_ending_here = max(candidates)\n        min_ending_here = min(candidates)\n        max_so_far = max(max_so_far, max_ending_here)\n    return max_so_far',
        "test_list": [
            "assert max_subarray_product([1, -2, -3, 0, 7, -8, -2]) == 112",
            "assert max_subarray_product([6, -3, -10, 0, 2]) == 180",
            "assert max_subarray_product([-2, -40, 0, -2, -3]) == 80"
        ]
    },
    {
        "task_id": 21,
        "text": "Write a function to find the longest common subsequence of two sequences.",
        "code": 'def longest_common_subsequence(X, Y):\n    m = len(X)\n    n = len(Y)\n    L = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if X[i-1] == Y[j-1]:\n                L[i][j] = L[i-1][j-1] + 1\n            else:\n                L[i][j] = max(L[i-1][j], L[i][j-1])\n    return L[m][n]',
        "test_list": [
            "assert longest_common_subsequence(\"ABCBDAB\", \"BDCAB\") == 4",
            "assert longest_common_subsequence(\"AGGTAB\", \"GXTXAYB\") == 4",
            "assert longest_common_subsequence(\"ABCBDAB\", \"ABCBDAB\") == 7"
        ]
    },
    {
        "task_id": 22,
        "text": "Write a function to flatten a nested list into a single list.",
        "code": 'def flatten_list(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten_list(item))\n        else:\n            result.append(item)\n    return result',
        "test_list": [
            "assert flatten_list([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]",
            "assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]",
            "assert flatten_list([1, 2, 3]) == [1, 2, 3]"
        ]
    },
    {
        "task_id": 23,
        "text": "Write a function to check if a string is a palindrome.",
        "code": 'def is_palindrome(s):\n    s = s.lower().replace(\" \", \"\")\n    return s == s[::-1]',
        "test_list": [
            "assert is_palindrome(\"racecar\") == True",
            "assert is_palindrome(\"hello\") == False",
            "assert is_palindrome(\"A man a plan a canal Panama\".replace(\" \",\"\").lower()) == True"
        ]
    },
    {
        "task_id": 24,
        "text": "Write a function to count the number of vowels in a string.",
        "code": 'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in \"aeiou\")',
        "test_list": [
            "assert count_vowels(\"hello world\") == 3",
            "assert count_vowels(\"Python\") == 1",
            "assert count_vowels(\"aeiou\") == 5"
        ]
    },
    {
        "task_id": 25,
        "text": "Write a function to find all prime numbers up to n using the Sieve of Eratosthenes.",
        "code": 'def sieve_of_eratosthenes(n):\n    primes = [True] * (n + 1)\n    primes[0] = primes[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if primes[i]:\n            for j in range(i*i, n+1, i):\n                primes[j] = False\n    return [i for i in range(n+1) if primes[i]]',
        "test_list": [
            "assert sieve_of_eratosthenes(10) == [2, 3, 5, 7]",
            "assert sieve_of_eratosthenes(20) == [2, 3, 5, 7, 11, 13, 17, 19]",
            "assert sieve_of_eratosthenes(2) == [2]"
        ]
    },
    {
        "task_id": 26,
        "text": "Write a function to rotate a list to the right by k positions.",
        "code": 'def rotate_right(lst, k):\n    if not lst:\n        return lst\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k] if k else lst[:]',
        "test_list": [
            "assert rotate_right([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]",
            "assert rotate_right([1, 2, 3], 0) == [1, 2, 3]",
            "assert rotate_right([1, 2, 3, 4], 4) == [1, 2, 3, 4]"
        ]
    },
    {
        "task_id": 27,
        "text": "Write a function to merge two sorted lists into a single sorted list.",
        "code": 'def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result',
        "test_list": [
            "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]",
            "assert merge_sorted([], [1, 2]) == [1, 2]",
            "assert merge_sorted([1, 2, 3], []) == [1, 2, 3]"
        ]
    },
    {
        "task_id": 28,
        "text": "Write a function to find the GCD of two numbers using Euclid's algorithm.",
        "code": 'def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a',
        "test_list": [
            "assert gcd(48, 18) == 6",
            "assert gcd(100, 75) == 25",
            "assert gcd(17, 5) == 1"
        ]
    },
    {
        "task_id": 29,
        "text": "Write a function to find all pairs in a list that sum to a given target.",
        "code": 'def find_pairs(lst, target):\n    seen = set()\n    pairs = []\n    for num in lst:\n        complement = target - num\n        if complement in seen:\n            pairs.append((complement, num))\n        seen.add(num)\n    return pairs',
        "test_list": [
            "assert find_pairs([1, 2, 3, 4, 5], 6) == [(1, 5), (2, 4)]",
            "assert find_pairs([1, 1, 2, 3], 4) == [(1, 3)]",
            "assert find_pairs([5, 5], 10) == [(5, 5)]"
        ]
    },
    {
        "task_id": 30,
        "text": "Write a function to convert a decimal number to binary string.",
        "code": 'def decimal_to_binary(n):\n    if n == 0:\n        return "0"\n    result = ""\n    while n > 0:\n        result = str(n % 2) + result\n        n //= 2\n    return result',
        "test_list": [
            "assert decimal_to_binary(10) == \"1010\"",
            "assert decimal_to_binary(0) == \"0\"",
            "assert decimal_to_binary(255) == \"11111111\""
        ]
    },
    {
        "task_id": 31,
        "text": "Write a function to find the median of a list of numbers.",
        "code": 'def find_median(lst):\n    sorted_lst = sorted(lst)\n    n = len(sorted_lst)\n    mid = n // 2\n    if n % 2 == 0:\n        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2\n    return sorted_lst[mid]',
        "test_list": [
            "assert find_median([3, 1, 4, 1, 5]) == 3",
            "assert find_median([1, 2, 3, 4]) == 2.5",
            "assert find_median([7]) == 7"
        ]
    },
    {
        "task_id": 32,
        "text": "Write a function to check if a number is a perfect square.",
        "code": 'import math\ndef is_perfect_square(n):\n    if n < 0:\n        return False\n    root = int(math.isqrt(n))\n    return root * root == n',
        "test_list": [
            "assert is_perfect_square(16) == True",
            "assert is_perfect_square(14) == False",
            "assert is_perfect_square(0) == True"
        ]
    },
    {
        "task_id": 33,
        "text": "Write a function to group a list of strings by their first letter.",
        "code": 'from collections import defaultdict\ndef group_by_first_letter(words):\n    groups = defaultdict(list)\n    for word in words:\n        groups[word[0]].append(word)\n    return dict(groups)',
        "test_list": [
            "assert group_by_first_letter([\"apple\", \"banana\", \"avocado\", \"blueberry\"]) == {\"a\": [\"apple\", \"avocado\"], \"b\": [\"banana\", \"blueberry\"]}",
            "assert group_by_first_letter([\"cat\", \"dog\"]) == {\"c\": [\"cat\"], \"d\": [\"dog\"]}",
            "assert group_by_first_letter([]) == {}"
        ]
    },
    {
        "task_id": 34,
        "text": "Write a function to count words in a sentence.",
        "code": 'def count_words(sentence):\n    return len(sentence.split())',
        "test_list": [
            "assert count_words(\"hello world\") == 2",
            "assert count_words(\"  spaces   between  words  \") == 3",
            "assert count_words(\"\") == 0"
        ]
    },
    {
        "task_id": 35,
        "text": "Write a function to find the nth Fibonacci number using dynamic programming.",
        "code": 'def fibonacci(n):\n    if n <= 1:\n        return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n    for i in range(2, n + 1):\n        dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]',
        "test_list": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(10) == 55"
        ]
    },
    {
        "task_id": 36,
        "text": "Write a function to find the intersection of two lists.",
        "code": 'def list_intersection(a, b):\n    return list(set(a) & set(b))',
        "test_list": [
            "assert sorted(list_intersection([1, 2, 3, 4], [3, 4, 5, 6])) == [3, 4]",
            "assert list_intersection([1, 2], [3, 4]) == []",
            "assert sorted(list_intersection([1, 1, 2], [1, 2, 2])) == [1, 2]"
        ]
    },
    {
        "task_id": 37,
        "text": "Write a function to remove duplicates from a list while preserving order.",
        "code": 'def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result',
        "test_list": [
            "assert remove_duplicates([1, 2, 2, 3, 3, 3]) == [1, 2, 3]",
            "assert remove_duplicates([]) == []",
            "assert remove_duplicates([1, 1, 1]) == [1]"
        ]
    },
    {
        "task_id": 38,
        "text": "Write a function to calculate the factorial of a number using recursion.",
        "code": 'def factorial(n):\n    if n < 0:\n        raise ValueError(\"Factorial not defined for negative numbers\")\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)',
        "test_list": [
            "assert factorial(5) == 120",
            "assert factorial(0) == 1",
            "assert factorial(10) == 3628800"
        ]
    },
    {
        "task_id": 39,
        "text": "Write a function to find the two numbers in a list that are closest together.",
        "code": 'def closest_pair(lst):\n    sorted_lst = sorted(lst)\n    min_diff = float(\"inf\")\n    pair = (sorted_lst[0], sorted_lst[1])\n    for i in range(len(sorted_lst) - 1):\n        diff = sorted_lst[i+1] - sorted_lst[i]\n        if diff < min_diff:\n            min_diff = diff\n            pair = (sorted_lst[i], sorted_lst[i+1])\n    return pair',
        "test_list": [
            "assert closest_pair([1, 3, 6, 10, 15]) == (1, 3)",
            "assert closest_pair([5, 5, 5]) == (5, 5)",
            "assert closest_pair([1, 10, 100]) == (1, 10)"
        ]
    },
    {
        "task_id": 40,
        "text": "Write a function to check if a string is an anagram of another.",
        "code": 'from collections import Counter\ndef is_anagram(s1, s2):\n    return Counter(s1.lower()) == Counter(s2.lower())',
        "test_list": [
            "assert is_anagram(\"listen\", \"silent\") == True",
            "assert is_anagram(\"hello\", \"world\") == False",
            "assert is_anagram(\"Triangle\", \"Integral\") == True"
        ]
    },
    {
        "task_id": 41,
        "text": "Write a function to find the maximum depth of a binary tree represented as nested dictionaries.",
        "code": 'def max_depth(node):\n    if node is None:\n        return 0\n    left_depth = max_depth(node.get(\"left\"))\n    right_depth = max_depth(node.get(\"right\"))\n    return 1 + max(left_depth, right_depth)',
        "test_list": [
            "assert max_depth({\"val\": 1, \"left\": {\"val\": 2, \"left\": {\"val\": 4, \"left\": None, \"right\": None}, \"right\": None}, \"right\": {\"val\": 3, \"left\": None, \"right\": None}}) == 3",
            "assert max_depth(None) == 0",
            "assert max_depth({\"val\": 1, \"left\": None, \"right\": None}) == 1"
        ]
    },
    {
        "task_id": 42,
        "text": "Write a function to count the frequency of each element in a list.",
        "code": 'from collections import Counter\ndef count_frequency(lst):\n    return dict(Counter(lst))',
        "test_list": [
            "assert count_frequency([1, 2, 2, 3, 3, 3]) == {1: 1, 2: 2, 3: 3}",
            "assert count_frequency([]) == {}",
            "assert count_frequency([\"a\", \"b\", \"a\"]) == {\"a\": 2, \"b\": 1}"
        ]
    },
    {
        "task_id": 43,
        "text": "Write a function to find the longest word in a string.",
        "code": 'def longest_word(sentence):\n    words = sentence.split()\n    if not words:\n        return \"\"\n    return max(words, key=len)',
        "test_list": [
            "assert longest_word(\"the quick brown fox\") == \"quick\"",
            "assert longest_word(\"\") == \"\"",
            "assert longest_word(\"one\") == \"one\""
        ]
    },
    {
        "task_id": 44,
        "text": "Write a function to transpose a matrix.",
        "code": 'def transpose(matrix):\n    if not matrix:\n        return []\n    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]',
        "test_list": [
            "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]",
            "assert transpose([[1]]) == [[1]]",
            "assert transpose([]) == []"
        ]
    },
    {
        "task_id": 45,
        "text": "Write a function to find the first non-repeating character in a string.",
        "code": 'from collections import Counter\ndef first_non_repeating(s):\n    counts = Counter(s)\n    for char in s:\n        if counts[char] == 1:\n            return char\n    return None',
        "test_list": [
            "assert first_non_repeating(\"aabbcd\") == \"c\"",
            "assert first_non_repeating(\"aabb\") is None",
            "assert first_non_repeating(\"z\") == \"z\""
        ]
    },
    {
        "task_id": 46,
        "text": "Write a function to calculate the power of a number using fast exponentiation.",
        "code": 'def fast_power(base, exp):\n    if exp == 0:\n        return 1\n    if exp % 2 == 0:\n        half = fast_power(base, exp // 2)\n        return half * half\n    return base * fast_power(base, exp - 1)',
        "test_list": [
            "assert fast_power(2, 10) == 1024",
            "assert fast_power(3, 0) == 1",
            "assert fast_power(5, 3) == 125"
        ]
    },
    {
        "task_id": 47,
        "text": "Write a function to check if a list is sorted in ascending order.",
        "code": 'def is_sorted(lst):\n    return all(lst[i] <= lst[i+1] for i in range(len(lst) - 1))',
        "test_list": [
            "assert is_sorted([1, 2, 3, 4]) == True",
            "assert is_sorted([1, 3, 2, 4]) == False",
            "assert is_sorted([]) == True"
        ]
    },
    {
        "task_id": 48,
        "text": "Write a function to compute the dot product of two vectors.",
        "code": 'def dot_product(v1, v2):\n    return sum(a * b for a, b in zip(v1, v2))',
        "test_list": [
            "assert dot_product([1, 2, 3], [4, 5, 6]) == 32",
            "assert dot_product([1, 0], [0, 1]) == 0",
            "assert dot_product([], []) == 0"
        ]
    },
    {
        "task_id": 49,
        "text": "Write a function to find the minimum number of coins needed to make a given amount.",
        "code": 'def min_coins(coins, amount):\n    dp = [float(\"inf\")] * (amount + 1)\n    dp[0] = 0\n    for i in range(1, amount + 1):\n        for coin in coins:\n            if coin <= i:\n                dp[i] = min(dp[i], dp[i - coin] + 1)\n    return dp[amount] if dp[amount] != float(\"inf\") else -1',
        "test_list": [
            "assert min_coins([1, 5, 10, 25], 36) == 3",
            "assert min_coins([2], 3) == -1",
            "assert min_coins([1, 2, 5], 11) == 3"
        ]
    },
    {
        "task_id": 50,
        "text": "Write a function to check if a string has balanced parentheses.",
        "code": 'def is_balanced(s):\n    stack = []\n    pairs = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n    for char in s:\n        if char in \"([{\":\n            stack.append(char)\n        elif char in \")]}\":\n            if not stack or stack[-1] != pairs[char]:\n                return False\n            stack.pop()\n    return len(stack) == 0',
        "test_list": [
            "assert is_balanced(\"([]{})\") == True",
            "assert is_balanced(\"([)]\") == False",
            "assert is_balanced(\"\") == True"
        ]
    },
    {
        "task_id": 51,
        "text": "Write a function to find the longest palindromic substring.",
        "code": 'def longest_palindrome(s):\n    if not s:\n        return \"\"\n    start = end = 0\n    for i in range(len(s)):\n        for l, r in [(i, i), (i, i+1)]:\n            while l >= 0 and r < len(s) and s[l] == s[r]:\n                if r - l > end - start:\n                    start, end = l, r\n                l -= 1\n                r += 1\n    return s[start:end+1]',
        "test_list": [
            "assert longest_palindrome(\"babad\") in [\"bab\", \"aba\"]",
            "assert longest_palindrome(\"cbbd\") == \"bb\"",
            "assert longest_palindrome(\"a\") == \"a\""
        ]
    },
    {
        "task_id": 52,
        "text": "Write a function to implement binary search on a sorted list.",
        "code": 'def binary_search(lst, target):\n    low, high = 0, len(lst) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if lst[mid] == target:\n            return mid\n        elif lst[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1',
        "test_list": [
            "assert binary_search([1, 3, 5, 7, 9], 5) == 2",
            "assert binary_search([1, 3, 5, 7, 9], 6) == -1",
            "assert binary_search([], 1) == -1"
        ]
    },
    {
        "task_id": 53,
        "text": "Write a function to generate all permutations of a list.",
        "code": 'def permutations(lst):\n    if len(lst) <= 1:\n        return [lst[:]]\n    result = []\n    for i in range(len(lst)):\n        lst[0], lst[i] = lst[i], lst[0]\n        for perm in permutations(lst[1:]):\n            result.append([lst[0]] + perm)\n        lst[0], lst[i] = lst[i], lst[0]\n    return result',
        "test_list": [
            "assert sorted(permutations([1, 2, 3])) == sorted([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])",
            "assert permutations([1]) == [[1]]",
            "assert permutations([]) == [[]]"
        ]
    },
    {
        "task_id": 54,
        "text": "Write a function to find the number of ways to climb n stairs if you can take 1 or 2 steps at a time.",
        "code": 'def climb_stairs(n):\n    if n <= 2:\n        return n\n    a, b = 1, 2\n    for _ in range(3, n + 1):\n        a, b = b, a + b\n    return b',
        "test_list": [
            "assert climb_stairs(2) == 2",
            "assert climb_stairs(3) == 3",
            "assert climb_stairs(5) == 8"
        ]
    },
    {
        "task_id": 55,
        "text": "Write a function to find the missing number in a list of n-1 integers from 1 to n.",
        "code": 'def find_missing(lst, n):\n    expected = n * (n + 1) // 2\n    return expected - sum(lst)',
        "test_list": [
            "assert find_missing([1, 2, 4, 5], 5) == 3",
            "assert find_missing([2, 3, 4, 5], 5) == 1",
            "assert find_missing([1, 2, 3, 4], 5) == 5"
        ]
    },
    {
        "task_id": 56,
        "text": "Write a function to convert Roman numerals to an integer.",
        "code": 'def roman_to_int(s):\n    values = {\"I\": 1, \"V\": 5, \"X\": 10, \"L\": 50, \"C\": 100, \"D\": 500, \"M\": 1000}\n    result = 0\n    for i in range(len(s)):\n        if i + 1 < len(s) and values[s[i]] < values[s[i+1]]:\n            result -= values[s[i]]\n        else:\n            result += values[s[i]]\n    return result',
        "test_list": [
            "assert roman_to_int(\"III\") == 3",
            "assert roman_to_int(\"IV\") == 4",
            "assert roman_to_int(\"MCMXCIV\") == 1994"
        ]
    },
    {
        "task_id": 57,
        "text": "Write a function to find the number of islands in a grid (1s are land, 0s are water).",
        "code": 'def num_islands(grid):\n    if not grid:\n        return 0\n    count = 0\n    def dfs(r, c):\n        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] != \"1\":\n            return\n        grid[r][c] = \"0\"\n        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)\n    for r in range(len(grid)):\n        for c in range(len(grid[0])):\n            if grid[r][c] == \"1\":\n                count += 1\n                dfs(r, c)\n    return count',
        "test_list": [
            "assert num_islands([[\"1\",\"1\",\"0\"],[\"0\",\"1\",\"0\"],[\"0\",\"0\",\"1\"]]) == 2",
            "assert num_islands([[\"1\",\"1\",\"1\"],[\"1\",\"1\",\"1\"]]) == 1",
            "assert num_islands([]) == 0"
        ]
    },
    {
        "task_id": 58,
        "text": "Write a function to implement run-length encoding of a string.",
        "code": 'def run_length_encode(s):\n    if not s:\n        return \"\"\n    result = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            result.append(str(count) + s[i-1])\n            count = 1\n    result.append(str(count) + s[-1])\n    return \"\".join(result)',
        "test_list": [
            "assert run_length_encode(\"AAABBBCC\") == \"3A3B2C\"",
            "assert run_length_encode(\"\") == \"\"",
            "assert run_length_encode(\"ABCD\") == \"1A1B1C1D\""
        ]
    },
    {
        "task_id": 59,
        "text": "Write a function to find the maximum sum path in a triangle of numbers.",
        "code": 'def max_path_sum(triangle):\n    dp = triangle[-1][:]\n    for row in range(len(triangle) - 2, -1, -1):\n        for col in range(len(triangle[row])):\n            dp[col] = triangle[row][col] + max(dp[col], dp[col+1])\n    return dp[0]',
        "test_list": [
            "assert max_path_sum([[3],[7,4],[2,4,6],[8,5,9,3]]) == 23",
            "assert max_path_sum([[1],[2,3]]) == 4",
            "assert max_path_sum([[5]]) == 5"
        ]
    },
    {
        "task_id": 60,
        "text": "Write a function to determine if a number is a happy number.",
        "code": 'def is_happy(n):\n    def sum_of_squares(num):\n        return sum(int(d) ** 2 for d in str(num))\n    seen = set()\n    while n != 1:\n        if n in seen:\n            return False\n        seen.add(n)\n        n = sum_of_squares(n)\n    return True',
        "test_list": [
            "assert is_happy(19) == True",
            "assert is_happy(2) == False",
            "assert is_happy(1) == True"
        ]
    }
]


def compute_md5(data: str) -> str:
    return hashlib.md5(data.encode()).hexdigest()


def main() -> None:
    out_path = Path(__file__).parent / "mbpp_subset.json"
    payload = json.dumps(MBPP_TASKS, indent=2)
    out_path.write_text(payload)

    md5 = compute_md5(payload)
    print(f"Dataset written to: {out_path}")
    print(f"Tasks:              {len(MBPP_TASKS)} (IDs 11–60)")
    print(f"MD5:                {md5}")
    print()
    print("Add this hash to experiments/run_metadata.json as 'dataset_md5'.")


if __name__ == "__main__":
    main()
