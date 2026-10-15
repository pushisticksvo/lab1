"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    abc = "aeiouAEIOU"
    cnt = 0
    for ch in text:
        if ch in abc:
            cnt += 1
    return cnt


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        p = 1
        for d in str(number):
            p *= int(d)
        number = p
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    s = 0
    for i in range(len(predicted)):
        s += (predicted[i] - expected[i]) ** 2
    return s / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    res = ""
    d = 2
    while d * d <= number:
        if number % d == 0:
            k = 0
            while number % d == 0:
                number //= d
                k += 1
            if k == 1:
                res += f"({d})"
            else:
                res += f"({d}**{k})"
        d += 1
    if number > 1:
        res += f"({number})"
    return res


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    s = 0
    k = 1
    while s < cube_count:
        s += k ** 2
        if s == cube_count:
            return k
        k += 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    half = (len(s) - 1) // 2
    if half == 0:
        return True
    left = 0
    right = 0
    for ch in s[:half]:
        left += int(ch)
    for ch in s[len(s) - half:]:
        right += int(ch)
    return left == right


def tests():
    assert count_vowels(TextInput("HelloWorld")) == 3
    assert count_vowels(TextInput("AaEeIiOoUu")) == 10
    assert count_vowels(TextInput("bcdfg")) == 0
    assert count_vowels(TextInput("")) == 0

    assert has_unique_characters(TextInput("abcdef")) == True
    assert has_unique_characters(TextInput("hello")) == False
    assert has_unique_characters(TextInput("a")) == True
    assert has_unique_characters(TextInput("")) == True

    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(7)) == 3
    assert count_one_bits(PositiveIntegerInput(16)) == 1
    assert count_one_bits(PositiveIntegerInput(255)) == 8

    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1

    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([1, 2, 3], [4, 5, 6])) == 9.0
    assert mse(VectorPairInput([5], [2])) == 9.0

    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(100)) == "(2**2)(5**2)"

    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(15)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"

    assert is_balanced_number(PositiveIntegerInput(1234006)) == True
    assert is_balanced_number(TextInput("7") if False else PositiveIntegerInput(7)) == True
    assert is_balanced_number(PositiveIntegerInput(123456)) == False
    assert is_balanced_number(PositiveIntegerInput(95)) == True
    assert is_balanced_number(PositiveIntegerInput(10201)) == True

    print("тесты пройдены")


tests()