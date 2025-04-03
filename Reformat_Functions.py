def reduce_array_by_one_dimension(array):
    if array[0] is not list:
        raise Exception("Cannot reduce the Dimensions of a 1-dimensional List.")
    result = []
    for elem in array:
        result += elem
    return result


print(reduce_array_by_one_dimension([1, 2, 3]))

