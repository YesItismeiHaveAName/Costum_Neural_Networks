def reduce_array_by_one_dimension(array):
    if array[0] is not list:
        raise Exception("Cannot reduce the Dimensions of a 1-dimensional List.")
    result = []
    for elem in array:
        result += elem
    return result

def normalize_value(value, max_value):
    if max_value < value:
        raise ValueError("Value is greater than maximum Value defined for normalization!")
    return value / max_value