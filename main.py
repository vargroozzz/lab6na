import numpy as np


def test_non_linear_system(xs):
    return np.array([xs[0] ** 2 / xs[1] ** 2.0 - np.cos(xs[1]) - 2, xs[0] ** 2 + xs[1] ** 2 - 6.0])


def test_non_linear_system_derivative(xs):
    x = xs[0]
    y = xs[1]
    result = np.zeros((len(xs), len(xs)))
    result[0][0] = 2.0 * x / (y ** 2.0)
    result[0][1] = np.sin(y) - 2.0 * (x ** 2.0) / (y ** 3.0)
    result[1][0] = 2.0 * x
    result[1][1] = 2.0 * y
    return result


def second_test_non_linear_system(xs):
    n = len(xs)
    result = np.zeros(n)
    for i in range(n):
        for j in range(n):
            result[i] += (xs[j] ** 3 - j ** 3 if i == j else xs[j] ** 2 - j ** 2)

    return result


def second_test_non_linear_system_derivative(xs):
    n = len(xs)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i][j] = (3 * xs[j] ** 2 if i == j else 2 * xs[j])
    return result


def relaxation_method(f, f_jacobian, x0):
    tau = 1.0 / np.linalg.norm(f_jacobian(x0))
    xk = x0
    while not np.allclose(f(xk), [0.0, 0.0]):
        xk = xk - tau * f(xk)

    return xk


def newton_method(f, f_jacobian, x0):
    jacobian_inversed = np.linalg.inv(f_jacobian(x0))
    xk = x0
    while not np.allclose(f(xk), [0.0, 0.0]):
        print(xk)
        xk = xk - np.dot(jacobian_inversed, f(xk))
        jacobian_inversed = np.linalg.inv(f_jacobian(xk))
    return xk


def modified_newton_method(f, f_jacobian, x0):
    jacobian_inversed = np.linalg.inv(f_jacobian(x0))
    xk = x0
    while not np.allclose(f(xk), [0.0, 0.0]):
        xk = xk - np.dot(jacobian_inversed, f(xk))
    return xk


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # relaxation_res = relaxation_method(
    #     test_non_linear_system,
    #     test_non_linear_system_derivative,
    #     np.array([1.0, 1.0]))
    # print(relaxation_res)
    # print(test_non_linear_system(relaxation_res))
    #
    # relaxation_res_2 = relaxation_method(
    #     second_test_non_linear_system,
    #     second_test_non_linear_system_derivative,
    #     np.array([1.0, 1.0]))
    # print(relaxation_res_2)
    # print(second_test_non_linear_system(relaxation_res_2))

    newton_res = newton_method(
        test_non_linear_system,
        test_non_linear_system_derivative,
        np.array([1.0, 1.0]))
    print(newton_res)
    print(test_non_linear_system(newton_res))

    # newton_res_2 = newton_method(
    #     second_test_non_linear_system,
    #     second_test_non_linear_system_derivative,
    #     np.array([1.0, 1.0]))
    # print(newton_res_2)
    # print(second_test_non_linear_system(newton_res_2))
    #
    # modified_newton_res = modified_newton_method(
    #     test_non_linear_system,
    #     test_non_linear_system_derivative,
    #     np.array([1.0, 1.0]))
    # print(modified_newton_res)
    # print(test_non_linear_system(modified_newton_res))
    #
    # modified_newton_res_2 = modified_newton_method(
    #     second_test_non_linear_system,
    #     second_test_non_linear_system_derivative,
    #     np.array([1.0, 1.0]))
    # print(modified_newton_res_2)
    # print(second_test_non_linear_system(modified_newton_res_2))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
