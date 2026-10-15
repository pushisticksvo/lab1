"""Заготовки задач на NumPy."""

import numpy as np
import matplotlib.pyplot as plt
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    n = matrices[0].shape[0]
    res = np.zeros((n, 1), dtype=np.result_type(*matrices, *vectors))
    for i in range(len(matrices)):
        v = np.asarray(vectors[i]).reshape(n, 1)
        res = res + matrices[i] @ v
    return res


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(np.asarray(matrix) > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    res = []
    for row in np.asarray(matrix):
        res.append(np.unique(row).tolist())
    return res


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    res = []
    for col in np.asarray(matrix).T:
        res.append(np.unique(col).tolist())
    return res


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    mat = rng.normal(mean, std, size=(rows, columns))

    r_means = np.mean(mat, axis=1)
    c_means = np.mean(mat, axis=0)
    r_vars = np.var(mat, axis=1)
    c_vars = np.var(mat, axis=0)

    return MatrixStatistics(
        matrix=mat,
        row_means=r_means,
        column_means=c_means,
        row_variances=r_vars,
        column_variances=c_vars,
    )


def plot_histograms(mat: np.ndarray):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for i in range(mat.shape[0]):
        plt.hist(mat[i], alpha=0.5, label=f"Строка {i}")
    plt.title("Строки")
    plt.legend()

    plt.subplot(1, 2, 2)
    for j in range(mat.shape[1]):
        plt.hist(mat[:, j], alpha=0.5, label=f"Столбец {j}")
    plt.title("Столбцы")
    plt.legend()
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    board = np.full((rows, columns), first, dtype=type(first + second))
    board[::2, 1::2] = second
    board[1::2, ::2] = second
    return board


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    img[:] = background_color

    y1 = max(0, (image_height - height) // 2)
    y2 = min(image_height, y1 + height)
    x1 = max(0, (image_width - width) // 2)
    x2 = min(image_width, x1 + width)

    img[y1:y2, x1:x2] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    img[:] = background_color

    x0 = image_width / 2
    y0 = image_height / 2
    y, x = np.indices((image_height, image_width))

    mask = ((x - x0) ** 2) / (semi_axis_x ** 2) + ((y - y0) ** 2) / (semi_axis_y ** 2) <= 1
    img[mask] = shape_color
    return img


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    arr = np.asarray(values, dtype=float)

    m = float(np.mean(arr))
    v = float(np.var(arr))
    s = float(np.std(arr))

    max_idx = []
    min_idx = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            max_idx.append(i)
        elif arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            min_idx.append(i)

    ma = []
    for i in range(len(arr) - window + 1):
        ma.append(np.mean(arr[i : i + window]))
    ma = np.array(ma, dtype=float)

    return TimeSeriesStatistics(
        mean=m,
        variance=v,
        std=s,
        local_maxima_indices=max_idx,
        local_minima_indices=min_idx,
        moving_average=ma,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    arr = np.asarray(labels, dtype=int)
    if arr.size == 0:
        cols = 0 if class_count is None else class_count
        return np.zeros((0, cols), dtype=int)

    if class_count is None:
        cols = int(arr.max()) + 1
    else:
        cols = class_count

    res = np.zeros((len(arr), cols), dtype=int)
    for i in range(len(arr)):
        res[i, arr[i]] = 1
    return res


def tests():
    a1 = np.eye(2, dtype=int)
    a2 = np.array([[1, 2], [3, 4]])
    v1 = np.array([[1], [1]])
    v2 = np.array([[2], [0]])
    res = sum_prod(MatrixVectorBatchInput([a1, a2], [v1, v2]))
    assert res.shape == (2, 1)
    assert np.array_equal(res, np.array([[3], [7]]))
    assert np.array_equal(
        sum_prod(MatrixVectorBatchInput([np.array([[5]])], [np.array([[3]])])),
        np.array([[15]]),
    )

    mat = np.array([[1, 5], [3, 8]])
    assert np.array_equal(binarize(BinarizeInput(mat, 3)), np.array([[0, 1], [0, 1]]))
    assert np.array_equal(binarize(BinarizeInput(mat, 10)), np.array([[0, 0], [0, 0]]))
    assert np.array_equal(binarize(BinarizeInput(np.array([[0.2, 0.8]]))), np.array([[0, 1]]))

    m = np.array([[1, 1, 2], [3, 2, 3]])
    assert unique_rows(MatrixInput(m)) == [[1, 2], [2, 3]]
    assert unique_columns(MatrixInput(m)) == [[1, 3], [1, 2], [2, 3]]
    assert unique_rows(MatrixInput(np.array([[5]]))) == [[5]]
    assert unique_columns(MatrixInput(np.array([[5]]))) == [[5]]

    stats = matrix_statistics(RandomMatrixInput(3, 4, 0.0, 1.0, 42))
    assert stats.matrix.shape == (3, 4)
    assert len(stats.row_means) == 3
    assert len(stats.column_means) == 4
    stats_const = matrix_statistics(RandomMatrixInput(2, 2, 5.0, 0.0, 1))
    assert np.allclose(stats_const.row_means, [5.0, 5.0])
    assert np.allclose(stats_const.row_variances, [0.0, 0.0])

    assert np.array_equal(
        chess(ChessInput(3, 4, 0, 1)),
        np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]),
    )
    assert np.array_equal(chess(ChessInput(1, 1, 7, 9)), np.array([[7]]))
    assert np.array_equal(chess(ChessInput(1, 4, 2, 5)), np.array([[2, 5, 2, 5]]))

    img = draw_rectangle(RectangleInput(4, 6, 10, 12, (255, 0, 0), (0, 0, 0)))
    assert img.shape == (10, 12, 3)
    assert np.array_equal(img[5, 6], [255, 0, 0])
    assert np.array_equal(img[0, 0], [0, 0, 0])

    img = draw_ellipse(EllipseInput(3, 2, 10, 12, (0, 255, 0), (0, 0, 0)))
    assert img.shape == (10, 12, 3)
    assert np.array_equal(img[5, 6], [0, 255, 0])
    assert np.array_equal(img[0, 0], [0, 0, 0])

    ts = analyze_time_series(TimeSeriesInput(np.array([1, 3, 2, 5, 1, 4]), 2))
    assert list(ts.local_maxima_indices) == [1, 3]
    assert list(ts.local_minima_indices) == [2, 4]
    assert np.allclose(ts.moving_average, [2.0, 2.5, 3.5, 3.0, 2.5])

    ts_empty = analyze_time_series(TimeSeriesInput(np.array([1, 2, 3, 4]), 4))
    assert list(ts_empty.local_maxima_indices) == []
    assert list(ts_empty.local_minima_indices) == []
    assert np.allclose(ts_empty.moving_average, [2.5])

    assert np.array_equal(
        one_hot(OneHotInput(np.array([0, 2, 3, 0]))),
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]),
    )
    assert np.array_equal(
        one_hot(OneHotInput(np.array([0, 1]), 4)),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
    )
    assert np.array_equal(one_hot(OneHotInput(np.array([0]))), np.array([[1]]))

    print("тесты пройдены")


tests()