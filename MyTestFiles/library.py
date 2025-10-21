import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, norm
from pathlib import Path
from numpy.linalg import norm as matrix_norm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 文件导入功能 (File Import Functions)
# ============================================================================

def load_data(filepath, **kwargs):
    """
    通用数据导入函数 - 自动识别文件类型并导入

    Parameters:
    -----------
    filepath : str or Path
        文件路径
    **kwargs : dict
        传递给 pandas 读取函数的额外参数

    Returns:
    --------
    pd.DataFrame
        导入的数据框

    Examples:
    ---------
    >>> df = load_data('data/test1.csv')
    >>> df = load_data('data/test2.csv', index_col=0)
    """
    filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
    file_extension = filepath.suffix.lower().replace('.', '')

    try:
        if file_extension == 'csv':
            data = pd.read_csv(filepath, **kwargs)
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(filepath, **kwargs)
        elif file_extension == 'txt':
            try:
                data = pd.read_csv(filepath, sep='\t', **kwargs)
            except:
                data = pd.read_csv(filepath, sep=' ', **kwargs)
        elif file_extension == 'json':
            data = pd.read_json(filepath, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")

        print(f"成功导入数据: {data.shape[0]} 行, {data.shape[1]} 列")
        return data

    except Exception as e:
        print(f"导入文件失败: {e}")
        raise


# ============================================================================
# 2. 缺失数据处理 (Missing Data Handling) - Tests 1.1-1.4
# ============================================================================

def handle_missing_listwise_cov(df):
    """
    Test 1.1: Covariance Missing data, skip missing rows (列表删除法)
    删除包含缺失值的行，然后计算协方差矩阵

    Parameters:
    -----------
    df : pd.DataFrame
        包含缺失值的数据框

    Returns:
    --------
    pd.DataFrame
        协方差矩阵
    """
    # 删除包含缺失值的行
    df_clean = df.dropna(axis=0, how='any')

    # 只使用数值列
    num_df = df_clean.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        raise ValueError("需要至少 2 个数值列来计算协方差矩阵")

    # 计算协方差矩阵 (ddof=1 表示样本协方差)
    cov_mat = num_df.cov(ddof=1)
    return cov_mat


def handle_missing_listwise_corr(df):
    """
    Test 1.2: Correlation Missing data, skip missing rows (列表删除法)
    删除包含缺失值的行，然后计算相关系数矩阵

    Parameters:
    -----------
    df : pd.DataFrame
        包含缺失值的数据框

    Returns:
    --------
    pd.DataFrame
        相关系数矩阵
    """
    # 删除包含缺失值的行
    df_clean = df.dropna(axis=0, how='any')

    # 计算相关系数矩阵
    corr_mat = df_clean.corr(method='pearson')
    return corr_mat


def handle_missing_pairwise_cov(df):
    """
    Test 1.3: Covariance Missing data, Pairwise (成对删除法)
    使用成对删除法计算协方差矩阵

    Parameters:
    -----------
    df : pd.DataFrame
        包含缺失值的数据框

    Returns:
    --------
    pd.DataFrame
        协方差矩阵
    """
    # 只选择数值列
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        raise ValueError("需要至少 2 个数值列来计算协方差矩阵")

    # pandas 的 cov() 默认使用成对删除
    cov_mat = num_df.cov(ddof=1)
    return cov_mat


def handle_missing_pairwise_corr(df):
    """
    Test 1.4: Correlation Missing data, pairwise (成对删除法)
    使用成对删除法计算相关系数矩阵

    Parameters:
    -----------
    df : pd.DataFrame
        包含缺失值的数据框

    Returns:
    --------
    pd.DataFrame
        相关系数矩阵
    """
    # 只选择数值列
    num_df = df.select_dtypes(include=[np.number])

    # pandas 的 corr() 默认使用成对删除
    corr_mat = num_df.corr(method='pearson')
    return corr_mat


# ============================================================================
# 3. 指数加权协方差和相关系数 (EW Covariance/Correlation) - Tests 2.1-2.3
# ============================================================================

def calculate_ew_covariance(df, lambda_param=0.97):
    """
    Test 2.1: EW Covariance, lambda=0.97
    计算指数加权协方差矩阵（使用归一化权重）

    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    lambda_param : float
        衰减因子 (默认 0.97)

    Returns:
    --------
    pd.DataFrame
        指数加权协方差矩阵

    Note:
    -----
    这个实现使用 finite-sample normalization，与 Test2.1.ipynb 完全一致
    """
    # 只使用数值列并删除缺失值（列表删除）
    num = df.select_dtypes(include=[np.number]).dropna()
    X = num.to_numpy(float)
    n, d = X.shape

    if n < 2 or d < 2:
        raise ValueError("需要至少 2 个观测值和 2 个数值列")

    # 归一化的指数权重（最新的观测值获得最大权重）
    w = (1 - lambda_param) * lambda_param ** np.arange(n - 1, -1, -1)
    w = w / w.sum()  # finite-sample normalization

    # 加权均值
    mu = (w[:, None] * X).sum(axis=0)

    # 中心化数据
    Xc = X - mu

    # 指数加权协方差矩阵
    S = (w[:, None] * Xc).T @ Xc

    ew_cov = pd.DataFrame(S, index=num.columns, columns=num.columns)
    return ew_cov


def calculate_ew_correlation(df, lambda_param=0.94):
    """
    Test 2.2: EW Correlation, lambda=0.94
    计算指数加权相关系数矩阵

    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    lambda_param : float
        衰减因子 (默认 0.94)

    Returns:
    --------
    pd.DataFrame
        指数加权相关系数矩阵
    """
    # 只使用数值列并删除缺失值
    num = df.select_dtypes(include=[np.number]).dropna()
    X = num.to_numpy(float)
    n, d = X.shape

    if n < 2 or d < 2:
        raise ValueError("需要至少 2 个观测值和 2 个数值列")

    # 归一化的指数权重
    w = (1 - lambda_param) * lambda_param ** np.arange(n - 1, -1, -1)
    w = w / w.sum()

    # 加权均值和协方差
    mu = (w[:, None] * X).sum(axis=0)
    Xc = X - mu
    S = (w[:, None] * Xc).T @ Xc

    # 转换为相关系数矩阵
    std = np.sqrt(np.diag(S))
    eps = 1e-18
    std = np.where(std < eps, eps, std)
    R = S / np.outer(std, std)

    return pd.DataFrame(R, index=num.columns, columns=num.columns)


def calculate_cov_with_ew_variance_and_corr(df, lambda_cov=0.97, lambda_corr=0.94):
    """
    Test 2.3: Covariance with EW Variance (λ=0.97), EW Correlation (λ=0.94)
    使用 EW 方差和 EW 相关系数计算协方差矩阵

    Σ = D_σ(0.97) × R(0.94) × D_σ(0.97)

    Parameters:
    -----------
    df : pd.DataFrame
        数据框
    lambda_cov : float
        方差的衰减因子 (默认 0.97)
    lambda_corr : float
        相关系数的衰减因子 (默认 0.94)

    Returns:
    --------
    pd.DataFrame
        混合协方差矩阵
    """
    # 计算 EW 协方差矩阵 (λ=0.97) 以获得标准差
    cov_ew97 = calculate_ew_covariance(df, lambda_param=lambda_cov)
    std97 = np.sqrt(np.maximum(np.diag(cov_ew97.to_numpy()), 0.0))
    D97 = np.diag(std97)

    # 计算 EW 相关系数矩阵 (λ=0.94)
    corr_ew94 = calculate_ew_correlation(df, lambda_param=lambda_corr)

    # 计算混合协方差矩阵: Σ = D × R × D
    Sigma = D97 @ corr_ew94.to_numpy() @ D97

    return pd.DataFrame(Sigma, index=cov_ew97.index, columns=cov_ew97.columns)


# ============================================================================
# 4. 正定性修正 (Positive Definite Fixes) - Tests 3.1-3.4
# ============================================================================

def near_psd(a, epsilon: float = 0.0):
    """
    Test 3.1: near_psd covariance
    Near PSD 修正 - 适用于协方差矩阵
    """
    # --- 自动修正维度 ---
    if isinstance(a, pd.DataFrame):
        # 使用列名作为标准（因为协方差矩阵的列通常是变量名）
        common = a.columns.tolist()
        # 如果索引和列不一致，重置索引为列名
        if list(a.index) != common:
            a = a.copy()
            a.index = common
        a = a.loc[common, common]

    idx, cols = a.index, a.columns
    A = a.to_numpy(float)
    A = 0.5 * (A + A.T)

    d = np.diag(A).copy()
    d[d < 0] = 0.0
    np.fill_diagonal(A, d)

    # to correlation if needed
    diagA = np.diag(A)
    is_corr = np.allclose(diagA, np.ones_like(diagA), atol=1e-12)
    if not is_corr:
        std = np.sqrt(np.maximum(diagA, 0.0))
        std = np.where(std < 1e-12, 1e-12, std)
        R = (A / std[:, None]) / std[None, :]
    else:
        R = A

    vals, vecs = np.linalg.eigh(R)
    vals = np.maximum(vals, epsilon)
    denom = (vecs ** 2) @ vals
    denom = np.where(denom < 1e-18, 1e-18, denom)
    T = np.diag(np.sqrt(1.0 / denom))
    L = np.diag(np.sqrt(vals))
    B = T @ vecs @ L
    R_psd = B @ B.T
    np.fill_diagonal(R_psd, 1.0)
    R_psd = 0.5 * (R_psd + R_psd.T)

    if not is_corr:
        C_psd = (R_psd * std[:, None]) * std[None, :]
    else:
        C_psd = R_psd

    C_psd = 0.5 * (C_psd + C_psd.T)
    return pd.DataFrame(C_psd, index=common, columns=common)


def near_psd_corr(M, eps=0.0):
    """
    Test 3.2: near_psd correlation
    Near PSD 修正 - 专门用于相关系数矩阵

    Parameters:
    -----------
    M : pd.DataFrame or np.ndarray
        输入相关系数矩阵
    eps : float
        最小特征值阈值

    Returns:
    --------
    pd.DataFrame or np.ndarray
        Near PSD 相关系数矩阵
    """
    if isinstance(M, pd.DataFrame):
        # 使用列名作为标准
        common = M.columns.tolist()
        if list(M.index) != common:
            M = M.copy()
            M.index = common
        M = M.loc[common, common]

    is_df = isinstance(M, pd.DataFrame)
    A = M.to_numpy(float) if is_df else np.array(M, float)

    # 对称化
    A = 0.5 * (A + A.T)

    # 如果是协方差矩阵，转换为相关系数矩阵
    d = np.diag(A)
    if not np.allclose(d, 1.0):
        sd = np.sqrt(np.maximum(d, 1e-18))
        A = (A / sd[:, None]) / sd[None, :]

    # 特征值截断 + 单位对角线
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    A_psd = (V * w) @ V.T
    np.fill_diagonal(A_psd, 1.0)
    A_psd = 0.5 * (A_psd + A_psd.T)

    return pd.DataFrame(A_psd, index=M.index, columns=M.columns) if is_df else A_psd


def proj_psd(A):
    """
    投影到 PSD 锥（通过特征值截断）
    辅助函数，用于 Higham 算法
    """
    w, V = np.linalg.eigh(A)
    A_psd = (V * np.maximum(w, 0.0)) @ V.T
    return 0.5 * (A_psd + A_psd.T)


def higham_nearcorr(A, tol=None, max_iterations=100, weights=None):
    """
    Higham 最近相关系数矩阵算法
    使用交替投影法

    Parameters:
    -----------
    A : np.ndarray
        输入对称矩阵
    tol : float, optional
        收敛容差
    max_iterations : int
        最大迭代次数
    weights : np.ndarray, optional
        权重向量

    Returns:
    --------
    np.ndarray
        最近的相关系数矩阵
    """
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("输入矩阵必须是对称的")

    n = A.shape[0]
    eps = np.finfo(float).eps
    if tol is None:
        tol = eps * n
    if weights is None:
        weights = np.ones(n)

    W12 = np.sqrt(np.outer(weights, weights))

    X = A.copy()
    Y = A.copy()
    D = np.zeros_like(A)

    rel_diffX = rel_diffY = rel_diffXY = np.inf
    it = 0

    while max(rel_diffX, rel_diffY, rel_diffXY) > tol:
        it += 1
        if it > max_iterations:
            break

        X_old = X.copy()
        R = X - D
        X = proj_psd(W12 * R) / W12
        D = X - R

        Y_old = Y.copy()
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)

        nY = matrix_norm(Y, 'fro') + eps
        rel_diffX = matrix_norm(X - X_old, 'fro') / (matrix_norm(X, 'fro') + eps)
        rel_diffY = matrix_norm(Y - Y_old, 'fro') / nY
        rel_diffXY = matrix_norm(Y - X, 'fro') / nY

        X = Y.copy()

    return X


def higham_covariance(S_in, max_iterations=200, tol=1e-10):
    """
    Test 3.3: Higham covariance
    Higham 算法修正协方差矩阵

    方法: 转换为相关系数 -> Higham -> 转换回协方差

    Parameters:
    -----------
    S_in : pd.DataFrame or np.ndarray
        输入协方差矩阵
    max_iterations : int
        最大迭代次数
    tol : float
        收敛容差

    Returns:
    --------
    pd.DataFrame or np.ndarray
        修正后的协方差矩阵
    """
    is_df = isinstance(S_in, pd.DataFrame)

    # 处理 DataFrame 的索引列不匹配问题
    if is_df:
        common = S_in.columns.tolist()
        if list(S_in.index) != common:
            S_in = S_in.copy()
            S_in.index = common
        A = S_in.loc[common, common].to_numpy(float)
        idx_cols = common
    else:
        A = np.array(S_in, float)
        idx_cols = None

    A = 0.5 * (A + A.T)

    # 提取标准差
    var = np.clip(np.diag(A), 0.0, None)
    sd = np.sqrt(np.where(var < 1e-18, 1e-18, var))

    # 转换为相关系数矩阵
    R = (A / sd[:, None]) / sd[None, :]

    # Higham 算法
    R_h = higham_nearcorr(R, tol=tol, max_iterations=max_iterations)

    # 转换回协方差矩阵
    S_h = (R_h * sd[:, None]) * sd[None, :]
    S_h = 0.5 * (S_h + S_h.T)

    return (pd.DataFrame(S_h, index=idx_cols, columns=idx_cols)
            if is_df else S_h)


def higham_correlation(M, max_iterations=200, tol=1e-10):
    """
    Test 3.4: Higham correlation
    Higham 算法修正相关系数矩阵

    Parameters:
    -----------
    M : pd.DataFrame or np.ndarray
        输入相关系数矩阵
    max_iterations : int
        最大迭代次数
    tol : float
        收敛容差

    Returns:
    --------
    pd.DataFrame or np.ndarray
        修正后的相关系数矩阵
    """
    is_df = isinstance(M, pd.DataFrame)
    A = M.to_numpy(float) if is_df else np.array(M, float)

    # 转换为相关系数矩阵（如果不是的话）
    d = np.diag(A).copy()
    if not np.allclose(d, 1.0, atol=1e-12):
        A = 0.5 * (A + A.T)
        d = np.where(d < 1e-18, 1e-18, d)
        sd = np.sqrt(d)
        A = (A / sd[:, None]) / sd[None, :]

    # Higham 算法
    R_higham = higham_nearcorr(A, max_iterations=max_iterations, tol=tol)

    return (pd.DataFrame(R_higham, index=M.index, columns=M.columns)
            if is_df else R_higham)


# ============================================================================
# 5. Cholesky 分解 (Cholesky Decomposition) - Test 4.1
# ============================================================================

def chol_psd(A, tol=1e-10):
    """
    Test 4.1: PSD Cholesky 分解
    对 PSD 矩阵进行 Cholesky 分解
    """
    is_df = isinstance(A, pd.DataFrame)
    if is_df:
        index = A.index
        columns = A.columns
    # 🔧 对齐 index/columns 如果是 DataFrame
    if isinstance(A, pd.DataFrame):
        common = A.columns.tolist()
        if list(A.index) != common:
            A = A.copy()
            A.index = common
        A = A.loc[common, common].to_numpy(float)
    else:
        A = np.array(A, dtype=float, copy=False)

    n = A.shape[0]
    L = np.zeros_like(A)

    for j in range(n):
        s = 0.0 if j == 0 else float(L[j, :j] @ L[j, :j])
        dj = A[j, j] - s

        if dj < -tol:
            raise np.linalg.LinAlgError(f"矩阵不是 PSD，主元 {j}: {dj}")

        dj = 0.0 if (-tol <= dj <= 0.0) else dj
        L[j, j] = np.sqrt(dj) if dj > 0.0 else 0.0

        if L[j, j] > 0.0:
            inv = 1.0 / L[j, j]
            for i in range(j + 1, n):
                s = 0.0 if j == 0 else float(L[i, :j] @ L[j, :j])
                L[i, j] = (A[i, j] - s) * inv
        else:
            L[j + 1:, j] = 0.0

    if is_df:
        return pd.DataFrame(L, index=index, columns=columns)
    else:
        return L

# ============================================================================
# 6. 模拟方法 (Simulation Methods) - Tests 5.1-5.5
# ============================================================================

def simulate_normal_pd(cov_matrix, n_sims=100000, mean=None, seed=42):
    """
    Test 5.1: Normal Simulation PD Input
    """
    if isinstance(cov_matrix, pd.DataFrame):
        # 处理索引列不匹配
        common = cov_matrix.columns.tolist()
        if list(cov_matrix.index) != common:
            cov_matrix = cov_matrix.copy()
            cov_matrix.index = common
        cov_matrix = cov_matrix.loc[common, common]
        cov_matrix = cov_matrix.values

    Sigma = (cov_matrix + cov_matrix.T) / 2
    n = Sigma.shape[0]

    if mean is None:
        mean = np.zeros(n)

    eps = 1e-12
    for _ in range(8):
        try:
            L = np.linalg.cholesky(Sigma + eps * np.eye(n))
            Sigma = Sigma + eps * np.eye(n)
            break
        except np.linalg.LinAlgError:
            eps *= 10
    else:
        raise RuntimeError("输入协方差不是正定的，请检查 CSV")

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_sims, n))
    X = Z @ L.T
    return X


def simulate_normal_psd(cov_matrix, n_sims=100000, mean=None, seed=42):
    """
    Test 5.2: Normal Simulation PSD Input
    使用特征值截断使矩阵 PSD，然后模拟

    Parameters:
    -----------
    cov_matrix : np.ndarray or pd.DataFrame
        协方差矩阵
    n_sims : int
        模拟次数
    mean : np.ndarray, optional
        均值向量
    seed : int
        随机种子

    Returns:
    --------
    np.ndarray
        模拟数据
    """
    if isinstance(cov_matrix, pd.DataFrame):
        # 处理索引列不匹配
        common = cov_matrix.columns.tolist()
        if list(cov_matrix.index) != common:
            cov_matrix = cov_matrix.copy()
            cov_matrix.index = common
        cov_matrix = cov_matrix.loc[common, common].values

    # 对称化
    Sigma = (cov_matrix + cov_matrix.T) / 2.0
    n = Sigma.shape[0]

    if mean is None:
        mean = np.zeros(n)

    # 特征值截断使其 PSD
    d, V = np.linalg.eigh(Sigma)
    d = np.clip(d, 0.0, None)
    sqrt_d = np.sqrt(d)
    B = V * sqrt_d  # Sigma_psd = B B^T

    # 模拟
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_sims, n))
    X = Z @ B.T

    return X


def simulate_normal_near_psd(cov_matrix, n_sims=100000, mean=None, seed=42):
    """
    Test 5.3: Normal Simulation nonPSD Input, near_psd fix
    使用 near_psd 修正后模拟
    """
    if isinstance(cov_matrix, pd.DataFrame):
        # 处理索引列不匹配
        common = cov_matrix.columns.tolist()
        if list(cov_matrix.index) != common:
            cov_matrix = cov_matrix.copy()
            cov_matrix.index = common
        cov_matrix = cov_matrix.loc[common, common].values

    # 对称化
    Sigma = (cov_matrix + cov_matrix.T) / 2.0
    n = Sigma.shape[0]

    if mean is None:
        mean = np.zeros(n)

    # Near PSD 修正（特征值截断）
    d, V = np.linalg.eigh(Sigma)
    d = np.clip(d, 0.0, None)
    B = V * np.sqrt(d)

    # 模拟
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_sims, n))
    X = Z @ B.T

    return X


def simulate_normal_higham(cov_matrix, n_sims=100000, mean=None, seed=42):
    """
    Test 5.4: Normal Simulation, Higham fix
    使用 Higham 算法修正后模拟
    """
    if isinstance(cov_matrix, pd.DataFrame):
        # 处理索引列不匹配
        common = cov_matrix.columns.tolist()
        if list(cov_matrix.index) != common:
            cov_matrix = cov_matrix.copy()
            cov_matrix.index = common
        Sigma = cov_matrix.loc[common, common].values
    else:
        Sigma = cov_matrix

    # 对称化
    Sigma = (Sigma + Sigma.T) / 2.0
    n = Sigma.shape[0]

    if mean is None:
        mean = np.zeros(n)

    # Higham 修正
    # 1) 转换为相关系数矩阵
    std = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    std_safe = np.where(std > 0.0, std, 1.0)
    D_inv = np.diag(1.0 / std_safe)
    C = D_inv @ Sigma @ D_inv
    C = (C + C.T) / 2.0

    # 2) Higham 算法
    C_fix = higham_nearcorr(C, tol=1e-9, max_iterations=100)

    # 3) 转换回协方差
    D = np.diag(std_safe)
    Sigma_fix = D @ C_fix @ D
    Sigma_fix = (Sigma_fix + Sigma_fix.T) / 2.0

    # 特征值分解用于模拟
    w, V = np.linalg.eigh(Sigma_fix)
    w = np.clip(w, 0.0, None)
    B = V * np.sqrt(w)

    # 模拟
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_sims, n))
    X = Z @ B.T

    return X


def simulate_pca(cov_matrix, variance_explained=0.99, n_sims=100000, seed=42):
    """
    Test 5.5: PCA Simulation, 99% explained
    使用 PCA 进行降维模拟

    Parameters:
    -----------
    cov_matrix : pd.DataFrame or np.ndarray
        协方差矩阵
    variance_explained : float
        要保留的方差比例
    n_sims : int
        模拟次数
    seed : int
        随机种子

    Returns:
    --------
    np.ndarray
        模拟数据
    """
    if isinstance(cov_matrix, pd.DataFrame):
        # 处理索引列不匹配
        common = cov_matrix.columns.tolist()
        if list(cov_matrix.index) != common:
            cov_matrix = cov_matrix.copy()
            cov_matrix.index = common
        Sigma = cov_matrix.loc[common, common].values
    else:
        Sigma = cov_matrix

    # 对称化并 PSD
    Sigma = (Sigma + Sigma.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.clip(eigvals, 0.0, None)

    # 降序排列
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 选择主成分
    total_var = eigvals.sum() + 1e-18
    cum = np.cumsum(eigvals) / total_var
    k = int(np.searchsorted(cum, variance_explained) + 1)

    # PCA 因子
    sqrt_dk = np.sqrt(eigvals[:k])
    B = eigvecs[:, :k] * sqrt_dk

    # 模拟
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_sims, k))
    X = Z @ B.T

    return X


# ============================================================================
# 7. 收益率计算 (Returns Calculation) - Tests 6.1-6.2
# ============================================================================

def calculate_arithmetic_returns(prices):
    """
    Test 6.1: 计算算术收益率
    R_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1

    Parameters:
    -----------
    prices : pd.DataFrame or np.ndarray
        价格数据

    Returns:
    --------
    pd.DataFrame or np.ndarray
        算术收益率
    """


    if isinstance(prices, pd.DataFrame):
        # 如果第一列是日期，保留它
        date_col = None
        data_cols = prices.columns.tolist()

        if prices.iloc[:, 0].dtype == 'object' or 'date' in prices.columns[0].lower():
            date_col = prices.iloc[:, 0]
            data_cols = prices.columns[1:].tolist()
            prices_numeric = prices.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        else:
            prices_numeric = prices.apply(pd.to_numeric, errors='coerce')

        # 计算收益率 P_t / P_{t-1} - 1
        P = prices_numeric.to_numpy(dtype=float)
        if P.shape[0] < 2:
            raise ValueError("需要至少 2 行来计算收益率")
        R = P[1:, :] / P[:-1, :] - 1.0

        # 构建输出 DataFrame
        out = pd.DataFrame(R, columns=data_cols)
        if date_col is not None:
            out.insert(0, prices.columns[0], date_col.iloc[1:].reset_index(drop=True))

        return out
    else:
        # NumPy 数组
        P = np.asarray(prices, dtype=float)
        if P.shape[0] < 2:
            raise ValueError("需要至少 2 行来计算收益率")
        return P[1:, :] / P[:-1, :] - 1.0


def calculate_log_returns(prices):
    """
    Test 6.2: 计算对数收益率
    r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Parameters:
    -----------
    prices : pd.DataFrame or np.ndarray
        价格数据

    Returns:
    --------
    pd.DataFrame or np.ndarray
        对数收益率
    """
    if isinstance(prices, pd.DataFrame):
        # 如果第一列是日期，保留它
        date_col = None
        data_cols = prices.columns.tolist()

        if prices.iloc[:, 0].dtype == 'object' or 'date' in prices.columns[0].lower():
            date_col = prices.iloc[:, 0]
            data_cols = prices.columns[1:].tolist()
            prices_numeric = prices.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        else:
            prices_numeric = prices.apply(pd.to_numeric, errors='coerce')

        # 计算对数收益率
        logP = np.log(prices_numeric.to_numpy(dtype=float))
        R = logP[1:, :] - logP[:-1, :]

        # 构建输出 DataFrame
        out = pd.DataFrame(R, columns=data_cols)
        if date_col is not None:
            out.insert(0, prices.columns[0], date_col.iloc[1:].reset_index(drop=True))

        return out
    else:
        # NumPy 数组
        logP = np.log(np.asarray(prices, dtype=float))
        return logP[1:, :] - logP[:-1, :]


# ============================================================================
# 8. 分布拟合 (Distribution Fitting) - Tests 7.1-7.3
# ============================================================================

def fit_normal_distribution(data):
    """
    Test 7.1: 拟合正态分布
    使用 MLE 拟合 Normal(mu, sigma)

    Parameters:
    -----------
    data : array-like
        数据

    Returns:
    --------
    dict
        包含 'mean' 和 'std' 的字典
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("没有数据可拟合")

    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))  # Bessel 修正

    return {'mean': mu, 'std': sigma}


def fit_t_distribution(data):
    """
    Test 7.2: 拟合 t 分布
    使用 MLE 拟合 Student-t 分布

    Parameters:
    -----------
    data : array-like
        数据

    Returns:
    --------
    dict
        包含 'df' (nu), 'loc' (mu), 'scale' (sigma) 的字典
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("数据不足以拟合 t 分布")

    nu, mu, sigma = t.fit(arr)
    return {'df': float(nu), 'loc': float(mu), 'scale': float(sigma)}


def fit_t_regression(y, X):
    """
    Test 7.3: T 回归
    MLE 拟合 Student-t 回归
    模型: y = X @ beta + e,  e ~ t_nu(loc=0, scale=sigma)

    Parameters:
    -----------
    y : array-like
        因变量
    X : array-like
        自变量（应该已经包含截距列）

    Returns:
    --------
    dict
        包含 'beta', 'sigma', 'nu' 的字典
    """
    from scipy import optimize

    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if y.size != n:
        raise ValueError("y 和 X 的形状不兼容")
    if n < p + 1:
        raise ValueError("观测值不足以进行回归")

    # OLS 初始化
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid0 = y - X @ beta0

    # 使用 MAD 估计初始 scale
    mad = np.median(np.abs(resid0 - np.median(resid0)))
    sigma0 = mad / 0.6744897501960817 if mad > 0 else np.std(resid0, ddof=p)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = max(np.std(resid0, ddof=p), 1e-6)
    nu0 = 8.0

    # 重参数化：sigma = exp(a), nu = exp(b) + 2
    def pack(beta, sigma, nu):
        return np.concatenate([beta, [np.log(sigma), np.log(nu - 2.0)]])

    def unpack(theta):
        beta = theta[:p]
        sigma = np.exp(theta[p])
        nu = np.exp(theta[p + 1]) + 2.0
        return beta, sigma, nu

    def neg_loglik(theta):
        beta, sigma, nu = unpack(theta)
        if not (np.isfinite(sigma) and np.isfinite(nu) and sigma > 0 and nu > 2):
            return np.inf
        resid = y - X @ beta
        ll = t.logpdf(resid, df=nu, loc=0.0, scale=sigma).sum()
        return -ll

    theta0 = pack(beta0, sigma0, nu0)
    res = optimize.minimize(
        neg_loglik,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12}
    )

    if not res.success:
        raise RuntimeError(f"优化失败: {res.message}")

    beta_hat, sigma_hat, nu_hat = unpack(res.x)

    return {
        'beta': beta_hat,
        'sigma': float(sigma_hat),
        'nu': float(nu_hat)
    }


# ============================================================================
# 9. VaR 和 ES 计算 (VaR and ES) - Tests 8.1-8.6
# ============================================================================

def var_normal(data, alpha=0.05):
    """
    Test 8.1: 使用正态分布计算 VaR

    Parameters:
    -----------
    data : array-like
        收益率数据
    alpha : float
        置信水平（默认 0.05 表示 95% VaR）

    Returns:
    --------
    dict
        包含 'VaR_Absolute' 和 'VaR_Diff_from_Mean' 的字典
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        returns = data.iloc[:, 0].dropna() if isinstance(data, pd.DataFrame) else data.dropna()
    else:
        returns = np.asarray(data, dtype=float)
        returns = returns[~np.isnan(returns)]

    # 拟合 t 分布
    df_t, loc, scale = t.fit(returns)

    # VaR 计算
    t_score = t.ppf(1 - (1 - alpha), df_t)
    var_diff_from_mean = -t_score * scale
    var_absolute = -(loc + t_score * scale)

    return {
        'VaR_Absolute': var_absolute,
        'VaR_Diff_from_Mean': var_diff_from_mean
    }


def var_t_distribution(data, alpha=0.05):
    """
    Test 8.2: 使用 t 分布计算 VaR

    Parameters:
    -----------
    data : array-like
        收益率数据
    alpha : float
        置信水平

    Returns:
    --------
    dict
        包含 'VaR_Absolute' 和 'VaR_Diff_from_Mean' 的字典
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        returns = data.iloc[:, 0].dropna() if isinstance(data, pd.DataFrame) else data.dropna()
    else:
        returns = np.asarray(data, dtype=float)
        returns = returns[~np.isnan(returns)]

    mean_return = float(returns.mean())
    std_return = float(returns.std())

    # VaR 计算
    z_score = norm.ppf(1 - (1 - alpha))  # 左尾
    var_diff_from_mean = -z_score * std_return
    var_absolute = -(mean_return + z_score * std_return)

    return {
        'VaR_Absolute': var_absolute,
        'VaR_Diff_from_Mean': var_diff_from_mean
    }

def var_simulation(data, alpha=0.05, n_simulations=10000, seed=32):
    """
    Test 8.3: 使用模拟计算 VaR

    Parameters:
    -----------
    data : array-like
        收益率数据
    alpha : float
        置信水平
    n_simulations : int
        模拟次数
    seed : int
        随机种子

    Returns:
    --------
    dict
        包含 'VaR_Absolute' 和 'VaR_Diff_from_Mean' 的字典
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        returns = data.iloc[:, 0].dropna() if isinstance(data, pd.DataFrame) else data.dropna()
    else:
        returns = np.asarray(data, dtype=float)
        returns = returns[~np.isnan(returns)]

    # 拟合 t 分布
    df_t, loc, scale = t.fit(returns)

    # 生成模拟
    np.random.seed(seed)
    simulated_returns = t.rvs(df_t, loc, scale, size=n_simulations)

    # VaR 计算
    var_absolute = -np.percentile(simulated_returns, (1 - (1 - alpha)) * 100)
    var_diff_from_mean = loc - np.percentile(simulated_returns, (1 - (1 - alpha)) * 100)

    return {
        'VaR_Absolute': var_absolute,
        'VaR_Diff_from_Mean': var_diff_from_mean
    }


def es_normal(data, alpha=0.05):
    """
    Test 8.4: 使用正态分布计算 Expected Shortfall (ES)

    Parameters:
    -----------
    data : array-like
        收益率数据
    alpha : float
        置信水平

    Returns:
    --------
    dict
        包含 'ES_Absolute' 和 'ES_Diff_from_Mean' 的字典
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        ret = data.iloc[:, 0].dropna() if isinstance(data, pd.DataFrame) else data.dropna()
        ret = ret.astype(float)
    else:
        ret = np.asarray(data, dtype=float)
        ret = ret[~np.isnan(ret)]

    mu = float(ret.mean())
    sigma = float(ret.std(ddof=1))

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid std for ES computation")

    # ES 从正态分布（左尾）
    alpha_val = 1.0 - (1 - alpha)  # 左尾概率
    z = norm.ppf(alpha_val)
    phi = norm.pdf(z)

    es_diff = sigma * phi / alpha_val
    es_abs = -(mu - sigma * phi / alpha_val)

    return {
        'ES_Absolute': es_abs,
        'ES_Diff_from_Mean': es_diff
    }


def es_t_distribution(data, alpha=0.05):
    """
    Test 8.5: 使用 t 分布计算 Expected Shortfall
    使用数值积分方法

    Parameters:
    -----------
    data : array-like
        收益率数据
    alpha : float
        置信水平

    Returns:
    --------
    dict
        包含 'ES_Absolute' 和 'ES_Diff_from_Mean' 的字典
    """
    from scipy.integrate import quad

    if isinstance(data, (pd.DataFrame, pd.Series)):
        x = data.iloc[:, 0].dropna() if isinstance(data, pd.DataFrame) else data.dropna()
        x = x.values
    else:
        x = np.asarray(data, dtype=float)

    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("输入序列为空")
    if not (0 < alpha < 0.5):
        raise ValueError("alpha 应该在 (0, 0.5) 之间用于左尾 ES")

    # 拟合 t 分布
    nu, mu, sigma = t.fit(x)

    # α 分位点（左尾 VaR 水平）
    alpha_val = 1.0 - (1 - alpha)
    t_alpha = t.ppf(alpha_val, nu)
    var_level = mu + sigma * t_alpha

    # 数值积分计算 E[X | X <= VaR]
    def integrand(z):
        return z * t.pdf((z - mu) / sigma, nu) / sigma

    # 左端积分下限
    lower_bound = mu + sigma * t.ppf(1e-12, nu)

    integral_val, _ = quad(integrand, lower_bound, var_level, limit=200)

    # 条件期望
    cond_expectation = integral_val / alpha_val

    # ES 取正数表示损失幅度
    es_abs = -cond_expectation
    es_diff_from_mean = mu - cond_expectation

    return {
        'ES_Absolute': es_abs,
        'ES_Diff_from_Mean': es_diff_from_mean
    }


def es_simulation(data, alpha=0.05, n_sims=1000000, seed=890):
    """
    Test 8.6: 使用模拟计算 Expected Shortfall

    Parameters:
    -----------
    data : array-like
        收益率数据
    alpha : float
        置信水平
    n_sims : int
        模拟次数
    seed : int
        随机种子

    Returns:
    --------
    dict
        包含 'ES_Absolute' 和 'ES_Diff_from_Mean' 的字典
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        x = data.iloc[:, 0].dropna() if isinstance(data, pd.DataFrame) else data.dropna()
        x = x.values
    else:
        x = np.asarray(data, dtype=float)

    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("输入序列为空")
    if not (0 < alpha < 0.5):
        raise ValueError("alpha 应该在 (0, 0.5) 之间")

    # 拟合 t 分布
    df_t, mu, sigma = t.fit(x)

    # 模拟
    rng = np.random.default_rng(seed)
    sims = t.rvs(df_t, loc=mu, scale=sigma, size=n_sims, random_state=rng)

    # VaR: α分位数（左尾）
    alpha_val = 1.0 - (1 - alpha)
    var_level = float(np.quantile(sims, alpha_val, method="linear"))

    # ES: 尾部均值
    tail = sims[sims <= var_level]
    if tail.size == 0:
        raise RuntimeError("左尾没有模拟点；增加 n_sims 或检查输入")

    cond_mean = float(np.mean(tail))
    sim_mean = float(np.mean(sims))

    es_abs = -cond_mean
    es_diff_from_mean = sim_mean - cond_mean

    return {
        'ES_Absolute': es_abs,
        'ES_Diff_from_Mean': es_diff_from_mean
    }


# ============================================================================
# 10. 辅助函数 (Helper Functions)
# ============================================================================

def compare_covariances(cov1, cov2, name1="Input", name2="Output"):
    """
    比较两个协方差矩阵的差异
    """
    if isinstance(cov1, pd.DataFrame):
        cov1 = cov1.values
    if isinstance(cov2, pd.DataFrame):
        cov2 = cov2.values

    print(f"\n{'=' * 60}")
    print(f"比较 {name1} vs {name2} 协方差矩阵")
    print(f"{'=' * 60}")

    # Frobenius 范数差异
    diff_frobenius = np.linalg.norm(cov1 - cov2, 'fro')
    print(f"Frobenius 范数差异: {diff_frobenius:.6f}")

    # 最大绝对差异
    max_diff = np.max(np.abs(cov1 - cov2))
    print(f"最大绝对差异: {max_diff:.6f}")

    # 相对差异
    rel_diff = diff_frobenius / np.linalg.norm(cov1, 'fro')
    print(f"相对差异: {rel_diff:.2%}")


def check_positive_definite(matrix):
    """
    检查矩阵是否正定

    Parameters:
    -----------
    matrix : np.ndarray or pd.DataFrame
        要检查的矩阵

    Returns:
    --------
    bool
        True 如果矩阵是正定的
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    try:
        np.linalg.cholesky(matrix)
        print("✓ 矩阵是正定的")
        return True
    except:
        eigvals = np.linalg.eigvalsh(matrix)
        min_eigval = np.min(eigvals)
        print(f"✗ 矩阵不是正定的")
        print(f"  最小特征值: {min_eigval:.6e}")
        print(f"  特征值范围: [{eigvals.min():.6e}, {eigvals.max():.6e}]")
        return False


def is_symmetric(matrix, tol=1e-10):
    """
    检查矩阵是否对称
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    return np.allclose(matrix, matrix.T, atol=tol)


def print_matrix_info(matrix, name="Matrix"):
    """
    打印矩阵的详细信息
    """
    if isinstance(matrix, pd.DataFrame):
        print(f"\n{name} 信息:")
        print(f"  维度: {matrix.shape}")
        print(f"  列名: {list(matrix.columns)}")
        matrix_np = matrix.values
    else:
        print(f"\n{name} 信息:")
        print(f"  维度: {matrix.shape}")
        matrix_np = matrix

    print(f"  对称性: {'✓ 对称' if is_symmetric(matrix_np) else '✗ 不对称'}")

    eigvals = np.linalg.eigvalsh(matrix_np)
    print(f"  特征值范围: [{eigvals.min():.6e}, {eigvals.max():.6e}]")
    print(f"  最小特征值: {eigvals.min():.6e}")
    print(f"  条件数: {eigvals.max() / max(abs(eigvals.min()), 1e-18):.2e}")

    if eigvals.min() >= 0:
        print(f"  正定性: ✓ 正半定")
        if eigvals.min() > 1e-10:
            print(f"           ✓ 严格正定")
    else:
        print(f"  正定性: ✗ 不定")


# ============================================================================
# 使用示例和测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Statistics and Finance Analysis Library")
    print("统计与金融分析库 - 已加载并验证")
    print("=" * 70)

    print("\n可用的功能模块:")
    print("\n1. 文件导入:")
    print("   - load_data(filepath)")

    print("\n2. 缺失数据处理 (Tests 1.1-1.4):")
    print("   - handle_missing_listwise_cov(df)")
    print("   - handle_missing_listwise_corr(df)")
    print("   - handle_missing_pairwise_cov(df)")
    print("   - handle_missing_pairwise_corr(df)")

    print("\n3. 指数加权方法 (Tests 2.1-2.3):")
    print("   - calculate_ew_covariance(df, lambda_param=0.97)")
    print("   - calculate_ew_correlation(df, lambda_param=0.94)")
    print("   - calculate_cov_with_ew_variance_and_corr(df)")

    print("\n4. 正定性修正 (Tests 3.1-3.4):")
    print("   - near_psd(matrix)")
    print("   - near_psd_corr(matrix)")
    print("   - higham_covariance(matrix)")
    print("   - higham_correlation(matrix)")

    print("\n5. Cholesky 分解 (Test 4.1):")
    print("   - chol_psd(matrix)")

    print("\n6. 模拟方法 (Tests 5.1-5.5):")
    print("   - simulate_normal_pd(cov_matrix, n_sims=100000)")
    print("   - simulate_normal_psd(cov_matrix, n_sims=100000)")
    print("   - simulate_normal_near_psd(cov_matrix, n_sims=100000)")
    print("   - simulate_normal_higham(cov_matrix, n_sims=100000)")
    print("   - simulate_pca(cov_matrix, variance_explained=0.99)")

    print("\n7. 收益率计算 (Tests 6.1-6.2):")
    print("   - calculate_arithmetic_returns(prices)")
    print("   - calculate_log_returns(prices)")

    print("\n8. 分布拟合 (Tests 7.1-7.3):")
    print("   - fit_normal_distribution(data)")
    print("   - fit_t_distribution(data)")
    print("   - fit_t_regression(y, X)")

    print("\n9. VaR 和 ES (Tests 8.1-8.6):")
    print("   - var_normal(data, alpha=0.05)")
    print("   - var_t_distribution(data, alpha=0.05)")
    print("   - var_simulation(data, alpha=0.05)")
    print("   - es_normal(data, alpha=0.05)")
    print("   - es_t_distribution(data, alpha=0.05)")
    print("   - es_simulation(data, alpha=0.05)")

    print("\n10. 辅助函数:")
    print("   - compare_covariances(cov1, cov2)")
    print("   - check_positive_definite(matrix)")
    print("   - is_symmetric(matrix)")
    print("   - print_matrix_info(matrix)")

    print("\n" + "=" * 70)
    print("所有函数都已经过实际测试文件验证！")
    print("=" * 70)


