import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((y.size, K), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y


def softmax(Z: np.ndarray) -> np.ndarray:
    # Softmax estável numericamente
    Zs = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Zs)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def loss_ce(P: np.ndarray, Y: np.ndarray, eps: float = 1e-15) -> float:
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))


def predict_class(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    P = softmax(X @ W)
    return np.argmax(P, axis=1)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(y_pred == y_true))


def grad_softmax(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Retorna:
      - gradiente dW (d x K)
      - loss
    """
    n = X.shape[0]
    P = softmax(X @ W)
    dW = (X.T @ (P - Y)) / n
    return dW, loss_ce(P, Y)


def fit_gd(X: np.ndarray, Y: np.ndarray, lr: float = 0.1, epochs: int = 2000) -> np.ndarray:
    d = X.shape[1]
    K = Y.shape[1]
    W = np.zeros((d, K), dtype=float)

    for t in range(epochs):
        dW, L = grad_softmax(X, Y, W)
        W -= lr * dW

        # log simples a cada 200 iterações
        if (t + 1) % 200 == 0:
            print(f"[GD] epoch={t+1:4d} loss={L:.6f}")

    return W


def build_hessian(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    n, d = X.shape
    K = P.shape[1]
    H = np.zeros((d * K, d * K), dtype=float)

    for k in range(K):
        pk = P[:, k]
        for l in range(K):
            pl = P[:, l]
            if k == l:
                r = pk * (1.0 - pk)
            else:
                r = -(pk * pl)

            Xw = X * r[:, None]          # (n x d)
            block = (Xw.T @ X) / n       # (d x d)

            H[k*d:(k+1)*d, l*d:(l+1)*d] = block

    return H

# Treino por Newton
def fit_newton(
    X: np.ndarray,
    Y: np.ndarray,
    iters: int = 30,
    damping: float = 1e-2,
) -> np.ndarray:
    n, d = X.shape
    K = Y.shape[1]
    W = np.zeros((d, K), dtype=float)

    for t in range(iters):
        P = softmax(X @ W)
        L = loss_ce(P, Y)

        # grad: d x K -> vetor dK
        dW = (X.T @ (P - Y)) / n
        g = dW.reshape(-1)

        H = build_hessian(X, P)
        H += damping * np.eye(d * K)

        # resolve H * step = g
        step = np.linalg.solve(H, g)

        W -= step.reshape(d, K)

        print(f"[Newton] iter={t+1:2d} loss={L:.6f} |step|={np.linalg.norm(step):.6e}")

    return W


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    K = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # intercepto: adiciona coluna de 1s
    X_train_ = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test_ = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    Y_train = one_hot(y_train, K)

    print("\n=== Treino: Gradiente Descendente (Softmax) ===")
    W_gd = fit_gd(X_train_, Y_train, lr=0.1, epochs=2000)
    pred_gd = predict_class(X_test_, W_gd)
    print("Acurácia (GD):", accuracy(pred_gd, y_test))

    print("\n=== Treino: Newton (Softmax) ===")
    W_newton = fit_newton(X_train_, Y_train, iters=20, damping=1e-6)
    pred_newton = predict_class(X_test_, W_newton)
    print("Acurácia (Newton):", accuracy(pred_newton, y_test))


if __name__ == "__main__":
    main()
