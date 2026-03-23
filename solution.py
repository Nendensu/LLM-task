import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data(path='data.csv'):
    X, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row['x1']), float(row['x2'])])
            y.append(int(row['label']))
    return np.array(X), np.array(y, dtype=float)


def load_weights(path='model_weights.npz'):
    d = np.load(path)
    return d['W1'], d['b1'], d['W2'], d['b2']


def sigmoid(z):
    # TODO: your code
    pass

def relu(z):
    # TODO: your code
    pass

def relu_grad(z):
    # TODO: your code
    pass


def forward(X, weights):
    # TODO: your code
    pass

def bce_loss(y_hat, y):
    # TODO: your code
    pass


def compute_gradients(X, y, weights):
    # TODO: your code
    pass


def gradient_check(X, y, weights, eps=1e-5):
    # TODO: your code
    pass


def input_gradient(x, y_true, weights):
    # TODO: your code
    pass


def pgd_attack(X, y, weights, lr=0.05, steps=200):
    # TODO: your code
    pass


def plot_decision_boundary(X, y, weights, deltas, success, correct_mask,
                           save_path='adversarial_plot.png'):
    x_min, x_max = X[:,0].min()-0.3, X[:,0].max()+0.3
    y_min, y_max = X[:,1].min()-0.3, X[:,1].max()+0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz, _ = forward(grid, weights)
    zz = zz.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, title, show_adv in zip(axes,
            ['Start + decision boundary',
             'Adversarial examples'],
            [False, True]):

        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1],
                    colors=['#aec6e8','#f4a97a'], alpha=0.35)
        ax.contour(xx, yy, zz, levels=[0.5],
                   colors='#333333', linewidths=1.2)

        ax.scatter(X[y==0, 0], X[y==0, 1], c='#3578b5', s=14,
                   alpha=0.6, label='Class 0', zorder=3)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='#d6604d', s=14,
                   alpha=0.6, label='Class 1', zorder=3)

        if show_adv:
            adv_idx = np.where(success)[0]
            X_adv = X[adv_idx] + deltas[adv_idx]
            norms = np.linalg.norm(deltas[adv_idx], axis=1)

            sc = ax.scatter(X_adv[:, 0], X_adv[:, 1],
                            c=norms, cmap='YlOrRd',
                            s=30, zorder=5, edgecolors='k',
                            linewidths=0.3, label='Adversarial')
            plt.colorbar(sc, ax=ax, label='‖δ‖₂')

            for j in adv_idx[:60]:
                ax.annotate('', xy=X[j]+deltas[j], xytext=X[j],
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            lw=0.5, alpha=0.5))
            ax.set_title(f'{title}\n'
                         f'{success.sum()} successfull attacks '
                         f'{correct_mask.sum()} correct prerdictions\n'
                         f'Median value ‖δ‖₂ = {np.median(norms):.3f}')
        else:
            ax.set_title(title)

        ax.legend(fontsize=8, markerscale=1.5)
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")



if __name__ == '__main__':
    print("=" * 60)
    print("Loading weights")
    print("=" * 60)
    X, y = load_data('data.csv')
    weights = load_weights('model_weights.npz')
    W1, b1, W2, b2 = weights
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"W1: {W1.shape}, b1: {b1.shape}, W2: {W2.shape}, b2: {b2.shape}")

    print("\n" + "=" * 60)
    print("Verifying forward pass")
    print("=" * 60)
    y_hat, _ = forward(X, weights)
    ref = np.load('reference_predictions.npy')
    max_diff = np.abs(y_hat - ref).max()
    print(f"Max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, "ERROR: bad forward pass!"
    print("Forward pass is ok (< 1e-5)")

    acc = ((y_hat > 0.5) == y.astype(bool)).mean()
    print(f"Dataset accuracy: {acc:.4f}")

    print("\n" + "=" * 60)
    print("Gradient check")
    print("=" * 60)
    idx = np.random.choice(len(X), 50, replace=False)
    gc_results = gradient_check(X[idx], y[idx], list(weights))

    all_passed = True
    for name, res in gc_results.items():
        status = 'ok' if res['passed'] else 'error'
        print(f"  {status} {name:3s}  max_abs_diff={res['max_abs_diff']:.2e}"
              f"  max_rel_diff={res['max_rel_diff']:.2e}"
              f"  {'PASS' if res['passed'] else 'FAIL'}")
        if not res['passed']:
            all_passed = False
    print("Gradients verified" if all_passed
          else "Error in gradients!")

    print("\n" + "=" * 60)
    print("Adversarial examples")
    print("=" * 60)
    deltas, success, correct_mask = pgd_attack(X, y, weights,
                                                lr=0.05, steps=300)

    norms = np.linalg.norm(deltas[success], axis=1)
    print(f"Correct predictions: {correct_mask.sum()}")
    print(f"Successfull attacks: {success.sum()}")
    print(f"‖δ‖₂ — min: {norms.min():.4f}")
    print(f"‖δ‖₂ — median: {np.median(norms):.4f}")
    print(f"‖δ‖₂ — max: {norms.max():.4f}")

    plot_decision_boundary(X, y, weights, deltas, success, correct_mask)
