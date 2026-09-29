"""Draw the system architecture diagram -> reports/figures/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

COL = {"data": "#dbeafe", "ml": "#dcfce7", "ci": "#fef3c7", "k8s": "#ede9fe",
       "mon": "#fee2e2", "user": "#f3f4f6"}
fig, ax = plt.subplots(figsize=(13, 7.6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.6)
ax.axis("off")


def box(x, y, w, h, text, kind, size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=COL[kind], ec="#374151", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")


def group(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec="#9ca3af", lw=1.2, ls="--"))
    ax.text(x + 0.12, y + h - 0.22, title, fontsize=9.5, fontweight="bold", color="#374151")


def arrow(x1, y1, x2, y2, label="", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.3,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.13, label, fontsize=7.5,
                ha="center", color="#4b5563")


ax.text(6.5, 7.3, "Heart Disease Risk - MLOps Architecture", ha="center",
        fontsize=14, fontweight="bold")

# Development / training
group(0.2, 4.1, 6.4, 2.9, "1. Development & training (local / CI)")
box(0.4, 5.3, 1.5, 1.0, "UCI dataset\n(download script)", "data")
box(2.2, 5.3, 1.5, 1.0, "Cleaning + EDA\ndata_processing.py\neda.py", "data")
box(4.0, 5.3, 2.4, 1.0, "Training\nsklearn Pipeline\nLogReg / RandomForest\nGridSearchCV (5-fold)", "ml")
box(4.0, 4.3, 2.4, 0.7, "MLflow tracking\nparams · metrics · plots", "ml")
box(0.4, 4.3, 3.3, 0.7, "models/model.joblib + metadata.json\n(best model by CV ROC-AUC)", "ml", bold=True)
arrow(1.9, 5.8, 2.2, 5.8)
arrow(3.7, 5.8, 4.0, 5.8)
arrow(5.2, 5.3, 5.2, 5.0)
arrow(4.0, 4.65, 3.7, 4.65)

# CI/CD
group(6.9, 4.1, 5.9, 2.9, "2. CI/CD - GitHub Actions")
box(7.1, 5.3, 1.6, 1.0, "push / PR\nto main", "ci")
box(8.95, 5.3, 1.75, 1.0, "flake8 lint\npytest (22 tests)\ntrain + artifacts", "ci")
box(10.95, 5.3, 1.65, 1.0, "docker build\nrun container\nsmoke test /predict", "ci")
box(7.1, 4.3, 5.5, 0.7, "Pipeline fails on lint / test / training / container errors", "ci", size=8.5)
arrow(8.7, 5.8, 8.95, 5.8)
arrow(10.7, 5.8, 10.95, 5.8)
arrow(6.4, 5.8, 7.1, 5.8, "git push")

# Kubernetes
group(0.2, 0.2, 9.4, 3.6, "3. Kubernetes (Minikube) - runtime")
box(0.4, 1.9, 2.0, 1.2, "Docker image 1.1.0\nFastAPI + model baked in\n(COPY models/)", "k8s")
box(2.9, 2.55, 2.0, 0.7, "Pod 1 - /predict\n/health /metrics", "k8s", size=8.5)
box(2.9, 1.65, 2.0, 0.7, "Pod 2 - /predict\n/health /metrics", "k8s", size=8.5)
box(5.5, 1.9, 1.9, 1.2, "Service\nLoadBalancer :80\n-> pods :8000", "k8s")
box(2.9, 0.4, 2.0, 0.85, "Prometheus\nscrapes /metrics every 10s", "mon")
box(5.5, 0.4, 1.9, 0.85, "Grafana\ndashboard (7 panels)", "mon")
arrow(2.4, 2.65, 2.9, 2.9)
arrow(2.4, 2.35, 2.9, 2.0)
arrow(4.9, 2.9, 5.5, 2.7)
arrow(4.9, 2.0, 5.5, 2.3)
arrow(3.9, 1.65, 3.9, 1.25)
arrow(4.9, 0.82, 5.5, 0.82, "PromQL")

# Users
group(9.9, 0.2, 2.9, 3.6, "4. Consumers")
box(10.1, 2.2, 2.5, 1.0, "Client / curl / app\nPOST /predict (JSON)\n-> prediction + confidence", "user", size=8.5)
box(10.1, 0.4, 2.5, 1.0, "Engineer\nGrafana dashboard\nkubectl logs", "user", size=8.5)
arrow(10.1, 2.7, 7.4, 2.7, "HTTP")
arrow(7.4, 0.82, 10.1, 0.9, "view")

# cross-link: packaged model is baked into the image
arrow(2.0, 4.3, 1.5, 3.1)

plt.tight_layout()
plt.savefig("reports/figures/architecture.png", dpi=170)
