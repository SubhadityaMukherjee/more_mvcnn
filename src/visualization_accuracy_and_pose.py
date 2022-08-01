"""
Matplotlib scripts to view results visually
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 21})


fig = plt.figure(
    figsize=(25, 20),
)
ax = fig.add_subplot(221)
# m10 m40
data = [[0.8586, 0.4384], [0.8794, 0.4833]]  # top1  # top5
X = np.arange(2)
plt.grid(axis="y")
limits = [-0.5, 1.5, 0.3, 1]
plt.axis(limits)


ax.bar(X - 0.125, data[0], color="r", width=0.25, label="Top-1 accuracy")
ax.bar(X + 0.125, data[1], color="blue", width=0.25, label="Top-5 accuracy")

ax.text(0 - 0.28, data[0][0] + 0.02, str(data[0][0]), color="red")
ax.text(0 + 0.25, data[1][0] - 0.05, str(data[1][0]), color="blue")

ax.text(1 - 0.28, data[0][1] + 0.02, str(data[0][1]), color="red")
ax.text(1 + 0.25, data[1][1] - 0.05, str(data[1][1]), color="blue")


plt.xticks([0, 1], ["ModelNet10", "ModelNet40"])
plt.yticks(np.arange(0.3, 1.1, 0.1))

plt.title("Our model with VGG-16 backbone")
# plt.xlabel("Methods")
plt.ylabel("Pose Estimation Accuracy")
plt.legend()

ax = fig.add_subplot(222)
data = [[0.8205, 0.4785], [0.8706, 0.5150]]
X = np.arange(2)
plt.grid(axis="y")
limits = [-0.5, 1.5, 0.3, 1]
plt.axis(limits)

plt.yticks(np.arange(0.3, 1.1, 0.1))

ax.bar(X - 0.125, data[0], color="r", width=0.25, label="Top-1 accuracy")
ax.bar(X + 0.125, data[1], color="blue", width=0.25, label="Top-5 accuracy")

ax.text(0 - 0.28, data[0][0] + 0.02, str(data[0][0]), color="red")
ax.text(0 + 0.25, data[1][0] - 0.05, str(data[1][0]), color="blue")

ax.text(1 - 0.28, data[0][1] + 0.02, str(data[0][1]), color="red")
ax.text(1 + 0.25, data[1][1] - 0.05, str(data[1][1]), color="blue")

plt.xticks([0, 1], ["ModelNet10", "ModelNet40"])
plt.title("Our model with MobileNet backbone")
# plt.xlabel("Methods")
# plt.legend()


# ///////////// RECOGNITION //////////////////


ax = fig.add_subplot(223)
data = [[0.9652, 0.7427], [0.9825, 0.8105]]
X = np.arange(2)
plt.grid(axis="y")
limits = [-0.5, 1.5, 0.3, 1]
plt.axis(limits)


ax.bar(X - 0.125, data[0], color="r", width=0.25, label="Top-1 accuracy")
ax.bar(X + 0.125, data[1], color="blue", width=0.25, label="Top-5 accuracy")
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

ax.text(0 - 0.28, data[0][0] + 0.02, str(data[0][0]), color="red")
ax.text(0 + 0.25, data[1][0] - 0.05, str(data[1][0]), color="blue")

ax.text(1 - 0.28, data[0][1] + 0.02, str(data[0][1]), color="red")
ax.text(1 + 0.25, data[1][1] - 0.05, str(data[1][1]), color="blue")


plt.xticks([0, 1], ["ModelNet10", "ModelNet40"])
plt.yticks(np.arange(0.3, 1.1, 0.1))

plt.ylabel("Object Recognition Accuracy")
# plt.legend()


# ////////////////////////////////////////////////////////////////

ax = fig.add_subplot(224)
data = [[0.9510, 0.8649], [0.9826, 0.9652]]

X = np.arange(2)
plt.grid(axis="y")
limits = [-0.5, 1.5, 0.3, 1]
plt.axis(limits)

plt.yticks(np.arange(0.3, 1.1, 0.1))

ax.bar(X - 0.125, data[0], color="r", width=0.25, label="Top-1 accuracy")
ax.bar(X + 0.125, data[1], color="blue", width=0.25, label="Top-5 accuracy")

ax.text(0 - 0.28, data[0][0] + 0.02, str(data[0][0]), color="red")
ax.text(0 + 0.25, data[1][0] - 0.05, str(data[1][0]), color="blue")

ax.text(1 - 0.28, data[0][1] + 0.02, str(data[0][1]), color="red")
ax.text(1 + 0.25, data[1][1] - 0.05, str(data[1][1]), color="blue")

plt.xticks([0, 1], ["ModelNet10", "ModelNet40"])

# plt.show()
plt.savefig("results/final_comparison.png")
