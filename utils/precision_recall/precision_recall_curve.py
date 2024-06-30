import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json
# font_path="/disks/sata4/huangyue/hy_DSH/DeepHash-hw/fonts/ttf/SimHei.ttf"
# # 加载字体
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
# plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# Precision Recall Curve data
pr_data = {

    "alpha=0.2": "/disks/sata4/huangyue/hy_DSH/DeepHash-hw/log/alpha/DSH_cifar10_24_0.2.json",
    "alpha=0.1": "/disks/sata4/huangyue/hy_DSH/DeepHash-hw/log/alexnet/cifar/DSH_cifar10_12.json",
    "alpha=0.05": "/disks/sata4/huangyue/hy_DSH/DeepHash-hw/log/alpha/DSH_cifar10_24_0.05.json",
    "alpha=0.15": "/disks/sata4/huangyue/hy_DSH/DeepHash-hw/log/alpha/DSH_cifar10_24_0.15.json",
    
    # "new":"/disks/sata4/huangyue/hy_DSH/DeepHash-hw/log/ResNet/pre_renewed/DSH_cifar10_48_new.json",
    # "pretrain":"/disks/sata4/huangyue/hy_DSH/DeepHash-hw/log/ResNet/pre_renewed/DSH_cifar10_48_pre.json",
}
N = 150
# N = -1
for key in pr_data:
    path = pr_data[key]
    pr_data[key] = json.load(open(path))


# markers = "DdsPvo*xH1234h"
markers = ".........................."
method2marker = {}
i = 0
for method in pr_data:
    method2marker[method] = markers[i]
    i += 1

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.suptitle('Different bits with alpha=0.1')  # Adding the suptitle

for method in pr_data:
    P, R,draw_range = pr_data[method]["P"],pr_data[method]["R"],pr_data[method]["index"]
    print(len(P))
    print(len(R))
    plt.plot(R, P, linestyle="-", marker=method2marker[method], label=method)
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.subplot(132)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, R, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('recall')
plt.legend()

plt.subplot(133)
for method in pr_data:
    P, R,draw_range = pr_data[method]["P"][:N],pr_data[method]["R"][:N],pr_data[method]["index"][:N]
    plt.plot(draw_range, P, linestyle="-", marker=method2marker[method], label=method)
plt.xlim(0, max(draw_range))
plt.grid(True)
plt.xlabel('The number of retrieved samples')
plt.ylabel('precision')
plt.legend()
plt.savefig("pr_bits_cifar10.png")
plt.show()
