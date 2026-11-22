# MRNet ResNet18-CBAM 论文复现

## 可变切片数批处理

MRNet 和 KAMRNet DICOM 的训练入口都支持 `--batch-size`，默认是 1。批次内的切片一起通过共享主干，再按检查和视角边界分别池化。Baseline 和 SIP 都支持该参数，评估和续训也可指定批大小。

```powershell
python train.py --mode sip --batch-size 2 --output runs/resnet_cbam_sip_b2
python train_kamrnet_dicom.py --mode sip --batch-size 2 --output runs/kamrnet_dicom_sip_b2
```

显存占用随每批总切片数变化。增大批大小也会改变训练时的 BatchNorm 统计和每轮优化器更新次数。

## SIP 原型与视角注意力（MRNet）

```powershell
python train.py --mode sip --prototype-samples 10 --prototype-lambda 0.01
python train.py --evaluate runs/resnet_cbam_sip/best.pt
```

默认 `--mode baseline`。SIP 模式用 one-hot 初始化可学习标签输入，分别编码高斯均值和正方差，训练时按 SIP 式 (9) 将多次重参数化采样取平均。评估使用高斯期望以获得确定性结果。三个视角各自的 512 维特征作为注意力的 Key 和 Value，三个原型作为 Query，各标签经自己的线性头输出 logit。原型聚合使用视角特征的均值，与正标签原型的均值计算平方 L2 距离。SIP 日志分别记录总损失、BCE 和原型聚合损失，默认输出到 `runs/resnet_cbam_sip`。

在项目目录安装依赖并启动训练。请使用与本机 GPU 匹配的 PyTorch 环境。默认初始化会通过 torchvision 加载 ImageNet 权重。

```powershell
python -m pip install -r requirements.txt
python train.py
python train.py --evaluate runs/resnet_cbam/best.pt
```

默认读取 `D:/dataset/MRNet/MRNet_Multilabel_Metadata.xlsx`。可以用 `--data-root` 和 `--metadata` 覆盖位置。工作簿中的 `train` 用于训练，`test` 对应本地公开的 120 例验证数据。模型按 Abnormal、ACL、Meniscus 的顺序输出三个独立的 logits。以患者为批次，将轴位、矢状位、冠状位沿切片维拼接，保留原始切片数。ResNet18 提取切片特征，CBAM 位于 layer4 的两个残差块之间。对空间维做平均池化，再按视图分别沿切片维做最大池化，最后拼接三个视图的特征进行分类。

默认训练 100 轮，使用 Adam、初始学习率 1e-5 和按标签正负样本频率加权的 BCE。程序保存每轮逐标签 ROC AUC 和宏平均 AUC 到 `metrics.jsonl`，并保存最佳与最后一轮权重。`--no-pretrained` 可用于离线调试。

论文未提供调度器类型、增强幅度、CBAM 精确插入位置和 dropout 设置。本实现采用余弦退火、±10° 旋转、每轴最多 5% 平移、0.5 概率水平翻转、ImageNet 通道归一化，以及 16 倍通道压缩。两个消融变体使用单独的输出目录。

```powershell
python train.py --no-cbam --output runs/resnet
python train.py --dropout 0.5 --output runs/resnet_cbam_dropout
```

Dropout 对照沿用已有的 `runs/resnet_cbam/metrics.jsonl` 基线。自己运行上方的 dropout 命令后，用下方命令比较两组共同完成的轮次。脚本报告各组最高验证宏平均 ROC AUC、最低加权 BCE，以及最后一个共同轮次的两项指标。

```powershell
python compare_dropout.py
```

论文报告的验证宏平均 AUC 为 0.925（CBAM）、0.916（无 CBAM）和 0.924（CBAM + dropout）。这些数值是论文的实验结果。
