# 自定义数据集

在本节中，您将了解如何使用自定义数据集训练和测试预训练模型。

基本步骤如下：

1. 准备数据
2. 准备配置文件
3. 在自定义数据集上训练，测试和推理模型

## 数据准备

理想情况下我们可以重新组织自定义的原始数据并将标注格式转换成 KITTI 风格。但是，考虑到对于自定义数据集而言，一些校准文件和 KITTI 格式的 3D 标注难以获得，我们在文档中介绍基本的数据格式。

### 基本数据格式

#### 点云格式

目前，我们只支持 '.bin' 格式的点云用于训练和推理。在训练自己的数据集之前，需要将其他格式的点云文件转换成 '.bin' 文件。常见的点云数据格式包括 `.pcd` 和 `.las`，我们列出一些开源工具作为参考。

1. pcd 转换成 bin：https://github.com/leofansq/Tools_RosBag2KITTI
2. las 转换成 bin：常见的转换流程为 las -> pcd -> bin，las -> pcd 的转换可以用该[工具](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor)。

#### 标签格式

最基本的信息：每个场景的 3D 边界框和类别标签应该包含在标注 `.txt` 文件中。每一行代表特定场景的一个 3D 框，如下所示：

```python
# format: [x, y, z, dx, dy, dz, yaw, category_name]
1.23 1.42 0.23 3.96 1.65 1.55 1.56 Car
3.51 2.15 0.42 1.05 0.87 1.86 1.23 Pedestrian
...
```

**注意**：对于自定义数据集评估目前我们只支持 KITTI 评估方法。

3D 框应存储在统一的 3D 坐标系中。

#### 校准格式

对于每个激光雷达收集的点云数据，通常会进行融合并转换到特定的激光雷达坐标系。因此，校准信息文件中应该包含每个相机的内参矩阵和激光雷达到每个相机的外参转换矩阵，并保存在校准 `.txt` 文件中，其中 `Px` 表示 `camera_x` 的内参矩阵，`lidar2camx` 表示 `lidar` 到 `camera_x` 的外参转换矩阵。

```
P0
P1
P2
P3
P4
...
lidar2cam0
lidar2cam1
lidar2cam2
lidar2cam3
lidar2cam4
...
```

### 原始数据格式

#### 基于激光雷达的 3D 检测

基于激光雷达的 3D 目标检测原始数据通常组织成如下格式，其中 `ImageSets` 包含划分文件，指明哪些文件数据属于训练/验证集，`points` 包含存储成 `.bin` 格式的点云数据，`labels` 包含 3D 检测的标签文件。

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

## 基于视觉的 3D 检测

基于视觉的 3D 目标检测原始数据通常组织成如下格式，其中 `ImageSets` 包含划分文件，指明哪些文件数据属于训练/验证集，`images` 包含来自不同相机的图像，例如 `camera_x` 获得的图像应放在 `images/images_x` 下，`calibs` 包含校准信息文件，其中存储了每个相机的内参矩阵，`labels` 包含 3D 检测的标签文件。

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── calibs
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
│   │   ├── images
│   │   │   ├── images_0
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   ├── ...
│   │   │   ├── images_1
│   │   │   ├── images_2
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### 多模态 3D 检测

多模态 3D 目标检测原始数据通常组织成如下格式。不同于基于视觉的 3D 目标检测，`calibs` 里的校准信息文件存储了每个相机的内参矩阵和外参矩阵。

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── calibs
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── images
│   │   │   ├── images_0
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   ├── ...
│   │   │   ├── images_1
│   │   │   ├── images_2
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── 000000.txt
│   │   │   ├── 000001.txt
│   │   │   ├── ...
```

#### 基于激光雷达的 3D 语义分割

基于激光雷达的 3D 语义分割原始数据通常组织成如下格式，其中 `ImageSets` 包含划分文件，指明哪些文件数据属于训练/验证集，`points` 包含点云数据，`semantic_mask` 包含逐点级标签。

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── custom
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── points
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
│   │   ├── semantic_mask
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   ├── ...
```

### 数据转换

按照我们的说明准备好原始数据后，您可以直接使用以下命令生成训练/验证信息文件。

```
python tools/create_data.py base --root-path ./data/custom --out-dir ./data/custom
```

## 自定义数据集示例

在完成数据准备后，我们可以在 `mmdet3d/datasets/my_dataset.py` 中创建一个新的数据集来加载数据。

```python
import mmengine

from mmdet3d.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class MyDataset(Det3DDataset):

    # 替换成自定义 pkl 信息文件里的所有类别
    METAINFO = {
       'CLASSES': ('Pedestrian', 'Cyclist', 'Car')
    }

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`

        Args:
            info (dict): Info dict.

        Returns:
            dict | None: Processed `ann_info`
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # 空实例
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # 过滤掉没有在训练中使用的类别
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
```

数据预处理后，用户可以通过两个步骤来训练自定义数据集：

1. 修改配置文件来使用自定义数据集。
2. 验证自定义数据集的正确性。

这里我们以在自定义数据集上训练 PointPillars 为例：

### 准备配置

这里我们演示一个纯点云训练的配置示例：

#### 准备数据集配置

在 `configs/_base_/datasets/custom.py` 中：

```python
# 数据集设置
dataset_type = 'MyDataset'
data_root = 'data/custom/'
class_names = ['Pedestrian', 'Cyclist', 'Car']  # 替换成自己的数据集类别
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # 根据你的数据集进行调整
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(CLASSES=class_names)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # 替换成你的点云数据维度
        use_dim=4),  # 替换成在训练和推理时实际使用的维度
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # 替换成你的点云数据维度
        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# 为可视化阶段的数据和 GT 加载构造流水线
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
]
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='custom_infos_train.pkl', # 指定你的训练 pkl 信息
            data_prefix=dict(pts='points'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file='custom_infos_val.pkl', # 指定你的验证 pkl 信息
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val.pkl', # 指定你的验证 pkl 信息
    metric='bbox')
```

#### 准备模型配置

对于基于体素化的检测器如 SECOND，PointPillars 及 CenterPoint，点云范围（point cloud range）和体素大小（voxel size）应该根据你的数据集做调整。
理论上，`voxel_size` 和 `point_cloud_range` 的设置是相关联的。设置较小的 `voxel_size` 将增加体素数以及相应的内存消耗。此外，需要注意以下问题：

如果将 `point_cloud_range` 和 `voxel_size` 分别设置成 `[0, -40, -3, 70.4, 40, 1]` 和 `[0.05, 0.05, 0.1]`，则中间特征图的形状为 `[(1-(-3))/0.1+1, (40-(-40))/0.05, (70.4-0)/0.05]=[41, 1600, 1408]`。更改 `point_cloud_range` 时，请记得依据 `voxel_size` 更改 `middle_encoder` 里中间特征图的形状。

关于 `anchor_range` 的设置，一般需要根据数据集做调整。需要注意的是，`z` 值需要根据点云的位置做相应调整，具体请参考此 [issue](https://github.com/open-mmlab/mmdetection3d/issues/986)。

关于 `anchor_size` 的设置，通常需要计算整个训练集中目标的长、宽、高的平均值作为 `anchor_size`，以获得最好的结果。

`configs/_base_/models/pointpillars_hv_secfpn_custom.py`：

```python
voxel_size = [0.16, 0.16, 4]  # 根据你的数据集做调整
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # 根据你的数据集做调整
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    # `output_shape` 需要根据 `point_cloud_range` 和 `voxel_size` 做相应调整
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        # 根据你的数据集调整 `ranges` 和 `sizes`
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # 模型训练和测试设置
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
```

#### 准备整体配置

我们将上诉的所有配置组合在 `configs/pointpillars/pointpillars_hv_secfpn_8xb6_custom.py` 文件中：

```python
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_custom.py',
    '../_base_/datasets/custom.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
```

#### 可视化数据集（可选）

为了验证准备的数据和配置是否正确，我们建议在训练和验证前使用 `tools/misc/browse_dataest.py` 脚本可视化数据集和标注，更多细节请参考[可视化](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/zh_cn/user_guides/visualization.md)文档。

## 评估

准备好数据和配置之后，你可以遵循我们的文档直接运行训练/测试脚本。

**注意**：我们为自定义数据集提供了 KITTI 风格的评估实现方法。在数据集配置中需要包含如下内容：

```python
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val.pkl', # 指定你的 验证 pkl 信息
    metric='bbox')
```