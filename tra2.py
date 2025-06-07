import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 点云数据读取函数
def read_bin_file(bin_path):
    """读取点云bin文件"""
    point_cloud = np.fromfile(bin_path, dtype=np.float32)
    return point_cloud.reshape((-1, 4))  # 每点4个值 (x, y, z, intensity)

# 2. 3D边界框计算函数 - 适配新格式
def calculate_3d_bbox(center, size, rotation_z):
    """
    计算3D边界框的8个顶点坐标 (适配新格式)
    参数:
        center: [x, y, z] 中心坐标
        size: [l, w, h] 长宽高
        rotation_z: 绕Z轴的旋转角度(弧度)
    """
    l, w, h = size
    
    # 创建基础立方体 (物体坐标系)
    # 坐标系: x-前, y-左, z-上
    corners = np.array([
        [l/2, w/2, h/2],    # 前右上
        [l/2, w/2, -h/2],   # 前右下
        [l/2, -w/2, h/2],   # 前左上
        [l/2, -w/2, -h/2],  # 前左下
        [-l/2, w/2, h/2],   # 后右上
        [-l/2, w/2, -h/2],  # 后右下
        [-l/2, -w/2, h/2],  # 后左上
        [-l/2, -w/2, -h/2]  # 后左下
    ])
    
    # 创建绕Z轴的旋转矩阵
    cos_z = math.cos(rotation_z)
    sin_z = math.sin(rotation_z)
    rot_matrix = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    
    # 应用旋转
    rotated_corners = np.dot(corners, rot_matrix.T)
    
    # 应用平移 (转换到世界坐标系)
    world_corners = rotated_corners + np.array(center)
    
    return world_corners

# 3. 可视化函数
def visualize_point_cloud_with_bbox(points, bbox_corners_list, title="Point Cloud with 3D Bounding Boxes", colors=None, labels=None):
    """可视化点云和多个3D边界框"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    ax.scatter(
        points[:, 0],  # x (前)
        points[:, 1],  # y (左)
        points[:, 2],  # z (上)
        c=points[:, 3], # 反射强度作为颜色
        s=0.5,
        alpha=0.5,
        cmap='viridis'
    )
    
    # 定义边界框的边
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # 前表面
        [4, 5], [5, 7], [7, 6], [6, 4],  # 后表面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接前后
    ]
    
    # 为每个边界框使用不同颜色
    if colors is None:
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    
    # 类别颜色映射 - 更新为四种类别
    class_colors = {
        'Helicopter': 'red',
        'Jet': 'blue',
        'Propeller': 'green',
        'UAV': 'purple',
        'Unknown': 'magenta'
    }
    
    # 绘制所有边界框
    for idx, bbox_corners in enumerate(bbox_corners_list):
        # 获取类别对应的颜色
        if labels and idx < len(labels):
            color = class_colors.get(labels[idx], 'magenta')
        else:
            color = colors[idx % len(colors)]
        
        for edge in edges:
            ax.plot3D(
                [bbox_corners[edge[0], 0], bbox_corners[edge[1], 0]],  # X坐标
                [bbox_corners[edge[0], 1], bbox_corners[edge[1], 1]],  # Y坐标
                [bbox_corners[edge[0], 2], bbox_corners[edge[1], 2]],  # Z坐标
                color,
                linewidth=2
            )
        
        # 在边界框中心添加类别标签
        if labels and idx < len(labels):
            center = np.mean(bbox_corners, axis=0)
            ax.text(center[0], center[1], center[2], labels[idx], 
                    color='white', backgroundcolor=color, fontsize=10)
    
    # 设置坐标轴标签
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_zlabel('Z (Up)')
    
    # 设置视角
    ax.view_init(elev=20, azim=-120)  # 调整视角
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 4. 解析JSON标签文件函数 - 适配新格式
def parse_sustechpoints_json(label_path):
    """解析JSON标签文件"""
    bboxes = []
    try:
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            
            # 格式: 对象列表
            for obj in label_data:
                psr = obj.get("psr", {})
                position = psr.get("position", {})
                scale = psr.get("scale", {})
                rotation = psr.get("rotation", {})
                
                # 提取3D边界框参数
                center_x = position.get("x", 0)
                center_y = position.get("y", 0)
                center_z = position.get("z", 0)
                
                # 尺寸参数
                length = scale.get("x", 0)  # 长度（前向）
                width = scale.get("y", 0)   # 宽度（侧向）
                height = scale.get("z", 0)  # 高度（垂直）
                
                # 旋转参数 - 绕Z轴的旋转角度
                rotation_z = rotation.get("z", 0)
                
                bboxes.append({
                    'type': obj.get("obj_type", "Unknown"),
                    'center': [center_x, center_y, center_z],
                    'size': [length, width, height],
                    'rotation_z': rotation_z
                })
    except Exception as e:
        print(f"解析标签文件 {label_path} 时出错: {e}")
    
    return bboxes

# 5. 点云数据处理函数
def extract_object_points(points, bbox, padding=0.2):
    """
    从点云中提取边界框内的点
    :param points: 完整点云 (N, 4)
    :param bbox: 边界框信息字典
    :param padding: 边界框扩展范围
    :return: 属于该物体的点云 (M, 4)
    """
    center = np.array(bbox['center'])
    size = np.array(bbox['size']) + padding
    rotation_z = bbox['rotation_z']
    
    # 计算边界框角点
    corners = calculate_3d_bbox(center, size, rotation_z)
    
    # 计算边界框范围
    min_bound = corners.min(axis=0)
    max_bound = corners.max(axis=0)
    
    # 提取边界框内的点
    mask = (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) & \
           (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) & \
           (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
    
    return points[mask]

# 6. 自定义数据集类 - 修复类别映射问题和数据维度问题
class PointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=1024, train=True, test_size=0.2, random_seed=42):
        self.data_dir = data_dir
        self.num_points = num_points
        self.train = train
        self.point_clouds = []  # 存储原始点云数据 (包括强度)
        self.labels = []
        
        # 动态构建类别映射
        self.class_map = self._build_class_map(data_dir)
        self.inverse_class_map = {v: k for k, v in self.class_map.items()}
        
        # 收集所有数据
        self._load_data(test_size, random_seed)
    
    def _build_class_map(self, data_dir):
        """扫描所有标签文件，构建完整的类别映射"""
        label_dir = os.path.join(data_dir, "label_json")
        json_files = glob.glob(os.path.join(label_dir, "*.json"))
        
        # 更新为四种类别
        class_map = {
            'Helicopter': 0,
            'Jet': 1,
            'Propeller': 2,
            'UAV': 3
        }
        
        # 扫描所有文件，添加新的类别
        class_counter = len(class_map)
        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    label_data = json.load(f)
                    for obj in label_data:
                        obj_type = obj.get("obj_type", "Unknown")
                        if obj_type not in class_map:
                            # 如果遇到新类别，将其映射到"Unknown"
                            class_map[obj_type] = class_counter
                            class_counter += 1
            except Exception as e:
                print(f"扫描标签文件 {json_path} 时出错: {e}")
        
        return class_map
    
    def _load_data(self, test_size, random_seed):
        """加载所有点云数据和标签"""
        velodyne_dir = os.path.join(self.data_dir, "velodyne")
        label_dir = os.path.join(self.data_dir, "label_json")
        
        bin_files = glob.glob(os.path.join(velodyne_dir, "*.bin"))
        
        # 分割训练集和测试集
        train_files, test_files = train_test_split(
            bin_files, test_size=test_size, random_state=random_seed
        )
        
        selected_files = train_files if self.train else test_files
        
        for bin_path in tqdm(selected_files, desc="Loading data"):
            file_id = os.path.splitext(os.path.basename(bin_path))[0]
            label_path = os.path.join(label_dir, f"{file_id}.json")
            
            if not os.path.exists(label_path):
                continue
            
            points = read_bin_file(bin_path)
            bboxes = parse_sustechpoints_json(label_path)
            
            for bbox in bboxes:
                # 提取属于当前物体的点
                obj_points = extract_object_points(points, bbox)
                
                # 如果点太少，跳过
                if len(obj_points) < 10:
                    continue
                
                # 随机采样固定数量的点
                if len(obj_points) > self.num_points:
                    indices = np.random.choice(len(obj_points), self.num_points, replace=False)
                    obj_points = obj_points[indices]
                else:
                    # 不足的点用零填充
                    padding = np.zeros((self.num_points - len(obj_points), 4))
                    obj_points = np.vstack([obj_points, padding])
                
                # 归一化点云 (仅对xyz坐标)
                centroid = np.mean(obj_points[:, :3], axis=0)
                obj_points[:, :3] -= centroid
                
                # 添加到数据集
                self.point_clouds.append(obj_points)  # 存储完整的4通道数据
                
                # 获取类别标签，如果类别不在映射中，使用'Unknown'
                obj_type = bbox.get('type', 'Unknown')
                label_idx = self.class_map.get(obj_type, 4)  # 4代表Unknown
                self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        # 获取完整点云数据 (4通道: x, y, z, intensity)
        full_point_cloud = self.point_clouds[idx]
        
        # 只取前3个通道 (x, y, z) 作为模型输入
        point_cloud_xyz = full_point_cloud[:, :3].copy()
        
        label = self.labels[idx]
        
        # 转换为PyTorch张量
        point_cloud_xyz = torch.tensor(point_cloud_xyz, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return point_cloud_xyz, label

# 7. PointNet++模型定义
class TNet(nn.Module):
    """T-Net for spatial transformation"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        identity_matrix = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        
        x = x.view(-1, self.k, self.k) + identity_matrix
        return x

class PointNetPP(nn.Module):
    """PointNet++ for point cloud classification"""
    def __init__(self, num_classes):
        super(PointNetPP, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points)
        
        # Input transformation
        transform = self.input_transform(x)
        x = torch.bmm(transform, x)
        
        # First MLP
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Feature transformation
        transform_feat = self.feature_transform(x)
        x = torch.bmm(transform_feat, x)
        
        # Second MLP
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        return x

# 8. 模型训练函数
# 在文件顶部添加必要的导入
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# 修改train_model函数，添加混淆矩阵功能
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    """训练分类模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 用于存储混淆矩阵数据
    all_val_preds = []
    all_val_labels = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            points, labels = points.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for points, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                points, labels = points.to(device), labels.to(device)
                
                outputs = model(points)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 保存验证集的预测结果用于混淆矩阵
        all_val_preds = all_preds.copy()
        all_val_labels = all_labels.copy()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印分类报告
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    
    # 绘制混淆矩阵
    plt.subplot(2, 2, (3, 4))
    
    # 动态生成类名列表
    class_names = list(train_loader.dataset.inverse_class_map.values())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_val_labels, all_val_preds)
    
    # 将混淆矩阵转换为DataFrame以便更好地显示
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    # 打印最终分类报告
    print("\nClassification Report:")
    print(classification_report(all_val_labels, all_val_preds, target_names=class_names))
    
    # 打印混淆矩阵的详细数据
    print("\nConfusion Matrix Details:")
    print(cm_df)
    
    return model



# 9. 预测函数 (修改后，返回预测类别和置信度)
def predict_point_cloud(model, point_cloud, device):
    """预测单个点云的类别和置信度"""
    model.eval()
    # point_cloud 已经是3维数据 (xyz)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)  # 添加batch维度
    point_cloud = point_cloud.to(device)
    
    with torch.no_grad():
        outputs = model(point_cloud)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()




# 10. 完整检测和分类流程 (修改后)
def detect_and_classify(data_dir, model_path=None, train_model_flag=True, num_epochs=20):
    """完整的点云目标检测和分类流程"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集
    train_dataset = PointCloudDataset(data_dir, train=True)
    val_dataset = PointCloudDataset(data_dir, train=False)
    
    # 打印类别映射信息
    print("类别映射信息:")
    for cls, idx in train_dataset.class_map.items():
        print(f"{cls}: {idx}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 初始化模型
    num_classes = len(train_dataset.class_map)
    model = PointNetPP(num_classes)
    
    # 训练模型或加载预训练模型
    if train_model_flag:
        print("开始训练模型...")
        model = train_model(model, train_loader, val_loader, num_epochs=num_epochs)
    elif model_path and os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("未提供预训练模型路径，且未设置训练标志，无法进行预测")
        return
    
    model = model.to(device)
    
    # 对验证集中的样本进行预测和可视化
    inverse_class_map = train_dataset.inverse_class_map
    
    # 创建一个目录来保存预测结果
    output_dir = os.path.join(data_dir, "predicted_labels")
    os.makedirs(output_dir, exist_ok=True)
    print(f"预测结果将保存到: {output_dir}")
    
    # 只处理前5个验证集场景
    max_scenes = min(5, len(val_dataset))
    print(f"\n只处理前{max_scenes}个验证集场景进行预测和可视化")
    
    for scene_idx in range(max_scenes):
        # 获取点云和标签
        points_xyz, true_label = val_dataset[scene_idx]
        
        # 获取完整点云数据（4维，用于可视化）
        full_points = val_dataset.point_clouds[scene_idx]
        
        # 获取点云原始文件信息
        velodyne_dir = os.path.join(data_dir, "velodyne")
        label_dir = os.path.join(data_dir, "label_json")
        
        # 找到对应的原始点云文件
        bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
        sample_file = bin_files[scene_idx % len(bin_files)]
        file_id = os.path.splitext(os.path.basename(sample_file))[0]
        label_path = os.path.join(label_dir, f"{file_id}.json")
        
        # 读取完整点云和标签
        full_scene_points = read_bin_file(sample_file)
        bboxes = parse_sustechpoints_json(label_path)
        
        # 预测每个目标的类别和置信度
        predicted_bboxes = []
        for bbox in bboxes:
            # 提取属于当前物体的点
            obj_points = extract_object_points(full_scene_points, bbox)
            
            # 如果点太少，跳过
            if len(obj_points) < 10:
                continue
            
            # 随机采样固定数量的点
            if len(obj_points) > val_dataset.num_points:
                indices = np.random.choice(len(obj_points), val_dataset.num_points, replace=False)
                obj_points = obj_points[indices]
            else:
                # 不足的点用零填充
                padding = np.zeros((val_dataset.num_points - len(obj_points), 4))
                obj_points = np.vstack([obj_points, padding])
            
            # 归一化点云 (仅对xyz坐标)
            centroid = np.mean(obj_points[:, :3], axis=0)
            obj_points[:, :3] -= centroid
            
            # 预测类别和置信度
            predicted_class, confidence = predict_point_cloud(model, obj_points[:, :3], device)
            predicted_label = inverse_class_map.get(predicted_class, "Unknown")
            
            # 创建预测边界框信息（添加置信度）
            predicted_bbox = {
                "type": predicted_label,
                "center": bbox["center"],
                "size": bbox["size"],
                "rotation_z": bbox["rotation_z"],
                "confidence": confidence  # 添加置信度
            }
            predicted_bboxes.append(predicted_bbox)
        
        # 保存预测结果到JSON文件（包含置信度）
        output_path = os.path.join(output_dir, f"{file_id}.json")
        with open(output_path, 'w') as f:
            json.dump(predicted_bboxes, f, indent=4)
        print(f"保存预测结果到: {output_path}")
        
        # 计算边界框角点用于可视化
        bbox_corners_list = []
        bbox_labels = []
        
        for bbox in bboxes:
            corners = calculate_3d_bbox(
                bbox['center'], 
                bbox['size'], 
                bbox['rotation_z']
            )
            bbox_corners_list.append(corners)
            bbox_labels.append(bbox['type'])
        
        # 可视化完整点云和边界框
        visualize_point_cloud_with_bbox(
            full_scene_points, 
            bbox_corners_list,
            title=f"原始点云: {file_id}",
            labels=bbox_labels
        )
        
        # 可视化预测结果（显示置信度）
        predicted_corners_list = []
        predicted_labels = []
        
        for bbox in predicted_bboxes:
            corners = calculate_3d_bbox(
                bbox['center'], 
                bbox['size'], 
                bbox['rotation_z']
            )
            predicted_corners_list.append(corners)
            # 创建带置信度的标签
            label_with_confidence = f"{bbox['type']}: {bbox['confidence']:.2f}"
            predicted_labels.append(label_with_confidence)
        
        visualize_point_cloud_with_bbox(
            full_scene_points, 
            predicted_corners_list,
            title=f"预测结果: {file_id}",
            labels=predicted_labels
        )
        
        # 打印预测结果（包含置信度）
        print(f"\n场景 {file_id} 预测结果:")
        for i, bbox in enumerate(predicted_bboxes):
            print(f"目标 {i+1}: 预测类别 = {bbox['type']}, 置信度 = {bbox['confidence']:.4f}")

# ================== 主程序 ==================
if __name__ == "__main__":
    # 设置数据路径
    data_dir = "./"  # 主数据目录
    
    # 执行检测和分类
    detect_and_classify(
        data_dir, 
        model_path=None,  # 预训练模型路径，如果没有则设为None
        train_model_flag=True,  # 是否训练模型
        num_epochs=10  # 训练轮数
    )