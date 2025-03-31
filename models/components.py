import torch
import torch.nn as nn
import open_clip
import math
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


# class TextGuidedGAT(nn.Module):
#     """文本引导的图注意力层"""
#
#     def __init__(self, point_feat_dim, text_feat_dim, out_dim):
#         super().__init__()
#         self.text_proj = nn.Sequential(
#             nn.Linear(text_feat_dim, point_feat_dim),
#             nn.GELU()
#         )
#         self.query = nn.Conv1d(point_feat_dim, out_dim, 1)
#         self.key = nn.Conv1d(point_feat_dim, out_dim, 1)
#         self.value = nn.Conv1d(point_feat_dim, out_dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, point_feats, text_feats):
#         """
#         point_feats: [B, C, N] 点云特征
#         text_feats: [B, D] 文本特征
#         """
#         B, C, N = point_feats.size()
#
#         # 文本特征投影并与点特征融合
#         text_feats = self.text_proj(text_feats).unsqueeze(-1)  # [B, C, 1]
#         fused_feats = point_feats + text_feats
#
#         # 注意力计算
#         q = self.query(point_feats)  # [B, C', N]
#         k = self.key(fused_feats)  # [B, C', N]
#         v = self.value(point_feats)  # [B, C', N]
#
#         attn = torch.bmm(q.transpose(1, 2), k)  # [B, N, N]
#         attn = F.softmax(attn, dim=-1)
#
#         out = torch.bmm(v, attn)  # [B, C', N]
#         return self.gamma * out + point_feats  # 残差连接

class TextGuidedGAT(nn.Module):
    """增强版文本引导的图注意力层，包含多头注意力和文本特征增强"""

    def __init__(self, point_feat_dim, text_feat_dim, out_dim, num_heads=4):
        super().__init__()

        # 文本特征投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, point_feat_dim),
            nn.GELU()
        )

        # 多头注意力机制
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # 定义多头注意力的卷积层
        self.query = nn.Conv1d(point_feat_dim, out_dim, 1)
        self.key = nn.Conv1d(point_feat_dim, out_dim, 1)
        self.value = nn.Conv1d(point_feat_dim, out_dim, 1)

        # MLP 用于进一步特征处理
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        # 学习的参数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, point_feats, text_feats):
        """
        point_feats: [B, C, N] 点云特征
        text_feats: [B, D] 文本特征
        """
        B, C, N = point_feats.size()

        # 1. 文本特征投影并与点特征融合
        text_feats = self.text_proj(text_feats).unsqueeze(-1)  # [B, C, 1]
        fused_feats = point_feats + text_feats  # [B, C, N]

        # 2. 多头注意力计算
        q = self.query(point_feats).view(B, self.num_heads, self.head_dim, N)   # [B, heads, head_dim, N]
        k = self.key(fused_feats).view(B, self.num_heads, self.head_dim, N)        # [B, heads, head_dim, N]
        v = self.value(point_feats).view(B, self.num_heads, self.head_dim, N)      # [B, heads, head_dim, N]

        # 计算注意力权重，使用正确的einsum string
        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k)  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)  # [B, heads, N, N]

        # 3. 加权求和
        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)  # [B, heads, head_dim, N]
        out = out.reshape(B, -1, N)  # Use reshape instead of view to handle non-contiguous tensors

        # 4. 特征处理
        # mlp expects input of shape [B, N, out_dim], so we need to transpose
        out = out.transpose(1, 2)  # [B, N, out_dim]
        out = self.mlp(out)        # [B, N, out_dim]
        out = out.transpose(1, 2)  # [B, out_dim, N]

        # 5. 残差连接
        return self.gamma * out + point_feats  # 残差连接




class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding for time step.
    """
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        time = time * self.scale
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1 + 1e-5)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def __len__(self):
        return self.dim
    

class TimeNet(nn.Module):
    """
    Time Embeddings
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):
        return self.net(t)
   
 
class TextEncoder(nn.Module):
    """
    Text Encoder to encode the text prompt.
    """
    def __init__(self, device):
        super(TextEncoder, self).__init__()
        self.device = device
        self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k",
                                                                      device=self.device)
    
    def forward(self, texts):
        """
        texts can be a single string or a list of strings.
        """
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(texts).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device)
        return text_features


class PointNetPlusPlus(nn.Module):
    """_summary_
    PointNet++ class.
    """
    def __init__(self):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])

        # 新增GAT层
        self.gat1 = TextGuidedGAT(128, 512, 128)
        self.gat2 = TextGuidedGAT(256, 512, 256)

        # self.conv1 = nn.Conv1d(128, 512, 1)
        # self.bn1 = nn.BatchNorm1d(512)

        # 修改最终的特征融合层
        self.final_conv = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.GELU()
        )

    def forward(self, xyz, text_feats=None):
        """_summary_
        Return point-wise features and point cloud representation.
        """
        # Set Abstraction layers
        xyz = xyz.contiguous().transpose(1, 2)
        l0_xyz = xyz
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # c = l3_points.squeeze()

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        # 执行FP1后添加GAT处理
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        if self.gat1 and text_feats is not None:
            l0_points = self.gat1(l0_points, text_feats)  # GAT特征增强

        # 使用新的final_conv代替原始的conv1+bn1
        point_features = self.final_conv(l0_points)

        # 全局特征整合文本信息
        c = l3_points.squeeze(-1)  # l3_points原本是 [B, 1024, 1]
        if text_feats is not None:
            c = torch.cat([c, text_feats], dim=1)  # [B, 1024+512]
        # 🆕 修改部分结束 -------------------------------------------------

        return point_features, c  # [B, 512, 2048], [B, 1536]


class PoseNet(nn.Module):
    """_summary_
    ContextPoseNet class. This class is for a denoising step in the diffusion.
    """

    def __init__(self):
        super(PoseNet, self).__init__()
        self.cloud_net0 = nn.Sequential(
            nn.Linear(1536, 512),  # Changed from 1024 to 1536
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32)
        )

        self.cloud_net3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 6)
        )
        self.cloud_net2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 4)
        )
        self.cloud_net1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )
        self.cloud_influence_net3 = nn.Sequential(
            nn.Linear(6 + 7 + 6 + 6, 6),  # 25 -> 6
            nn.GELU(),
            nn.Linear(6, 6)
        )
        self.cloud_influence_net2 = nn.Sequential(
            nn.Linear(4 + 7 + 4 + 4, 4),  # 19 -> 4
            nn.GELU(),
            nn.Linear(4, 4)
        )
        self.cloud_influence_net1 = nn.Sequential(
            nn.Linear(2 + 6 + 2 + 2, 16),  # 12 -> 16 (choose an intermediate dimension)
            nn.GELU(),
            nn.Linear(16, 2)  # Final output: 2
        )

        self.text_net0 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32)
        )
        self.text_net3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 6)
        )
        self.text_net2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 4)
        )
        self.text_net1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )
        self.text_influence_net3 = nn.Sequential(
            nn.Linear(6 + 7 + 6 + 6, 6),  # 25 -> 6
            nn.GELU(),
            nn.Linear(6, 6)
        )
        self.text_influence_net2 = nn.Sequential(
            nn.Linear(4 + 7 + 4 + 4, 4),  # 19 -> 4
            nn.GELU(),
            nn.Linear(4, 4)
        )
        self.text_influence_net1 = nn.Sequential(
            nn.Linear(2 + 7 + 2 + 2, 16),  # 13 -> 16
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )

        # self.time_net3 = SinusoidalPositionEmbeddings(dim=6)
        # self.time_net2 = SinusoidalPositionEmbeddings(dim=4)
        # self.time_net1 = SinusoidalPositionEmbeddings(dim=2)
        self.time_net3 = TimeNet(dim=6)
        self.time_net2 = TimeNet(dim=4)
        self.time_net1 = TimeNet(dim=2)

        self.down1 = nn.Sequential(
            nn.Linear(7, 6),
            nn.GELU(),
            nn.Linear(6, 6)
        )
        self.down2 = nn.Sequential(
            nn.Linear(6, 4),
            nn.GELU(),
            nn.Linear(4, 4)
        )
        self.down3 = nn.Sequential(
            nn.Linear(4, 2),
            nn.GELU(),
            nn.Linear(2, 2)
        )

        self.up1 = nn.Sequential(
            nn.Linear(2 + 4, 4),
            nn.GELU(),
            nn.Linear(4, 4)
        )
        self.up2 = nn.Sequential(
            nn.Linear(4 + 6, 6),
            nn.GELU(),
            nn.Linear(6, 6)
        )
        self.up3 = nn.Sequential(
            nn.Linear(6 + 7, 7),
            nn.GELU(),
            nn.Linear(7, 7)
        )

    def forward(self, g, c, t, context_mask, _t):
        """_summary_
        Args:
            g: pose representations, size [B, 7]
            c: point cloud representations, size [B, 1024]
            t: affordance texts, size [B, 512]
            context_mask: masks {0, 1} for the contexts, size [B, 1]
            _t is for the timesteps, size [B,]
        """
        c = c * context_mask
        c0 = self.cloud_net0(c)
        c1 = self.cloud_net1(c0)
        c2 = self.cloud_net2(c0)
        c3 = self.cloud_net3(c0)

        # 将文本特征与点云特征一起传入网络
        t = t * context_mask
        t0 = self.text_net0(t)
        t1 = self.text_net1(t0)
        t2 = self.text_net2(t0)
        t3 = self.text_net3(t0)

        # 添加时间步信息
        _t0 = _t.unsqueeze(1)
        _t1 = self.time_net1(_t0)
        _t2 = self.time_net2(_t0)
        _t3 = self.time_net3(_t0)

        g = g.float()
        g_down1 = self.down1(g)  # 6
        g_down2 = self.down2(g_down1)  # 4
        g_down3 = self.down3(g_down2)  # 2

        # 这里对文本和点云特征进行融合，增强pose生成的引导
        c1_influence = self.cloud_influence_net1(
            torch.cat((c1, g[:, :6], _t1, t1), dim=1)
        )

        t1_influence = self.text_influence_net1(torch.cat((t1, g, _t1, c1), dim=1))  # 添加点云特征

        influences1 = F.softmax(torch.cat((c1_influence.unsqueeze(1), t1_influence.unsqueeze(1)), dim=1), dim=1)
        ct1 = (c1 * influences1[:, 0, :] + t1 * influences1[:, 1, :])
        up1 = self.up1(torch.cat((g_down3 * ct1 + _t1, g_down2), dim=1))

        c2_influence = self.cloud_influence_net2(torch.cat((c2, g, _t2, t2), dim=1))
        t2_influence = self.text_influence_net2(torch.cat((t2, g, _t2, c2), dim=1))
        influences2 = F.softmax(torch.cat((c2_influence.unsqueeze(1), t2_influence.unsqueeze(1)), dim=1), dim=1)
        ct2 = (c2 * influences2[:, 0, :] + t2 * influences2[:, 1, :])
        up2 = self.up2(torch.cat((up1 * ct2 + _t2, g_down1), dim=1))

        c3_influence = self.cloud_influence_net3(torch.cat((c3, g, _t3, t3), dim=1))
        t3_influence = self.text_influence_net3(torch.cat((t3, g, _t3, c3), dim=1))
        influences3 = F.softmax(torch.cat((c3_influence.unsqueeze(1), t3_influence.unsqueeze(1)), dim=1), dim=1)
        ct3 = (c3 * influences3[:, 0, :] + t3 * influences3[:, 1, :])
        up3 = self.up3(torch.cat((up2 * ct3 + _t3, g), dim=1))  # size [B, 7]

        return up3





