import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Rotater(nn.Module):
    def __init__(self, name='rotater', mode='traj', aug=False, carla=False):
        super().__init__()
        self.name = name
        self.mode = mode
        self.random_aug = aug  # False
        self.carla = carla

    def forward(self, states, mask, curr_frame=None, aug=True, random_angle=None):
        if self.mode == 'traj':
            return self._make_rotations(states, mask, curr_frame, aug)
        elif self.mode == 'fut':
            return self._rotate_fut(states, curr_frame, aug)
        else:
            return self._make_map_rotations(states, mask, curr_frame, aug, random_angle)

    def _rotate_fut(self, states, curr_frames, aug=True):

        '''
        input;[batch,timestep,2]
        curr_frame:[batch,5]
        '''
        mask = (states != 0)[:, :, 0]
        # [batch, timestep]
        mask = mask.float()
        yaw = curr_frames[:, 2]
        cos_a = torch.cos(yaw).unsqueeze(-1)
        sin_a = torch.sin(yaw).unsqueeze(-1)
        # [batch, 1]
        x = states[:, :, 0] - curr_frames[:, 0].unsqueeze(-1)
        y = states[:, :, 1] - curr_frames[:, 1].unsqueeze(-1)
        new_x = cos_a * x + sin_a * y # (b, t)
        new_y = -sin_a * x + cos_a * y
        rotated_state = torch.stack([new_x, new_y], dim=-1)
        mask = mask.unsqueeze(-1)
        return rotated_state * mask, mask

    def _make_rotations(self, states, mask, curr_frame=None, aug=True):
        # states [32, 6 10, 5]
        # maks  (32, 6, 10)
        # get curr_frame location according to the mask
        mask = mask.to(torch.int32)
        ind = mask[:, 0, :].sum(dim=-1) # ego车在哪一帧停止
        # (32)
        ind = torch.clamp(ind - 1, 0, 100)
        # gather indices is [batch_no , 0(ego) , mask_ind]
        # return [batch,hidden(4)]
        if curr_frame is None:
            batch_indices = torch.arange(mask.size(0)).to(mask.device)
            curr_frames = states[batch_indices, 0, ind]
            # torch.Size([32, 5])
        else:
            # for future representations
            curr_frames = curr_frame
        if self.random_aug:
            r_range = np.pi / 2
            random_angle = torch.empty_like(curr_frames[:, 2]).uniform_(-r_range, r_range)
            cos_r = torch.cos(random_angle).unsqueeze(-1).unsqueeze(-1)
            sin_r = torch.sin(random_angle).unsqueeze(-1).unsqueeze(-1)
            # torch.Size([32,1,1])

        yaw = curr_frames[:, 2]
        cos_a = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)
        x = states[:, :, :, 0] - curr_frames[:, 0].unsqueeze(-1).unsqueeze(-1)
        # [32, 6, 10]
        y = states[:, :, :, 1] - curr_frames[:, 1].unsqueeze(-1).unsqueeze(-1)
        angle = states[:, :, :, 2] - yaw.unsqueeze(-1).unsqueeze(-1)
        if aug:
            new_x = cos_a * x + sin_a * y
            new_y = -sin_a * x + cos_a * y
            if self.random_aug:
                n_x = cos_r * new_x + sin_r * new_y
                n_y = -sin_r * new_x + cos_r * new_y
                new_x, new_y = n_x, n_y
        else:
            new_x, new_y = x, y
        vx = states[:, :, :, 3] - curr_frames[:, 3].unsqueeze(-1).unsqueeze(-1)
        vy = states[:, :, :, 4] - curr_frames[:, 4].unsqueeze(-1).unsqueeze(-1)
        if aug:
            new_vx = cos_a * vx + sin_a * vy
            new_vy = -sin_a * vx + cos_a * vy
            if self.random_aug:
                n_vx = cos_r * new_vx + sin_r * new_vy
                n_vy = -sin_r * new_vx + cos_r * new_vy
                new_vx, new_vy = n_vx, n_vy
        else:
            new_vx, new_vy = vx, vy
        rotated_state = torch.stack([-new_x, new_y, angle, -new_vx, new_vy], dim=-1)
        if not self.carla:
            rotated_state = torch.stack([new_x, new_y, angle, new_vx, new_vy], dim=-1)
        mask = mask.unsqueeze(-1).float()
        if self.random_aug:
            return rotated_state * mask, curr_frames, random_angle
        return rotated_state * mask, curr_frames, None

    def _make_map_rotations(self, states, mask, curr_frames, aug=True, random_angle=None):
        # states [32, 18, 10, 2]
        # mask (32, 6*3, 10)
        # curr_frames [32, 5]
        # curr_frame:[batch,5]
        # mask = torch.tensor(mask, dtype=torch.int32)
        # states = torch.tensor(states, dtype=torch.float32).to(mask.device)
        # curr_frames = torch.tensor(curr_frames, dtype=torch.float32).to(mask.device)

        yaw = curr_frames[:, 2]
        # [32，]
        cos_a = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)
        # [32，1， 1]
        # angle = states[:,:,:,2] - tf.reshape(yaw,[-1,1,1])
        x = states[:, :, :, 0] - curr_frames[:, 0].unsqueeze(-1).unsqueeze(-1)
        y = states[:, :, :, 1] - curr_frames[:, 1].unsqueeze(-1).unsqueeze(-1)
        # (32, 6 * 3, 10)
        if self.random_aug:
            cos_r = torch.cos(random_angle).unsqueeze(-1).unsqueeze(-1)
            sin_r = torch.sin(random_angle).unsqueeze(-1).unsqueeze(-1)

        if aug:
            new_x = cos_a * x + sin_a * y
            new_y = -sin_a * x + cos_a * y
            if self.random_aug:
                n_x = cos_r * new_x + sin_r * new_y
                n_y = -sin_r * new_x + cos_r * new_y
                new_x, new_y = n_x, n_y
        else:
            new_x, new_y = x, y
        rotated_state = torch.stack([-new_x, new_y], dim=-1)
        # (32, 6 * 3, 10， 2)
        if not self.carla:
            angle = states[:, :, :, 2] - yaw.unsqueeze(-1).unsqueeze(-1)
            rotated_state = torch.stack([new_x, new_y, angle, states[:, :, :, 3], states[:, :, :, 4]], dim=-1)
        mask = mask.unsqueeze(-1).float()
        # [batch, ego + neighbours, timesteps, 1]
        return rotated_state * mask


class MapEncoder(nn.Module):
    def __init__(self, return_attention_scores=False, carla=False):
        super(MapEncoder, self).__init__()
        self.return_attention_scores = return_attention_scores
        self.carla = carla
        if self.carla:
            self.self_line = nn.Linear(2, 3*64)
        else:
            self.self_line = nn.Linear(3, 128 + 64)


        self.node_attention = nn.MultiheadAttention(embed_dim=3*64, num_heads=2, dropout=0, batch_first=True)
        self.flatten = nn.AdaptiveMaxPool1d(1)  # 全局最大池化层，将序列数据压缩为单一向量
        self.vector_feature = nn.Linear(2, 64)
        self.sublayer = nn.Linear(64*4, 128)  # 拼接后的维度为 128 + 64

    def forward(self, inputs, mask, test):
        # (32, 10, 2)  (32, 10)
        inputs = inputs.to(device)
        mask = mask.to(device)

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.bool)

        # 创建 key_padding_mask

        if self.carla:
            nodes = inputs[:, :, :2]
            # (32, 10, 2)
        else:
            nodes = inputs[:, :, :3]

        nodes = self.self_line(nodes)
        # (32, 10, 3 * 64)

        mask = (~mask).float()  # 保持梯度连续性
        # (32, 10)
        if self.return_attention_scores:
            nodes, attention_weights = self.node_attention(
                nodes, nodes, nodes,
                key_padding_mask=mask,
                # average_attn_weights=False,
            )
            # (32, 10, 192)， (32, (num_heads=2), 10, 10) 每个头计算一个 10x10 的分数矩阵
        else:
            nodes, _ = self.node_attention(
                nodes, nodes, nodes,
                key_padding_mask=mask,
            )
        # torch.Size([32, 10, 192])  torch.Size([32, 10, 10])
        nodes = F.relu(nodes)
        nodes = self.flatten(nodes.transpose(1,2)).squeeze(-1)  # (batch_size, embed_dim)

        # (32, 96*3)
        # 处理向量特征

        vector = self.vector_feature(inputs[:, 0, -2:])
        # (32, 64)
        # 拼接节点特征和向量特征
        out = torch.cat([nodes, vector], dim=1)  # (batch_size, embed_dim + 64)
        # (32, 256)
        polyline_feature = self.sublayer(out)  # (32, 128)

        if self.return_attention_scores:
            # 计算注意力分数的均值
            attention_weights = attention_weights.mean(dim=1) # (batch_size, seq_len)
            # (32, 128)  (32, 10)

            return polyline_feature, attention_weights
        return polyline_feature



class MultiModal_Attention(nn.Module):
    def __init__(self, num_modes, key_dim, head_num=1):
        super(MultiModal_Attention, self).__init__()
        self.num_modes = num_modes
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=key_dim, num_heads=head_num, batch_first=True)
            for _ in range(num_modes)
        ])
        self.norm1 = nn.LayerNorm(key_dim)
        self.norm2 = nn.LayerNorm(key_dim)
        # self.FFN1 = nn.Linear(key_dim, 4 * key_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.FFN2 = nn.Linear(key_dim, key_dim)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, query, key, mask=None, training=True):
        output = []
        for i in range(self.num_modes):
            value, _ = self.attention[i](query, key, key, key_padding_mask=~mask)

            output.append(value.squeeze(1))

        value = F.relu(torch.stack(output, dim=1))

        return value, None


import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, out_dim)
        self.relu = nn.LeakyReLU(0.01)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class Hierachial_Transformer(nn.Module):
    def __init__(self, state_shape, name='hier_encoder', units=256,
                 use_trans_encode=False, num_heads=2, drop_rate=0.1, neighbours=5,
                 make_rotation=True, time_step=8, num_modes=1, final_head_num=2,
                 random_aug=False, no_ego_fut=False, no_neighbor_fut=False, carla=False,
                 leaky_relu_slope=0.01, rezero=True, n_heads=2):
        super(Hierachial_Transformer, self).__init__()
        self.inp_dim = state_shape[-1]
        # (6, 10,5)
        self.n_heads = n_heads
        self.map_layer = MapEncoder(return_attention_scores=True, carla=carla)

        self.neighbours = neighbours
        self.make_rotation = make_rotation
        self.time_step = time_step
        self.embedder = nn.Linear(self.inp_dim, units)



        self.time_layer = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=0, batch_first=True)
        self.time_pooling = nn.AdaptiveMaxPool1d(1)

        self.use_trans = use_trans_encode
        self.rel_layer = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=0, batch_first=True)

        if self.make_rotation:
            self.rotater = Rotater(mode='traj', aug=random_aug, carla=carla)
            self.map_rotater = Rotater(mode='map', aug=random_aug, carla=carla)

        self.map_attention = nn.MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=0, batch_first=True)

        self.final_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=units, num_heads=final_head_num, dropout=0, batch_first=True)
            for _ in range(num_modes)
        ])
        self.num_modes = num_modes

        self.no_ego_fut = no_ego_fut
        self.no_neighbor_fur = no_neighbor_fut
        self.carla = carla

        self.att_proj = nn.Linear(self.inp_dim + 1,  1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.Rezero = nn.Parameter(torch.zeros(units)) if rezero else None
        self.LayerNorm = nn.LayerNorm(units)
        self.rezero_enabled = rezero
        self.FF = FeedForwardNetwork(units, units)
        self.edge_attr_proj = nn.Linear(6, units)
        self.D_head = units // self.n_heads
        self.att_proj_emb = nn.Parameter(torch.randn(self.n_heads, 3 * self.D_head))

    def compute_edge_attr(self, states):
        """
        states: [B, N, T, 5] → [x, y, vx, vy, yaw]
        returns: edge_attr: [B, T, N, N, 6], where edge_attr[:, t, i, j, :] is from node j → i
        """
        x = states[..., 0]  # [B, N, T]
        y = states[..., 1]
        vx = states[..., 2]
        vy = states[..., 3]
        yaw = states[..., 4]

        # 转置为 [B, T, N] 方便广播
        x = x.permute(0, 2, 1)  # [B, T, N]
        y = y.permute(0, 2, 1)
        vx = vx.permute(0, 2, 1)
        vy = vy.permute(0, 2, 1)
        yaw = yaw.permute(0, 2, 1)

        # 构造 pair-wise 差值（i - j）: i 是接收节点，j 是发送节点
        dx = x.unsqueeze(2) - x.unsqueeze(3)  # [B, T, N, N]
        dy = y.unsqueeze(2) - y.unsqueeze(3)
        dvx = vx.unsqueeze(2) - vx.unsqueeze(3)
        dvy = vy.unsqueeze(2) - vy.unsqueeze(3)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        dyaw = yaw.unsqueeze(2) - yaw.unsqueeze(3)

        # 堆叠为边属性
        edge_attr = torch.stack([dx, dy, dvx, dvy, dist, dyaw], dim=-1)  # [B, T, N, N, 6]
        return edge_attr


    def forward(self, states, test=False, map_state=None, aug=True):
        # states (32, 6, 10, 5)  map_state (32, 18, 10, 2)
        training = not test
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(device)
        else:
            states = states.to(device).float()
        if isinstance(map_state, np.ndarray):
            map_state = torch.from_numpy(map_state).float().to(device)
        else:
            map_state = map_state.to(device).float()


        # Create mask
        mask = (states != 0)[:, :, :, 0]
        # (32, 6, 10)

        if self.make_rotation:
            states, curr_frames, rg = self.rotater(states, mask, aug=aug)
            # [32, 6, 10, 5]  [32, 5] None

        embedder_state = self.embedder(states)
        # states (32, 6, 10, 128)
        ego_states, neighbor_states = states[:, 0, :, :], states[:, 1:, :, :]
        # [32, 10, 5]  [32, 5, 10, 5]

        ego_mask, neighbor_mask = mask[:, 0, :], mask[:, 1:, :]
        # [32, 10]  [32, 5, 10]

        ego_embedder, neighbor_embedder = embedder_state[:, 0, :, :], embedder_state[:, 1:, :, :]
        # [32, 10, 128]  [32, 5, 10, 128]
        actor_mask = (torch.cat([torch.ones_like(ego_states).unsqueeze(1), neighbor_states], dim=1) != 0)[:, :, 0, 0]
        # [32, 6]


        ego = self._timestep_attention(ego_embedder, training, ego_mask)
        # (32, 128)


        # Process neighbor states
        neighbors = [
            self._timestep_attention(neighbor_embedder[:, i, :, :], training, neighbor_mask[:, i, :])
            for i in range(self.neighbours)
        ]


        # Process map states
        map_mask = (map_state != 0)[:, :, :, 0]
        # torch.Size([32, 18, 10])
        map_traj_mask = (map_state != 0)[:, :, 0, 0]
        # torch.Size([32, 18])
        if self.make_rotation:
            map_state = self.map_rotater(map_state, map_mask, curr_frames, aug, rg)
        # (32, 18, 10, 2)
        map = []
        val = []
        for i in range(map_state.size(1)):
            m, v = self.map_layer(map_state[:, i], map_mask[:, i, :], test)
            # (32, 10, 2)  (32, 10)
            #  (32, 128)  (32, 10)

            map.append(m)
            if test:
                val.append(v)
        map = torch.stack(map, dim=1)
        # (32, 18, 128)


        if test:
            val = torch.stack(val, dim=1)
            # (32, 18, 128)

        if self.carla:
            ego_map, neighbor_map = map[:, :3, :], map[:, 3:, :]
            # (32, 3, 128)   (32, 15, 128)
            ego_map_traj_mask, neighbor_map_traj_mask = map_traj_mask[:, :3], map_traj_mask[:, 3:]
            # (32, 3)   (32, 15)

        else:
            ego_map, neighbor_map = map[:, :2, :], map[:, 2:, :]
            ego_map_traj_mask, neighbor_map_traj_mask = map_traj_mask[:, :2], map_traj_mask[:, 2:]

        neighbor_rel_val = [
            self._map_vehicle_rel(neighbors[i], neighbor_map, neighbor_map_traj_mask, i * 3)[0]
            for i in range(self.neighbours)
            # 【32，128】 (32, 15, 128) (32, 15)
            # torch.Size([32, 128])  # torch.Size([32, 3])
        ]

        if self.no_neighbor_fur:
            neighbor_rel_val = neighbors

        if True:
            neighbor_val = [
                self._map_vehicle_rel(neighbors[i], neighbor_map, neighbor_map_traj_mask, i * 3)[1]
                for i in range(self.neighbours)
            ]
            # 【32，128】 (32, 15, 128) (32, 15)
            # torch.Size([32, 128])  # torch.Size([32, 3])
        actor = torch.cat([ego.unsqueeze(1), torch.stack(neighbor_rel_val, dim=1)], dim=1)
        # (32, 6, 128)


        # # v1版本
        # edge_attr = self.compute_edge_attr(states)  # [B, T, N, N, 6]
        # att_logits = self.att_proj(edge_attr).squeeze(-1)  # [B, T, N, N]
        # att_logits = self.leaky_relu(att_logits)
        # valid = (states != 0)[:, :, :, 0].permute(0, 2, 1)  # [B, T, N]
        # att_logits = att_logits.masked_fill(~valid.unsqueeze(2), -1e9)
        # att_weights = torch.softmax(att_logits, dim=-1)  # [B, T, N, N]
        # att_avg = att_weights.mean(dim=1)  # [B, N, N]
        # actor = torch.einsum('bij,bjd->bid', att_avg, actor)  # [B, N, D]
        # if self.rezero_enabled:
        #     actor = actor + self.Rezero * self.FF(actor)
        # else:
        #     actor = actor + self.FF(actor)
        #
        # actor = self.LayerNorm(actor)

        # v2版本考虑所有时间步
        # edge_attr = self.compute_edge_attr(states)  # [B, T, N, N, 6]
        # # states (32, 6, 10, 5)
        # # === 基本维度信息 ===
        # B, N, T, _ = states.shape
        # D = actor.size(-1)
        # H = self.n_heads
        # D_head = D // H
        #
        # # === 多头表示 actor: [B, N, D] → [B, N, H, D_head] ===
        # actor_multi = actor.view(B, N, H, D_head)
        # # === 构造 h_i 和 h_j ===
        # h_i = actor_multi.unsqueeze(1).unsqueeze(2).expand(B, T, N, N, H, D_head)  # [B,T,N,N,H,D']
        # h_j = actor_multi.unsqueeze(1).unsqueeze(3).expand(B, T, N, N, H, D_head)  # [B,T,N,N,H,D']
        #
        # # === 投影边属性 edge_attr: [B,T,N,N,6] → [B,T,N,N,H,D_head] ===
        # edge_attr_proj = self.edge_attr_proj(edge_attr).view(B, T, N, N, H, D_head)
        # # === 拼接注意力输入：[h_i | edge_attr | h_j] → [B, T, N, N, H, 3*D'] ===
        # att_input = torch.cat([h_i, edge_attr_proj, h_j], dim=-1)
        #
        # # === 注意力打分: einsum 代替 Linear，LeakyReLU + softmax ===
        # att_logits = torch.einsum("bijnhd,hd->bijnh", att_input, self.att_proj_emb)  # [B, T, N, N, H]
        # att_logits = self.leaky_relu(att_logits)
        #
        # # === Mask 掉无效邻接节点 ===
        # valid = (states != 0)[..., 0].permute(0, 2, 1)  # [B, N, T]
        # att_logits = att_logits.masked_fill(~valid.unsqueeze(2).unsqueeze(-1), -1e9)
        # # === softmax 实现 exp(LeakyReLU(...)) 并归一化 ===
        # att_weights = torch.softmax(att_logits, dim=3)  # [B, T, N, N, H]
        # att_weights = att_weights.mean(dim=1).unsqueeze(-1)  # → [B, N, N, H, 1]
        # # === actor 作为 value（不投影）：[B, N, H, D'] → v_j ===
        # v_j = actor_multi.unsqueeze(1).expand(B, N, N, H, D_head)  # [B, N, N, H, D']
        # # === 加权聚合 → sum over j → [B, N, H, D']
        # att_output = (att_weights * v_j).sum(dim=2)
        # # === 多头拼接 → [B, N, D]
        # actor = att_output.reshape(B, N, D)
        #
        #
        # if self.rezero_enabled:
        #     actor = actor + self.Rezero * self.FF(actor)
        # else:
        #     actor = actor + self.FF(actor)
        #
        # actor = self.LayerNorm(actor)

        # === 基本维度 ===
        # # v3最后时间
        edge_attr = self.compute_edge_attr(states)  # [B, T, N, N, 6]
        edge_attr = edge_attr[:, -1, :, :] # [B, N, N, 6]

        states = states[:, :, -1, :]  # [B, N, 5]

        B, N, _ = states.shape
        D = actor.size(-1)
        H = self.n_heads
        D_head = D // H

        # === 多头 actor: [B, N, D] → [B, N, H, D']
        actor_multi = actor.view(B, N, H, D_head)

        # === 构造 h_i 和 h_j: [B, N, H, D'] → [B, N, N, H, D']
        h_i = actor_multi.unsqueeze(2).expand(B, N, N, H, D_head)  # [B, N, N, H, D']
        h_j = actor_multi.unsqueeze(1).expand(B, N, N, H, D_head)  # [B, N, N, H, D']

        # === edge_attr: [B, N, N, 6] → [B, N, N, H, D']
        edge_attr_proj = self.edge_attr_proj(edge_attr).view(B, N, N, H, D_head)

        # === 拼接注意力输入: [h_i | edge_attr | h_j] → [B, N, N, H, 3*D']
        att_input = torch.cat([h_i, edge_attr_proj, h_j], dim=-1)

        # === 注意力打分 + 激活 ===
        att_logits = torch.einsum("bjnhd,hd->bjnh", att_input, self.att_proj_emb)  # [B, N, N, H]
        att_logits = self.leaky_relu(att_logits)

        # === 构造 valid mask（是否为有效 agent）
        valid = (states != 0)[..., 0]  # [B, N]
        att_logits = att_logits.masked_fill(~valid.unsqueeze(1).unsqueeze(-1), -1e9)  # mask j

        # === softmax 得到注意力权重
        att_weights = torch.softmax(att_logits, dim=2).unsqueeze(-1)  # [B, N, N, H, 1]

        # === actor_multi: [B, N, H, D'] → [B, 1, N, H, D'] → broadcast
        v_j = actor_multi.unsqueeze(1).expand(B, N, N, H, D_head)

        # === 聚合
        att_output = (att_weights * v_j).sum(dim=2)  # [B, N, H, D']
        actor = att_output.reshape(B, N, D)  # [B, N, D]

        # === FFN + ReZero + LN
        if self.rezero_enabled:
            actor = actor + self.Rezero * self.FF(actor)
        else:
            actor = actor + self.FF(actor)
        actor = self.LayerNorm(actor)

        actor_rel, _ = self.rel_layer(
            ego.unsqueeze(1),
            actor,
            actor,
            key_padding_mask=(~actor_mask).float()  # ✅ 正确维度：[B, T]
        )
        # torch.Size([32, 1, 128])
        actor_rel = F.relu(actor_rel.squeeze(1))
        # (32, 128)

        goals, ego_val = self._goal_layer(actor_rel.unsqueeze(1), ego_map, ego_map_traj_mask.unsqueeze(1))
        # (32, 1, 128)  (32, 3, 128)  (32, 1, 3)

        # goals(32, num_modes, 128), ego_val(32, 3)三种模式打分

        ego_states = actor_rel.unsqueeze(1).repeat(1, self.num_modes, 1)
        # torch.Size([32, num_modes, 128])
        if self.no_ego_fut:
            states = ego_states
        else:
            states = goals + ego_states
        if test:
            neighbor_val = [ego_val] + neighbor_val
            neighbor_val = torch.cat(neighbor_val, dim=-1).unsqueeze(-1)
            return states, neighbor_val
        # (32, num_modes, 128)
        return states

    def _timestep_attention(self, states, training, mask):
        # states [32, 10, 128]
        # mask [32, 10]

        t = states.shape[1]  # 时序长度

        # === Step 1: 处理 mask ===
        # key_padding_mask: [B, T], bool, True 表示“需要被 mask”，即 padding
        key_padding_mask = (~mask).float()

        # causal_mask: [T, T], bool
        causal_mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(states.device)


        # === Step 3: 调用多头注意力 ===
        attn_output, _ = self.time_layer(
            states, states, states,
            attn_mask=causal_mask,  # PyTorch 语义：True 表示 mask 掉
            key_padding_mask=key_padding_mask  # batch-wise mask
        )
        # [32, 10, 128]
        # === Step 4: 后处理 ===
        attn_output = F.relu(attn_output)  # [B, T, C]
        state_val = self.time_pooling(attn_output.transpose(1, 2)).squeeze(-1)  # [B, C]
        # state_val(32, 128)
        return state_val

    def _map_vehicle_rel(self, value, map_state, map_mask, i):#value (32, 128)
        # 【32，128】 (32, 15, 128)(32, 15)
        use_map = map_state[:, i:i + 3, :] #(32, 3, 128)
        use_map_mask = torch.cat([
            torch.ones_like(map_mask[:, 0]).unsqueeze(1),
            map_mask[:, i:i + 3]
        ], dim=1)
        # torch.Size([32, 4])
        mv_rel = torch.cat([value.unsqueeze(1), use_map], dim=1)
        # torch.Size([32, 4, 128])
        key_padding_mask = (~use_map_mask).float()
        mv_val, val = self.map_attention(
            value.unsqueeze(1),
            mv_rel,
            mv_rel,
            key_padding_mask=key_padding_mask,
            # average_attn_weights = False  # <== 关键参数
        )
        # torch.Size([32, 1, 128])  torch.Size([32, 1, 4])
        val = val.squeeze(-2)[:, 1:]
        mv_val = F.relu(mv_val.squeeze(1))
        # torch.Size([32, 128])  # torch.Size([32, 3])

        return mv_val, val

    def _goal_layer(self, query, key, mask=None, training=True):
        # (32, 1, 128)  (32, 3, 128)  (32, 1, 3)

        output = []
        key_padding_mask = (~mask.squeeze(1)).float()
        # (32, 3)
        # torch.Size([32, 1, 128])   torch.Size([32, 1, 3])
        v = []
        for i in range(self.num_modes):
            value, val = self.final_attention[i](
                query,
                key,
                key,
                key_padding_mask=key_padding_mask if mask is not None else None,
                # average_attn_weights = False  # <== 关键参数
            )

            # torch.Size([32, 1, 128]) torch.Size([32, 1, 3])
            output.append(value.squeeze(1))
            v.append(val.squeeze(1))
            # torch.Size([32, 128]) torch.Size([32, 3])
        v = torch.stack(v, dim=1).mean(dim=-2)
        # v = v[0].squeeze(-2).mean(dim=-2)
        value = F.relu(torch.stack(output, dim=1))
        # torch.Size([32, num_modes, 128])     torch.Size([32, 3])
        return value, v


class Represent_Learner(nn.Module):
    def __init__(self, encoder=None, target_encoder=None, hidden_activation="relu", name='rl_encoder', head_dim=1, random_aug=False):
        super().__init__()
        dim_shape = 128
        action_dim = 2
        self.name = name
        self.recurrent_layer = nn.MultiheadAttention(embed_dim=192, num_heads=head_dim, batch_first=True)
        self.embedder = nn.Linear(192, 128)
        self.action_layer = nn.Linear(action_dim, 64)
        self.projection_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU() if hidden_activation == "relu" else nn.Tanh(),
            nn.Linear(256, dim_shape),
            nn.ReLU() if hidden_activation == "relu" else nn.Tanh()
        )
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.random_aug = random_aug

        if not random_aug:
            self.projection_layers_target = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU() if hidden_activation == "relu" else nn.Tanh(),
                nn.Linear(256, dim_shape),
                nn.ReLU() if hidden_activation == "relu" else nn.Tanh()
            )
            self.soft_update(self.projection_layers_target, self.projection_layers, tau=1.0)  # or tau=0.005

        self.pred_layer = nn.Linear(dim_shape, 128)
        self.back_layer = nn.Linear(dim_shape, 128)

        self.pred_step = 1
        self.similarity_loss = nn.CosineSimilarity(dim=-1)


    def soft_update(self, target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    def _timestep_attention(self, states, mask):
        # states: [B, T, 192]; mask: [B, T] (True=valid)
        B, T, _ = states.shape
        mask = mask.to(states.device).bool()
        key_padding_mask = ~mask  # True 表示要屏蔽
        causal_mask = torch.triu(torch.ones(T, T, device=states.device, dtype=torch.bool), diagonal=1)
        out, _ = self.recurrent_layer(states, states, states,
                                      attn_mask=causal_mask,
                                      key_padding_mask=key_padding_mask)
        return out

    def _action_transition(self, feat, actions, init_state=None, re=False):
        if self.params['use_map'] or self.params['use_hier']:
            ensembles = feat.size(1)
            act_feature = self.action_layer(actions).unsqueeze(1)
            features = torch.cat([feat, act_feature.repeat(1, ensembles, 1)], dim=2)
        else:
            act_feature = self.action_layer(actions)
            features = torch.cat([feat, act_feature], dim=-1)
        for layer in self.transition_layers:
            features = layer(features)
        return features

    def _make_autonomous_forward_cycle_loss(self, states, map_state, actions, next_states, next_map_state, mask=None,
                                           test=False, init_state=None):
        """
        嵌入自动驾驶预测结构：
        - states: 当前帧图像状态，shape: [B, C, H, W] 或 [B, T, C, H, W]
        - next_states: 多步图像状态 [B, pred_step, C, H, W]
        - actions: 动作序列 [B, pred_step, action_dim]
        - 返回:
            forward_pred_loss, cycle_loss
        """
        # === 当前时刻 latent（z_t） ===
        latent, _ = self.encoder(states, mask=mask, test=test, init_state=init_state, map_state=map_state)
        latent = latent[:, 0]  # 取 batch 第一帧

        pred_latents = [latent]
        forward_actions = []

        # === 对应 target latent ===
        target_latents = []
        for i in range(self.pred_step):
            with torch.no_grad():
                tgt_latent, _ = self.target_encoder(next_states[:, i], mask=mask, test=test, init_state=init_state,
                                                    map_state=next_map_state[:, i])
                tgt_latent = tgt_latent[:, 0]  # [B, C]
                target_latents.append(tgt_latent)
        target_latents = torch.stack(target_latents, dim=1).flatten(0, 1)  # [B*T, C]

        # === forward dynamics rollout ===
        for i in range(self.pred_step):
            cur_latent = pred_latents[-1]  # [B, latent_dim]
            cur_action = actions[:, i]  # [B, act_dim]
            next_latent, _ = self.dynamics_model(cur_latent, cur_action)
            pred_latents.append(next_latent)
            forward_actions.append(cur_action)

        forward_latents = torch.stack(pred_latents[1:], dim=1).flatten(0, 1)  # [B*T, C]
        for_pred_loss = self.byol_loss(forward_latents, target_latents, mode='spr', batch_size=states.size(0),
                                       T=self.pred_step)

        # === backward consistency (real cycle) ===
        if self.real_cycle:
            recon_latent = pred_latents[-1]
            for i in reversed(range(self.pred_step)):
                act = forward_actions[i]
                # === 使用同一个 dynamics_model 进行逆预测 ===
                recon_latent, _ = self.dynamics_model(recon_latent, act, reverse=True)

            with torch.no_grad():
                cycle_target, _ = self.target_encoder(states, mask=mask, test=test, init_state=init_state,
                                                      map_state=map_state)
                cycle_target = cycle_target[:, 0]  # [B, C]

            if self.space == 'z':
                cycle_loss = F.mse_loss(recon_latent, cycle_target, reduction='none').mean(dim=-1)
            else:
                cycle_loss = self.yspace_loss(recon_latent, cycle_target, no_grad=False)
        else:
            cycle_loss = 0

        return for_pred_loss, cycle_loss

    def info_nce_loss(self, p_f_z, f_z_f, temperature=0.1, mask=None):
        """
        InfoNCE loss for batched sequence inputs.

        Inputs:
            p_f_z: [B, T, C] - predicted embedding
            f_z_f: [B, T, C] - target embedding
            temperature: float
            mask: [B, T] - optional binary mask (0=ignore, 1=keep)
        Returns:
            scalar InfoNCE loss
        """
        B, T, C = p_f_z.shape

        # === 1. Flatten and normalize ===
        if p_f_z.dim() == 3:
            B, T, C = p_f_z.shape
            p_f_z = p_f_z.reshape(B * T, C)
            f_z_f = f_z_f.reshape(B * T, C)
            if mask is not None:
                mask = mask.reshape(B * T)

        # normalize
        p = F.normalize(p_f_z, dim=-1)
        f = F.normalize(f_z_f, dim=-1)
        z = torch.cat([p, f], dim=0)  # [2N, C]

        # === 2. Cosine similarity matrix ===
        sim_matrix = torch.matmul(z, z.T) / temperature  # [2N, 2N]

        # === 3. Mask diagonal (self-similarity) ===
        N = p.size(0)
        eye_mask = torch.eye(2 * N, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(eye_mask, -1e9)

        # === 4. Create targets: i ↔ i + N
        targets = torch.arange(N, device=z.device)
        targets = torch.cat([targets + N, targets], dim=0)  # [2N]

        # === 5. Apply optional mask (step-wise valid)
        if mask is not None:

            valid = mask.reshape(B * T)  # [N]
            valid = valid.bool()

            valid = torch.cat([valid, valid], dim=0)  # [2N]
            sim_matrix = sim_matrix[valid][:, valid]
            targets = targets[valid]

        # === 6. Cross-entropy loss ===
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

    def _make_recurrent_cycle_rep(self, states, map_state, actions, next_states, next_map_state,
                                  mask=None, test=False, init_state=None):
        """
        Recurrent cycle-based representation loss:
        Predict forward latents with attention, then flip for backward cycle,
        and constrain final prediction to match ground truth embeddings.
        """
        B = states.size(0)
        next_states = next_states.unsqueeze(1)
        next_map_state = next_map_state.unsqueeze(1)
        # === Step 1: Encode each time step into latent (forward) ===
        f_z_seq = []
        for i in range(self.pred_step):
            if i == 0:
                latent, _ = self.encoder(states, mask=mask, test=test,
                                         init_state=init_state, map_state=map_state)
            else:
                latent, _ = self.encoder(next_states[:, i - 1], mask=mask, test=test,
                                         init_state=init_state, map_state=next_map_state[:, i - 1])
            f_z_seq.append(latent)
        f_z = torch.stack(f_z_seq, dim=1)[:, :, 0, :]  # [B, T, 128]

        # === Step 2: Get target representation (detach to stop grad) ===
        true_f_z = self._single_projection(f_z.detach())  # [B, T, D]

        # === Step 3: Encode actions and fuse with latent ===
        action_embed = self.action_layer(actions)  # [B, T, D_a]
        forward_input = torch.cat([f_z, action_embed], dim=-1)  # [B, T, D+D_a]

        # === Step 4: Forward recurrent attention prediction ===
        pred_forward = self._timestep_attention(forward_input, mask=(next_states[:, :, 0, 0, 0] != 0))
        pred_forward = self.embedder(pred_forward)
        pred_forward = self._single_projection(pred_forward)
        pred_forward = self.pred_layer(pred_forward)  # [B, T, D]

        # === Step 5: Reverse (cycle) prediction ===
        reversed_forward = torch.flip(pred_forward, dims=[1])
        reversed_action = torch.flip(action_embed, dims=[1])
        reverse_input = torch.cat([reversed_forward, reversed_action], dim=-1)
        step_mask = torch.flip((next_states[:, :, 0, 0, 0] != 0), dims=[1])  # ✅ 正确的 flip

        rep_f_z = self._timestep_attention(reverse_input, mask=step_mask)
        rep_f_z = self.embedder(rep_f_z)
        rep_f_z = self._single_projection(rep_f_z)
        rep_f_z = self.back_layer(rep_f_z)

        # === Step 6: Cosine similarity loss ===
        cosine_sim = self.similarity_loss(
            F.normalize(true_f_z, dim=-1),
            F.normalize(rep_f_z, dim=-1)
        )
        step_mask = (next_states[:, :, 0, 0, 0] != 0)

        step_mask = step_mask.float()
        loss = torch.mean(step_mask * (1 - cosine_sim))

        return loss

    def _make_recurrent_rep(self, states, map_state, actions, next_states, next_map_state, mask=None, test=False,
                            init_state=None):
        # (states.shape, next_states.shape, map_state.shape, next_map_state.shape)
        # torch.Size([32, 6, 10, 5])
        # torch.Size([32, 1, 6, 10, 5])
        # torch.Size([32, 18, 10, 2])
        # torch.Size([32, 1, 18, 10, 2])
        # actions  torch.Size([32, 1, 2])
        f_z, f_z_f = [], []
        next_states = next_states.unsqueeze(1)
        next_map_state = next_map_state.unsqueeze(1)
        for i in range(self.pred_step):
            if i == 0:
                f, _ = self.encoder(states, mask=mask, test=test, init_state=init_state, map_state=map_state)
            else:
                f, _ = self.encoder(next_states[:, i - 1], mask=mask, test=test, init_state=init_state,
                                    map_state=next_map_state[:, i - 1])
            f_z.append(f)
            if self.random_aug:
                f_n, _ = self.encoder(next_states[:, i], mask=mask, test=test, init_state=init_state,
                                      map_state=next_map_state[:, i], curr_frames=None)
            else:
                f_n, _ = self.target_encoder(next_states[:, i], mask=mask, test=test, init_state=init_state,
                                             map_state=next_map_state[:, i], curr_frames=None)
            f_z_f.append(f_n)
        f_z, f_z_f = torch.stack(f_z, dim=1), torch.stack(f_z_f, dim=1).detach()

        # torch.Size([32, 1, 1, 128])
        step_mask = (next_states[:, :, 0, 0, 0] != 0)
        f_z = f_z[:, :, 0, :]
        f_z_f = f_z_f[:, :, 0, :]
        f_a = self.action_layer(actions)
        p_f_z = torch.cat([f_z, f_a], dim=-1)
        # (32, 1, 192)



        # torch.Size([32, 1, 192])  torch.Size([32, 1])
        z = self._timestep_attention(p_f_z, mask=step_mask)
        z = self.embedder(z)
        # torch.Size([32, 1, 128])


        p_f_z, f_z_f = self._projection(z, f_z_f)
        p_f_z, f_z_f = self.pred_layer(p_f_z), f_z_f.detach()
        # (32, 1, 128)
        # ✅ 正则化：将两侧 embedding 单位归一化（L2）
        p_f_z = F.normalize(p_f_z, dim=-1)
        f_z_f = F.normalize(f_z_f, dim=-1)

        step_mask = step_mask.float()
        ince_loss = self.info_nce_loss(p_f_z, f_z_f, mask=step_mask, temperature=0.1)

        loss = torch.mean(step_mask * (1 - self.similarity_loss(f_z_f, p_f_z)))
        # loss = torch.mean(step_mask * self.similarity_loss(f_z_f, p_f_z))
        return loss, ince_loss

    def _projection(self, feat, feat_target):
        if self.random_aug:
            for layer in self.projection_layers:
                feat, feat_target = layer(feat), layer(feat_target)
        else:
            for layer, target_layer in zip(self.projection_layers, self.projection_layers_target):
                feat, feat_target = layer(feat), target_layer(feat_target)
        return feat, feat_target

    def _update_params(self, tau=5e-3):
        if self.random_aug:
            return
        for target_param, param in zip(self.projection_layers_target.parameters(), self.projection_layers.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def _single_projection(self, feat):
        if self.random_aug:
            for layer in self.projection_layers:
                feat = layer(feat)
        else:
            for layer, target_layer in zip(self.projection_layers, self.projection_layers_target):
                feat = layer(feat)
        return feat



    def forward(self, prev_obs, actions, next_obs, mask=None, test=False):
        lambda_cosine = 1.0
        lambda_infonce = 0.1   # InfoNCE量级~10x cosine，0.1使两者贡献相近
        lambda_cycle = 0.5     # cycle loss与cosine同量级，作为辅助时间一致性正则


        states = prev_obs['neighbor_trajs']
        map_state = prev_obs['neighbor_waypoints']
        init_state = prev_obs['ego_state']
        next_states = next_obs['neighbor_trajs']
        next_map_state = next_obs['neighbor_waypoints']

        loss, ince_loss = self._make_recurrent_rep(states, map_state, actions, next_states, next_map_state,
                                                   mask, test, init_state)
        cycle_loss = self._make_recurrent_cycle_rep(states, map_state, actions, next_states, next_map_state,
                                                    mask, test, init_state)
        simi_loss = lambda_cosine * loss + lambda_infonce * ince_loss + lambda_cycle * cycle_loss

        return simi_loss




class RLEncoder(nn.Module):
    def __init__(self, state_shape, action_dim, units=[256] * 3, hidden_activation="relu", name='rl_encoder',
                 lstm=False, trans=False, cnn_lstm=False, ego_surr=False,
                 use_trans=False, neighbours=5, time_step=8, debug=False, make_rotation=True, make_prediction=False,
                 use_mask=False, use_map=False, num_traj=5, cnn=False, path_length=0, head_dim=1, use_hier=False,
                 random_aug=False, no_ego_fut=False, no_neighbor_fut=False, carla=False):
        super().__init__()
        self.lstm = lstm
        self.cnn = cnn
        self.cnn_lstm = cnn_lstm
        self.ego_surr = ego_surr
        self.trans = trans
        self.debug = debug
        self.use_map = use_map
        self.neighbours = neighbours
        self.num_traj = num_traj
        self.use_mask = use_mask
        self.use_hier = use_hier

        print('Using Hierarchical Transformer')
        self.h_layer = Hierachial_Transformer(state_shape, units=128, use_trans_encode=True, num_heads=2,
                                                  drop_rate=0, neighbours=neighbours, make_rotation=make_rotation,
                                                  time_step=time_step, num_modes=num_traj,
                                                  final_head_num=head_dim, random_aug=random_aug,
                                                  no_ego_fut=no_ego_fut, no_neighbor_fut=no_neighbor_fut, carla=carla)

    def forward(self, states, mask=None, test=False, init_state=None, map_state=None, curr_frames=None, aug=True):
        # states [32, 6, 10, 5]
        # map_state [32, 18, 10, 2]
        if test:
            states, val = self.h_layer(states, test, map_state, aug)
            return states, val
        states = self.h_layer(states, test, map_state, aug)
        # [32, 1, 128]

        return states, None








