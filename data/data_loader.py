from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Dataset_wdxcjnb1(Dataset):
    def __init__(self, root_path, data_path, flag='train',
                 seq_len=96, pred_len=24,
                 scale=True, ratios=(0.7, 0.1, 0.2),
                 samplerate=1000,
                 gt_path=None, gt_with_lag=False):
        """
        wdxcjnb1: 逐通道独立 z-score 标准化

        新增参数:
        - gt_path     : str or None
            因果真值表 CSV 路径（相对于 root_path）
            格式: 有表头, 三列 [src, tgt, lag], src/tgt 从0开始, lag=0无延迟
        - gt_with_lag : bool
            False -> 构建 [C, C] 无延迟邻接矩阵（忽略 lag 列）
            True  -> 构建 [C, C, max_lag+1] 含延迟矩阵（最后一维=lag值, 0=无延迟）
        """
        assert flag in ['train', 'val', 'test', 'veri']
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.ratios = ratios
        self.samplerate = samplerate
        self.gt_path = gt_path
        self.gt_with_lag = gt_with_lag

        # 解析 ratios：支持 3 段或 4 段
        _r = self._parse_ratios(self.ratios)
        use_four = (len(_r) == 4)

        if use_four:
            type_map = {'train': 0, 'val': 1, 'test': 2, 'veri': 3}
        else:
            type_map = {'train': 0, 'val': 1, 'test': 2}

        if (not use_four) and (flag == 'veri'):
            raise ValueError("flag='veri' requires 4 ratios: 'train,val,test,veri'")

        self.set_type = type_map[flag]

        # 逐通道标准化用的统计量
        self.mean_ = None  # [1, C]
        self.std_ = None   # [1, C]

        # 因果真值（与 train/val/test 分割无关，全局共享）
        # gt_with_lag=False: np.ndarray [C, C]
        # gt_with_lag=True : np.ndarray [C, C, max_lag+1]，最后一维=lag值（0=无延迟）
        self.causal_gt = None
        self.causal_gt_sign = None  # [C,C] sign矩阵，+1=激活，-1=抑制，0=未知

        self.__read_data__()
        self.__read_gt__()

    def __read_data__(self):
        # 1) read csv
        # 标准格式：第一行为表头（ch0,ch1,...），行=时间点，列=通道
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        num_rows, num_cols = df_raw.shape

        # 检测格式：CSV 第一列通常是时间轴（行数 >> 列数）
        # 如果列数 > 行数，说明格式是 [C, T]（通道×时间），需要转置
        # 注意：只在行数明显少于列数时才转置，避免 C >= T 时误判
        if num_cols > num_rows * 2:
            df_raw = df_raw.T
            num_rows, num_cols = df_raw.shape

        # ensure purely numeric matrix
        df_data = df_raw.select_dtypes(include=[np.number]).copy()
        data = df_data.values.astype(np.float32)  # [T, C]

        num_total = len(df_data)

        # 2) split into train/val/test(/veri) in time order
        _r = self._parse_ratios(self.ratios)
        use_four = (len(_r) == 4)

        if use_four:
            r_train, r_val, r_test, r_veri = _r
            num_train = int(num_total * r_train)
            num_val   = int(num_total * r_val)
            num_test  = int(num_total * r_test)
            num_veri  = num_total - num_train - num_val - num_test

            border1s = [0,
                        num_train - self.seq_len,
                        num_train + num_val - self.seq_len,
                        num_train + num_val + num_test - self.seq_len]
            border2s = [num_train,
                        num_train + num_val,
                        num_train + num_val + num_test,
                        num_total]
        else:
            r_train, r_val, r_test = _r
            num_train = int(num_total * r_train)
            num_test  = int(num_total * r_test)
            num_val   = num_total - num_train - num_test

            border1s = [0,
                        num_train - self.seq_len,
                        num_total - num_test - self.seq_len]
            border2s = [num_train,
                        num_train + num_val,
                        num_total]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 逐通道标准化：每个通道各自用训练段统计量
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]  # [T_train, C]
            self.mean_ = train_data.mean(axis=0, keepdims=True)  # [1, C]
            self.std_  = train_data.std(axis=0, keepdims=True)   # [1, C]
            eps = 1e-6
            self.std_ = np.where(self.std_ < eps, 1.0, self.std_)
            data_scaled = (data - self.mean_) / self.std_
        else:
            data_scaled = data

        # 3) build synthetic time stamps: [T,1]
        total_times = data_scaled.shape[0]
        t_full = np.linspace(
            0,
            total_times / self.samplerate,
            num=total_times,
            endpoint=False,
            dtype='float32'
        )
        data_stamp = t_full[border1:border2]
        if data_stamp.ndim == 1:
            data_stamp = data_stamp.reshape(-1, 1)

        self.data_x = data_scaled[border1:border2]  # [T_split, C]
        self.data_y = self.data_x
        self.data_stamp = data_stamp  # [T_split, 1]

    def __read_gt__(self):
        """
        加载因果真值表（与时序 split 无关，全数据集共享）

        文件格式（CSV，有表头）:
            src,tgt,lag
            0,1,0        <- 变量0->变量1，无延迟 (lag=0)
            0,2,1        <- 变量0->变量2，延迟1步
            1,3,2        <- 变量1->变量3，延迟2步
        src/tgt: 0-indexed; lag=0 无延迟，lag=k 延迟 k 步
        """
        if self.gt_path is None:
            return

        full_path = os.path.join(self.root_path, self.gt_path)
        if not os.path.exists(full_path):
            print(f"[WARNING] GT file not found: {full_path}, skipping.")
            return

        df = pd.read_csv(full_path, header=0)
        if df.shape[1] < 2:
            raise ValueError(f"GT CSV must have at least 2 columns, got {df.shape[1]}")

        cols = df.iloc[:, :3].values.astype(int)   # [N, 2 or 3]
        has_lag_col = (cols.shape[1] >= 3)
        # ── 可选：读取 sign 列（第4列）──
        has_sign_col = (df.shape[1] >= 4)
        C = self.data_x.shape[-1]

        if not self.gt_with_lag:
            # ---- 无延迟：[C, C] ----
            mat = np.zeros((C, C), dtype=np.float32)
            for row in cols:
                src, tgt = int(row[0]), int(row[1])
                if src == tgt or not (0 <= src < C and 0 <= tgt < C):
                    continue
                mat[src, tgt] = 1.0
            self.causal_gt = mat
            print(f"[INFO] GT loaded (no-lag): shape={mat.shape}, "
                  f"positive_edges={int(mat.sum())}")
            # ── 可选：读取 sign 列（第4列）──
            if has_sign_col:
                sign_mat = np.zeros((C, C), dtype=np.float32)
                sign_vals = df.iloc[:, 3].values
                for idx_row, row in enumerate(df.iloc[:, :2].values.astype(int)):
                    src, tgt = int(row[0]), int(row[1])
                    if src == tgt or not (0 <= src < C and 0 <= tgt < C):
                        continue
                    sign_mat[src, tgt] = float(sign_vals[idx_row])
                self.causal_gt_sign = sign_mat
                print(f"[INFO] GT sign loaded: +1×{int((sign_mat>0).sum())}  -1×{int((sign_mat<0).sum())}")
        else:
            # ---- 含延迟：[C, C, max_lag+1]，最后一维=lag值 ----
            if not has_lag_col:
                raise ValueError(
                    "gt_with_lag=True but GT CSV has only 2 columns. "
                    "Please add a 'lag' column.")
            max_lag = int(cols[:, 2].max())
            mat = np.zeros((C, C, max_lag + 1), dtype=np.float32)
            skipped = 0
            for row in cols:
                src, tgt, lag = int(row[0]), int(row[1]), int(row[2])
                if src == tgt:
                    continue
                if not (0 <= src < C and 0 <= tgt < C) or lag > max_lag:
                    skipped += 1
                    continue
                mat[src, tgt, lag] = 1.0
            self.causal_gt = mat
            print(f"[INFO] GT loaded (with-lag): shape={mat.shape} [C,C,lag], "
                  f"positive_triplets={int(mat.sum())}, skipped={skipped}")
            # ── 可选：读取 sign 列（第4列），折叠为 [C,C]）──
            if has_sign_col:
                sign_mat = np.zeros((C, C), dtype=np.float32)
                sign_vals = df.iloc[:, 3].values
                for idx_row, row in enumerate(df.iloc[:, :2].values.astype(int)):
                    src, tgt = int(row[0]), int(row[1])
                    if src == tgt or not (0 <= src < C and 0 <= tgt < C):
                        continue
                    sign_mat[src, tgt] = float(sign_vals[idx_row])
                self.causal_gt_sign = sign_mat
                print(f"[INFO] GT sign loaded: +1×{int((sign_mat>0).sum())}  -1×{int((sign_mat<0).sum())}")

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end
        r_end   = r_begin + self.pred_len

        seq_x      = self.data_x[s_begin:s_end]
        seq_y      = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # data: [..., C]
        if (self.mean_ is None) or (self.std_ is None):
            return data
        return data * self.std_ + self.mean_

    def _parse_ratios(self, ratios):
        if isinstance(ratios, str):
            parts = ratios.split(',')
            if len(parts) not in (3, 4):
                raise ValueError("ratios must be 'a,b,c' or 'a,b,c,d'")
            parts = [float(x) for x in parts]
            if abs(sum(parts) - 1.0) > 1e-6:
                raise ValueError("ratios must sum to 1.")
            return parts
        elif isinstance(ratios, (list, tuple)):
            parts = list(map(float, ratios))
            if len(parts) not in (3, 4):
                raise ValueError("ratios list/tuple must have length 3 or 4.")
            if abs(sum(parts) - 1.0) > 1e-6:
                raise ValueError("ratios must sum to 1.")
            return parts
        else:
            raise TypeError("ratios must be str or list")