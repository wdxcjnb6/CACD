from torch.utils.data import DataLoader
from data.data_loader import Dataset_wdxcjnb1 # ensure imported correctly

data_dict = {
    'wdxcjnb1': Dataset_wdxcjnb1
}

def data_provider(args, flag):
    """
    flag ∈ {train, val, test, veri, pred}

    数据顺序由 Dataset 决定：
    train → val → test → veri
    """

    Data = data_dict[args.data]

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
    elif flag == 'val':
        shuffle_flag = True
        drop_last = True
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
    elif flag == 'veri':
        shuffle_flag = False
        drop_last = False
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
    else:
        raise ValueError(f"Unknown flag {flag}")

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        scale=args.scale,
        ratios=args.ratios,
        samplerate=args.samplerate,
        gt_path=getattr(args, 'gt_path', None),
        gt_with_lag=getattr(args, 'gt_with_lag', False),
    )

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader