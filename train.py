import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter

from utils import data_importer, get_logger, netflix_reviews, collate_fn
from models import sentiment_analysis
from args import get_train_args

def main(args):
    logger = get_logger(logging_dir=args.logging_dir, name=args.exp_name)

    logger.info(f"Saving data and metadata from {args.exp_name} do {args.save_dir}")

    tbx = SummaryWriter(args.save_dir)

    logger.info(f"Using random seed {args.random_seed}...")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    logger.info(f"Using device type: {device}")

    train_dataset = netflix_reviews(data_path=args.clean_train_data)
    train_loader = data.DataLoader(train_dataset,
                                   shuffle=True, 
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn)
    test_dataset = netflix_reviews(data_path=args.clean_test_data)
    test_loader = data.DataLoader(test_dataset,
                                   shuffle=True, 
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn)

    # model stuff happens here
    model = sentiment_analysis(args=args)
    model = nn.DataParallel(model, gpu_ids)
    model.to(device)
    model.train()

    # optimizer happens here

    for epoch in range(args.num_epochs):
        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for ids, rw_tokens, sentiments, lengths in train_loader:
                res = model(rw_tokens, lengths)
                break
                progress_bar.update(args.batch_size)
                progress_bar.set_postfix(Epoch=(epoch + 1))
        break






if __name__ == "__main__":
    args = get_train_args()
    main(args)
    