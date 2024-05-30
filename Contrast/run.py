import dgl
import yaml
import json
import time
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from loguru import logger
from model import GraphEncoder
from moco import MemoryMoCo
from criterion import NCESoftmaxLoss
from dataset import GraphDataset, batcher
from utils import moment_update, clip_grad_norm, warmup_linear, adjust_learning_rate, set_bn_train


warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, model, model_ema, contrast, criterion, optimizer, config):
    n_batch = loader.dataset.total // config['batch_size']
    model.train()
    model_ema.eval()
    model_ema.apply(set_bn_train)

    loss_buffer = list()
    time_end = time.time()
    pro_bar = tqdm(enumerate(loader), total=len(loader))
    for index, batch in pro_bar:
        time_data = time.time() - time_end
        batch_graph_q, batch_graph_k = batch
        for graph_q, graph_k in zip(batch_graph_q, batch_graph_k):
            graph_q.to(torch.device(config['gpu']))
            graph_k.to(torch.device(config['gpu']))

        # Forward
        feat_q = model(batch_graph_q)
        with torch.no_grad():
            feat_k = model_ema(batch_graph_k)

        out = contrast(feat_q, feat_k, config['gpu'])
        prob = out[:, 0].mean()

        # Backward
        optimizer.zero_grad()
        loss = criterion(out, config['gpu'])
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), config['clip_norm'])

        global_step = epoch * n_batch + index
        lr_this_step = config['learning_rate'] * warmup_linear(
            global_step / (config['epochs'] * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        moment_update(model, model_ema, config['alpha'])

        torch.cuda.synchronize()
        time_batch = time.time() - time_end
        time_end = time.time()

        loss_buffer.append(loss.item())
        pro_bar.set_description(f"[Epoch {epoch}, batch {index + 1}/{n_batch}]")
        pro_bar.set_postfix_str(f"loss: {loss.item():.3f}, prob: {prob.item():.3f}, lr: {lr_this_step:.6f}")
    return sum(loss_buffer) / len(loss_buffer)


def main(dataset, gpu):
    # load config
    config = dict()
    with open(f'config/{dataset}.yaml', 'r', encoding='utf-8') as f:
        config.update(yaml.safe_load(f.read()))
    config['gpu'] = int(gpu)

    logger.add(config['log_path'])
    logger.info(json.dumps(config, indent=4))

    # random seed
    dgl.random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # dataset
    dataset = GraphDataset(
        graph_path=config['graph_path'],
        meta_path=config['meta_path'],
        random_walk_hops=config['random_walk_hops'],
        restart_prob=config['restart_prob'],
        sample_num=config['sample_num']
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=batcher,
    )

    model, model_ema = [
        GraphEncoder(
            n_embedding=dataset.n_embedding,
            emb_dim=config['emb_dim'],
            n_relation=config['n_relation'],
            n_bases=config['n_bases']
        )
        for _ in range(2)
    ]

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        input_size=config['emb_dim'],
        output_size=None,
        queue_size=config['queue_size'],
        temperature=config['temperature'],
        use_softmax=config['use_softmax']
    ).cuda(config['gpu'])

    criterion = NCESoftmaxLoss().cuda(config['gpu'])
    model = model.cuda(config['gpu'])
    model_ema = model_ema.cuda(config['gpu'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay']
    )

    start_epoch = 1
    lr = config['learning_rate']
    lr_decay_rate = config['lr_decay_rate']
    lr_decay_epochs = config['lr_decay_epochs']
    for epoch in range(start_epoch, config['epochs']):
        adjust_learning_rate(epoch, lr, lr_decay_rate, lr_decay_epochs, optimizer)
        time_start = time.time()
        loss = train_epoch(epoch, loader, model, model_ema, contrast, criterion, optimizer, config)
        time_end = time.time()
        logger.info(f"[Epoch {epoch} finished, total time {time_end - time_start:.2f}, loss {loss:.6f}]")

        # save checkpoint
        state = {
            'config': config,
            'encoder': model.encoder.state_dict(),
            'epoch': epoch
        }
        save_file = f"{config['checkpoint_path']}/{epoch}-epoch.pth"
        torch.save(state, save_file)
        del state
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='redial')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args, _ = parser.parse_known_args()
    main(args.dataset, args.gpu)
