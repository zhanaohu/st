import time
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from model.encoder import CoSTEncoder
from model.model import Dual_ACoST
from utils.otherutils import split_with_nan, centerize_vary_length_series
from utils.otherutils import PretrainDataset
from model.gan import Generator, Discriminator
import memory_profiler as memory
from tqdm import trange


def train(config: dict):
    train_data = config['train_data']
    batch_size = config['batch_size']
    lr = config['lr']
    epochs = config['epochs']
    input_dims = config['input_dims']
    output_dims = config['output_dims']
    kernels = config['kernels']
    alpha = config['alpha']
    max_train_length = config['max_train_length']
    hidden_dims = config['hidden_dims']
    depth = config['depth']
    device = config['device']
    num_workers = config['num_workers']
    length_queue = config['length_queue']
    similarity = config['similarity']

    gan_config = config['gan_config']

    encoder = CoSTEncoder(input_dims=input_dims, output_dims=output_dims, kernels=kernels, length=max_train_length,
                          hidden_dims=hidden_dims, depth=depth).to(device)

    model = Dual_ACoST(encoder, copy.deepcopy(encoder), dim=encoder.component_dims, alpha=alpha, K=length_queue).to(
        device)

    assert train_data.ndim == 3

    if max_train_length is not None:
        sections = train_data.shape[1] // max_train_length
        if sections >= 2:
            train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
            pass

    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)
        pass
    train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
    multiplier = 1 if train_data.shape[0] >= batch_size else math.ceil(batch_size / train_data.shape[0])

    gan_config['data'] = torch.from_numpy(np.float32(train_data))
    gan_config['sections'] = sections
    gen_model = train_gan(config=gan_config)
    train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), sigma=0.5, gen_model=gen_model,
                                    multiplier=multiplier, similarity=similarity, device=device)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True,
                              drop_last=True, num_workers=num_workers)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=lr, momentum=0.9, weight_decay=1e-4)
    loss_log = []
    for epoch in trange(epochs, desc="Training Task2 -- Representation Learning"):
        sum_loss = 0
        n_epoch_iter = 0
        for batch in train_loader:
            x_q, x_k = map(lambda x: x.to(device), batch)
            if max_train_length is not None and x_q.size(1) > max_train_length:
                window_offset = np.random.randint(x_q.size(1) - max_train_length + 1)
                x_q = x_q[:, window_offset: window_offset + max_train_length]
                pass
            if max_train_length is not None and x_k.size(1) > max_train_length:
                window_offset = np.random.randint(x_k.size(1) - max_train_length + 1)
                x_k = x_k[:, window_offset: window_offset + max_train_length]
                pass
            optimizer.zero_grad()
            loss = model(x_q, x_k)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            n_epoch_iter += 1
            pass
        sum_loss /= n_epoch_iter
        loss_log.append(sum_loss)
        pass
    return model, loss_log


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        pass
    return lr


def train_gan(config):
    data = config['data']
    dropout = config['dropout']
    device = config['device']
    lr_g = config['lr_g']
    lr_d = config['lr_d']
    epochs = config['epochs']

    num_ts, seq_len, dim = data.shape

    data = data.reshape(num_ts * seq_len, dim)
    data_np = data.numpy()
    data_np[np.isnan(data_np)] = 0.0
    data = torch.from_numpy(data_np)

    generator1 = Generator(output_size=dim, device=device)
    generator2 = Generator(output_size=dim, device=device)
    discriminator1 = Discriminator(input_size=dim, dropout=dropout, device=device)
    discriminator2 = Discriminator(input_size=dim, dropout=dropout, device=device)

    optimizer_g = torch.optim.Adam(list(generator1.parameters()) + list(generator2.parameters()), lr=lr_g)
    optimizer_d = torch.optim.Adam(list(discriminator1.parameters()) + list(discriminator2.parameters()), lr=lr_d)

    adversarial_loss = nn.BCELoss()
    l1 = nn.L1Loss()

    for epoch in trange(epochs, desc="Training Task1 -- CycleGAN"):
        real_label = torch.ones((data.shape[0], 1)).to(device)
        fake_label = torch.zeros((data.shape[0], 1)).to(device)
        real_data = data.to(device)

        # train discriminator
        z = torch.randn(data.shape[0], 512).to(device)
        fake_data_1 = generator1(z).detach().to(device)
        d1_loss_real = adversarial_loss(discriminator1(real_data), real_label)
        d1_loss_fake = adversarial_loss(discriminator1(fake_data_1), fake_label)
        d1_loss = d1_loss_fake + d1_loss_real
        fake_data_1 = nn.Linear(fake_data_1.shape[-1], 512).to(device)(fake_data_1)
        fake_data_2 = generator2(fake_data_1).detach()
        d2_loss_real = adversarial_loss(discriminator2(real_data), real_label)
        d2_loss_fake = adversarial_loss(discriminator2(fake_data_2), fake_label)
        d2_loss = d2_loss_fake + d2_loss_real
        d_loss = (d1_loss + d2_loss) / 2
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # train generator
        z = torch.randn(data.shape[0], 512).to(device)
        g_fake_data = generator1(z)
        g1_loss = adversarial_loss(discriminator1(g_fake_data), real_label)
        g_fake_data = nn.Linear(g_fake_data.shape[-1], 512).to(device)(g_fake_data)
        g2_loss = adversarial_loss(discriminator2(generator2(g_fake_data)), real_label)

        cycle_loss = l1(generator2(g_fake_data), real_data)

        g_loss = g1_loss + g2_loss + cycle_loss
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        pass

    return generator1
