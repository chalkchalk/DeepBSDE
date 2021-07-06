import time
from solver import FeedForwardModel
from config import get_config
from equation import get_equation
import logging
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
import torch

# torch.backends.cudnn.benchmark=True

def train(config,bsde):
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s')
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)

    # build and train
    net = FeedForwardModel(config, bsde)
    # net.cuda()

    optimizer = optim.SGD(net.parameters(), config.lr_values)
    start_time = time.time()
    # to save iteration results
    training_history = []
    # for validation
    dw_valid, x_valid = bsde.sample(config.valid_size)
    loss_train_log = deque(maxlen=10000)
    y0log = deque(maxlen=10000)
    # begin sgd iteration
    for step in range(config.num_iterations + 1):
        if step % config.logging_frequency == 0:
            net.eval()
            # loss, init = net(x_valid.cuda(), dw_valid.cuda())
            loss, init = net(x_valid, dw_valid)
            elapsed_time = time.time() - start_time
            training_history.append([step, loss, init.item(), elapsed_time])
            loss_train_log.append(loss)
            y0log.append(init.item())
            plt.ion()  # 打开交互模式
            plt.clf()
            plt.subplot(121)
            plt.plot(loss_train_log, label='loss')
            plt.ylabel('loss')
            plt.ylim((0, 5))
            plt.xlabel('t')
            plt.legend()
            plt.subplot(122)
            plt.plot(y0log, label='y0')
            plt.ylabel('y0')
            plt.xlabel('t')
            plt.legend()
            plt.show()
            plt.pause(0.001)
            if config.verbose:
                logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                    step, loss, init.item(), elapsed_time))

        dw_train, x_train = bsde.sample(config.batch_size)
        # print(dw_train.shape, x_train.shape) # torch.Size([64, 100, 20]) torch.Size([64, 100, 21])
        optimizer.zero_grad()
        net.train()
        # loss, _ = net(x_train.cuda(), dw_train.cuda())
        loss, _ = net(x_train, dw_train)
        loss.backward()

        optimizer.step()

    training_history = np.array(training_history)

    if bsde.y_init:
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(
                         abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init))


    np.savetxt('{}_training_history.csv'.format(bsde.__class__.__name__),
                training_history,
                fmt=['%d', '%.5e', '%.5e', '%d'],
                delimiter=",",
                header="step,loss_function,target_value,elapsed_time",
                comments='')


if __name__ == '__main__':
    model = "LiMan2"
    # model = "AllenCahn"
    #cfg = get_config('AllenCahn')
    cfg = get_config(model)
    bsde = get_equation(model, cfg.dim, cfg.total_time, cfg.num_time_interval)
    train(cfg, bsde)
