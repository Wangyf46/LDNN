import os
import sys
sys.path.insert(0, '/home/wangyf/codes/LDNN')
import time
import argparse
import ipdb
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from src.model import UNet
from src.DIV2K import DIV2K
from src.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train_net(args, net):
    DATE = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    dir_log = os.path.join('log', DATE)
    dir_checkpoint = os.path.join('checkpoints/', DATE)
    if not os.path.isdir(dir_log):
        os.makedirs(dir_log)
    log_file = open(os.path.join(dir_log, 'record.txt'), 'w')
    tblogger = SummaryWriter(dir_log)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.ToTensor()])
                                    # transforms.Normalize(mean, std)])

    listDataset = DIV2K(args,
                        transform)

    train_loader = DataLoader(listDataset,
                              batch_size=args.bz,
                              shuffle=True,
                              pin_memory=True)

    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.999),
                           eps=1e-08)

    # criterion = nn.MSELoss(size_average=True)
    criterion = SANetLoss(3).cuda()

    itr = 0
    max_itr = args.epochs * len(train_loader)
    print(itr, max_itr, len(train_loader))
    net.train()
    for epoch in range(args.epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))

        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        for i_batch, (Iin, LD, name) in enumerate(train_loader):
            data_time.update(time.time() - end)       # measure batch_size data loading time
            now_lr = adjust_lr(optimizer, epoch, args.lr)
            Iin = Iin.cuda()                          # torch.float32, [0.0-1.0]
            LD = LD.cuda()                            # torch.float32-[0.0-255.0]
            Icp = net(Iin)                            # torch.float32-[0.0-1.0]            TODO
            Iout = get_Iout(Icp, LD)                  # torch.float32-[0.0-1.0]            TODO



            loss = criterion(Iout, Iin)
            losses.update(loss.item(), args.bz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            print_str = 'Epoch: [{0}/{1}]\t'.format(epoch, args.epochs)
            print_str += 'Batch: [{0}]/{1}\t'.format(i_batch + 1, listDataset.__len__() // args.bz)
            print_str += 'LR: {0}\t'.format(now_lr)
            print_str += 'Data time {data_time.cur:.3f}({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_str += 'Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
            print_str += 'Loss {loss.cur:.4f}({loss.avg:.4f})\t'.format(loss=losses)
            log_print(print_str, log_file, color="green", attrs=["bold"])


            # cv2.imshow('Iin', Iin.detach().cpu().numpy()[0].transpose((1,2,0)))
            # cv2.imshow('LD', np.uint8(LD.detach().cpu().numpy()[0]))
            # cv2.imshow('Icp', Icp.detach().cpu().numpy()[0].transpose((1,2,0)))
            # cv2.imshow('Iout', Iout.detach().cpu().numpy()[0].transpose((1, 2, 0)))
            # cv2.waitKey(0)

            ## torch.float32-CHW-[0.0-1.0]
            tblogger.add_scalar('loss', losses.avg, itr)
            tblogger.add_scalar('lr', now_lr, itr)
            # tblogger.add_image('Iin', Iin[0].cpu(), itr)   ## CHW
            # tblogger.add_image('Icp', Icp[0].cpu(), itr)
            # tblogger.add_image('Iout', Iout[0].cpu(), itr)

            end = time.time()
            itr += 1
        if not os.path.isdir(dir_checkpoint):
            os.makedirs(dir_checkpoint)
        save_path = os.path.join(dir_checkpoint, '%s_%s_itr%d.pth' % (DATE, epoch, itr))
        torch.save(net.state_dict(), save_path)
        print('%s has been saved' % save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LDNN Train")
    parser.add_argument('--period', default='train')
    parser.add_argument('--path', default='/home/wangyf/datasets')
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--bz', default=2)
    parser.add_argument('--lr',  default=0.0001)
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--pre', default=False, help='load file model')
    parser.add_argument('--base_size', type=int, default=[1080, 1920])
    parser.add_argument('--block_size', type=int, default=[9, 16])
    parser.add_argument('--bl', type=str, default='LUT')
    parser.add_argument('--BMA', type=int, default=4)
    args = parser.parse_args()

    ## 固定随机种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    net = UNet(3, 3)
    # net = nn.DataParallel(net)

    if args.pre:
        net.load_state_dict(torch.load(args.pre))
        print('Model loaded from {}'.format(args.pre))
    if args.gpu:

        net.cuda()
    try:
        train_net(args, net)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)