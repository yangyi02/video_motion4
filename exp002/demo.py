import os
import sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from learning_args import parse_args
from base_demo import BaseDemo
from model import Net, GtNet
from visualizer import Visualizer
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class Demo(BaseDemo):
    def __init__(self, args):
        super(Demo, self).__init__(args)
        self.model, self.model_gt = self.init_model(self.data.m_kernel)
        self.visualizer = Visualizer(args, self.data.reverse_m_dict)

    def init_model(self, m_kernel):
        self.model = Net(self.im_size, self.im_size, self.im_channel, self.num_frame - 1,
                         m_kernel.shape[1], self.m_range, self.net_depth)
        self.model_gt = GtNet(self.im_size, self.im_size, self.im_channel, self.num_frame - 1,
                              m_kernel.shape[1], self.m_range)
        if torch.cuda.is_available():
            # model = torch.nn.DataParallel(model).cuda()
            self.model = self.model.cuda()
            self.model_gt = self.model_gt.cuda()
        if self.init_model_path is not '':
            self.model.load_state_dict(torch.load(self.init_model_path))
        return self.model, self.model_gt

    def train(self):
        writer = SummaryWriter(self.tensorboard_path)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        base_loss_all, train_loss_all = [], []
        for epoch in range(self.train_epoch):
            optimizer.zero_grad()
            if self.data.name in ['box', 'mnist', 'chair']:
                im, _, _, _ = self.data.get_next_batch(self.data.train_images)
            elif self.data.name in ['robot', 'mpii', 'viper', 'kitti', 'robotc']:
                im = self.data.get_next_batch(self.data.train_images)
            else:
                logging.error('%s data not supported' % self.data.name)
                sys.exit()
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
            im_pred, flow = self.model(im_input, im_output)
            im_diff = im_pred - im_input[:, -self.im_channel:, :, :]  # inverse warping loss
            loss = torch.abs(im_diff).sum() / self.batch_size
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss', loss.data[0], epoch)
            train_loss_all.append(loss.data[0])
            if len(train_loss_all) > 100:
                train_loss_all.pop(0)
            ave_train_loss = sum(train_loss_all) / float(len(train_loss_all))
            im_base = im_input[:, -self.im_channel:, :, :]
            base_loss = torch.abs(im_base - im_output).sum() / self.batch_size
            base_loss_all.append(base_loss.data[0])
            if len(base_loss_all) > 100:
                base_loss_all.pop(0)
            ave_base_loss = sum(base_loss_all) / float(len(base_loss_all))
            logging.info('epoch %d, train loss: %.2f, average train loss: %.2f, base loss: %.2f',
                         epoch, loss.data[0], ave_train_loss, ave_base_loss)
            if (epoch+1) % self.save_interval == 0:
                logging.info('epoch %d, saving model', epoch)
                with open(os.path.join(self.save_dir, '%d.pth' % epoch), 'w') as handle:
                    torch.save(self.model.state_dict(), handle)
            if (epoch+1) % self.test_interval == 0:
                logging.info('epoch %d, testing', epoch)
                test_loss, test_epe = self.validate()
                writer.add_scalar('test_loss', test_loss, epoch)
                if test_epe is not None:
                    writer.add_scalar('test_epe', test_epe, epoch)
        writer.close()

    def test(self):
        base_loss_all, test_loss_all = [], []
        test_epe_all = []
        motion = None
        for epoch in range(self.test_epoch):
            if self.data.name in ['box', 'mnist', 'chair']:
                im, motion, _, _ = self.data.get_next_batch(self.data.test_images)
            elif self.data.name in ['robot', 'mpii', 'viper', 'kitti', 'robotc']:
                im, motion = self.data.get_next_batch(self.data.test_images), None
            elif self.data.name in ['mpii_sample', 'kitti_sample']:
                im, motion = self.data.get_next_batch(self.data.test_images), None
                im = im[:, -self.num_frame:, :, :, :]
            else:
                logging.error('%s data not supported' % self.data.name)
                sys.exit()
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float(), volatile=True)
            im_output = Variable(torch.from_numpy(im_output).float(), volatile=True)
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
            im_pred, flow = self.model(im_input, im_output)
            flow = flow * self.im_size / 2  # resize flow from [-1, 1] back to image scale
            im_diff = im_pred - im_input[:, -self.im_channel:, :, :]  # inverse warping loss
            loss = torch.abs(im_diff).sum() / self.batch_size

            test_loss_all.append(loss.data[0])
            im_base = im_input[:, -self.im_channel:, :, :]
            base_loss = torch.abs(im_base - im_output).sum() / self.batch_size
            base_loss_all.append(base_loss.data[0])

            if motion is None:
                gt_motion = None
            else:
                gt_motion = motion[:, -2, :, :, :]
                gt_motion = Variable(torch.from_numpy(gt_motion).float())
                if torch.cuda.is_available():
                    gt_motion = gt_motion.cuda()
                epe = (flow - gt_motion) * (flow - gt_motion)
                epe = torch.sqrt(epe.sum(1))
                epe = epe.sum() / epe.numel()
                test_epe_all.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                 'test_%d.png' % epoch)
            if self.display_all:
                for i in range(self.batch_size):
                    self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                     'test_%d.png' % i, i)
        test_loss = numpy.mean(numpy.asarray(test_loss_all))
        base_loss = numpy.mean(numpy.asarray(base_loss_all))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        if motion is not None:
            test_epe = numpy.mean(numpy.asarray(test_epe_all))
            logging.info('average test endpoint error: %.2f', test_epe)
        else:
            test_epe = None
        return test_loss, test_epe, improve_percent

    def test_gt(self):
        base_loss_all, test_loss_all = [], []
        test_epe_all = []
        for epoch in range(self.test_epoch):
            if self.data.name in ['box', 'mnist', 'chair']:
                im, motion, _, _ = self.data.get_next_batch(self.data.test_images)
            else:
                logging.error('%s data not supported in test_gt' % self.data.name)
                sys.exit()
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            gt_motion = motion[:, -2, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float(), volatile=True)
            im_output = Variable(torch.from_numpy(im_output).float(), volatile=True)
            gt_motion = Variable(torch.from_numpy(gt_motion).float(), volatile=True)
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
                gt_motion = gt_motion.cuda()
            if self.data.name in ['box', 'mnist', 'chair']:
                im_pred, flow = self.model_gt(im_input, im_output, gt_motion)
                flow = flow * self.im_size / 2  # resize flow from [-1, 1] back to image scale
            im_diff = im_pred - im_input[:, -self.im_channel:, :, :]  # inverse warping loss
            loss = torch.abs(im_diff).sum() / self.batch_size

            test_loss_all.append(loss.data[0])
            im_base = im_input[:, -self.im_channel:, :, :]
            base_loss = torch.abs(im_base - im_output).sum() / self.batch_size
            base_loss_all.append(base_loss.data[0])
            epe = (flow - gt_motion) * (flow - gt_motion)
            epe = torch.sqrt(epe.sum(1))
            epe = epe.sum() / epe.numel()
            test_epe_all.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                 'test_gt.png')
            if self.display_all:
                for i in range(self.batch_size):
                    self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                     'test_gt_%d.png' % i, i)
        test_loss = numpy.mean(numpy.asarray(test_loss_all))
        base_loss = numpy.mean(numpy.asarray(base_loss_all))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average ground truth test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        test_epe = numpy.mean(numpy.asarray(test_epe_all))
        logging.info('average ground truth test endpoint error: %.2f', test_epe)
        return improve_percent


def main():
    args = parse_args()
    logging.info(args)
    demo = Demo(args)
    if args.train:
        demo.train()
    if args.test:
        demo.test()
    if args.test_gt:
        demo.test_gt()
    if args.test_all:
        demo.test_all()


if __name__ == '__main__':
    main()
