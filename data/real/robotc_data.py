import numpy

from real_data import RealData
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class RobotcData(RealData):
    def __init__(self, args):
        super(RobotcData, self).__init__(args)
        self.name = 'robotc'
        self.train_dir = '/home/yi/code/video_motion_data/robot%d-contrast-train' % args.resolution
        self.test_dir = '/home/yi/code/video_motion_data/robot%d-contrast-test' % args.resolution
        self.train_images = self.get_meta(self.train_dir)
        self.test_images = self.get_meta(self.test_dir)
        if args.fixed_data:
            numpy.random.seed(args.seed)

    def get_next_batch(self, meta):
        im = self.generate_data(meta)
        return im


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = RobotcData(args)
    im = data.get_next_batch(data.train_images)
    data.display(im)


if __name__ == '__main__':
    unit_test()
