import os
import numpy
from PIL import Image
import cv2

from synthetic2_data import Synthetic2Data
import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class ChairData(Synthetic2Data):
    def __init__(self, args):
        super(ChairData, self).__init__(args)
        self.name = 'chair'
        self.fg_dir = '/home/yi/code/video_motion_data/rendered_chairs'
        self.bg_dir = '/media/yi/DATA/data-orig/non_person_images'
        self.train_images = self.get_meta(self.fg_dir, self.bg_dir)
        self.test_images = self.get_meta(self.fg_dir, self.bg_dir)
        self.m_range = args.motion_range
        self.resolution = args.resolution
        if args.fixed_data:
            numpy.random.seed(args.seed)

    def get_meta(self, fg_dir, bg_dir):
        meta = {}
        # get foreground image meta
        meta['fg'] = []
        for sub_dir in os.listdir(fg_dir):
            for sub_sub_dir in os.listdir(os.path.join(fg_dir, sub_dir)):
                image_files = os.listdir(os.path.join(fg_dir, sub_dir, sub_sub_dir))
                image_files.sort(key=lambda f: int(filter(str.isdigit, f)))
                image_names = [os.path.join(fg_dir, sub_dir, sub_sub_dir, f) for f in image_files]
                for image_name in image_names:
                    meta['fg'].append(image_name)
        # get background image meta
        image_files = os.listdir(bg_dir)
        image_names = [os.path.join(bg_dir, f) for f in image_files]
        meta['bg'] = image_names
        return meta

    def generate_source_image(self, meta):
        max_shift = self.im_size / 3
        batch_size, num_objects, im_size = self.batch_size, self.num_objects, self.im_size
        im_channel = self.im_channel
        # generate source foreground images and their masks
        src_fg = numpy.zeros((num_objects, batch_size, self.im_channel, im_size, im_size))
        src_mask = numpy.zeros((num_objects, batch_size, 1, im_size, im_size))
        image_names = meta['fg']
        for i in range(num_objects):
            idx = numpy.random.permutation(len(image_names))
            for j in range(batch_size):
                if im_channel == 1:
                    image = numpy.array(Image.open(image_names[idx[j]]).convert('L'))
                elif im_channel == 3:
                    image = numpy.array(Image.open(image_names[idx[j]]))
                image = cv2.resize(image, (self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
                if im_channel == 1:
                    image = numpy.expand_dims(image, 3)
                image = image / 255.0
                image = image.transpose((2, 0, 1))
                src_fg[i, j, :, :, :] = image
                if im_channel == 1:
                    nonzero_mask = src_fg[i, j, :, :, :].sum(0) < 0.75
                elif im_channel == 3:
                    nonzero_mask = src_fg[i, j, :, :, :].sum(0) < 2.25
                src_mask[i, j, 0, nonzero_mask] = num_objects - i
                shift = numpy.random.randint(-max_shift, max_shift, size=2)
                src_fg[i, j, :, :, :] = self.shift_image(src_fg[i, j, :, :, :], shift, max_shift)
                src_mask[i, j, :, :, :] = self.shift_image(src_mask[i, j, :, :, :], shift, max_shift)
        # generate source bg image
        m_range, num_frame = self.m_range, self.num_frame
        min_im_size = im_size + 2 * (num_frame - 1) * m_range
        src_bg = numpy.zeros((batch_size, self.im_channel, min_im_size, min_im_size))
        image_names = meta['bg']
        idx = numpy.random.permutation(len(image_names))
        i, j = 0, 0
        while i < batch_size:
            if im_channel == 1:
                image = numpy.array(Image.open(image_names[idx[j]]).convert('L'))
                image = numpy.expand_dims(image, 3)
            elif im_channel == 3:
                image = numpy.array(Image.open(image_names[idx[j]]))
            image = image / 255.0
            height, width = image.shape[0], image.shape[1]
            if height < min_im_size or width < min_im_size:
                j += 1
                continue
            idx_h = numpy.random.randint(0, height + 1 - min_im_size)
            idx_w = numpy.random.randint(0, width + 1 - min_im_size)
            image = image.transpose((2, 0, 1))
            src_bg[i, :, :, :] = image[:, idx_h:idx_h + min_im_size, idx_w:idx_w + min_im_size]
            i += 1
        return src_fg, src_mask, src_bg

    @staticmethod
    def shift_image(im, shift, max_shift):
        [im_channel, im_height, im_width] = im.shape
        im_big = numpy.zeros((im_channel, im_height + 2 * max_shift, im_width + 2 * max_shift))
        im_big[:, max_shift:-max_shift, max_shift:-max_shift] = im
        x = max_shift + shift[0]
        y = max_shift + shift[1]
        im = im_big[:, y:y + im_height, x:x + im_width]
        return im

    def get_next_batch(self, meta):
        src_fg, src_mask, src_bg = self.generate_source_image(meta)
        im, motion, motion_label, seg_layer = self.generate_data(src_fg, src_mask, src_bg)
        return im, motion, motion_label, seg_layer


def unit_test():
    args = learning_args.parse_args()
    logging.info(args)
    data = ChairData(args)
    im, motion, motion_label, seg_layer = data.get_next_batch(data.train_images)
    data.display(im, motion, seg_layer)


if __name__ == '__main__':
    unit_test()
