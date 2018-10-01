import os, pickle, random, math, shutil
import numpy as np
import cv2

def image_preprocessing(image, args):
    image = cv2.resize(image, (args.IMAGE_WIDTH, args.IMAGE_HEIGHT))
    image = image.astype(np.float32) / 255. - 0.5
    image = np.reshape(image, (image.shape[0], image.shape[1], args.IMAGE_CHANNELS))
    return image


class MasterLoader:
    def __init__(self, main_loader, sub_loader):
        self.main_loader = main_loader
        self.sub_loader = sub_loader

        self.is_main_turn = True

    def size(self):
        return self.main_loader.size() + self.sub_loader.size()

    def get_batch(self):
        if self.is_main_turn:
            self.is_main_turn = False
            return self.main_loader.get_batch(), True
        else:
            self.is_main_turn = True
            return self.sub_loader.get_batch(), False

class RecognizeDataLoader:
    def __init__(self, voca_i2c, voca_c2i, data_dir, pickle_name, args):
        # data path
        self.data_dir = data_dir

        # args
        self.args = args

        # make voca
        self.voca_i2c = voca_i2c
        self.voca_c2i = voca_c2i

        # load data
        with open("{}/{}.pkl".format(self.data_dir, pickle_name), 'rb') as f:
            data = pickle.load(f)
        self.data = data

        # initialize
        self.init()

    def init(self):
        self.iteration = 0
        self.image_paths = list(self.data.keys())
        random.shuffle(self.image_paths)

    def size(self):
        return len(self.image_paths)

    def get_batch(self):
        images = []
        dummy_masks = []
        labels = np.zeros([self.args.BATCH_SIZE, self.args.SEQ_LENGTH], np.int32)

        idx = 0
        while True:
            if self.iteration >= len(self.image_paths):
                # end of batch
                self.init()
                self.iteration = 0
                return None
            try:
                image_path = self.image_paths[self.iteration]
                self.iteration += 1
                if len(list(self.data[image_path])) > self.args.SEQ_LENGTH:
                    continue

                image = cv2.imread(os.path.join(self.data_dir, image_path), cv2.IMREAD_GRAYSCALE)
                image = image_preprocessing(image, args=self.args)

            except:
                print("Failed to open {}".format(os.path.join(self.data_dir, image_path)))
                continue

            images.append(image)
            annotation = list(self.data[image_path].lower())
            for char_idx, char in enumerate(annotation):
                labels[idx][char_idx] = self.voca_c2i[char]
           
            idx += 1
            if idx >= self.args.BATCH_SIZE:
                break

        images = np.array(images)
        dummy_masks = np.zeros_like(images)
        # labels = np.array(labels)
        return images, dummy_masks, labels

if __name__ == "__main__":
    voca_i2c = list("abcdefghijklmnopqrstuvwxyz0123456789")
    tmp = voca_i2c[0]
    voca_i2c[0] = 'None'
    voca_i2c += [tmp, 'SOS']
    voca_c2i = {c: i for i, c in enumerate(voca_i2c)}
    
    # Data
    train_data = RecognizeDataLoader(voca_i2c=voca_i2c, voca_c2i=voca_c2i, data_dir='train_data/data_recognition_11.6M', pickle_name='train')
    images, masks, labels = train_data.get_batch()
    print("images", images.shape)
    print("masks", masks.shape)
    print("labels", labels.shape)
    print("labels[0]", labels[0])
    print("masks[0]", masks[:, :32])