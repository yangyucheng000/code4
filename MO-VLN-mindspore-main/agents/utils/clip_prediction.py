import numpy as np
from PIL import Image
import mindspore as ms
from mindformers import pipeline


class SemanticPredCLIP():

    def __init__(self, args, categories):
        self.categories = categories
        self.num_sem_categories = args.num_sem_categories

        self.classifier = pipeline("zero_shot_image_classification",
                                   model='clip_vit_b_32',
                                   candidate_labels=categories)

        self.thresh = args.det_thresh

    def get_prediction(self, img):
        h, w, _ = img.shape
        img = img.transpose(1, 0, 2)

        try:
            predict_result = self.classifier(ms.Tensor(img))[0]
        except Exception as e:
            print(e)
            predict_result = self.classifier(ms.Tensor(img))[0]

        semantic_input = np.zeros((h, w, self.num_sem_categories))

        for result in predict_result:
            label = result['label']
            score = result['score']

            if score > self.thresh:
                class_idx = self.categories.index(label)
                semantic_input[..., class_idx] = score

        return semantic_input, img

