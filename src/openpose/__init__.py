import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import numpy as np
from . import util
from .body import Body
#from .hand import Hand
#from .face import Face
import jax.numpy as jnp

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')

body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
#hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
#face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
 #   faces = pose['faces']
 #   hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

 #   if draw_hand:
 #       canvas = util.draw_handpose(canvas, hands)

 #   if draw_face:
 #       canvas = util.draw_facepose(canvas, faces)

    return canvas


class OpenposeDetector:
    def __init__(self):
        body_modelpath = os.path.join(annotator_ckpts_path, "body_pose_model.pth")
 #       hand_modelpath = os.path.join(annotator_ckpts_path, "hand_pose_model.pth")
 #       face_modelpath = os.path.join(annotator_ckpts_path, "facenet.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(body_model_path, model_dir=annotator_ckpts_path)

#        if not os.path.exists(hand_modelpath):
#            from basicsr.utils.download_util import load_file_from_url
#            load_file_from_url(hand_model_path, model_dir=annotator_ckpts_path)

#        if not os.path.exists(face_modelpath):
#            from basicsr.utils.download_util import load_file_from_url
#            load_file_from_url(face_model_path, model_dir=annotator_ckpts_path)

        self.body_estimation = Body(body_modelpath)
#        self.hand_estimation = Hand(hand_modelpath)
#        self.face_estimation = Face(face_modelpath)

    def __call__(self, oriImg, hand_and_face=False, return_is_index=False):
        oriImg = jnp.array(oriImg)[:, :, :, ::-1]
        print(oriImg.shape)
        B, H, W, C = oriImg.shape
        candidates, subsets = self.body_estimation(oriImg)
        hands = []
        faces = []
        outputs = []
        for candidate, subset in zip(candidates, subsets):
            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)
            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            if return_is_index:
                outputs.append(pose)
            else:
                outputs.append(draw_pose(pose, H, W))
        return outputs
