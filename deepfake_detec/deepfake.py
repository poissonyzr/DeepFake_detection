import os
import math
import pickle
from functools import partial
from collections import defaultdict

from PIL import Image
from glob import glob
import cv2
import numpy as np
import skimage.measure
import albumentations as A
from tqdm import tqdm 
from albumentations.pytorch.transforms import ToTensor 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.video import mc3_18, r2plus1d_18

from facenet_pytorch import MTCNN
import logging
from re import S
import numpy
import yaml
from factory.builder import build_model, build_dataloader
import towhee
from towhee.operator.base import NNOperator, OperatorFlag
from towhee.types.arg import arg, to_image_color
from towhee import register

from PIL import Image as PILImage

import warnings

warnings.filterwarnings('ignore')
log = logging.getLogger()

@register(output_schema=['list'],
          flag=OperatorFlag.STATELESS | OperatorFlag.REUSEABLE)

class DeepFake2D(NNOperator):
    '''
    Args:
        video_path (`str`):
            Video path in string.
        
        Returns:
            ('Tuple[('label','score')])

    '''
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open('./experiment001.yaml') as f:
            CFG = yaml.load(f, Loader=yaml.FullLoader)
        
        CFG['model']['params']['pretrained'] = None
        self.model = build_model(CFG['model']['name'], CFG['model']['params'])
        self.model.load_state_dict(torch.load('./SRXT50_094_VM-0.2504.PTH'))
        self.model.to(self.device)
        self.model.eval()
        self.loader = build_dataloader(CFG, data_info={'vidfiles': [], 'labels': []}, mode='predict')
    
    def get_roi_for_each_face(self,faces_by_frame, probs, video_shape, temporal_upsample, upsample=1):
    # Create boolean face array
        TWO_FRAME_OVERLAP = False
        MIN_FRAMES_FOR_FACE = 30
        MAX_FRAMES_FOR_FACE = 100
        frames_video, rows_video, cols_video, channels_video = video_shape
        frames_video = math.ceil(frames_video)
        boolean_face_3d = np.zeros((frames_video, rows_video, cols_video), dtype=np.bool)  # Remove colour channel
        proba_face_3d = np.zeros((frames_video, rows_video, cols_video)).astype('float32')
        for i_frame, faces in enumerate(faces_by_frame):
            if faces is not None:  # May not be a face in the frame
                for i_face, face in enumerate(faces):
                    left, top, right, bottom = face
                    boolean_face_3d[i_frame, int(top):int(bottom), int(left):int(right)] = True
                    proba_face_3d[i_frame, int(top):int(bottom), int(left):int(right)] = probs[i_frame][i_face]
                
        # Replace blank frames if face(s) in neighbouring frames with overlap
        for i_frame, frame in enumerate(boolean_face_3d):
            if i_frame == 0 or i_frame == frames_video-1:  # Can't do this for 1st or last frame
                continue
            if True not in frame:
                if TWO_FRAME_OVERLAP:
                    if i_frame > 1:
                        pre_overlap = boolean_face_3d[i_frame-1] | boolean_face_3d[i_frame-2]
                    else:
                        pre_overlap = boolean_face_3d[i_frame-1]
                    if i_frame < frames_video-2:
                        post_overlap = boolean_face_3d[i_frame+1] | boolean_face_3d[i_frame+2]
                    else:
                        post_overlap = boolean_face_3d[i_frame+1]
                    neighbour_overlap = pre_overlap & post_overlap
                else:
                    neighbour_overlap = boolean_face_3d[i_frame-1] & boolean_face_3d[i_frame+1]
                boolean_face_3d[i_frame] = neighbour_overlap

        # Find faces through time
        id_face_3d, n_faces = skimage.measure.label(boolean_face_3d, return_num=True)
        region_labels, counts = np.unique(id_face_3d, return_counts=True)
        # Get rid of background=0
        region_labels, counts = region_labels[1:], counts[1:]
        ###################
        # DESCENDING SIZE #
        ###################
        descending_size = np.argsort(counts)[::-1]
        labels_by_size = region_labels[descending_size]
        ####################
        # DESCENDING PROBS #
        ####################
        probs = [np.mean(proba_face_3d[id_face_3d == i_face]) for i_face in region_labels]
        descending_probs = np.argsort(probs)[::-1]
        labels_by_probs = region_labels[descending_probs]
        # Iterate over faces in video
        rois = [] ; face_maps = []
        for i_face in labels_by_probs:#labels_by_size:
            # Find the first and last frame containing the face
            frames = np.where(np.any(id_face_3d == i_face, axis=(1, 2)) == True)
            starting_frame, ending_frame = frames[0].min(), frames[0].max()

            # Iterate over the frames with faces in and find the min/max cols/rows (bounding box)
            cols, rows = [], []
            for i_frame in range(starting_frame, ending_frame + 1):
                rs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=1) == True)
                rows.append((rs[0].min(), rs[0].max()))
                cs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=0) == True)
                cols.append((cs[0].min(), cs[0].max()))
            frame_from, frame_to = starting_frame*temporal_upsample, ((ending_frame+1)*temporal_upsample)-1
            rows_from, rows_to = np.array(rows)[:, 0].min(), np.array(rows)[:, 1].max()
            cols_from, cols_to = np.array(cols)[:, 0].min(), np.array(cols)[:, 1].max()
        
            frame_to = min(frame_to, frame_from + MAX_FRAMES_FOR_FACE)
        
            if frame_to - frame_from >= MIN_FRAMES_FOR_FACE:
                tmp_face_map = id_face_3d.copy()
                tmp_face_map[tmp_face_map != i_face] = 0
                tmp_face_map[tmp_face_map == i_face] = 1
                face_maps.append(tmp_face_map[frame_from//temporal_upsample:frame_to//temporal_upsample+1])
                rois.append(((frame_from, frame_to),
                            (int(rows_from*upsample), int(rows_to*upsample)),
                            (int(cols_from*upsample), int(cols_to*upsample))))
            
        return np.array(rois), face_maps

    def get_coords(self,faces_roi):
        coords = np.argwhere(faces_roi == 1)
        #print(coords)
        if coords.shape[0] == 0:
            return None
        y1, x1 = coords[0]
        y2, x2 = coords[-1]
        return x1, y1, x2, y2

    def get_center(self,bbox):
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2, (y1+y2)/2

    def interpolate_center(self,c1, c2, length):
        x1, y1 = c1
        x2, y2 = c2
        xi, yi = np.linspace(x1, x2, length), np.linspace(y1, y2, length)
        return np.vstack([xi, yi]).transpose(1,0)

    def get_faces(self,faces_roi, upsample): 
        all_faces = []
        rows = faces_roi[0].shape[1]
        cols = faces_roi[0].shape[2]
        for i in range(len(faces_roi)):
            faces = np.asarray([self.get_coords(faces_roi[i][j]) for j in range(len(faces_roi[i]))])
            if faces[0] is None:  faces[0] = faces[1]
            if faces[-1] is None: faces[-1] = faces[-2]
            if None in faces:
                #print(faces)
                log.error('This should not have happened ...')
            all_faces.append(faces)

        extracted_faces = []
        for face in all_faces:
            # Get max dim size
            max_dim = np.concatenate([face[:,2]-face[:,0],face[:,3]-face[:,1]])
            max_dim = np.percentile(max_dim, 90)
            # Enlarge by 1.2
            max_dim = int(max_dim * 1.2)
            # Get center coords
            centers = np.asarray([self.get_center(_) for _ in face])
            # Interpolate
            centers = np.vstack([self.interpolate_center(centers[i], centers[i+1], length=10) for i in range(len(centers)-1)]).astype('int')
            x1y1 = centers - max_dim // 2
            x2y2 = centers + max_dim // 2 
            x1, y1 = x1y1[:,0], x1y1[:,1]
            x2, y2 = x2y2[:,0], x2y2[:,1]
            # If x1 or y1 is negative, turn it to 0
            # Then add to x2 y2 or y2
            x2[x1 < 0] -= x1[x1 < 0]
            y2[y1 < 0] -= y1[y1 < 0]
            x1[x1 < 0] = 0
            y1[y1 < 0] = 0
            # If x2 or y2 is too big, turn it to max image shape
            # Then subtract from y1
            y1[y2 > rows] += rows - y2[y2 > rows]
            x1[x2 > cols] += cols - x2[x2 > cols]
            y2[y2 > rows] = rows
            x2[x2 > cols] = cols
            vidface = np.asarray([[x1[_],y1[_],x2[_],y2[_]] for _,c in enumerate(centers)])
            vidface = (vidface*upsample).astype('int')
            extracted_faces.append(vidface)

        return extracted_faces

    def detect_face_with_mtcnn(self,mtcnn_model, pil_frames, facedetection_upsample, video_shape, face_frames):
        boxes, _probs = mtcnn_model.detect(pil_frames, landmarks=False)
        faces, faces_roi = self.get_roi_for_each_face(faces_by_frame=boxes, probs=_probs, video_shape=video_shape, temporal_upsample=face_frames, upsample=facedetection_upsample)
        coords = [] if len(faces_roi) == 0 else self.get_faces(faces_roi, upsample=facedetection_upsample)
        return faces, coords
    
    def face_detection_wrapper(self,mtcnn_model, videopath, every_n_frames, facedetection_downsample, max_frames_to_load):
        video, pil_frames, rescale = self.load_video(videopath, every_n_frames=every_n_frames, to_rgb=True, rescale=facedetection_downsample, inc_pil=True, max_frames=max_frames_to_load)
        if len(pil_frames):
            try:
                faces, coords = self.detect_face_with_mtcnn(mtcnn_model=mtcnn_model, 
                                                       pil_frames=pil_frames, 
                                                       facedetection_upsample=1/rescale, 
                                                       video_shape=video.shape, 
                                                       face_frames=every_n_frames)
            except RuntimeError:  # Out of CUDA RAM
                log.error(f"Failed to process {videopath} ! Downsampling x2 ...")
                video, pil_frames, rescale = self.load_video(videopath, every_n_frames=every_n_frames, to_rgb=True, rescale=facedetection_downsample/2, inc_pil=True, max_frames=max_frames_to_load)

                try:
                    faces, coords = self.detect_face_with_mtcnn(mtcnn_model=mtcnn_model, 
                                           pil_frames=pil_frames, 
                                           facedetection_upsample=1/rescale, 
                                           video_shape=video.shape, 
                                           face_frames=every_n_frames)
                except RuntimeError:
                    log.error(f"Failed on downsample ! Skipping...")
                    return [], []
                
        else:
            log.error('Failed to fetch frames ! Skipping ...')
            return [], []
        
        if len(faces) == 0:
            log.error('Failed to find faces ! Upsampling x2 ...')
            try:
                video, pil_frames, rescale = self.load_video(videopath, every_n_frames=every_n_frames, to_rgb=True, rescale=facedetection_downsample*2, inc_pil=True, max_frames=max_frames_to_load)
                faces, coords = self.detect_face_with_mtcnn(mtcnn_model=mtcnn_model, 
                                                       pil_frames=pil_frames, 
                                                       facedetection_upsample=1/rescale, 
                                                       video_shape=video.shape, 
                                                       face_frames=every_n_frames)
            except Exception as e:
                log.error(e)
                return [], []
    
        return faces, coords

    def load_video(self,filename, every_n_frames=None, specific_frames=None, to_rgb=True, rescale=None, inc_pil=False, max_frames=None):
        """Loads a video.
        Called by:
    
        1) The finding faces algorithm where it pulls a frame every FACE_FRAMES frames up to MAX_FRAMES_TO_LOAD at a scale of FACEDETECTION_DOWNSAMPLE, and then half that if there's a CUDA memory error.
    
        2) The inference loop where it pulls EVERY frame up to a certain amount which it the last needed frame for each face for that video"""
    
        assert every_n_frames or specific_frames, "Must supply either every n_frames or specific_frames"
        assert bool(every_n_frames) != bool(specific_frames), "Supply either 'every_n_frames' or 'specific_frames', not both"
    
        cap = cv2.VideoCapture(filename)
        n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        if rescale:
            rescale = rescale * 1920./np.max((width_in, height_in))
    
        width_out = int(width_in*rescale) if rescale else width_in
        height_out = int(height_in*rescale) if rescale else height_in
    
        if max_frames:
            n_frames_in = min(n_frames_in, max_frames)
    
        if every_n_frames:
            specific_frames = list(range(0,n_frames_in,every_n_frames))
    
        n_frames_out = len(specific_frames)
    
        out_pil = []

        out_video = np.empty((n_frames_out, height_out, width_out, 3), np.dtype('uint8'))

        i_frame_in = 0
        i_frame_out = 0
        ret = True

        while (i_frame_in < n_frames_in and ret):
        
            try:
                try:
        
                    if every_n_frames == 1:
                        ret, frame_in = cap.read()  # Faster if reading all frames
                    else:
                        ret = cap.grab()

                        if i_frame_in not in specific_frames:
                            i_frame_in += 1
                            continue

                        ret, frame_in = cap.retrieve()
                    
#                   print(f"Reading frame {i_frame_in}")

                    if rescale:
                        frame_in = cv2.resize(frame_in, (width_out, height_out))
                    if to_rgb:
                        frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
                
                except Exception as e:
                    log.error(f"Error for frame {i_frame_in} for video {filename}: {e}; using 0s")
                    frame_in = np.zeros((height_out, width_out, 3))

        
                out_video[i_frame_out] = frame_in
                i_frame_out += 1

                if inc_pil:
                    try:  
                        pil_img = Image.fromarray(frame_in)
                    except Exception as e:
                        log.error(f"Using a blank frame for video {filename} frame {i_frame_in} as error {e}")
                        pil_img = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))  # Use a blank frame
                    out_pil.append(pil_img)

                i_frame_in += 1
            
            except Exception as e:
                log.error(f"Error for file {filename}: {e}")

        cap.release()
    
        if inc_pil:
            return out_video, out_pil, rescale
        else:
            return out_video, rescale

    def get_last_frame_needed_across_faces(self,faces):
        last_frame = 0
    
        for face in faces:
            (frame_from, frame_to), (row_from, row_to), (col_from, col_to) = face
            last_frame = max(frame_to, last_frame)
        
        return last_frame

    def __call__(self,vedio_path:str) -> (list):
        # Face detection
        MAX_FRAMES_TO_LOAD = 100
        MIN_FRAMES_FOR_FACE = 30
        MAX_FRAMES_FOR_FACE = 100
        FACE_FRAMES = 10
        MAX_FACES_HIGHTHRESH = 5
        MAX_FACES_LOWTHRESH = 1
        FACEDETECTION_DOWNSAMPLE = 0.25
        MTCNN_THRESHOLDS = (0.8, 0.8, 0.9)  # Default [0.6, 0.7, 0.7]
        MTCNN_THRESHOLDS_RETRY = (0.5, 0.5, 0.5)
        MMTNN_FACTOR = 0.71  # Default 0.709 p
        TWO_FRAME_OVERLAP = False

        # Inference
        PROB_MIN, PROB_MAX = 0.001, 0.999
        REVERSE_PROBS = True
        DEFAULT_MISSING_PRED = 0.5
        USE_FACE_FUNCTION = np.mean

        # 3D inference
        RATIO_3D = 1
        OUTPUT_FACE_SIZE = (256, 256)
        PRE_INFERENCE_CROP = (224, 224)

        # 2D
        RATIO_2D = 1
        FRAMES2D = 32
        videopaths = sorted(glob(os.path.join(vedio_path, "*.mp4")))
        mtcnn = MTCNN(margin=0, keep_all=True, post_process=False, select_largest=False, device='cuda:0', thresholds=MTCNN_THRESHOLDS, factor=MMTNN_FACTOR)
        faces_by_videopath = {}
        coords_by_videopath = {}

        for i_video, videopath in enumerate(tqdm(videopaths)):
            faces, coords = self.face_detection_wrapper(mtcnn, videopath, every_n_frames=FACE_FRAMES, facedetection_downsample=FACEDETECTION_DOWNSAMPLE, max_frames_to_load=MAX_FRAMES_TO_LOAD)
            #print(len(faces))
            if len(faces):
                faces_by_videopath[videopath]  = faces[:MAX_FACES_HIGHTHRESH]
                coords_by_videopath[videopath] = coords[:MAX_FACES_HIGHTHRESH]
            else:
                print(f"Found no faces for {videopath} !")
        faces_by_videopath = dict(sorted(faces_by_videopath.items(), key=lambda x: x[0]))
        videopaths_missing_faces = {p for p in videopaths if p not in faces_by_videopath}
        #print(coords_by_videopath)
        predictions = defaultdict(list)

        for videopath,faces in tqdm(faces_by_videopath.items(),total=len(faces_by_videopath)):
            try:
                if len(faces):
                    last_frame_needed = self.get_last_frame_needed_across_faces(faces)
                    video, rescale = self.load_video(videopath, every_n_frames=1, to_rgb=True, rescale=None, inc_pil=False, max_frames=last_frame_needed)
                    #print(video)
                else:
                    print(f"Skipping {videopath} as no faces found")
                    continue
                try:
                    coords = coords_by_videopath[videopath]
                    #print(coords)
                    videoname = os.path.basename(videopath)
                    preds_video = []
                    for i_coord, coordinate in enumerate(coords):
                        (frame_from, frame_to), (row_from, row_to), (col_from, col_to) = faces[i_coord]
                        x = []
                        for coord_ind, frame_number in enumerate(range(frame_from, min(frame_from+FRAMES2D, frame_to-1))):
                            if coord_ind >= len(coordinate):
                                break
                            x1, y1, x2, y2 = coordinate[coord_ind]

                            x.append(video[frame_number, y1:y2, x1:x2])
                        x = np.asarray(x)
                            # Reverse back to BGR because it will get reversed to RGB when preprocessed
                            #x = x[...,::-1]
                            # Preprocess
                        x = self.loader.dataset.process_video(x)
                            #x = np.asarray([loader.dataset.process_image(_) for _ in x])
                            # Flip every other frame
                        #print(x)
                        x[:,::2] = x[:,::2,:,::-1]
                            # RGB reverse every 3rd frame
                            #x[:,::3] = x[::-1,::3]
                        #print(x)
                        with torch.no_grad():
                            out = self.model(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).cuda())
                            #out = np.median(out.cpu().numpy())
                        preds_video.append(out.cpu().numpy())
                        print(out.cpu().numpy())
                    if len(preds_video) > 0:
                        predictions[videoname].extend([USE_FACE_FUNCTION(preds_video)] * RATIO_2D)
                    
                except:
                    pass

            except Exception as e:
                log.error(f"ERROR: Video {videopath}:{e}")
        #print(predictions)
        return predictions


if __name__ == "__main__":
    video_path = "/home/xuyu/Deepfake/deepfake_detec"
    op = DeepFake2D()
    pred = op(video_path)
    print(pred)