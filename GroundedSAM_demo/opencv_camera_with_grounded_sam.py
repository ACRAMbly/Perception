import argparse
import cv2
import distinctipy

import numpy as np

from GroundedSAM_demo.grounded_sam import GroundedSAM

from typing_extensions import Union


def get_grounded_segmentation(device: Union[int, str],
                              checkpoints_dir: str, grounded_dino_config_dir: str, grounded_dino_use_vitb: bool,
                              box_threshold: float,  text_threshold: float,
                              use_yolo_sam: bool, sam_vit_model: str, mask_threshold: float,
                              prompt_text: str,
                              background: str):
    print("Starting process to get the grounded segmentation masks of video frames.")

    # initialize grounded sam
    print("Initializing models.")
    grounded_sam = GroundedSAM.load_grounded_sam_model(checkpoints_dir=checkpoints_dir,
                                                       grounded_dino_config_dir=grounded_dino_config_dir,
                                                       grounded_dino_use_vitb=grounded_dino_use_vitb,
                                                       box_threshold=box_threshold,
                                                       text_threshold=text_threshold,
                                                       use_yolo_sam=use_yolo_sam,
                                                       sam_vit_model=sam_vit_model,
                                                       mask_threshold=mask_threshold,
                                                       prompt_text=prompt_text,
                                                       segmentor_width_size=None,
                                                       device=None)

    # prepare camera
    try:
        # try to cast to int
        device = int(device)
    except ValueError:
        # real string
        pass
    camera = cv2.VideoCapture(device)

    print("Initialized models and started usb camera.")

    # expected maximal number of different objects/needed color in an image
    num_max_objs = 50
    colors = distinctipy.get_colors(num_max_objs)

    print("Start getting the grounded segmentation masks for all frames.")

    cv2.namedWindow("GroundedSam", cv2.WINDOW_NORMAL)

    while True:
        # read frame
        ret, frame = camera.read()

        if not ret:
            print("Failed to grab frame.")
            continue

        # stabilise contrast/brightness
        cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)

        # get masks
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = grounded_sam.generate_masks(frame)

        if detections:
            # sort masks based on bounding box position
            bboxes = detections["boxes"].cpu().numpy()
            masks = detections["masks"].squeeze(1).cpu().numpy()
            sort_indicis = np.lexsort([bboxes[:, 1], bboxes[:, 0]])
            #bboxes = bboxes[sort_indicis]
            masks = masks[sort_indicis]

            # get background image
            if background == "rgb":
                pass
            elif background == "gray":
                # convert to gray image
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            elif background == "black":
                frame = np.zeros_like(frame, dtype=np.uint8)
            elif background == "white":
                frame = np.ones_like(frame, dtype=np.uint8)
            else:
                raise NotImplementedError(
                    "Value for 'background' must be one of ['rgb', 'gray', 'black', 'white'], but saw <{}>".format(
                        background))

            # combine masks with background image
            alpha = 0.33
            for mask_idx, mask in enumerate(masks):
                r = int(255 * colors[mask_idx][0])
                g = int(255 * colors[mask_idx][1])
                b = int(255 * colors[mask_idx][2])

                frame[mask, 0] = alpha * r + (1 - alpha) * frame[mask, 0]
                frame[mask, 1] = alpha * g + (1 - alpha) * frame[mask, 1]
                frame[mask, 2] = alpha * b + (1 - alpha) * frame[mask, 2]

        # show frame
        cv2.imshow("GroundedSam", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if not isinstance(device, int) and camera.get(cv2.CAP_PROP_POS_FRAMES) >= camera.get(cv2.CAP_PROP_FRAME_COUNT):
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("Reached video end, restart.")

        k = cv2.waitKey(25) & 0xFF  # reduce waiting time for higher framerate
        if k == 27:
            # ESC pressed
            print("Escape pressed, closing.")
            break

    # clean up
    camera.release()
    cv2.destroyAllWindows()

    print("Finished processing the frames.")
    print("End")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Applies 'GroundedSam' to all video frames.")

    parser.add_argument("-i", "--device", dest="device", type=str, default=0,
                        help="Device/Video file for the video camera.")

    parser.add_argument("-e", "--checkpoints_dir", dest="checkpoints_dir", type=str, default="GroundedSAM_demo/checkpoints",
                    help="Path of the root dir containing the checkpoints.")
    parser.add_argument("-d", "--grounded_dino_config_dir", dest="grounded_dino_config_dir", type=str, default="GroundedSAM_demo/cfg/gdino",
                        help="Path of the dir containing the configuration files for 'GroundedDino'.")
    parser.add_argument("-u", "--grounded_dino_use_vitb", dest="grounded_dino_use_vitb", action="store_true", default=False,
                        help="Use the vit_b backbone for GroundingDino.")

    parser.add_argument("-m", "--box_threshold", dest="box_threshold", type=float, default=0.1,
                        help="The minimum confidence score of a bounding box to be used as prompt.")
    parser.add_argument("-n", "--text_threshold", dest="text_threshold", type=float, default=0.1,
                        help="The minimum confidence score that the bounding box class matches the phrase.")

    parser.add_argument("-y", "--use_yolo_sam", dest="use_yolo_sam", action="store_true", default=False,
                        help="Use Yolo implementation of SAM.")
    parser.add_argument("-s", "--sam_vit_model", dest="sam_vit_model", type=str, default="vit_b",
                        help="Which SAM model/backbone size to use.")
    parser.add_argument("-q", "--mask_threshold", dest="mask_threshold", type=float, default=0.01,
                        help="The minimum confidence score of a segmentation masks.")

    parser.add_argument("-p", "--prompt_text", dest="prompt_text", type=str, default="objects",
                        help="Prompt for the bounding box search.")

    parser.add_argument("-b", "--background", dest="background", choices=["rgb", "gray", "black", "white"],
                        default="gray",
                        help="Choose which background image for the renderings to use: 'rgb': scene image, 'gray': grayscale scene image, 'black'/'white': black/white background.")

    args = parser.parse_args()

    get_grounded_segmentation(device=args.device,
                              checkpoints_dir=args.checkpoints_dir,
                              grounded_dino_config_dir=args.grounded_dino_config_dir,
                              grounded_dino_use_vitb=args.grounded_dino_use_vitb,
                              box_threshold=args.box_threshold,
                              text_threshold=args.text_threshold,
                              use_yolo_sam=args.use_yolo_sam,
                              sam_vit_model=args.sam_vit_model,
                              mask_threshold=args.mask_threshold,
                              prompt_text=args.prompt_text,
                              background=args.background)
