## :luggage: Checkpoints
You can download the checkpoint files for groundeddino from the following links and put them into the `GroundedSAM_demo/checkpoints/gdino`
<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>Data</th>
      <th>box AP on COCO</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>GroundingDINO-T</td>
      <td>Swin-T</td>
      <td>O365,GoldG,Cap4M</td>
      <td>48.4 (zero-shot) / 57.2 (fine-tune)</td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth">GitHub link</a> | <a href="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth">HF link</a></td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py">link</a></td>
    </tr>
    <tr>
      <th>2</th>
      <td>GroundingDINO-B</td>
      <td>Swin-B</td>
      <td>COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO</td>
      <td>56.7 </td>
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth">GitHub link</a>  | <a href="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth">HF link</a> 
      <td><a href="https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinB_cfg.py">link</a></td>
    </tr>
  </tbody>
</table>

### Run Grounded-Light-HQSAM Demo

- Firstly, download the pretrained Light-HQSAM weight [here](https://github.com/SysCV/sam-hq#model-checkpoints), pretrained MobileSAM weight [here](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights), pretrained EfficientSAM checkpoint from [here](https://github.com/yformer/EfficientSAM#model) and put it into the `GroundedSAM_demo/checkpoints/yolosam` folder.





Installation:
	pip install -r requirements.txt

Examples:
	demo video: 
		python3 -m opencv_camera_with_grounded_sam -i "demo.mp4" -e "checkpoints" -d "cfg/gdino" -m 0.3 -n 0.3 -y -s "mobile_sam.pt" -q 0.01 -p "red box" -b "gray"

	webcame:
		python3 -m opencv_camera_with_grounded_sam -e "checkpoints" -d "cfg/gdino" -m 0.1 -n 0.1 -y -s "mobile_sam.pt" -q 0.01 -p "objects" -b "gray"
