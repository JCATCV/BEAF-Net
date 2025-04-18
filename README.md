# [TCSVT-2025] Bidirectional Error-Aware Fusion Network 

**[Update]: We uploaded the code of our model. The training framework is the same as E2FGVI, ProPainter, Fuseformer and so on.**

![overall_structure](./figs/overview.png)
### ⚡ Highlights:
Our propose model has the following *merits* that others have not:
- **Bidirectional inpainting**: Our method bidirectionally incorporates both *past inpainted frames* and forward reference frames to make the current generation become more temporally-consistent. This bidirectional design can fully exploit available information from the entire video to enhance temporal consistency.
- **Error-aware inpainting**: Our method exploits *location priors* for video inpainting to mark each token when calculating correlation in self-attention based on the given masks, which allows the model to distinguish different tokens with the awareness of error so as to produce more faithful results.

## Demo
We place some video examples produced by our model below (click for details):

<table>
<tr>
   <td> 
      <details> 
      <summary> 
      <strong>Dance (Object removal)</strong>
      </summary> 
      <img src="./demo/dance.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Hokey (Object removal)</strong>
      </summary> 
      <img src="./demo/hokey.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Motor (Object removal)</strong>
      </summary> 
      <img src="./demo/motor.gif">
      </details>
   </td>
</tr>
<td> 
      <details> 
      <summary> 
      <strong>Parkour (Object removal)</strong>
      </summary> 
      <img src="./demo/parkour.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Swing (Object removal)</strong>
      </summary> 
      <img src="./demo/swing.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Tennis (Object removal)</strong>
      </summary> 
      <img src="./demo/tennis.gif">
      </details>
   </td>
</tr>
<td> 
      <details> 
      <summary> 
      <strong>Bumps (Corruption restoration)</strong>
      </summary> 
      <img src="./demo/bmx_bumps.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Car (Corruption restoration)</strong>
      </summary> 
      <img src="./demo/car_drift.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Breakflare (Corruption restoration)</strong>
      </summary> 
      <img src="./demo/breakflare.gif">
      </details>
   </td>
</tr>
<td> 
      <details> 
      <summary> 
      <strong>Bus (Dis-occlusion)</strong>
      </summary> 
      <img src="./demo/bus.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Surf (Dis-occlusion)</strong>
      </summary> 
      <img src="./demo/surf.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Fly (Dis-occlusion)</strong>
      </summary> 
      <img src="./demo/fly.gif">
      </details>
   </td>
</tr>
</table>

## Citation
If you find this work is helpful, please cite our paper:
```bibtex
@article{hou2024bidirectional,
  title={Bidirectional Error-Aware Fusion Network for Video Inpainting},
  author={Hou, Jiacheng and Ji, Zhong and Yang, Jinyu and Zheng, Feng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```


## Reference
We acknowledge the following works for their open source:<br/>
[1] ProPainter: Improving Propagation and Transformer for Video Inpainting, Zhou et al., In ICCV 2023.<br/>
[2] Towards End-to-End Flow-Guided Video Inpainting, Li et al., In CVPR 2022.<br/>
[3] Fusing Fine-grained Information in Transformers for Video Inpainting, Liu et al., In ICCV 2021.<br/>
[4] Learning Joint Spatio-Temporal Transformations for Video Inpainting, Zeng et al., In ECCV 2020.<br/>
 <br/>
