# Bidirectional Error-Aware Fusion Network 
![overall_structure](./figs/overview.png)
### ⚡ Highlights:
Our propose model has the following *merits* that others have not:
- **Memorize the Past**: Our method bidirectionally incorporates both past inpainted frames and forward reference frames to make the current generation become more temporally-consistent.
- **Distinguish the Reality**: Our method exploits mask priors for video inpainting to mark each token when calculating correlation in self-attention, which allows the model to distinguish different tokens according to their sources so as to guide the model to produce more faithful results.

## Demo
We place several examples below (click for details):

<table>
<tr>
   <td> 
      <details> 
      <summary> 
      <strong>Dance</strong>
      </summary> 
      <img src="./demo/dance.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Hokey</strong>
      </summary> 
      <img src="./demo/hokey.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Motor</strong>
      </summary> 
      <img src="./demo/motor.gif">
      </details>
   </td>
</tr>
<td> 
      <details> 
      <summary> 
      <strong>Parkour</strong>
      </summary> 
      <img src="./demo/parkour.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Swing</strong>
      </summary> 
      <img src="./demo/swing.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Tennis</strong>
      </summary> 
      <img src="./demo/tennis.gif">
      </details>
   </td>
</tr>
</table>

More demo videos are going to uploaded...


## Reference
We acknowledge the following works for their open source:<br/>
[1] ProPainter: Improving Propagation and Transformer for Video Inpainting, Zhou et al., In ICCV 2023.<br/>
[2] Towards End-to-End Flow-Guided Video Inpainting, Li et al., In CVPR 2022.<br/>
[3] Fusing Fine-grained Information in Transformers for Video Inpainting, Liu et al., In ICCV 2021.<br/>
[4] Learning Joint Spatio-Temporal Transformations for Video Inpainting, Zeng et al., In ECCV 2020.<br/>
 <br/>