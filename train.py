import torch
import lpips
import numpy as np
from config import parser
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.loss import AdversarialLoss
from models.e2fgvi import InpaintGenerator as E2FGVI
from models.flow_net import FlowCompletionNet as FlowModel
from models.model import InpaintGenerator as IG
from models.model import Discriminator
from models.modules.flow_comp_raft import RAFT_bi, FlowLoss, EdgeLoss
from models.canny.canny_filter import Canny
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import random
import numpy as np
from core.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", dataset="davis"):
        self.num_local_frames = args.num_local
        self.num_ref_frames = args.num_non_local
        self.num_memory = args.num_memory
        self.size = self.w, self.h = (432, 240)
        self.data_root = "your path"
        self.mask_root = "your root"
        self.video_names = sorted(os.listdir(self.data_root)) 
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

        self.mode = mode

    def __len__(self):
        return len(self.video_names)

        
    def __getitem__(self, index):
        item = self.load_item(index)
        return item
    

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(self.num_memory, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        memory_idx = complete_idx_set[pivot - self.num_memory : pivot]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index, memory_idx

    def load_item(self, index):
        video_name = self.video_names[index]
        frame_list = sorted(os.listdir(os.path.join(self.data_root, video_name)))
        video_lens = len(frame_list)
        # create masks
        all_masks = create_random_shape_with_random_motion(
            video_lens, imageHeight=self.h, imageWidth=self.w)
        
        # create sample index
        if self.mode == "train":
            selected_index, memory_idx = self._sample_index(video_lens, self.num_local_frames, self.num_ref_frames)
        else:
            local_index = list(range(self.num_memory, self.num_memory + self.num_local_frames))
            non_local_step = (video_lens - local_index[-1]) // self.num_ref_frames
            non_local_index = [i for i in range(self.num_memory + self.num_local_frames + non_local_step, video_lens, non_local_step)]
            selected_index = local_index + non_local_index
            memory_idx = list(range(self.num_memory))
        frames = []
        masks = []
        memory_frames = []
        memory_masks = []
        for idx in selected_index:
            img_path = os.path.join(self.data_root, video_name, frame_list[idx])
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
            if len(frames) == self.num_local_frames and random.random() < 0.5 and self.mode == "train": # random reverse
                frames.reverse()
                masks.reverse()

        for idx in memory_idx:
            img_path = os.path.join(self.data_root, video_name, frame_list[idx])
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.size)
            memory_frames.append(img)
            memory_masks.append(all_masks[idx])
            
        if self.mask_root is not None:
            img_path = os.path.join(self.mask_root, video_name, "0000.png")
            img = Image.open(img_path).convert('L')
            img = img.resize(self.size)
            masks = [img] * len(selected_index)
            memory_masks = [img] * len(memory_idx)
        elif self.mode == "train":
            frames = GroupRandomHorizontalFlip()(frames)
            memory_frames = GroupRandomHorizontalFlip()(memory_frames)

        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        memory_tensors = self._to_tensors(memory_frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        mask_memory_tensors = self._to_tensors(memory_masks)

        
        return frame_tensors, memory_tensors, mask_tensors, mask_memory_tensors
    


class train(object):
    def __init__(self):
        
        self.ckpt = {}
        self.model = IG()
        self.templator = E2FGVI()
        if args.weight_flow is None:
            self.flow_model = FlowModel("pretrain_rfc.pth")
        else:
            self.flow_model = FlowModel(args.weight_flow)
        self.raft = RAFT_bi("./checkpoints/raft-things.pth")
        self.discriminator = Discriminator(in_channels=3, use_sigmoid=False)
        self.gpu = list(range(torch.cuda.device_count()))
        if True:
            dist.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            self.model.to(self.device)
            self.templator.to(self.device)
            self.raft.to(self.device)
            self.flow_model.to(self.device)
            self.discriminator.to(self.device)
            Trainset, Validset = TrainDataset(mode="train", dataset="youtube"), TrainDataset(mode="valid")
            self.train_sampler, self.valid_sampler = DistributedSampler(Trainset), DistributedSampler(Validset)
            self.train_loader = DataLoader(Trainset, batch_size=1, shuffle=False, num_workers=4,pin_memory=True,
                                           sampler=self.train_sampler)
            self.valid_loader = DataLoader(Validset, batch_size=1, shuffle=False, num_workers=4,pin_memory=True,
                                           sampler=self.valid_sampler)
            self.interp_mode = "nearest"

        self.L1 = nn.L1Loss()
        self.Lpips_func = lpips.LPIPS(net='vgg')
        self.Lpips_func.to(self.device)

        self.flow_loss = FlowLoss().to(self.device)
        self.edge_loss = EdgeLoss().to(self.device)
        self.canny = Canny(sigma=(2,2), low_threshold=0.1, high_threshold=0.2).to(self.device)
        self.adversarial_loss = AdversarialLoss(type='hinge').to(self.device)

        self.optimG = optim.Adam(
            params=self.model.parameters(),
            lr=args.lr,
            betas=(0,0.99),
            eps=1e-8,
        )
        self.optimD = optim.Adam(
            params=self.discriminator.parameters(),
            lr = 1e-4,
            betas=(0, 0.99),
        )
        self.model = DDP(self.model, broadcast_buffers=False, find_unused_parameters=True)
        self.templator = DDP(self.templator, broadcast_buffers=False, find_unused_parameters=False)
        self.discriminator = DDP(self.discriminator, broadcast_buffers=True, find_unused_parameters=False)   
        for epoch in tqdm(range(1, args.epoch + 1)):
            self.train()
        print("Finished!")

    def train(self):
        self.model.train(); self.raft.eval(); self.flow_model.eval(); self.discriminator.train(); self.templator.eval()
        for frames, memory, masks, memory_masks in self.train_loader:
            frames, memory, masks, memory_masks = frames.to(self.device), memory.to(self.device), masks.to(self.device), memory_masks.to(self.device)
            valids = 1 - masks
            b, t, c, h, w = frames.size()
            _, m_t, _, _, _ = memory.size()
            masked_frames = frames * valids
            masked_memory = memory * (1 - memory_masks)
            l_t = args.num_local
            gt_local_frames = frames[:, :l_t, ...]
            local_masks = masks[:, :l_t, ...].contiguous()
            masked_local_frames = gt_local_frames * (1 - local_masks)
            
            with torch.no_grad():
                gt_flows_bi = self.raft(gt_local_frames)
                
                pred_flows_bi, _ = self.flow_model.forward_bidirect_flow(gt_flows_bi, local_masks)
                pred_flows_bi = self.flow_model.combine_flow(gt_flows_bi, pred_flows_bi, local_masks)

                prop_imgs, updated_local_masks = self.model.module.img_propagation(masked_local_frames, pred_flows_bi, local_masks, interpolation=self.interp_mode)
                updated_masks = masks.clone()
                updated_masks[:, :l_t, ...] = updated_local_masks.view(b, l_t, 1, h, w)
                updated_frames = masked_frames.clone()
                prop_local_frames = gt_local_frames * (1-local_masks) + prop_imgs.view(b, l_t, 3, h, w) * local_masks # merge
                updated_frames[:, :l_t, ...] = prop_local_frames

                pred_memory, _ = self.templator(masked_memory, args.num_local)
                pred_memory = pred_memory.unsqueeze(0) * memory_masks + memory * (1-memory_masks)

            
            pred_img, pred_memory, _ = self.model(updated_frames, pred_flows_bi, masks, updated_masks, args.num_local, pred_memory, memory_masks)
            comp_img = pred_img * masks + frames * valids
            hole_loss = self.L1(pred_img * masks, frames * masks)/ torch.mean(masks)
            valid_loss = self.L1(pred_img * valids, frames * valids) / torch.mean(valids)
            perceptual_loss = torch.sum(self.Lpips_func((comp_img.squeeze(0)+1.)/2., (frames.squeeze(0)+1.)/2., normalize=True))
            if True:
                # discriminator adversarial loss
                real_clip = self.discriminator(frames)
                fake_clip = self.discriminator(comp_img.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss = (dis_real_loss + dis_fake_loss) / 2

                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                gen_clip = self.discriminator(comp_img)
                gan_loss = self.adversarial_loss(gen_clip, True, False)

            total_loss = hole_loss + valid_loss + perceptual_loss + gan_loss
            total_loss.backward()
            self.optimG.step()
            self.optimG.zero_grad()

    
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = parser.parse_args()
    train()
