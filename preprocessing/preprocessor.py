import kornia as K
import torch
import torch.nn.functional as F


class Preprocessor:
    def __init__(self, patch_size=32, num_patches=8, device='cuda'):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.device = device
        self.stride = patch_size // 4

    @torch.no_grad()
    def process_batch(self, batch_imgs):
        batch_imgs = batch_imgs.to(self.device)

        B, C, H, W = batch_imgs.shape

        grads = K.filters.spatial_gradient(batch_imgs, mode='sobel')
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]
        mag = torch.sqrt(gx ** 2 + gy ** 2)

        sigma = (2.5, 2.5)
        kernel_size = (11, 11)
        mag_s = K.filters.gaussian_blur2d(mag, kernel_size, sigma)
        j11 = K.filters.gaussian_blur2d(gx * gx, kernel_size, sigma)
        j22 = K.filters.gaussian_blur2d(gy * gy, kernel_size, sigma)
        j12 = K.filters.gaussian_blur2d(gx * gy, kernel_size, sigma)

        coh = torch.sqrt((j11 - j22) ** 2 + 4 * j12 ** 2) / (j11 + j22 + 1e-6)
        score = mag_s * coh

        flattened_scores = score.view(B, -1)
        k = int(0.85 * flattened_scores.size(1))
        thresh, _ = torch.kthvalue(flattened_scores, k, dim=1)
        mask = (score > thresh.view(B, 1, 1, 1)).float()

        mask_lowres = F.interpolate(mask, scale_factor=0.25, mode='area')

        padding = 5
        kernel_size = 11

        mask_lowres = F.avg_pool2d(mask_lowres, kernel_size, stride=1, padding=padding)
        mask_lowres = (mask_lowres > 0.1).float()
        mask_lowres = F.avg_pool2d(mask_lowres, kernel_size, stride=1, padding=padding)
        mask_lowres = (mask_lowres > 0.9).float()

        mask = F.interpolate(mask_lowres, size=(H, W), mode='nearest')

        local_mean = K.filters.gaussian_blur2d(batch_imgs, (13, 13), (3.0, 3.0))
        centered = batch_imgs - local_mean
        local_var = K.filters.gaussian_blur2d(centered ** 2, (13, 13), (3.0, 3.0))
        local_std = torch.sqrt(local_var + 1e-5).clamp(min=0.05)
        normalized_imgs = centered / local_std

        patches = F.unfold(
            normalized_imgs,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.patch_size // 2
        )

        L = patches.shape[-1]
        patches = patches.view(B, 1, self.patch_size, self.patch_size, L)
        patches = patches.permute(0, 4, 1, 2, 3)

        mask_patches = F.unfold(
            mask,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.patch_size // 2
        )
        mask_ratio = mask_patches.mean(dim=1)

        final_batch = []
        for i in range(B):
            valid_idx = (mask_ratio[i] >= 0.8).nonzero(as_tuple=True)[0]
            if self.num_patches is None:
                idx = valid_idx
            elif len(valid_idx) < self.num_patches:
                idx = torch.randint(0, len(valid_idx) if len(valid_idx) > 0 else 1, (self.num_patches,))
            else:
                perm = torch.randperm(len(valid_idx))[:self.num_patches]
                idx = valid_idx[perm]

            final_batch.append(patches[i, idx])

        return torch.stack(final_batch)
