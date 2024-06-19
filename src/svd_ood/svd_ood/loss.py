import torch
from torch.nn import functional as F


class LoCoOpLoss:
    def __init__(self, locoop_lambda=0.25, locoop_top_k=200):
        """LoCoOpLoss class to calculate locoop loss.
        Args:
            locoop_lambda (float): lambda value for locoop OOD loss
            locoop_top_k (int): top k value for locoop OOD loss
        """
        self.locoop_lambda = locoop_lambda
        self.locoop_top_k = locoop_top_k

    def cal_loss_id(self, logits_global, labels):
        """Calculate ID classification loss.
        Args:
            logits_global (tensor): (image_batch_size, text_batch_size)
                NOTICE: LOGITS **WITH** logit_scale.exp() !!!
                cosine similarity scores for global image and text features of CLIP model;
                or global image features of vision-only model
            labels (tensor): (image_batch_size,)
                labels for the image batch
        Returns:
            tensor: ID classification loss
        """
        # Note: F.cross_entropy receives "Predicted unnormalized logits" as input
        loss_id = F.cross_entropy(logits_global, labels)
        return loss_id

    def cal_loss_ood(self, logits_local, labels=None, is_ood_patch=None):
        """Calculate OOD regularization loss.
        Args:
            logits_local (tensor): (image_batch_size, weight * height, text_batch_size)
                NOTICE: LOGITS **WITH** logit_scale.exp() !!!
                cosine similarity scores for local image and text features of CLIP model;
                or global image features of vision-only model
            labels (tensor): (image_batch_size,)
                labels for the image batch
            is_ood_patch (tensor): (image_batch_size, weight*height)
                optional, precomputed is_ood_patch tensor
        Returns:
            tensor: OOD regularization loss
        """
        assert labels is not None or is_ood_patch is not None, "neither `lables` nor `is_ood_patch` provided"

        #### extract OOD patches ####
        if is_ood_patch is None:
            # get the top_k local patch labels, shape: [batch, weight*height, top_k]
            _, topk_indices = torch.topk(logits_local, k=self.locoop_top_k, dim=2)
            # labels: [batch] -> labels_: [batch, weight*height, top_k]
            batch_size, num_local_feat = logits_local.shape[0], logits_local.shape[1]
            labels_ood_repeat = labels.view(batch_size, 1, 1).expand(-1, num_local_feat, self.locoop_top_k)
            # whether local patch's top-k labels contain id label (true: ID patch, false: OOD patch)
            is_id_patch = topk_indices.eq(labels_ood_repeat).any(dim=2)      # [batch, weight*height]
            is_ood_patch = ~is_id_patch                                      # [batch, weight*height]

        probs = F.softmax(logits_local, dim=-1) # [batch, weight*height, cls]
        probs_selected = probs[is_ood_patch]    # [ood_patch_num, cls]

        if probs_selected.shape[0] == 0:
            return torch.tensor([0]).cuda()
        # entropy maximization for the extracted OOD features
        entropy_select_topk = -torch.mean(torch.sum(probs_selected * torch.log(probs_selected+1e-5), 1))
        loss_ood = - entropy_select_topk
        return loss_ood

    def cal_loss(self, logits_global, logits_local, labels):
        """Calculate locoop_loss.
        Args:
            logits_global (tensor): (image_batch_size, text_batch_size)
                NOTICE: LOGITS **WITH** logit_scale.exp() !!!
                cosine similarity scores for global image and text features of CLIP model;
                or global image features of vision-only model
            logits_local (tensor): (image_batch_size, weight*height, text_batch_size)
                NOTICE: LOGITS **WITH** logit_scale.exp() !!!
                cosine similarity scores for local image and text features of CLIP model;
                or global image features of vision-only model
            labels (tensor): (image_batch_size,)
                labels for the image batch
        Returns:
            dict: {"locoop_loss": tensor, "loss_id": tensor, "loss_ood": tensor}
        """
        # calculate ID classification loss
        loss_id = self.cal_loss_id(logits_global, labels)

        # calculate OOD regularization loss
        assert len(logits_local.shape) == 3, "logits_local should be (batch_size, weight*height, cls)"
        loss_ood = self.cal_loss_ood(logits_local, labels)

        # calculate total loss for locoop
        locoop_loss = loss_id + self.locoop_lambda * loss_ood
        return {
            "locoop_loss": locoop_loss,
            "loss_id": loss_id,
            "loss_ood": loss_ood
        }


if __name__ == "__main__":
    # Test LoCoOpLoss
    labels = torch.randint(0, 1000, (32,))    # batch_size=32, num_classes=1000
    locoop_lambda = 0.25
    locoop_top_k = 200
    logits_global = torch.randn(32, 1000)        # batch_size=32, num_classes=1000
    logits_local = torch.randn(32, 14*14, 1000) # batch_size=32, local_feat=14x14, num_classes=1000
    loss_fn = LoCoOpLoss(locoop_lambda, locoop_top_k)
    loss = loss_fn.cal_loss(logits_global, logits_local, labels)
    print(loss)