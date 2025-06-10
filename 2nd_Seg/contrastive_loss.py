import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_same_image_mask(num_pixels):
    """두 픽셀이 같은 이미지에 속하는지를 나타내는 마스크를 생성

    Args:
        num_pixels: 각 이미지에 포함된 픽셀 수를 나타내는 리스트로,
                         리스트 길이는 이미지 개수와 동일
    Returns:
        크기가 [1, num_total_pixels, num_total_pixels]인 텐서를 반환하며,
    이 텐서는 어떤 픽셀 쌍들이 같은 이미지에 속하는지를 나타낸다.
    """
  
    image_ids = []
    for img_id, count in enumerate(num_pixels):
        image_ids.extend([img_id] * count)

    # shape : [num_total_pixels, 1]
    image_ids = torch.tensor(image_ids).view(1, -1)
    # shape : [num_total_pixels, num_total_pixels]
    same_image_mask = (image_ids == image_ids.T).float()
    # shape : [1, num_total_pixels, num_total_pixels]
    same_image_mask = same_image_mask.unsqueeze(0)
    
    return same_image_mask


def generate_ignore_mask(labels, ignore_labels):
    """Contrastive loss에 사용되는 ignore mask를 생성

    Args:
        labels: A tensor of shape [batch_size, height, width, 1]
        ignore_labels: 무시할 라벨들의 리스트이다. 이 라벨을 가진 픽셀들은 loss 계산에서 제외

    Returns:
        A tensor of shape [batch, num_pixels, num_pixels],
        이 텐서는 어떤 픽셀 쌍이 무시되어야 하는지를 나타냄 (1이면 무시)
    """
    
    B, H, W, _ = labels.shape
    N = H*W
    
    # shape : [B, H*W, 1]
    labels_flat = labels.view(B, -1, 1)
    # shape : [1, 1, k] ignore_lables의 개수 : k 개개
    ignore_tensor = torch.tensor(ignore_labels,
                                 dtype=labels.dtype,
                                 device=labels.device).view(1, 1, -1)
    # shape : [B, H*W, k] => [B, H*W]
    # 하나라도 해당되면 1로 반환
    ignore_mask_1d = (labels_flat == ignore_tensor).any(dim=2).float()
    
    # [B, N, 1] + [B, 1, N] => [B, N, N]
    # 두 픽셀 중 하나라도 무시 대상이면 1로 반환
    ignore_mask = torch.logical_or(
        ignore_mask_1d.unsqueeze(2).bool(),
        ignore_mask_1d.unsqueeze(1).bool()
    ).float()
    
    return ignore_mask


def generate_positive_and_negative_masks(labels):
    """Contrastive loss에 사용되는 양성 및 음성 마스크를 생성

    Args:
        labels: A tensor of shape [batch_size, height, width, 1]

    Returns:
        positive_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
        어떤 픽셀 쌍이 양성 쌍인지 표시
        negative_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
        어떤 픽셀 쌍이 음성 상인지 표시
    """
    
    B, H, W, _ = labels.shape
    N = H * W
    
    # shape : [B, H*W, 1]
    labels_flat = labels.view(B, -1, 1)
    
    # [B, N, 1] == [B, 1, N] -> [B, N, N]
    # 양성 마스크는 두 픽셀이 같은 라벨을 가지는 경우
    positive_mask = (labels_flat == labels_flat.transpose(1, 2)).float()
    # 음성 마스크는 양성 마스크를 제외한 나머지
    negative_mask = 1 - positive_mask
    
    return positive_mask, negative_mask


def collapse_spatial_dimensions(inputs):
    """height와 width 차원을 하나의 차원으로 병합

    Args:
        inputs: A tensor of shape [batch_size, height, width, num_channels]

    Returns:
        A tensor of shape [batch_size, height * width, num_channels]
    """

    B, C, H, W = inputs.shape  
    return inputs.permute(0, 2, 3, 1).reshape(B, H * W, C)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim):
        """Contrastive loss 전에 사용되는 projection head를 구현

        이 projection head는 1x1 Convolution을 사용

        Args:
            features: A tensor of shape [batch_size, height, width, num_input_channels]
            num_projection_layers: projection head의 layer 수
            num_projection_channels: projection layer 에서 사용할 채널 

        Returns:
            A tensor of shape [batch_size, num_pixels, num_proj_channels]
        """
        super(ProjectionHead, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
            
        layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=1))
        self.projection_head = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.projection_head(x) # [B, C, H, W]
        #x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        #x = x.reshape(x.shape[0], -1, x.shape[-1])  # [B, H*W, C]
        x = F.normalize(x, p=2, dim=1)  # 채널 방향으로 L2 normalization
        return x
    

def resize_and_project(features,
                       resize_size,
                       #num_layer,
                       #num_channels,
                       proj_head):
    """입력 feature를 resize한 후 projection head에 통과

    Args:
        features: A [batch_size, height, width, num_channels] tensor
        resize_size: Contrastive loss 계산 전에 feature/label을 resize할 크기 => (height, width)
        num_projection_layers: Projection head의 레이어 개수
        num_projection_channels: Projection head에서 사용할 채널 수

    Returns:
        A [batch_size, resize_height, resize_width, num_projection_channels] tensor
    """
    # 보간 방법을 사용해서 resize
    resized_features = F.interpolate(features, size=resize_size, mode='bilinear', align_corners=True)
    projected = proj_head(resized_features)

    return projected


def compute_contrastive_loss(logits,
                             positive_mask,
                             negative_mask,
                             ignore_mask):
    
    """Contrastive loss function.

    Args:
        logits: A tensor of shape [batch, num_pixels, num_pixels]
			      각 값은 픽셀 쌍 간의 유사도를 나타냄 
        positive_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
							     어떤 픽셀 쌍이 양의 쌍인지를 나타냄
        negative_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
							     어떤 픽셀 쌍이 음의 쌍인지를 나타냄
        ignore_mask: A tensor of shape [batch, num_pixels, num_pixels], indicating
						     무시해야 할 쌍을 나타냄

    Returns:
        contrastive loss 값을 담은 스칼라 텐서
    """

    #무시 마스크를 반영하여 유효한 상만 남긴다.
    validity_mask = 1.0 - ignore_mask

    #무시할 픽셀 쌍을 양과 음의 마스크에서 0으로 만든다
    positive_mask = positive_mask * validity_mask
    negative_mask = negative_mask * validity_mask

    #픽셀 간의 유사도 계산 -> softmax 구성
    exp_logits = torch.exp(logits) * validity_mask

    #(anchor p, positive q) 쌍에 대해 계산한 softmax 확률 비율
    # [B, N, 1]
    denom = exp_logits + (exp_logits * negative_mask).sum(dim=2, keepdim=True)
    normalized_exp_logits = torch.dived(
        exp_logits,
        denom.clamp(min=1e-6)
    )

    # log likelihood
    # 점수가 클수록 log를 씌우면 값이 작아지기 때문 => -를 붙여 loss 방식으로로
    # ignore mask는 log(1) = 0
    log_probs = torch.log(normalized_exp_logits * validity_mask + ignore_mask + 1e-6)
    neg_log_likelihood = -log_probs

    # normalize weight and sum in dimension 2
    # j축 정규화 => 각 픽셀마다 positive 의 개수가 다르기 때문에에
    pos_sum = positive_mask.sum(dim=2, keepdim=True).clamp(min=1e-6)
    normalized_weight = positive_mask / pos_sum
    loss_matrix = (neg_log_likelihood * normalized_weight).sum(dim=2) # [B, N]

    # normalize weight and sum in dimension 1
    # i축 정규화 => 하나의 이미지 안에서도 positive가 아예 없는 픽셀들이 있을 수 있기 때문
    valid_index = (positive_mask.sum(dim=2) > 0).float()  # [B, N]
    weight_sum = valid_index.sum(dim=1, keepdim=True).clamp(min=1e-6)
    normalized_weight = valid_index / weight_sum  # [B, N]


    loss = (loss_matrix * normalized_weight).sum(dim=1)  # [B]

    return loss


def supervised_pixel_contrastive_loss(features_orig,
                                      features_aug,
                                      labels_orig,
                                      labels_aug,
                                      ignore_labels,
                                      resize_size,
                                      proj_head,
                                      num_projection_layers=2,
                                      num_projection_channels=256,
                                      temperature=0.07,
                                      within_image_loss = False):
    """픽셀 단위의 지도 학습 contrastive loss 계산

    Args:
        features_orig: A [batch_size, height, width, num_channels] tensor
							     원본 이미지로부터 추출된 특징 맵을 나타냄
        features_aug: A [batch_size, height, width, num_channels] tensor
							    증강된 이미지로부터 추출된 특징 맵을 나타
        labels_orig: A tensor of shape [batch_size, height, width, 1] representing
						     원본 이미지의 라벨 정보를 담고 있는 텐서
        labels_aug: A tensor of shape [batch_size, height, width, 1] representing
						    증강된 이미지의 라벨 정보를 담고 있는 텐서
        ignore_labels: contrastive loss 계산 시 무시할 라벨들의 리스트.
								  해당 라벨을 가진 픽셀들은 손실 계산에서 제외
        resize_size: contrastive loss 계산 전에
							   feature 및 label을 리사이징할 (height, width) 튜플
        num_projection_layers: Projection head에 사용할 layer 수
        num_projection_channels: Projection head에 사용할 channel 수
        temperature: Temperature to use in contrastive loss
        within_image_loss: True -> 한 이미지 내 픽셀 간 contrastive loss 계산
										   False -> 서로 다른 이미지 간 픽셀 간 contrastive loss 계산

    Returns:
    Contrastive loss tensor
    """

    # Projection head 구조 반환환
    features_orig = resize_and_project(features_orig, resize_size, proj_head)
    features_aug = resize_and_project(features_aug, resize_size, proj_head)

    #label 텐서 지정된 크기로 resize
    labels_orig = F.interpolate(labels_orig, size=resize_size, mode='nearest')
    labels_aug = F.interpolate(labels_aug, size=resize_size, mode='nearest')
    
    #height와 depth를 하나의 차원으로 반환환
    features_orig = collapse_spatial_dimensions(features_orig)
    features_aug = collapse_spatial_dimensions(features_aug)
    labels_orig = collapse_spatial_dimensions(labels_orig)
    labels_aug = collapse_spatial_dimensions(labels_aug)

    if within_image_loss:
        within_image_loss_orig = within_image_supervised_pixel_contrastive_loss(
        features=features_orig, labels=labels_orig,
        ignore_labels=ignore_labels, temperature=temperature)

        within_image_loss_aug = within_image_supervised_pixel_contrastive_loss(
        features=features_aug, labels=labels_aug,
        ignore_labels=ignore_labels, temperature=temperature)

        return within_image_loss_orig + within_image_loss_aug
    
    # 현재 배치에 몇개의 이미지가 있는지 가져온다
    batch_size = features_orig.size(0)
    indices = torch.randperm(batch_size)
    shuffled_features_aug = features_aug[indices]
    shuffled_labels_aug = labels_aug[indices]

    return cross_image_supervised_pixel_contrastive_loss(
        features1=features_orig,
        features2=shuffled_features_aug,
        labels1=labels_orig,
        labels2=shuffled_labels_aug,
        ignore_labels=ignore_labels,
        temperature=temperature
    )


def within_image_supervised_pixel_contrastive_loss(features,
                                                   labels,
                                                   ignore_labels,
                                                   temperature):
    """이미지 내부에서의 supervised pixel contrastive loss

    Args:
        features: A tensor of shape [batch_size, num_pixels, num_channels]
        labels: A tensor of shape [batch_size, num_pixels, 1]
        ignore_labels: 무시할 레이블 리스트
								   이 레이블을 가진 픽셀 상은 손실 계산에서 제외된다
        temperature: Temperature to use in contrastive loss

    Returns:
    Contrastive loss tensor
    """

    #유사도 계산
    logits = torch.bmm(features, features(1,2)) / temperature

    #양과 음의 마스크 생성
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels)

    #무시할 마스크 생성
    ignore_mask = generate_ignore_mask(labels, ignore_labels)

    return compute_contrastive_loss(
        logits, positive_mask, negative_mask, ignore_mask
    )


# within의 확장형태라고 생각하면 편함
def cross_image_supervised_pixel_contrastive_loss(features1,
                                                  features2,
                                                  labels1,
                                                  labels2,
                                                  ignore_labels,
                                                  temperature):
    """Computes cross-image suprvised pixel contrastive loss.

    Args:
        features1: A tensor of shape [batch_size, num_pixels, num_channels]
        features2: A tensor of shape [batch_size, num_pixels, num_channels]
        labels1: A tensor of shape [batch_size, num_pixels, 1]
        labels2: A tensor of shape [batch_size, num_pixels, 1]
        ignore_labels: A list of labels too ignore. Pixels with these labels will
        be ignored in the loss computation
        temperature: Temperature to use in contrastive loss

    Returns:
        Contrastive loss tensor
    """

    #각 이미지의 픽셀 수를 가져온다
    B, N, C = features1.shape
    num_pixels1 = N
    B, N, C = features2.shape
    num_pixels2 = N

    #두 이미지의 feature와 label을 픽셀 차원에서 이어 붙인다
    features = torch.cat([features1, features2], dim=1) #[B, 2N, C]
    labels = torch.cat([labels1, labels2], dim=1)

    # negative쌍은 같은 이미지에서만 나올 수 있도록 생성
    same_image_mask = generate_same_image_mask([num_pixels1, num_pixels2])

    # 유사도 계산
    logits = torch.bmm(features, features.transpose(1, 2)) / temperature
    # 양과 음의 마스크 생성
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels)
    negative_mask = negative_mask * same_image_mask

    ignore_mask = generate_ignore_mask(labels, ignore_labels)

    return compute_contrastive_loss(
        logits, positive_mask, negative_mask, ignore_mask
    )