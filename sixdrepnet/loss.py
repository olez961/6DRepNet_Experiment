import torch.nn as nn
import torch

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
         
        return torch.mean(theta)

# 这里实际上是余弦距离损失函数
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, A, B):
        """
        A, B: 3D tensors of shape (batch_size, num_vectors, vector_size)
        """
        dot_product = torch.sum(torch.mul(A, B), dim=2)
        norm_A = torch.norm(A, p=2, dim=2)
        norm_B = torch.norm(B, p=2, dim=2)
        cosine_similarity = dot_product / (norm_A * norm_B)
        loss = 1 - cosine_similarity.mean()
        return loss