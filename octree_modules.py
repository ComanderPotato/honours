import torch
from ocnn.nn import OctreeConv
from ocnn.octree import Octree
from typing import List

# Custom layers for voxnet
class FcLeakyRelu(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.flatten = torch.nn.Flatten(start_dim=1)
    self.fc = torch.nn.Linear(in_channels, out_channels, bias=False)
    self.relu = torch.nn.LeakyReLU(inplace=True)
  def forward(self, data):
    r''''''

    out = self.flatten(data)
    out = self.fc(out)
    out = self.relu(out)
    return out

class OctreeConvLeakyRelu(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
      in_channels, out_channels, kernel_size, stride, nempty)
    self.leakyRelu = torch.nn.LeakyReLU()

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    out = self.conv(data, octree, depth)
    out = self.leakyRelu(out)
    return out
