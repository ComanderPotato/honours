import torch
import ocnn
from ocnn.octree import Octree
from octree_modules import OctreeConvLeakyRelu, FcLeakyRelu

class VoxNet(torch.nn.Module):


  def __init__(self, in_channels: int, out_channels: int, stages: int,
               nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stages = stages
    self.nempty = nempty
    channels = [in_channels] + [2 ** max(i+7-stages, 2) for i in range(stages)]
    # self.convs = torch.nn.ModuleList([OctreeConvLeakyRelu(
    #     channels[i], channels[i+1], nempty=nempty, stride=2 if i == 0 else 1) for i in range(stages)])
    self.layers = torch.nn.ModuleList([
      ocnn.modules.OctreeConvBnRelu(in_channels=4, out_channels=32, nempty=nempty, stride=2),
      ocnn.modules.OctreeConvBnRelu(in_channels=32, out_channels=32, nempty=nempty),
      ocnn.modules.OctreeConvBnRelu(in_channels=32, out_channels=32, nempty=nempty),
      ocnn.nn.OctreeMaxPool(nempty)])

    
    # self.pool = ocnn.nn.OctreeMaxPool(nempty)
    self.octree2voxel = ocnn.nn.Octree2Voxel(self.nempty)
    self.header = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        FcLeakyRelu(64 * 64, 128),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(128, out_channels))

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    for i in range(len(self.layers)):
      d = depth - i
      data = self.layers[i](data, octree, d)

    # data = self.pool(data, octree, self.stages)
    data = self.octree2voxel(data, octree, depth-self.stages)
    data = self.header(data)
    return data
