* Understanding the code:
They initially set `self.resnet_layer[str(i)] = nn.ModuleList([])`and then append to this list. This is not a regular list, therefore appending `eval('encoder.layer'+str(i+1))` would actually add a resnet block to the list upon which we can call `forward`. 
`self.cross_unit[i-1][tn][j]*ss_rep[i-1][j]` takes the cross_stitch from the previous layer. Therefore, cross stitch units start after the first conv layer and end before the 4th block. 

* This is a very biased implementation. According to the [paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf) Section **5**, it's better to initialize networks pre-trained for each task specifically. Here, the ImageNet pre-trained res-net is used which is exactly the opposite of what paper suggests. 

* Apart from being biased, it is only viable for the resnet architecture. 

* In resnet, we have 1 convolution layer with maxpooling and then several (4) blocks with the following:
```
self.conv1 = conv3x3(inplanes, planes, stride)
self.bn1 = norm_layer(planes)
self.relu = nn.ReLU(inplace=True)
self.conv2 = conv3x3(planes, planes)
self.bn2 = norm_layer(planes)
self.downsample = downsample
self.stride = stride
```

* A single stich unit is used `self.cross_unit = nn.Parameter(torch.ones(4, self.task_num, self.task_num))` for each encoder; not for each channel. 