import torch
from torch import nn
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        stride = 1 if inplanes == planes else 2
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None if inplanes == planes else nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1,512)
        x = self.fc(x)

        return x
    
def replace_strides_with_dilation(module, dilation_rate):
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)
            
class resnet34_encoder(resnet34):
    def __init__(self):
        super().__init__()
        del self.avgpool
        del self.fc
        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.layer_1 = nn.Sequential(self.maxpool, self.layer1)
        
    def forward(self, x):
        features = []
        
        x = self.layer0(x)
        features.append(x)
        x = self.layer_1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features
    
class NeckFPN(nn.Module):
    def __init__(self, inputs = [512, 256, 128, 64, 64], out = 128):
        super().__init__()
        self.out = out
        self.convs1x1 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(i,self.out,kernel_size=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.out)
                ) for i in inputs
            ])
        
        self.ups = nn.ModuleList([ nn.Upsample(mode="nearest", scale_factor=2) for i in range(len(inputs) - 1) ])
        
        self.act = nn.ReLU(inplace=True)
        self.convs3x3 =nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.out,self.out,kernel_size=3,padding=1,bias=False),
                    nn.BatchNorm2d(self.out)
                ) for i in range(len(inputs) - 1)
            ])
        
    def forward(self, input_features):
        output_features = []
        output = self.convs1x1[0](input_features[0])
        output_features.append(output)
        for skip_input, conv1x1, conv3x3, upsample in zip(
            input_features[1:], self.convs1x1[1:], self.convs3x3, self.ups
        ):
            output = upsample(output)
            skip_input = conv1x1(skip_input)
            output = output + skip_input
            output = self.act(output)
            skip_output = conv3x3(output)
            output_features.append(skip_output)
        return output_features

class YOLOLayer(nn.Module):
    def __init__(
        self,
        anchors,
        n_classes,
        conf_thresholds,
        image_size,
        stride,
        export_to_platform,
        apply_postprocess,
    ):
        super(YOLOLayer, self).__init__()
        self.anchors = (
            anchors  # e.g. shape is [3, 2], in size of pixels of original image
        )
        self.stride = stride  # layer stride
        self.n_anchors = len(anchors)  # (3)
        self.n_classes = n_classes  # (80)
        self.conf_thresholds = conf_thresholds
        self.n_outputs = n_classes + 5  # (85)
        self.nx, self.ny, self.n_grid = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = (
            self.anchors / self.stride
        )  # this rescales anchors (e.g. of shape [3, 2]) down to feature map size
        self.anchor_wh = self.anchor_vec.view(
            1, self.n_anchors, 1, 1, 2
        )  # just a reshape
        self.sigmoid = nn.Sigmoid()  # must be a module, for converter
        self.export_to_platform = export_to_platform
        self.image_size = image_size
        self.apply_postprocess = apply_postprocess

        if self.export_to_platform:
            self.training = False
            self.create_grids(
                (self.image_size[1] // stride, self.image_size[0] // stride)
            )  # number x, y grid points

    def create_grids(self, n_grid=(13, 13), device="cpu"):
        self.nx, self.ny = n_grid  # x and y grid size
        self.n_grid = torch.tensor(n_grid, dtype=torch.float)

        # build xy offsets
        y_grid, x_grid = torch.meshgrid(
            [
                torch.arange(self.ny, device=device),
                torch.arange(self.nx, device=device),
            ],
            indexing="ij",
        )
        # grid of coordinates of "pixels" of a feature map
        self.grid = (
            torch.stack((x_grid, y_grid), 2).view((1, 1, self.ny, self.nx, 2)).float()
        )  # shape is [1, 1, ny, nx, 2]

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def postprocess(self, output: torch.Tensor):
        batch_size, _, ny, nx = output.shape
        # print(output.shape)
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), output.device)

        output = output.view(
            batch_size, self.n_outputs, self.n_anchors, self.ny, self.nx
        )
        output = output.permute(0, 2, 3, 4, 1).contiguous()

        # calculate xywh
        output[..., :2] = output[..., :2] * 2.0 - 0.5 + self.grid  # xy
        output[..., 2:4] = (output[..., 2:4] * 2) ** 2 * self.anchor_wh  # wh
        output[..., :4] *= self.stride  # resize to pixels of original image

        output = output.view(batch_size, -1, self.n_outputs)

        # xywh -> xyxy
        boxes = output[..., :4].clone()
        boxes[..., 0] = output[..., 0] - output[..., 2] / 2  # x1
        boxes[..., 1] = output[..., 1] - output[..., 3] / 2  # y1
        boxes[..., 2] = output[..., 0] + output[..., 2] / 2  # x2
        boxes[..., 3] = output[..., 1] + output[..., 3] / 2  # y2

        # clip
        image_height, image_width = self.image_size
        boxes[..., 0] = torch.clamp(boxes[..., 0], min=0, max=image_width - 1)
        boxes[..., 1] = torch.clamp(boxes[..., 1], min=0, max=image_height - 1)
        boxes[..., 2] = torch.clamp(boxes[..., 2], min=0, max=image_width - 1)
        boxes[..., 3] = torch.clamp(boxes[..., 3], min=0, max=image_height - 1)

        # final class confidences
        if self.n_classes > 1:
            confs = output[..., 5:] * output[..., 4:5]
        else:
            confs = output[..., 4:5]

        # keep only max classes
        if self.n_classes > 1:
            max_conf, max_idx = confs.max(dim=-1, keepdim=True)
            output = torch.cat((boxes, max_conf, max_idx.float()), dim=2)
        else:
            output = torch.cat((boxes, confs, torch.zeros_like(confs)), dim=2)

        # per class confidence thresholding
        output_filtered = []
        for output_i in output:
            image_output_filtered = []
            for cls_idx, cls_conf_threshold in enumerate(self.conf_thresholds):
                boxes_cls = output_i[output_i[:, 5] == cls_idx]
                boxes_cls = boxes_cls[boxes_cls[:, 4] > cls_conf_threshold]
                image_output_filtered.append(boxes_cls)
            image_output_filtered = torch.cat(image_output_filtered, dim=0)
            output_filtered.append(image_output_filtered)
        return output_filtered

    def forward(
        self, output_raw: torch.Tensor
    ):
        #1111111111111111111111111111111111111111111111111111111111111111111111
        # if self.training or not self.apply_postprocess:
        #     return output_raw
        return output_raw
        #1111111111111111111111111111111111111111111111111111111111111111111111
    
        output = self.sigmoid(output_raw)

        if self.export_to_platform:
            # everything else is done in postprocessing layer in hardware
            return output

        # List[Tensor[-1, 6]]
        output = self.postprocess(output)

        return output_raw, output

def nms_one_class(boxes, scores, iou_threshold):

    keep = torch.zeros_like(scores, dtype=torch.int64)
    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = torch.mul(x2 - x1, y2 - y1)

    order = torch.argsort(scores, dim=0, descending=True)
    # or argmax in a loop if sorting is problematic

    nboxes = boxes.size(0)

    suppressed = torch.zeros_like(scores, dtype=torch.uint8)

    num_to_keep = 0
    print(7, nboxes)

    for _i in range(nboxes):
        i = order[_i]  # best box index
        if suppressed[i] == 1:
            continue

        keep[num_to_keep] = i
        num_to_keep += 1

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        print(8,_i + 1, nboxes)

        for _j in range(_i + 1, nboxes):
            j = order[_j]  # another box index

            if suppressed[j] == 1:
                continue

            # for intersection
            xx1 = torch.max(ix1, x1[j])
            yy1 = torch.max(iy1, y1[j])
            xx2 = torch.min(ix2, x2[j])
            yy2 = torch.min(iy2, y2[j])

            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)

            inter = w * h
            union = iarea + areas[j] - inter
            iou = inter / union
            if iou > iou_threshold:
                suppressed[j] = 1

    keep = keep[:num_to_keep]
    return keep


def nms_layer(detections, class_num, iou_threshold=0.65, top_k=400):
    boxes = detections[:, :4]
    scores = detections[:, 4]  # confidence scores

    if class_num > 0:  # has classes column in detections
        classes = detections[:, 5]
        has_class_column = True
    else:  # class_num is 0
        class_num = 1
        has_class_column = False

    # fixed size arrays for outputs
    if has_class_column:
        output = torch.full(
            (top_k, 6), fill_value=-1, dtype=torch.float32, device=boxes.device
        )
    else:
        output = torch.full(
            (top_k, 5), fill_value=-1, dtype=torch.float32, device=boxes.device
        )

    if boxes.size(0) == 0:
        return output

    keep_indices = torch.zeros_like(scores, dtype=torch.int64)
    box_count = 0

    # loop over classes
    for class_id in range(class_num):
        if class_num > 1:
            # selecting boxes of a class
            box_indices_of_class = torch.where(classes == class_id)[0]
            if box_indices_of_class.size(0) == 0:
                continue
            class_boxes = boxes[box_indices_of_class]
            class_scores = scores[box_indices_of_class]
        else:
            class_boxes = boxes
            class_scores = scores

        class_keep_indices = nms_one_class(class_boxes, class_scores, iou_threshold)
        num_class_boxes = class_keep_indices.size(0)
        if num_class_boxes == 0:
            continue

        # add class indices to the array of indices
        if class_num > 1:
            keep_indices[
                box_count : box_count + num_class_boxes
            ] = box_indices_of_class[class_keep_indices]
        else:
            keep_indices[:num_class_boxes] = class_keep_indices
        box_count += num_class_boxes

    if box_count == 0:
        return output

    sort_order = scores[keep_indices[:box_count]].argsort(descending=True)
    keep_indices[:box_count] = keep_indices[sort_order]  # sorting over all classes

    if box_count > top_k:  # limit detections
        box_count = top_k

    output[:box_count, :4] = boxes[keep_indices[:box_count]]
    output[:box_count, 4] = scores[keep_indices[:box_count]]
    if has_class_column:
        output[:box_count, 5] = classes[keep_indices[:box_count]]

    return output


class HeadDetection(nn.Module):
    def __init__(
        self,
        n_classes,
        neck_pyramid_channels,
        image_size = [224, 224],
        n_levels  = 5,
        conf_thresholds = 0.5,
        head_anchors = [[[224, 112],[112, 224],[112, 112]],
                       [[112, 64],[64, 112],[64, 64]],
                       [[64, 32],[32, 64],[32, 32]],
                       [[32, 16],[16, 16],[20, 16]],
                       [[16, 8],[8, 16],[8, 8]]],
        nms_iou_threshold = 0.65,
        nms_top_k = 10,
        use_bias = False,
        export_to_platform = False):
        
        super().__init__()
        self.levels = n_levels
        self.in_channels = neck_pyramid_channels
        self.n_classes = n_classes if n_classes > 1 else 0
        self.conf_thresholds = [conf_thresholds for _ in range(n_classes)]
        self.strides = [2**(x+1) for x in range(n_levels)]
        self.anchors = torch.tensor(head_anchors)
        self.out_channels = self.anchors.shape[1] * (1 + self.n_classes + 4)
        self.iou_threshold = nms_iou_threshold
        self.top_k = nms_top_k
        self.use_bias = use_bias
        self.export_to_platform = export_to_platform
        self.image_size = image_size
        self.indices = [x for x in range(5)[::-1]]
        self.final_convs = nn.ModuleList([
            nn.Sequential(
                    nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,padding=1,bias=False),
                )
                for _ in self.strides
        ])
        self.apply_postprocess = True
        self.yolo_layers = nn.ModuleList([
                YOLOLayer(
                    anchors=anchors,
                    n_classes=self.n_classes,
                    conf_thresholds=self.conf_thresholds,
                    image_size=self.image_size,
                    stride=stride,
                    export_to_platform=self.export_to_platform,
                    apply_postprocess=self.apply_postprocess,
                )
                for stride, anchors in zip(self.strides, self.anchors)
            ])
    def forward(self, input_features):
        yolo_output = []
        for index, conv, yolo in zip(self.indices, self.final_convs, self.yolo_layers):
            x = input_features[index]
            x = conv(x)
            x = yolo(x)
            yolo_output.append(x)
        #1111111111111111111111111111111111111111111111111111111111111111111111
        # if self.training or not self.apply_postprocess:
        #     # before sigmoid
        #     return yolo_output
        return yolo_output
        #1111111111111111111111111111111111111111111111111111111111111111111111

        if self.export_to_platform:
            # after sigmoid
            return yolo_output

        # output_processed: n_heads x batch_size x tensor[n_boxes, 6]
        output_raw, output_processed = zip(*yolo_output)
        output_raw = list(output_raw)
        output_nms = []
        for output_image in zip(*output_processed):
            # concat output from all heads for one image: -> n_boxes, 6
            output_image = torch.cat(output_image, dim=0)
            output_image = nms_layer(
                output_image, self.n_classes, self.iou_threshold, self.top_k
            )  # top_k x 6
            output_nms.append(output_image)
        output_nms = torch.stack(output_nms)  # batch_size x top_k x 6
        return output_raw, output_nms

class Detect(nn.Module):
    def __init__(self, n_classes = 12, Yolo_imput = 128, image_size = [224,224]):
        super().__init__()
        self.encoder = resnet34_encoder()
        self.fpn = NeckFPN(out = Yolo_imput)
        self.head = HeadDetection(n_classes, neck_pyramid_channels = Yolo_imput)
    def forward(self, x):
        features = self.encoder(x)[::-1]
        features = self.fpn(features)
        output = self.head(features)
        return output