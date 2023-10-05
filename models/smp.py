from torch import nn
from typing import Optional

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.deeplabv3.decoder import (
    DeepLabV3Decoder,
    DeepLabV3PlusDecoder,
)

import segmentation_models_pytorch as smp

########### 1 правка: segmentation_models_pytorch.encoders.resnet.ResNetEncoder ###########
class MyResNetEncoder(smp.encoders.resnet.ResNetEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l1 = nn.Identity()  # Не использовать необявленные переменные

    def forward(self, x):
        stages = self.get_stages()

        features = []
        # Не использовать индексирование в форе
        # for i in range(self._depth + 1):
        #     x = stages[i](x)
        #     features.append(x)
        x = self.l1(x)
        features.append(x)
        # Не использовать неинициализированные компоненты сети меняющие обращение к блокам (model.conv1 на model.L1[0].conv1)
        # nn.Sequential(self.conv1, self.bn1, self.relu),
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


########### 2 правка: segmentation_models_pytorch.base.SegmentationModel ###########
class MySegmentationModel(smp.base.SegmentationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # Не использавать логические конструкции в forward
        # self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


########### 3 правка: segmentation_models_pytorch.encoders.__init__.py ###########
from segmentation_models_pytorch.encoders import *  ########### чтобы не вытаскавать все ###########

resnet_encoders = {
    "resnet34": {
        "encoder": MyResNetEncoder,  ########### Нужно вернуть MyResNetEncoder ###########
        "pretrained_settings": smp.encoders.resnet.pretrained_settings["resnet34"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": smp.encoders.resnet.BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
}
encoders.update(
    resnet_encoders
)  ########### Добавить resnet_encoders с MyResNetEncoder в encoders
# Переинициализация ради замены resnet_encoders
def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs,
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError(
            "Wrong encoder name `{}`, supported encoders: {}".format(
                name, list(encoders.keys())
            )
        )

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


########### 4 правка: Наследуемся от MySegmentationModel ###########
class smpDeepLabV3(MySegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=8,
        )

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None


class smpDeepLabV3Plus(MySegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(
                    encoder_output_stride
                )
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
