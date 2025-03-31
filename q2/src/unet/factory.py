from enum import Enum


from src.unet.unet import UNet
from src.unet.vanilla import VanillaUNet
from src.unet.noskip import NoSkipUNet
from src.unet.residual import ResidualUNet


class Variant(Enum):
    Vanilla = "vanilla"
    NoSkip = "noskip"
    Residual = "residual"


def fetch_unet(variant: Variant) -> type[UNet]:
    match variant:
        case Variant.Vanilla:
            return VanillaUNet
        case Variant.NoSkip:
            return NoSkipUNet
        case Variant.Residual:
            return ResidualUNet
