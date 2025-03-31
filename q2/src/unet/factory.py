from enum import Enum


from src.unet.unet import UNet
from src.unet.vanilla import VanillaUNet


class Variant(Enum):
    Vanilla = "vanilla"


def fetch_unet(variant: Variant) -> type[UNet]:
    match variant:
        case Variant.Vanilla:
            return VanillaUNet
