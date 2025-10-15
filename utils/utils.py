from typing import List, Tuple, Optional, Union
import numpy as np
from PIL.Image import Image
from pathlib import Path
from typing import Union
from PIL import Image
import cv2




def open_image(image: Union[str, Path, Image.Image, np.ndarray], show: bool = False) -> Image.Image:
    """
    Load an image (from a path, OpenCV array, or PIL.Image) and return a PIL.Image in RGB.

    Args:
        image: 
        - Path or str: will be read via cv2.imread (fast C++ loader).
        - PIL.Image: will be converted to RGB.
        - np.ndarray: assumed to be an H×W×3 array (BGR or RGB—converted to RGB).
        show: If True, the image will be displayed using PIL's show() method.

    Returns:
        A PIL.Image in RGB mode.

    Raises:
        FileNotFoundError: if a path is given but the file doesn't exist or fails to load.
        ValueError: if the supplied object isn't a supported type.
    """
    # 1) Path or string → use OpenCV
    if isinstance(image, (str, Path)):
        path = str(image)
        arr = cv2.imread(path)
        if arr is None:
            raise FileNotFoundError(f"Image file not found or unreadable: '{path}'")
        # OpenCV gives BGR; convert to RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(arr)

    # 2) PIL.Image → ensure RGB
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    # 3) NumPy array → assume H×W×3, may be BGR or RGB
    elif isinstance(image, np.ndarray):
        arr = image
        # If dtype is uint8 and values seem in BGR order, we still convert
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Convert BGR→RGB
            arr = arr[:, :, ::-1]
        image = Image.fromarray(arr.astype("uint8")).convert("RGB")

    else:
        raise ValueError(
            f"Unsupported image type: {type(image)}. "
            "Expected str/Path, PIL.Image, or np.ndarray."
        )

    # If show is True, display the image
    if show:
        image.show()
    
    return image