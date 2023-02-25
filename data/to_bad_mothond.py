

from data.datachange import apply_mask, get_mask


class mri_mask():
    def __init__(self, mask_path, mask_type, mask_level):
        self.mask = get_mask(mask_path, mask_type, mask_level)
    def tobad(self,input):
        return apply_mask(input,mask=self.mask)