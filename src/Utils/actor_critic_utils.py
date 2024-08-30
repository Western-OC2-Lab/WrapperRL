import numpy as np
from torch import FloatTensor
from torch.autograd import Variable
import torch
import torch.nn.functional as F

class Utils:
    
    def __init__(self, img_dims):
        self.img_dims = img_dims
    
    def numpy_array_to_torch_tensor(self, numpy_array, dtype = np.float32, tensor_type = FloatTensor):
        """
        This function converts a numpy array to a pytorch tensor
        :param numpy_array (obj:`numpy array`): the numpy array to be converted
        :param dtype (obj:`numpy float type`): the dtype of the numpy array
        :param tensor_type (obj:`Pytorch Tensor`): the type of the final output tensor
        """

        if numpy_array.dtype != dtype:
            numpy_array = numpy_array.astype(dtype)
        return Variable(torch.from_numpy(numpy_array).type(tensor_type))
    
    def action_feature_map(self, feature_map, action_plans, type_ofc = 'zero', interpolate=False):
        if interpolate:
            w, h = feature_map.size()
            w_img, h_img = self.img_dims[0], self.img_dims[1]
            cls_w = F.interpolate(feature_map.view(1, 1, w, h), (w_img, h_img), mode = 'bilinear').view(w_img, h_img, 1)
        else:
            cls_w = feature_map
        
        r_weights = torch.zeros_like(cls_w)
        if type_ofc == 'zero':
            for a_plan in action_plans:
                a_plan = a_plan[0]
                x_plan, w_plan, y_plan, h_plan = int(a_plan[0].item()), int(a_plan[1].item()), int(a_plan[2].item()), int(a_plan[3].item())
                r_weights[:, :, x_plan:x_plan+w_plan, y_plan:y_plan+h_plan] = cls_w[:, :, x_plan:x_plan+w_plan, y_plan:y_plan+h_plan]

        return r_weights, cls_w
    

    def deselect_region(self, feature_map, set_actions, type_ofc = 'zero', interpolate=False):
        if interpolate:
            w, h = feature_map.size()
            w_img, h_img = self.img_dims[0], self.img_dims[1]
            cls_w = F.interpolate(feature_map.view(1, 1, w, h), (w_img, h_img), mode = 'bilinear').view(1, w_img,h_img)
        else:
            cls_w = feature_map
        
        r_weights = cls_w.clone()
        if type_ofc == 'zero':
            for a_plan in set_actions:
                x_plan, w_plan, y_plan, h_plan = int(a_plan[0]), int(a_plan[1]), int(a_plan[2]), int(a_plan[3])
                r_weights[:, :,  y_plan:y_plan+h_plan,x_plan:x_plan+w_plan] = 0

        return r_weights, cls_w
    
    def select_region(self, input_image, set_actions, type_ofc = 'zero', interpolate=False):
        # r_weights = cls_w.clone()
        # print(input_image.size())
        if len(input_image.size()) < 4:
            input_image = input_image[None, :]
        input_full = input_image.clone()
        r_weights = torch.zeros_like(input_full)
        if type_ofc == 'zero':
            for a_plan in set_actions:
                x_plan, w_plan, y_plan, h_plan = int(a_plan[0]), int(a_plan[1]), int(a_plan[2]), int(a_plan[3])
                # r_weights[:, :,  y_plan:y_plan+h_plan,x_plan:x_plan+w_plan] = 0
                r_weights[:, :,  y_plan:y_plan+h_plan,x_plan:x_plan+w_plan] = input_full[:, :, y_plan:y_plan+h_plan,x_plan:x_plan+w_plan]

        return r_weights, input_full
    
    def soft_update(self, target, source, tau):

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


    def hard_update(self, target, source):

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
