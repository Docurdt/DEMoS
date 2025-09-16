import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F

from statistics import mode, mean

class SaveValues():
    """
    Captures and stores activations and gradients from a specified neural network module.

    This class registers hooks to a given module to save the output of the forward pass and the gradients
    during the backward pass, allowing for analysis of the model's behavior.

    Args:
        m (torch.nn.Module): The neural network module from which to capture activations and gradients.

    Attributes:
        activations (Tensor): The output of the forward pass.
        gradients (Tensor): The gradients from the backward pass.
    """
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Initializes the CAM object with a model and target layer.

        This class computes the Class Activation Mapping for a given model and target layer, which is useful
        for visualizing the regions of an image that contribute to the model's predictions.

        Args:
            model (torch.nn.Module): The base model to get CAM from, which should have a global pooling and fully connected layer.
            target_layer (torch.nn.Module): The convolutional layer before Global Average Pooling.
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)

    def forward(self, x, idx=None):
        """
        Computes the Class Activation Mapping for the input image.

        This method performs a forward pass through the model, calculates the probabilities, and generates
        the CAM for the predicted class.

        Args:
            x (Tensor): The input image with shape (1, 3, H, W).
            idx (int, optional): The index of the predicted class. If None, it will be determined from the model's output.

        Returns:
            Tuple[Tensor, int]: The CAM for the predicted class and the predicted class index.
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data

        cam = self.getCAM(self.values, weight_fc, idx)

        return cam, idx

    def __call__(self, x):
        """
        Calls the forward method to compute the CAM.

        Args:
            x (Tensor): The input image with shape (1, 3, H, W).

        Returns:
            Tuple[Tensor, int]: The CAM for the predicted class and the predicted class index.
        """

    def getCAM(self, values, weight_fc, idx):
        """
        Generates the Class Activation Map using the activations and weights.

        This method computes the CAM based on the activations from the target layer and the weights from
        the fully connected layer.

        Args:
            values (SaveValues): The activations and gradients of the target layer.
            weight_fc (Tensor): The weights of the fully connected layer.
            idx (int): The predicted class index.

        Returns:
            Tensor: The Class Activation Map for the predicted class.
        """

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data


class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Initializes the GradCAM object with a model and target layer.

        This class extends the CAM class to compute Grad-CAM, which uses gradients to enhance the class
        activation mapping visualization.

        Args:
            model (torch.nn.Module): The base model to get CAM from, which need not have a global pooling and fully connected layer.
            target_layer (torch.nn.Module): The convolutional layer to visualize.
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class GradCAMpp(CAM):
    """ Grad CAM plus plus """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAMpp(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[0, idx].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class SmoothGradCAMpp(CAM):
    """ Smooth Grad CAM plus plus """

    def __init__(self, model, target_layer, n_samples=25, stdev_spread=0.15):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
            n_sample: the number of samples
            stdev_spread: standard deviationß
        """

        self.n_samples = n_samples
        self.stdev_spread = stdev_spread

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        stdev = self.stdev_spread / (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev

        indices = []
        probs = []

        for i in range(self.n_samples):
            self.model.zero_grad()

            x_with_noise = torch.normal(mean=x, std=std_tensor)
            x_with_noise.requires_grad_()

            score = self.model(x_with_noise)

            prob = F.softmax(score, dim=1)

            if idx is None:
                prob, idx = torch.max(prob, dim=1)
                idx = idx.item()
                probs.append(prob.item())

            indices.append(idx)

            score[0, idx].backward(retain_graph=True)

            activations = self.values.activations
            gradients = self.values.gradients
            n, c, _, _ = gradients.shape

            # calculate alpha
            numerator = gradients.pow(2)
            denominator = 2 * gradients.pow(2)
            ag = activations * gradients.pow(3)
            denominator += \
                ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator))
            alpha = numerator / (denominator + 1e-7)

            relu_grad = F.relu(score[0, idx].exp() * gradients)
            weights = \
                (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

            # shape => (1, 1, H', W')
            cam = (weights * activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)

            if i == 0:
                total_cams = cam.clone()
            else:
                total_cams += cam

        total_cams /= self.n_samples
        idx = mode(indices)
        prob = mean(probs)

        print("predicted class ids {}\t probability {}".format(idx, prob))

        return total_cams.data, idx

    def __call__(self, x):
        return self.forward(x)
