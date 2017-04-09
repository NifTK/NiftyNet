from network.base_layer import BaseLayer


class NetTemplate(BaseLayer):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training,
                 device_str):
        super(NetTemplate, self).__init__(is_training, device_str)
        self.batch_size = batch_size
        self.input_image_size = image_size
        self.input_label_size = label_size
        self.num_classes = num_classes
        self.name = "optional network name"

    def set_input_size(self, batch_size, image_size, label_size, num_classes):
        self.batch_size = batch_size
        self.input_image_size = image_size
        self.input_label_size = label_size
        self.num_classes = num_classes

    # images: [batch, width, height, depth, feature]
    def inference(self, images, layer_id=None):
        raise NotImplementedError
